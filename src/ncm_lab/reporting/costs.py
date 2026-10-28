"""Join actual immutable dependency costs once; missing observations stay unknown."""
from pathlib import Path
from ncm_lab.runtime.io import read, digest
from ncm_lab.runtime.provenance import validate_run


def cost_closure(path):
    seen, rows, missing, preparations = set(), [], [], {}
    def visit(directory, expected=None):
        directory=Path(directory)
        manifest=validate_run(directory)
        if expected and digest(directory/'manifest.json') != expected:
            raise ValueError('cost dependency manifest changed')
        if manifest['run_id'] in seen:
            return
        seen.add(manifest['run_id'])
        metrics=read(directory/'metrics.json')
        if (directory/'preparation.json').exists():
            prepared=read(directory/'preparation.json')
            preparations[prepared['reference']['sha256']]=prepared['evidence'].get('resources',{})
        resources=metrics.get('resources',{})
        if not resources:
            missing.append({'run_id':manifest['run_id'],'reason':'legacy resource observations unavailable'})
        rows.append({'run_id':manifest['run_id'],'kind':manifest['run_kind'],
            'origin':manifest['result_origin'],'status':manifest['status'],**resources})
        dependencies=directory/'dependencies.json'
        if dependencies.exists():
            for dependency in read(dependencies):
                visit(dependency['path'],dependency['manifest_sha256'])
    visit(path)
    keys=['executor_wall_seconds','target_processed_positions','optimizer_steps','teacher_generated_tokens','storage_bytes']
    totals={key:sum(row[key] for row in rows) if all(row.get(key) is not None for row in rows) else None for key in keys}
    totals['asset_preparation_phase_wall_seconds']=(sum(p['preparation_phase_wall_seconds'] for p in preparations.values())
        if preparations and all(p.get('preparation_phase_wall_seconds') is not None for p in preparations.values()) else None)
    totals['charged_download_bytes']=(sum(p['charged_download_bytes'] for p in preparations.values())
        if preparations and all(p.get('charged_download_bytes') is not None for p in preparations.values()) else None)
    if not preparations:
        missing.append({'reason':'asset/data preparation reference not provided; no preparation cost is inferred'})
    return {'runs':rows,'preparations':preparations,'total_once_per_dependency':totals,'missing':missing,
            'amortization':'no implicit amortization; shared dependency charged once in each displayed closure'}
