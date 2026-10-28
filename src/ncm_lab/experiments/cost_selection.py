import statistics
import math
from dataclasses import replace
from pathlib import Path
from ncm_lab.runtime.io import read, digest, write_json
from ncm_lab.runtime.provenance import validate_run


def training_cost_budget(config,budget):
    path = Path(config["teacher_cost_run"])
    manifest = validate_run(path)
    if digest(path/"manifest.json") != config["teacher_cost_manifest_sha256"]:
        raise ValueError("teacher cost evidence changed")
    if manifest["run_kind"] != "teacher" or manifest["status"] != "completed" or manifest["result_origin"] != "measured":
        raise ValueError("total-cost matching requires real completed teacher evidence")
    from ncm_lab.reporting.costs import cost_closure
    rows={r['run_id']:r for r in cost_closure(path)['runs']}
    receipts=list(config.get('input_receipts',[]))
    for key in ('training_data_reference','development_data_reference'):
        if config.get(key):
            reference=read(config[key])
            if reference.get('input_receipt'):
                receipts.append(reference['input_receipt'])
    for receipt_path in receipts:
        directory=Path(read(receipt_path)['path']).parent
        if (directory/'manifest.json').exists():
            rows.update({r['run_id']:r for r in cost_closure(directory)['runs']})
    if config.get('codec_selection'):
        codec_path=Path(config['codec_selection']).parent
        codec_manifest=validate_run(codec_path)
        if codec_manifest['run_kind']!='codec' or codec_manifest['status']!='completed' or codec_manifest['result_origin']!='measured':
            raise ValueError('total-cost matching requires completed measured codec-search evidence')
        rows.update({r['run_id']:r for r in cost_closure(codec_path)['runs']})
    preparation=[r for r in rows.values() if r['kind'] in {'teacher','codec','latent_cache'}]
    if any(r['origin']!='measured' or r.get('executor_wall_seconds') is None for r in preparation):
        raise ValueError('complete measured preparation costs, including retries/resumes, are required')
    teacher=sum(r['executor_wall_seconds'] for r in preparation if r['kind']=='teacher')
    codec=sum(r['executor_wall_seconds'] for r in preparation if r['kind'] in {'codec','latent_cache'})
    total = config["total_new_training_wall_seconds"]
    remaining = min(budget.max_wall_seconds,total-teacher-codec)
    if remaining <= 0:
        raise ValueError("preparation exhausted the declared total training wall budget")
    return replace(budget,max_wall_seconds=remaining),{'teacher_seconds':teacher,'codec_search_seconds':codec,
        'charged_runs':[r['run_id'] for r in preparation],
        'total_wall_cap':total,'remaining_student_wall_cap':remaining,'units':'measured wall seconds on declared hardware',
        'limitation':'released pretraining cost excluded; unknown external preparation remains separately disclosed'}


def select_cost(run_paths, output, tolerance=.1, reference="HIER_SKIP"):
    if not 0 <= tolerance <= 1:
        raise ValueError("predeclare a finite relative cost tolerance in [0,1]")
    candidates, references, manifests = [], {}, []
    for path in map(Path,run_paths):
        run = validate_run(path)
        if run['status'] != 'completed':
            raise ValueError('cost selection requires completed development profiles')
        manifests.append({'run_id':run['run_id'],'manifest_sha256':digest(path/'manifest.json')})
        for row in read(path/'component_timings.json'):
            if not math.isfinite(row['seconds']) or row['seconds']<=0:
                raise ValueError('development timing must have positive high-resolution observations')
            key=(row['batch'],row['prefix_length'])
            if row['method']==reference:
                references.setdefault(key,[]).append(row['seconds'])
            if row['method']=='FLAT_COST':
                candidates.append(row)
    if not candidates or not references:
        raise ValueError('profile actual reduced-depth candidates and the declared reference first')
    grouped={}
    for row in candidates:
        layers=tuple(row['resolved_method']['layer_indices'])
        key=(row['batch'],row['prefix_length'])
        if key not in references:
            raise ValueError('unpaired cost-profile condition')
        grouped.setdefault(layers,[]).append(abs(row['seconds']/statistics.median(references[key])-1))
    matched=[(statistics.mean(values),layers) for layers,values in grouped.items() if statistics.mean(values)<=tolerance]
    if not matched:
        raise ValueError('no reduced-depth candidate meets the predeclared tolerance')
    error,layers=min(matched)
    result={'schema_version':1,'split':'dev','layer_indices':list(layers),'relative_cost_error':error,
        'tolerance':tolerance,'reference':reference,'observations':manifests,
        'weighting':'uniform recorded timing observations; component cost only',
        'performance_claim':False}
    write_json(output,result,exclusive=True)
    return result
