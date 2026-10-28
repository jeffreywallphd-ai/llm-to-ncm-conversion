"""Recorded codec, transfer and cost tables. Missing measurements remain null."""
import csv
import json
from pathlib import Path
from ncm_lab.runtime.io import read, write_json, digest


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def analyze_codec_runs(paths, output, figures=False):
    paths=list(map(Path,paths))
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rates, curves, interventions, gaps, costs = [], [], [], [], []
    for base in paths:
        manifest = read(base/"manifest.json")
        origin = manifest["result_origin"]
        from ncm_lab.reporting.costs import cost_closure
        closure = cost_closure(base)
        costs.append({'run_id':manifest['run_id'],'origin':origin,**closure['total_once_per_dependency'],
                      'missing':closure['missing'],'dependency_count':len(closure['runs'])})
        for file in base.rglob("codec-selection.json"):
            selection = read(file)
            for observation in selection["candidates"]:
                for level, metrics in observation["metrics"].items():
                    rates.append({"run_id":manifest["run_id"],"origin":origin,"level":int(level),
                        "standalone_scalars":selection["rates"]["standalone_scalars"][level],
                        "all_level_scalars":selection["rates"]["all_levels_scalars"],
                        "candidate":observation["candidate"],"step":observation["step"],
                        "prediction_weight":observation["prediction_weight"],**metrics})
        for file in base.rglob("dev-*.json"):
            observation = read(file)
            if "dev_loss" not in observation:
                continue
            metrics = read(base/"metrics.json")
            curves.append({"run_id":manifest["run_id"],"origin":origin,"step":observation["step"],
                "development_loss":observation["dev_loss"],"valid_positions":observation["valid_positions"],
                "training_seed":manifest["seeds"]["training"],
                "actual_target_positions":observation.get("source_token_passes"),
                "distinct_source_positions":observation.get("distinct_source_positions"),
                "total_run_wall_seconds":metrics.get("elapsed_seconds"),
                "total_cost_missing_reason":"teacher/codec costs require joined dependency receipts"})
        for file in base.rglob("latent-diagnostics.json"):
            diagnostic = read(file)
            for level, intervention in diagnostic["interventions"].items():
                interventions.append({"run_id":manifest["run_id"],"origin":origin,"disabled_level":int(level),
                    "gate":diagnostic["gates"][level],'intervention':'bridge_removal',**intervention})
            for level,intervention in diagnostic.get('level_zeroing',{}).items():
                interventions.append({'run_id':manifest['run_id'],'origin':origin,'disabled_level':int(level),
                    'gate':diagnostic['gates'][level],'intervention':'zero_predicted_code',**intervention})
            for level, predicted in diagnostic["causal_predictions"].items():
                gaps.append({"run_id":manifest["run_id"],"origin":origin,"level":int(level),**predicted,
                    "oracle_normalized_mse":diagnostic["oracle_codec"][level]["normalized_mse"],
                    "interpretation":"oracle diagnostic and causal predictor; not a generation speed claim"})
    tables = {"rate_distortion":rates,"learning_curves":curves,"interventions":interventions,
              "oracle_predicted":gaps,"total_dependency_cost":costs}
    for name,rows in tables.items():
        write_csv(output/(name+".csv"),rows)
    write_json(output/"tables.json",tables)
    if figures and rates:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axis = plt.subplots()
        for origin in sorted({r["origin"] for r in rates}):
            values = [r for r in rates if r["origin"] == origin]
            axis.scatter([r["standalone_scalars"] for r in values],[r["normalized_mse"] for r in values],label=origin)
        axis.set(xlabel="Standalone latent scalars",ylabel="Held-out normalized state MSE")
        axis.legend()
        fig.savefig(output/"rate_distortion.png",dpi=160,bbox_inches="tight")
        plt.close(fig)
        for name,x,y in [('learning_curves','actual_target_positions','development_loss'),
                         ('interventions','disabled_level','max_logit_difference'),
                         ('oracle_predicted','oracle_normalized_mse','mse'),
                         ('total_dependency_cost','target_processed_positions','executor_wall_seconds')]:
            values=[r for r in tables[name] if r.get(x) is not None and r.get(y) is not None]
            if values:
                fig,axis=plt.subplots()
                axis.scatter([r[x] for r in values],[r[y] for r in values])
                axis.set(xlabel=x.replace('_',' '),ylabel=y.replace('_',' '),title='Recorded observations; inspect origin/scope')
                fig.savefig(output/(name+'.png'),dpi=160,bbox_inches='tight')
                plt.close(fig)
    from ncm_lab.runtime.provenance import implementation_hash
    receipt={'schema_version':1,'analysis':'codec_tables_v2','implementation_sha256':implementation_hash(),
        'inputs':[{'run_id':read(p/'manifest.json')['run_id'],'manifest_sha256':digest(p/'manifest.json')} for p in paths],
        'files':{p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='analysis_manifest.json'},
        'parameters':{'figures':figures},'research_claims':[]}
    write_json(output/'analysis_manifest.json',receipt)
    return {"output":str(output),"rows":{k:len(v) for k,v in tables.items()},"research_claims":[],
            'analysis_manifest':str(output/'analysis_manifest.json')}
