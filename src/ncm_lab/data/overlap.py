"""Gold-free, bounded source-prompt overlap audit for new adaptation data."""
import json
import math
from pathlib import Path
from ncm_lab.data.artifacts import verify_input
from ncm_lab.data.loaders import load_records,normalized,near_duplicate
from ncm_lab.runtime.io import digest,read,write_json


def audit(run,config,meter):
    threshold=config.get('threshold',.85)
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError('overlap threshold must be finite in (0,1]')
    training=verify_input(config['training_prompt_receipt'],'train','prompts')
    rows=[json.loads(line) for line in Path(training['path']).read_text(encoding='utf-8').splitlines() if line.strip()]
    if any(set(row)-{'question','group_id','original_id','split'} for row in rows):
        raise ValueError('overlap audit accepts allowlisted training prompts, never answers/programs')
    texts=[normalized(row['question']) for row in rows]
    def grams(text):
        words=text.split()
        return set(zip(words,words[1:],words[2:])) or {tuple(words)}
    inverted={}
    for i,text in enumerate(texts):
        for gram in grams(text):
            inverted.setdefault(gram,set()).add(i)
    matches,benchmarks=[],[]
    for receipt_path in config['benchmark_receipts']:
        receipt=verify_input(receipt_path)
        benchmarks.append({'receipt':receipt_path,'receipt_sha256':digest(receipt_path),
                           'data_sha256':receipt['sha256'],'schema':receipt['schema'],'split':receipt['split']})
        for item,_ in load_records(receipt['path'],receipt['schema'],receipt['split']):
            meter.check(step=False)
            # Only ModelInput reaches matching. Evaluator gold is never passed onward.
            text=normalized(item.question+' '+' '.join(item.context))
            candidates=set().union(*(inverted.get(gram,set()) for gram in grams(text)))
            for index in candidates:
                meter.check()
                if text==texts[index] or near_duplicate(text,texts[index],config.get('threshold',.85)):
                    matches.append({'training_id':rows[index]['original_id'],'benchmark_id':item.original_id,
                                    'dataset':item.dataset,'kind':'exact' if text==texts[index] else 'near'})
                meter.record(0,0)
    if not benchmarks:
        raise ValueError('a nonempty declared benchmark audit is required')
    result={'schema_version':1,'passed':not matches,'training_receipt':config['training_prompt_receipt'],
            'training_receipt_sha256':digest(config['training_prompt_receipt']),
            'training_data_sha256':training['sha256'],'training_original_ids':training['selected_ids'],
            'benchmarks':benchmarks,'matches':matches,'no_evaluator_gold_used':True,
            'rule':{'normalization':'word-normalized prompt plus permitted context','near_duplicate_trigram_jaccard':config.get('threshold',.85)},
            'limitation':'detects declared exact/near textual overlap; does not establish semantic disjointness'}
    write_json(run.path/'overlap_audit.json',result,exclusive=True)
    return {'status':'completed' if result['passed'] else 'failed','overlap_audit':result,'comparisons':meter.steps}


def validate_audit(path,benchmark_sha256=None):
    path=Path(path)
    audit=read(path)
    if not audit['passed'] or not audit['no_evaluator_gold_used']:
        raise ValueError('new adaptation data failed overlap audit')
    if not (path.parent/'manifest.json').is_file():
        raise ValueError('overlap audit must be a completed immutable data-audit run')
    from ncm_lab.runtime.provenance import validate_run
    manifest=validate_run(path.parent)
    if manifest['run_kind']!='overlap_audit' or manifest['status']!='completed':
        raise ValueError('overlap audit must be a completed immutable data-audit run')
    if digest(audit['training_receipt'])!=audit['training_receipt_sha256']:
        raise ValueError('overlap-audit training identity changed')
    verify_input(audit['training_receipt'],'train','prompts')
    for benchmark in audit['benchmarks']:
        if digest(benchmark['receipt'])!=benchmark['receipt_sha256']:
            raise ValueError('overlap-audit benchmark identity changed')
        verify_input(benchmark['receipt'])
    if benchmark_sha256 and not any(b['data_sha256']==benchmark_sha256 for b in audit['benchmarks']):
        raise ValueError('evaluated input was not covered by the adaptation overlap audit')
    return audit
