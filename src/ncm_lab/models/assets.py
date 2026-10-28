"""Cache-only validation of the actual pinned tensors/tokenizer bytes before loading."""
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
from ncm_lab.runtime.io import root,read
from ncm_lab.data.download import verify_asset


def verify_model_assets(config,roles,meter):
    lock=read(root()/'third_party/preparation_assets.lock.json')
    receipts=[]
    for role in roles:
        assets=[a for a in lock['assets'] if a['role']==role and a['repo_id']==config[role+'_id']
                and a['revision']==config[role+'_revision'] and a['filename'].endswith(('.json','.safetensors'))]
        if not assets:
            raise ValueError('model/tokenizer execution assets must have a pinned file-integrity lock')
        for asset in assets:
            if config.get(role+'_path'):
                path=Path(config[role+'_path'])/asset['filename']
            else:
                cached=try_to_load_from_cache(config[role+'_id'],asset['filename'],revision=config[role+'_revision'])
                if not isinstance(cached,str):
                    raise ValueError('required model asset is absent from cache')
                path=Path(cached)
            actual=verify_asset(path,asset,meter)
            if not actual:
                raise ValueError(f'missing/changed pinned model asset: {role}/{asset["filename"]}')
            receipts.append({'role':role,'filename':asset['filename'],'path':str(path),'sha256':actual,
                             'bytes':asset['size_bytes'],'revision':asset['revision']})
    return receipts
