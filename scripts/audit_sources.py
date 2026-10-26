"""Bounded metadata/code audit. No weights, training, or dataset releases."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.cache' / 'audit'
DEST.mkdir(parents=True, exist_ok=True)
remaining = 12_000_000


def get(url, name):
    global remaining
    request = urllib.request.Request(url, headers={'User-Agent': 'ncm-lab-source-audit'})
    with urllib.request.urlopen(request, timeout=40) as response:
        body = response.read(min(remaining, 2_000_000) + 1)
    if len(body) > min(remaining, 2_000_000):
        raise ValueError('metadata audit byte budget exceeded')
    remaining -= len(body)
    (DEST / name).write_bytes(body)
    return body


if __name__ == '__main__':
    lock = {'schema_version': 1, 'audit_date': '2026-10-25', 'sources': {}}
    repos = {
        'dflash': 'z-lab/dflash', 'dcore': 'ernie-research/DCore',
        'ncm': 'HazyResearch/ncm', 'longspark': 'Hao-Yuan-He/LongSpark',
        'finqa': 'czyssrs/FinQA', 'tatqa': 'NExTplusplus/TAT-QA',
        'gsm8k': 'openai/grade-school-math', 'evalplus': 'evalplus/evalplus',
        'longbench': 'THUDM/LongBench',
    }
    for key, repo in repos.items():
        try:
            commit = json.loads(get(f'https://api.github.com/repos/{repo}/commits/HEAD', f'{key}-commit.json'))['sha']
            tree = json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1', f'{key}-tree.json'))
            paths = [x['path'] for x in tree['tree'] if x['type'] == 'blob']
            selected = [p for p in paths if p.lower() in ('readme.md', 'license', 'license.txt', 'pyproject.toml')]
            if key == 'dflash':
                selected += [p for p in paths if p.endswith('.py') and ('model' in p or 'train' in p or 'generate' in p)]
            if key == 'finqa':
                selected += [p for p in paths if p in ('code/evaluate/evaluate.py', 'code/evaluate/general_utils.py', 'dataset/train.json')]
            if key == 'tatqa':
                selected += [p for p in paths if p in ('tatqa_metric.py', 'tatqa_utils.py')]
            if key == 'dcore':
                selected += [p for p in paths if 'dflash' in p.lower() and (p.endswith('.json') or p.endswith('.yaml') or p.endswith('.py'))][:9]
            files = {}
            for p in dict.fromkeys(selected):
                try:
                    body = get(f'https://raw.githubusercontent.com/{repo}/{commit}/{p}', key + '--' + p.replace('/', '__'))
                    files[p] = hashlib.sha256(body).hexdigest()
                except Exception as exc:
                    files[p] = {'error': str(exc)}
            lock['sources'][key] = {'url': f'https://github.com/{repo}', 'revision': commit, 'files': files}
            print(key, commit, flush=True)
        except Exception as exc:
            lock['sources'][key] = {'url': f'https://github.com/{repo}', 'blocker': str(exc)}
            print(key, str(exc), flush=True)
    for key, model in {'target': 'Qwen/Qwen3-8B', 'drafter': 'z-lab/Qwen3-8B-DFlash-b16', 'dcore_drafter': 'ernie-research/DCore-DFlash-Qwen3-8B-Budget16', 'oasst1': 'datasets/OpenAssistant/oasst1', 'docfinqa': 'datasets/kensho/DocFinQA'}.items():
        try:
            info = json.loads(get(f'https://huggingface.co/api/{"datasets" if model.startswith("datasets/") else "models"}/{model.removeprefix("datasets/")}', f'{key}-info.json'))
            rev = info['sha']
            files = {}
            for p in ('config.json', 'README.md'):
                try:
                    body = get(f'https://huggingface.co/{model}/resolve/{rev}/{p}', f'{key}--{p}')
                    files[p] = hashlib.sha256(body).hexdigest()
                except Exception as exc:
                    files[p] = {'error': str(exc)}
            lock['sources'][key] = {'url': f'https://huggingface.co/{model}', 'revision': rev, 'files': files}
            print(key, rev, flush=True)
        except Exception as exc:
            lock['sources'][key] = {'blocker': str(exc)}
    (DEST / 'sources.lock.json').write_text(json.dumps(lock, indent=2) + '\n', encoding='utf-8')
