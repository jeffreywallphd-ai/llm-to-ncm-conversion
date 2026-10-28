"""Content-bound consumed inputs. Receipt split is authoritative, never a launch label."""
import json
import hashlib
from pathlib import Path
from ncm_lab.runtime.io import digest, read, canonical_hash, write_json


def fixture_lineage(value):
    if isinstance(value,dict):
        return (value.get('fixture') is True or value.get('origin') in {'test_fixture','software_fixture','software_test_fixture'}
                or value.get('result_origin')=='test_fixture' or any(fixture_lineage(v) for v in value.values()))
    return isinstance(value,list) and any(fixture_lineage(v) for v in value)


def split_lineage(source,split,rows):
    official=source.get('official_split') or source.get('asset',{}).get('split') or source.get('split')
    policy=source.get('derived_split_policy')
    if official in {'dev','test'} and split=='train' or official=='test' and split=='dev':
        raise ValueError('official development/test data cannot be relabeled for fitting or selection')
    if official and official!=split and not policy and source.get('split')!=split:
        raise ValueError('derived split requires its explicit assignment policy')
    if policy:
        if policy.get('version')!='oasst_tree_hash_holdout_v1':
            raise ValueError('unsupported derived split assignment policy')
        fraction=policy['dev_fraction']
        for row in rows:
            group=row.get('group_id')
            if not group:
                raise ValueError('derived split requires complete group identities')
            expected='dev' if int(hashlib.sha256(group.encode()).hexdigest()[:8],16)/2**32<fraction else 'train'
            if expected!=split:
                raise ValueError('whole-group split assignment mismatch')
    return official,policy


def row_ids(rows, schema, split):
    if schema == "gsm8k":
        return [f"{split}:{i}" for i in range(len(rows))]
    if schema == "ifeval":
        return [str(r["key"]) for r in rows]
    if schema == 'humaneval_plus':
        return [r['task_id'] for r in rows]
    if schema == "longbench":
        return [str(r.get("_id", i)) for i, r in enumerate(rows)]
    return [r.get("original_id") for r in rows]


def lock_input(path, receipt, schema, split, family, source_receipt=None, fixture=False):
    path = Path(path).resolve()
    if not fixture and not source_receipt:
        raise ValueError("real inputs need an acquisition/transformation source receipt")
    if split not in {"train", "dev", "test"}:
        raise ValueError("invalid official split")
    source = read(source_receipt) if source_receipt else {"origin": "test_fixture"}
    fixture=fixture or fixture_lineage(source)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = row_ids(rows, schema, split)
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicated selected source IDs")
    if any(r.get("split", split) != split for r in rows):
        raise ValueError("data and receipt split disagree")
    official,policy=split_lineage(source,split,rows)
    value = {"schema_version": 1, "path": str(path), "sha256": digest(path), "schema": schema,
             "split": split, "family": family, "source": source, "fixture": fixture,
             'official_split':official,'derived_split_policy':policy,
             "selected_ids": ids, "order_sha256": canonical_hash(ids), "selection_rule": "explicit_file_order",
             "transformation_version": "response_records_v2" if schema == "teacher" else "source_release"}
    write_json(receipt, value, exclusive=True)
    return value


def verify_input(receipt_path, expected_split=None, expected_schema=None):
    receipt = read(receipt_path)
    if fixture_lineage(receipt['source']) and not receipt['fixture']:
        raise ValueError('fixture lineage cannot become research input')
    path = Path(receipt["path"])
    if digest(path) != receipt["sha256"]:
        raise ValueError("consumed data changed")
    if expected_split and receipt["split"] != expected_split or expected_schema and receipt["schema"] != expected_schema:
        raise ValueError("wrong input split/schema receipt")
    ids = receipt["selected_ids"]
    if len(ids) != len(set(ids)) or canonical_hash(ids) != receipt["order_sha256"]:
        raise ValueError("invalid selected IDs/order")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if row_ids(rows, receipt["schema"], receipt["split"]) != ids:
        raise ValueError("input rows disagree with selected IDs/order")
    if any(r.get("split", receipt["split"]) != receipt["split"] for r in rows):
        raise ValueError("input row split disagrees with receipt")
    split_lineage(receipt['source'],receipt['split'],rows)
    return receipt
