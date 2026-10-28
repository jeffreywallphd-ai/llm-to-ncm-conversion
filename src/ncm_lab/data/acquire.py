import json
from pathlib import Path
import urllib.request
from ncm_lab.runtime.io import root, read, write_json, digest


def manifest(dataset):
    path = root() / "data/manifests" / f"{dataset}.yaml"
    if not path.is_file():
        raise ValueError(f"no manifest registered for {dataset}")
    return read(path)


def fetch(dataset, split, max_bytes=None):
    spec = manifest(dataset)
    item = spec["splits"].get(split)
    if item is None:
        raise ValueError(f"{dataset} has no declared {split} split")
    if not item.get("url"):
        raise ValueError(item.get("blocker", "download URL has not been verified; inspect the source first"))
    cap = max_bytes if max_bytes is not None else spec.get("setup_download_budget_bytes", 0)
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError("supply a finite --max-bytes download/storage budget")
    if (item.get("size_bytes") or 0) > cap:
        raise ValueError(f"release requires {item['size_bytes']} bytes; budget is {cap}; no download started")
    dest = root() / "data/raw" / dataset / spec["revision"] / item["filename"]
    def verify(path):
        if item.get('size_bytes') is not None and path.stat().st_size!=item['size_bytes']:
            raise ValueError('release size mismatch')
        actual=digest(path)
        if item.get('sha256') and actual!=item['sha256']:
            raise ValueError('release hash mismatch')
        if item.get('git_blob_sha1'):
            from ncm_lab.data.download import verify_asset
            if verify_asset(path,{**item,'size_bytes':path.stat().st_size}) is None:
                raise ValueError('release Git blob mismatch')
        return actual
    if dest.exists():
        return {"path": str(dest), "sha256": verify(dest), "cached": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urllib.request.urlopen(item["url"], timeout=40) as response, tmp.open("wb") as out:
            used = 0
            while chunk := response.read(min(65536, cap - used + 1)):
                used += len(chunk)
                if used > cap:
                    raise ValueError("download byte budget exceeded; partial release removed")
                out.write(chunk)
        actual = verify(tmp)
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)
    receipt = {"dataset": dataset, "split": split, "revision": spec["revision"], "sha256": actual,
               "size_bytes": dest.stat().st_size, "source_url": item["url"], "license": spec["license"]}
    if dest.suffix=='.jsonl':
        with dest.open(encoding='utf-8') as stream:
            count=sum(1 for line in stream if line.strip())
        if item.get('expected_questions') is not None and count!=item['expected_questions']:
            raise ValueError('official release cardinality mismatch; acquired file retained for inspection')
        receipt.update(official_split=split,scope='full_official_split',official_split_record_count=count)
    write_json(dest.with_suffix(dest.suffix + ".receipt.json"), receipt)
    return {"path": str(dest), **receipt}


def training_sample(url, output, max_bytes=262144, count=2):
    """Stream only a small prefix of a public JSON array; never fetch test gold."""
    decoder, text, rows, consumed = json.JSONDecoder(), "", [], 0
    with urllib.request.urlopen(url, timeout=40) as response:
        while consumed < max_bytes and len(rows) < count:
            body = response.read(min(8192, max_bytes - consumed))
            if not body:
                break
            consumed += len(body)
            text += body.decode("utf-8")
            text = text.lstrip("\ufeff[\r\n\t ,")
            while text:
                try:
                    row, end = decoder.raw_decode(text)
                except json.JSONDecodeError:
                    break
                rows.append(row)
                text = text[end:].lstrip("\r\n\t ,")
                if len(rows) == count:
                    break
    if len(rows) < count:
        raise ValueError("could not parse requested training records within byte budget")
    write_json(output, rows)
    receipt = {"source_url": url, "split": "train", "records": len(rows), "bytes_read": consumed,
               "sha256": digest(output), "license": "MIT", "transformation": "first complete released JSON records",
               "usage": "software integration; not benchmark results"}
    write_json(Path(output).with_suffix(".provenance.json"), receipt)
    return receipt
