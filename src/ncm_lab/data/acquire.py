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
    if dest.exists():
        if item.get("sha256") and digest(dest) != item["sha256"]:
            raise ValueError("cached release hash mismatch")
        return {"path": str(dest), "sha256": digest(dest), "cached": True}
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
        actual = digest(tmp)
        if item.get("sha256") and actual != item["sha256"]:
            raise ValueError("downloaded release hash mismatch")
        tmp.rename(dest)
    finally:
        tmp.unlink(missing_ok=True)
    receipt = {"dataset": dataset, "split": split, "revision": spec["revision"], "sha256": actual,
               "size_bytes": dest.stat().st_size, "source_url": item["url"], "license": spec["license"]}
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
