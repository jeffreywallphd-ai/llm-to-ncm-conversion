"""Bounded HTTPS acquisition with hash checking and explicit range recovery.

Only pinned public URLs from the asset lock are consumed. No Hub code, pickle,
dataset script, model loader, or target generation is invoked here.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time
import urllib.request
from ncm_lab.runtime.io import canonical_hash, write_json


@dataclass(frozen=True)
class PreparationBudget:
    schema_version: int
    phase_budget_id: str
    max_download_bytes: int
    max_storage_bytes: int
    max_wall_seconds: int
    max_phase_wall_seconds: int
    reserve_free_bytes: int
    max_processed_bytes: int
    max_prompt_examples: int
    max_prompt_chars: int
    dev_fraction: float

    def __post_init__(self):
        for key,value in asdict(self).items():
            if key == "phase_budget_id":
                if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",value):
                    raise ValueError("preparation phase ID must be a safe nonempty identifier")
            elif isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite numeric limit")
            elif key not in {"reserve_free_bytes","dev_fraction"} and (not isinstance(value,int) or value <= 0):
                raise ValueError(f"{key} must be a positive integer")
        if self.schema_version != 1 or not 0 < self.dev_fraction < 1 or self.reserve_free_bytes < 0:
            raise ValueError("invalid preparation schema, reserve, or development fraction")
        if self.max_phase_wall_seconds < self.max_wall_seconds:
            raise ValueError("aggregate preparation time cap must cover an invocation")


def atomic_json(path,value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    write_json(temporary,value)
    temporary.replace(path)


def asset_path(base,asset):
    relative = PurePosixPath(asset["path"])
    if relative.is_absolute() or any(x in {"..","."} or ":" in x for x in relative.parts):
        raise ValueError("unsafe asset path in preparation lock")
    path = (base/Path(*relative.parts)).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("asset path escapes the selected asset directory")
    return path


class AcquisitionMeter:
    def __init__(self,base,budget,selection_hash):
        self.base,self.budget = Path(base),budget
        self.base.mkdir(parents=True,exist_ok=True)
        self.path = self.base/"preparations"/f"{budget.phase_budget_id}.ledger.json"
        self.path.parent.mkdir(parents=True,exist_ok=True)
        config_hash = canonical_hash(asdict(budget))
        self.state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {
            "schema_version":1,"budget_hash":config_hash,"selection_hash":selection_hash,
            "charged_download_bytes":0,"elapsed_seconds":0.0}
        if self.state["budget_hash"] != config_hash or self.state["selection_hash"] != selection_hash:
            raise ValueError("preparation phase budget/asset selection changed; select a deliberate new phase_budget_id")
        self.started = self.saved_at = time.monotonic()
        self.save()

    def check(self):
        now = time.monotonic()
        elapsed = now-self.started
        phase = self.state["elapsed_seconds"]+now-self.saved_at
        if elapsed >= self.budget.max_wall_seconds or phase >= self.budget.max_phase_wall_seconds:
            raise ValueError("preparation wall-time budget reached; partial files retained for recovery")

    def save(self):
        now = time.monotonic()
        self.state["elapsed_seconds"] += now-self.saved_at
        self.saved_at = now
        atomic_json(self.path,self.state)

    def charge(self,count):
        self.check()
        if self.state["charged_download_bytes"]+count > self.budget.max_download_bytes:
            raise ValueError("aggregate download budget exhausted; verified assets and partial files retained")
        # Persist before reading. Interrupted reads conservatively consume their reservation.
        self.state["charged_download_bytes"] += count
        self.save()


def verify_asset(path,asset,meter=None):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != asset["size_bytes"]:
        return None
    sha256,blob = hashlib.sha256(),hashlib.sha1(usedforsecurity=False)
    blob.update(f"blob {asset['size_bytes']}\0".encode())
    with path.open("rb") as stream:
        while body := stream.read(1024*1024):
            if meter:
                meter.check()
            sha256.update(body)
            blob.update(body)
    if asset.get("sha256") and sha256.hexdigest() != asset["sha256"]:
        return None
    if asset.get("git_blob_sha1") and blob.hexdigest() != asset["git_blob_sha1"]:
        return None
    if not asset.get("sha256") and not asset.get("git_blob_sha1"):
        raise ValueError("pinned asset requires a SHA256 or Git blob identity")
    return sha256.hexdigest()


def download_asset(base,asset,meter,opener=None,progress=None):
    opener = opener or urllib.request.urlopen
    dest = asset_path(Path(base),asset)
    actual = verify_asset(dest,asset,meter)
    if actual:
        return {"id":asset["id"],"path":str(dest),"sha256":actual,"size_bytes":asset["size_bytes"],"cached":True}
    if dest.exists():
        raise ValueError(f"cached asset is corrupt: {dest}; preserve/inspect it and remove that file before retrying")
    partial = dest.with_suffix(dest.suffix+".partial")
    dest.parent.mkdir(parents=True,exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset["size_bytes"]:
        raise ValueError(f"oversized partial asset: {partial}")
    if offset == asset["size_bytes"]:
        actual = verify_asset(partial,asset,meter)
        if not actual:
            raise ValueError(f"completed partial asset has wrong checksum: {partial}")
        partial.replace(dest)
        return {"id":asset["id"],"path":str(dest),"sha256":actual,"size_bytes":asset["size_bytes"],"cached":False}
    headers = {"User-Agent":"ncm-lab-prepare/0.1", "Accept-Encoding":"identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(asset["url"],headers=headers)
    meter.check()
    with opener(request,timeout=30) as response:
        status = getattr(response,"status",200)
        if status == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",response.headers.get("Content-Range",""))
            if not match or int(match[1]) != offset or int(match[3]) != asset["size_bytes"]:
                raise ValueError("server returned an incompatible resume range")
        elif status == 200:
            # A server may ignore Range. Restart explicitly; earlier transfers remain charged.
            offset = 0
        else:
            raise ValueError(f"unexpected download HTTP status: {status}")
        remaining = asset["size_bytes"]-offset
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != remaining:
            raise ValueError("remote content length differs from locked asset size")
        with partial.open("ab" if offset else "wb") as stream:
            last_progress = time.monotonic()
            while remaining:
                count = min(1024*1024,remaining)
                meter.charge(count)
                body = response.read(count)
                if not body:
                    raise ValueError("download interrupted before locked file size; partial retained")
                stream.write(body)
                remaining -= len(body)
                if progress and time.monotonic()-last_progress >= 5:
                    progress(f"  {asset['id']}: {(asset['size_bytes']-remaining)/1e6:.1f} / {asset['size_bytes']/1e6:.1f} MB")
                    last_progress = time.monotonic()
    actual = verify_asset(partial,asset,meter)
    if not actual:
        raise ValueError(f"download checksum mismatch: {partial}; partial retained for inspection")
    partial.replace(dest)
    return {"id":asset["id"],"path":str(dest),"sha256":actual,"size_bytes":asset["size_bytes"],"cached":False}
