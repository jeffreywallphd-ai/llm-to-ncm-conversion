"""Snapshot committed target tensors and recover an interrupted Block8 target stage."""

from contextlib import closing
from itertools import batched
from pathlib import Path
import os
import shutil
import sqlite3
import time
import uuid

from ncm_lab.block8.runtime import Attempt, charge_write, checked, compatible_implementation, read_settings
from ncm_lab.runtime.budget import BudgetStop
from ncm_lab.runtime.canvas_pipeline import exclusive_writer
from ncm_lab.runtime.io import atomic_json, digest, read, reference, write_json

SCHEMA = "block8_target_backup_v1"


def stage_inputs(state):
    refs = {}
    for stage, key in (("prepare", "dataset"), ("teacher", "features"), ("codec", "codec")):
        entry = state["stages"].get(stage, {})
        if entry.get("status") != "completed":
            raise ValueError(f"target recovery requires a completed {stage} stage")
        receipt = read(checked(entry["receipt"]))
        if receipt["action"] != stage or receipt["status"] != "completed":
            raise ValueError(f"invalid {stage} receipt")
        refs[key] = receipt["result"][key]
        checked(refs[key])
    return refs


def _attempt_records(folder, settings):
    budget = dict(settings["stage_budgets"].get("targets", settings["budget"]))
    budget["phase_id"] = f"block8-{settings['identity'][:24]}-targets"
    expected = dict(settings, stage="block8_targets", budget=budget)
    ledger = folder / "ledgers" / f"{budget['phase_id']}.json"
    runs = read(ledger)["runs"]
    if len(runs) > settings["resources"]["max_attempts_per_stage"]:
        raise ValueError("too many target attempts in the phase ledger")
    for name in reversed(runs):
        attempt = Path(name).resolve()
        if attempt.parent != folder / "attempts" / "targets":
            raise ValueError("invalid target attempt path in the phase ledger")
        yield attempt / "resolved_config.json", expected


def _verify_attempt_config(path, expected):
    value = read(path)
    value.pop("executor_sha256", None)
    if value != expected:
        raise ValueError("target attempt configuration differs from its sealed experiment")


def _boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _open_index(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)


def _rows(db, max_parents):
    return db.execute(
        "SELECT parent,split,length,file,sha,dtype,width,parent_sha FROM tensors ORDER BY parent LIMIT ?",
        (max_parents + 1,),
    )


def _target_file(folder, key, name):
    if type(key) is not int or key < 0 or name != f"parent-{key:08d}.pt":
        raise ValueError("invalid target tensor filename in index")
    file = folder / name
    if file.is_symlink():
        raise ValueError("target tensor files must be ordinary files")
    return file


def backup_targets(manifest, backup):
    """Publish a SQLite-consistent snapshot after every indexed tensor verifies."""
    folder = manifest.parent
    state = read(manifest)
    settings = read_settings(state)
    inputs = stage_inputs(state)
    config, expected = next(_attempt_records(folder, settings))
    _verify_attempt_config(config, expected)
    source = config.parent / "cache"
    if folder == backup or folder in backup.parents:
        raise ValueError("choose a backup directory outside the live experiment")
    backup.mkdir(parents=True, exist_ok=True, mode=0o700)
    with exclusive_writer(backup / "backup.lock"):
        if len(list(backup.glob("index-*.sqlite"))) >= settings["resources"]["max_attempts_per_stage"]:
            raise BudgetStop("target backup reached its finite snapshot allowance")
        previous = backup / "manifest.json"
        minimum_parents = 1
        if previous.exists():
            prior = read(previous)
            producer = prior.get("source_attempt", str(Path(prior["source_config"]["manifest"]).parent))
            if prior["experiment"] != str(manifest) or producer != str(config.parent):
                raise ValueError("use a separate backup directory for a different target attempt")
            minimum_parents = prior["parents"]
        return _snapshot(manifest, backup, config, source, settings, inputs, minimum_parents)


def _snapshot(manifest, backup, config, source, settings, inputs, minimum_parents):
    cache = backup / "cache"
    cache.mkdir(exist_ok=True)
    index = backup / f"index-{uuid.uuid4().hex}.sqlite"
    max_parents = settings["data"]["max_parents"]
    budget = settings["stage_budgets"].get("targets", settings["budget"])
    prior_metadata = sum(p.stat().st_size for p in backup.glob("index-*.*"))
    max_bytes = min(budget["max_storage_bytes"], budget["max_phase_storage_bytes"] - prior_metadata)
    deadline = time.monotonic() + 7200

    def check(*_):
        if time.monotonic() >= deadline:
            raise BudgetStop("target backup exceeded its two-hour allowance")

    with closing(_open_index(source / "index.sqlite")) as origin, closing(sqlite3.connect(index)) as snapshot:
        origin.backup(snapshot, pages=256, progress=check)
        count = 0
        total = index.stat().st_size
        for key, _, _, name, checksum, _, _, _ in _rows(snapshot, max_parents):
            check()
            count += 1
            original = _target_file(source, key, name)
            copy = _target_file(cache, key, name)
            size = original.stat().st_size
            total += size
            if count > max_parents or total > max_bytes:
                raise BudgetStop("target backup exceeds sealed parent or storage allowance")
            if copy.is_file() and copy.stat().st_size == size and digest(copy, check) == checksum:
                continue
            pending = cache / (name + ".pending")
            if shutil.disk_usage(cache).free < size + settings["resources"]["reserve_free_bytes"]:
                raise BudgetStop("target backup would consume the free-disk reserve")
            shutil.copyfile(original, pending)
            if digest(pending, check) != checksum:
                raise ValueError(f"target tensor hash changed during backup: {name}")
            os.replace(pending, copy)
    if count < minimum_parents:
        raise ValueError("the target cache has fewer committed examples than required for this backup")
    saved_config = index.with_suffix(".json")
    shutil.copyfile(config, saved_config)
    # Persist the data and directory entries before publishing a receipt for a reboot.
    os.sync()
    value = dict(
        schema=SCHEMA, experiment=str(manifest), configuration_identity=settings["identity"],
        inputs=inputs, source_attempt=str(config.parent), source_config=reference(saved_config),
        index=reference(index),
        parents=count, bytes=total, boot_id=_boot_id(),
    )
    write_json(index.with_suffix(".receipt.json"), value, exclusive=True)
    atomic_json(backup / "manifest.json", value)
    os.sync()
    return value


def _archive_stale_lock(lock, report, archive):
    if not lock.exists():
        return
    boot_seconds = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                        if line.startswith("btime "))
    if report["boot_id"] == _boot_id() or lock.stat().st_mtime >= boot_seconds:
        raise ValueError(f"writer may still be active: {lock}")
    archive.mkdir(parents=True, exist_ok=True)
    lock.rename(archive / lock.name)


def _recovery_rows(index, count, run, *, backup):
    try:
        with closing(_open_index(index)) as db:
            return _rows(db, count).fetchall()
    except sqlite3.DatabaseError:
        journal = index.with_name(index.name + "-journal")
        if backup or not journal.is_file():
            raise
        # Roll back an interrupted SQLite transaction on a bounded, private copy.
        size = index.stat().st_size + journal.stat().st_size
        if size > 3 * run.settings["data"]["max_index_bytes"]:
            raise ValueError("interrupted target index and journal exceed sealed allowance")
        run.meter.check_disk(size)
        charge_write(run, size)
        scratch = run.path / f"recovery-index-{uuid.uuid4().hex}"
        scratch.mkdir()
        try:
            shutil.copyfile(index, scratch / index.name)
            shutil.copyfile(journal, scratch / journal.name)
            with closing(sqlite3.connect(scratch / index.name)) as db:
                return _rows(db, count).fetchall()
        finally:
            shutil.rmtree(scratch)


def restore_tensors(writer, data, indexes, backup_report):
    """Keep valid entries from the newest attempt first, then use the saved snapshot."""
    run = writer.run
    expected = set(data.ids["train"] + data.ids["dev"])
    width = run.settings["architecture"]["latent_width"]
    restored = set()
    summary = {}
    for index, cache in indexes:
        is_backup = cache == backup_report.parent / "cache"
        try:
            rows = _recovery_rows(index, len(expected), run, backup=is_backup)
        except sqlite3.DatabaseError:
            if is_backup:
                raise
            print(f"Could not read {index}; continuing with other target copies", flush=True)
            continue
        if len(rows) > len(expected):
            raise ValueError("saved target index exceeds the admitted dataset")
        before = len(restored)
        for batch in batched(rows, 2048):
            keep = []
            for row in batch:
                key, split, length, name, checksum, dtype, row_width, parent_sha = row
                run.meter.check()
                if key not in expected or key in restored or dtype != "float32" or row_width != width:
                    continue
                parent = data.parent(key)
                if (split, length, parent_sha) != (parent.split, len(parent.answer_ids), parent.identity):
                    continue
                file = _target_file(cache, key, name)
                if file.is_file() and digest(file, run.meter.check) == checksum:
                    keep.append((file, row))
            if not keep:
                continue
            index_writes = len(keep) * 16384
            run.meter.check_disk(sum(file.stat().st_size for file, _ in keep) + index_writes)
            charge_write(run, index_writes)
            for file, row in keep:
                os.link(file, writer.path / file.name)
                writer.insert(*row)
                restored.add(row[0])
            writer.db.commit()
        if len(restored) > before:
            summary[str(index)] = len(restored) - before
    if not restored:
        raise ValueError("no valid saved target tensors could be recovered")
    run.meter.check_disk()
    write_json(run.path / "target_recovery.json", dict(backup=reference(backup_report), restored=summary),
               exclusive=True)
    print(f"Recovered {len(restored):,} target tensors; {len(expected - restored):,} remain", flush=True)
    return restored


def recover_targets(manifest, backup):
    """Finish only the target stage, leaving subsequent training to ordinary resume."""
    from ncm_lab.block8.targets import targets

    folder = manifest.parent
    report_path = backup / "manifest.json"
    report = read(report_path)
    state = read(manifest)
    settings = read_settings(state)
    inputs = stage_inputs(state)
    executor = compatible_implementation(settings)
    if (report["schema"], report["experiment"], report["configuration_identity"], report["inputs"]) != (
        SCHEMA, str(manifest), settings["identity"], inputs,
    ):
        raise ValueError("target backup does not match this experiment and its completed inputs")
    index = checked(report["index"])
    if index.parent != backup:
        raise ValueError("target backup index is outside its backup directory")
    if (backup / "cache").stat().st_dev != folder.stat().st_dev:
        raise ValueError("target recovery hard links require the local backup on the experiment filesystem")
    source = checked(report["source_config"])
    producer = Path(report.get("source_attempt", str(source.parent)))
    indexes = []
    records = dict(_attempt_records(folder, settings))
    expected = records.get(producer / "resolved_config.json")
    if expected is None:
        raise ValueError("backup source is outside this experiment's target attempts")
    _verify_attempt_config(source, expected)
    for config, expected in records.items():
        cache = config.parent / "cache"
        if config.is_file() and (cache / "index.sqlite").is_file():
            _verify_attempt_config(config, expected)
            indexes.append((cache / "index.sqlite", cache))
    indexes.append((index, backup / "cache"))
    sealed_report = index.with_suffix(".receipt.json")
    if sealed_report.is_file():
        if read(sealed_report) != report:
            raise ValueError("target backup receipt differs from its published manifest")
        report_path = sealed_report

    archive = folder / "recovery" / uuid.uuid4().hex
    writer_lock = folder / "writer.lock"
    _archive_stale_lock(writer_lock, report, archive)
    with exclusive_writer(writer_lock):
        state = read(manifest)
        if stage_inputs(state) != inputs:
            raise ValueError("target inputs changed during recovery admission")
        if state["stages"].get("targets", {}).get("status") == "completed":
            return dict(status="completed", experiment=str(manifest), already_complete=True)
        count = state["attempts"].get("targets", 0)
        if count >= settings["resources"]["max_attempts_per_stage"]:
            raise ValueError("target recovery would exceed the sealed retry allowance")
        phase = f"block8-{settings['identity'][:24]}-targets"
        for name in (f"{phase}.lock", "targets-writes.lock"):
            _archive_stale_lock(folder / "ledgers" / name, report, archive)
        state["attempts"]["targets"] = count + 1
        state["stages"]["targets"] = dict(status="running")
        state["status"] = "running"
        atomic_json(manifest, state)
        run = Attempt(folder, "targets", settings, executor_sha256=executor)
        print(f"Recovering Block8 targets in {run.path}", flush=True)
        try:
            result = targets(
                run, settings, inputs["dataset"], inputs["features"], inputs["codec"],
                restore=lambda writer, data: restore_tensors(writer, data, indexes, report_path),
            )
            run.meter.check_disk()
            result["recovery"] = reference(run.path / "target_recovery.json")
            status = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            status = "budget_stopped" if isinstance(exc, BudgetStop) else "failed"
            result = dict(error=f"{type(exc).__name__}: {exc}")
        receipt = run.close(status, result)
        state["status"] = status
        state["stages"]["targets"] = dict(status=status, receipt=receipt)
        if status != "completed":
            state["failure"] = dict(stage="targets", **result, receipt=receipt)
        elif state.get("failure", {}).get("stage") == "targets":
            state.pop("failure")
        atomic_json(manifest, state)
    return dict(status=status, experiment=str(manifest), receipt=receipt, **result)
