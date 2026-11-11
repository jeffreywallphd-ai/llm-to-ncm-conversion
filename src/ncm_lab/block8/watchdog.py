"""Watch the owned Block8 worker's durable progress without a lifetime deadline."""

import multiprocessing
from pathlib import Path
from ncm_lab.block8.runtime import experiment_directory
from ncm_lab.runtime.io import digest, read, reference, write_json

STOP_GRACE_SECONDS = 30


def progress_snapshot(manifest):
    """Stage/recovery transitions, charged work, and writes across the experiment."""
    identity = digest(manifest) if manifest.is_file() else None
    ledgers = {}
    for path in (manifest.parent / "ledgers").glob("*.json"):
        value = read(path)
        ledgers[path.name] = (value.get("charges"), value.get("written_bytes"))
    return identity, ledgers


def _worker(sender, stop, settings, options):
    with sender:
        try:
            from ncm_lab.block8.workflow import run_workflow

            result = run_workflow(settings, **options, stop_requested=stop.is_set)
        except (ValueError, OSError, KeyError, ImportError) as exc:
            result = dict(status="failed", error=str(exc))
        sender.send(result)


def _stop_worker(worker):
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=5)
    if worker.is_alive():
        raise RuntimeError("Block8 watchdog could not stop its worker")


def _forced_stop(manifest, timeout, worker):
    error = f"Block8 watchdog stopped an unresponsive worker after {timeout} seconds without progress"
    failure = dict(error=error)
    if manifest.is_file():
        identity = digest(manifest)
        path = manifest.parent / "watchdog" / (identity + ".json")
        if not path.exists():
            write_json(
                path,
                dict(
                    schema="block8_watchdog_stop_v1",
                    manifest_sha256=identity,
                    max_stall_seconds=timeout,
                    child_exit_status=worker.exitcode,
                    error=error,
                ),
                exclusive=True,
            )
        failure["receipt"] = reference(path)
    return dict(status="budget_stopped", experiment=str(manifest), failure=failure)


def supervise(settings, options, timeout):
    options = dict(options)
    if options["resume"]:
        manifest = Path(options["resume"]).resolve()
    else:
        folder = experiment_directory(options["experiment_dir"])
        options["experiment_dir"] = folder
        manifest = folder / "manifest.json"
    previous = progress_snapshot(manifest)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    stop = context.Event()
    worker = context.Process(target=_worker, args=(sender, stop, settings, options), name="block8-worker")
    with receiver, sender:
        worker.start()
        sender.close()
        try:
            while not receiver.poll(timeout):
                current = progress_snapshot(manifest)
                if current != previous:
                    previous = current
                    continue
                stop.set()
                if not receiver.poll(STOP_GRACE_SECONDS):
                    _stop_worker(worker)
                    if worker.exitcode != 0:
                        return _forced_stop(manifest, timeout, worker)
                break
            try:
                return receiver.recv()
            except EOFError:
                return dict(status="failed", experiment=str(manifest), error="Block8 worker exited without a result")
        finally:
            _stop_worker(worker)
            worker.close()
