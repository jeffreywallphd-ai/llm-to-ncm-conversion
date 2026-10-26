import asyncio
import math
import time


async def replay(requests, worker, concurrency=1, deadline_seconds=30, max_requests=100):
    """Controlled arrivals; worker is an async serving client, not host code execution.

    The driver measures actual load elapsed time. It records timeout/error requests
    in the denominator and never treats experimental arrivals as production logs.
    """
    if not 1 <= concurrency <= 128 or not 1 <= max_requests <= 100000:
        raise ValueError("finite positive concurrency/request limits required")
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("finite positive request deadline required")
    requests = list(requests)
    if len(requests) > max_requests:
        raise ValueError("declared request workload exceeds max_requests")
    semaphore, start = asyncio.Semaphore(concurrency), time.monotonic()

    async def one(index, request):
        arrival = time.monotonic()
        try:
            async def service():
                async with semaphore:
                    return await worker(request)
            result = await asyncio.wait_for(service(),timeout=deadline_seconds)
            error = None
        except TimeoutError:
            result,error = {},"timeout"
        except Exception as exc:
            result,error = {},f"{type(exc).__name__}: {exc}"
        return {"index":index,"original_id":request["original_id"],"group_id":request["group_id"],
                "correct":result.get("correct",False) and error is None,"error":error,
                "request_seconds":time.monotonic()-arrival,"deadline_met":error is None,
                "result":result}
    records = await asyncio.gather(*(one(i,r) for i,r in enumerate(requests)))
    from ncm_lab.evaluation.analysis import summarize
    return {"records":records,"summary":summarize(records,time.monotonic()-start),
            "scenario":"controlled simultaneous arrivals, not observed production traffic"}
