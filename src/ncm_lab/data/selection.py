"""Stable disk-backed fair selection; complete examples, never replacement."""

from collections import Counter
import heapq
from ncm_lab.runtime.io import canonical_hash

BANDS = ((1, 128), (129, 256), (257, 512), (513, 768), (769, 1024))
SHARES = (0.20, 0.20, 0.25, 0.20, 0.15)


def band(length):
    return next(i for i, (low, high) in enumerate(BANDS) if low <= length <= high)


def rank(identity, seed=17):
    return canonical_hash([seed, identity])


def fair_rows(database, split, shares=None, coverage=False):
    """Rows table: key, split, band, source, rank. Cursors retain no corpus bodies."""
    counts = {
        (b, s): n
        for b, s, n in database.execute(
            "SELECT band,source,count(*) FROM rows WHERE split=? GROUP BY band,source", (split,)
        )
    }
    totals = Counter()
    for (b, _), n in counts.items():
        totals[b] += n
    queue = []
    cursors = {}
    for (b, s), n in counts.items():
        weight = shares[b] * n / totals[b] if shares else n / sum(counts.values())
        cursor = database.execute(
            "SELECT key FROM rows WHERE split=? AND band=? AND source=? ORDER BY rank,key", (split, b, s)
        )
        cursors[b, s] = cursor
        heapq.heappush(queue, (0.0, b, s, 0, weight))
    while queue:
        _, b, s, used, weight = heapq.heappop(queue)
        row = cursors[b, s].fetchone()
        if row is None:
            continue
        yield row[0]
        used += 1
        priority = used / weight
        # Give each available dev source/band one slot before proportional filling.
        heapq.heappush(queue, (priority if not coverage or used else 0.0, b, s, used, weight))
