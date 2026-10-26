from dataclasses import dataclass


@dataclass(frozen=True)
class Partition:
    groups: tuple[tuple[int, ...], ...]
    horizon: int = 15

    def __post_init__(self):
        flattened = [p for group in self.groups for p in group]
        if any(not group for group in self.groups) or flattened != list(range(1, self.horizon + 1)):
            raise ValueError("ordered future partition must cover offsets 1..horizon exactly; anchor is separate")

    @property
    def representatives(self):
        return [(g[0] + g[-1]) // 2 for g in self.groups]

    def parents(self, child):
        if self.horizon != child.horizon:
            raise ValueError("partition horizon mismatch")
        out = []
        for group in child.groups:
            matches = [i for i, ancestor in enumerate(self.groups) if set(group) <= set(ancestor)]
            if len(matches) != 1:
                raise ValueError("expansion must refine a nested partition")
            out.append(matches[0])
        return out


def partition(count, horizon=15):
    if count not in {4, 8, horizon}:
        raise ValueError("supported resolutions are 4, 8, and full horizon")
    width = 4 if count == 4 else 2 if count == 8 else 1
    groups = tuple(tuple(range(i, min(i + width, horizon + 1))) for i in range(1, horizon + 1, width))
    if len(groups) != count:
        raise ValueError("this horizon needs explicit validated custom partitions")
    return Partition(groups, horizon)


def validate_schedule(schedule, horizon=15):
    if not schedule or schedule[-1] != horizon:
        raise ValueError("last layer must resolve every future token")
    parts = [partition(n, horizon) for n in schedule]
    for a, b in zip(parts, parts[1:]):
        a.parents(b)
    return parts
