"""Small strict JSON contracts. No Torch or optional dashboard imports."""

from dataclasses import dataclass, asdict
import math

SCHEMA = "native_monitor_record_v1"
MAX_RECORD_BYTES = 1048576


@dataclass(frozen=True)
class Monitoring:
    enabled: bool = True
    scalar_interval: int = 10
    scalar_seconds: float = 30
    detail_interval: int = 100
    first_steps: int = 3
    sample_interval: int = 500
    sample_prompts: int = 4
    sample_seconds: float = 60
    text_chars: int = 2048
    histograms: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.histograms) is not bool:
            raise ValueError("monitoring switches must be boolean")
        for key in ("scalar_interval", "detail_interval", "first_steps", "sample_interval", "text_chars"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("positive monitoring limit required: " + key)
        if type(self.sample_prompts) is not int or not 0 <= self.sample_prompts <= 4:
            raise ValueError("at most four sample prompts")
        for value in (self.scalar_seconds, self.sample_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive monitoring time required")
        if self.histograms:
            raise ValueError("histogram collection is not implemented; keep it disabled")


@dataclass(frozen=True)
class Measure:
    value: float | None
    count: int = 1
    unit: str = "dimensionless"
    reduction: str = "observed scalar"
    status: str = "valid"
    numerator: float | None = None

    def json(self):
        result = asdict(self)
        if self.count < 0:
            raise ValueError("negative observation count")
        if self.value is not None and not math.isfinite(self.value):
            result.update(value=None, numerator=None, status="nonfinite")
        elif self.numerator is not None and not math.isfinite(self.numerator):
            result.update(value=None, numerator=None, status="nonfinite")
        elif self.value is None or self.count == 0:
            result.update(value=None, status=self.status if self.status != "valid" else "unavailable")
        return result


def mean_measure(numerator, count, **kwargs):
    return Measure(numerator / count if count else None, count, numerator=numerator, **kwargs)


class Accumulator:
    """Count-weighted summaries, separate from equally weighted step objectives."""

    def __init__(self):
        self.values = {}

    def add(self, values):
        for name, measure in values.items():
            row = measure.json()
            old = self.values.setdefault(name, {"sum": 0.0, "count": 0, "row": row})
            if row["status"] != "valid":
                if row["status"] == "nonfinite":
                    old["row"] = row
                continue
            old["sum"] += row["numerator"] if row["numerator"] is not None else row["value"] * row["count"]
            old["count"] += row["count"]

    def finish(self):
        result = {}
        for name, item in self.values.items():
            row = item["row"]
            if row["status"] == "nonfinite":
                result[name] = Measure(None, status="nonfinite")
            elif row["reduction"].startswith("sum;"):
                result[name] = Measure(
                    item["sum"], item["count"], row["unit"], row["reduction"], numerator=item["sum"]
                )
            else:
                result[name] = mean_measure(
                    item["sum"], item["count"], unit=row["unit"], reduction=row["reduction"]
                )
        self.values.clear()
        return result


def mappings(architecture):
    resolutions = architecture["resolutions"]
    early = architecture["early_blocks"]
    updates = [
        {"update": i + 1, "group": i // 2 + 1, "block": early + i // 2 + 1, "resolution": resolution}
        for i, resolution in enumerate(resolutions)
    ]
    return {"updates": updates, "groups": [u for u in updates if u["update"] % 2 == 0]}


def validate_record(row):
    if row.get("schema") != SCHEMA or row.get("stream") not in {"metrics", "events", "samples"}:
        raise ValueError("incompatible monitoring record schema/stream")
    for key in (
        "run_id",
        "attempt_id",
        "sequence",
        "utc",
        "wall_time",
        "elapsed_seconds",
        "step",
        "phase",
        "split",
        "origin",
        "config_sha256",
    ):
        if key not in row:
            raise ValueError("incomplete monitoring identity: " + key)
    if not isinstance(row["sequence"], int) or row["sequence"] < 1:
        raise ValueError("invalid stream sequence")
    for measure in row.get("metrics", {}).values():
        if measure["value"] is not None and not math.isfinite(measure["value"]):
            raise ValueError("nonfinite JSON metric")
    return row
