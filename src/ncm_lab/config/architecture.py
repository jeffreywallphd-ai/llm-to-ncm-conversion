"""The initial embedded architecture; smaller canvases retain all refinements."""

from dataclasses import asdict, dataclass
import math

PIN = "b968826d9c46dd6066d109eabc6255188de91218"
RESOLUTIONS = (
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    17,
    20,
    24,
    29,
    34,
    41,
    49,
    59,
    70,
    84,
    100,
    120,
    143,
    171,
    205,
    245,
    293,
    350,
    419,
    501,
    599,
    716,
    856,
    1024,
)


@dataclass(frozen=True)
class Architecture:
    schema: str = "embedded_qwen_hierarchy_v1"
    canvas: int = 1024
    initial_queries: int = 4
    latent_width: int = 384
    early_blocks: int = 8
    hierarchy_blocks: int = 16
    late_blocks: int = 12
    resolutions: tuple = RESOLUTIONS
    gate_initial: float = 0.1
    longer_skips: bool = False
    activation_checkpointing: bool = True

    def __post_init__(self):
        object.__setattr__(self, "resolutions", tuple(self.resolutions))
        if self.schema != "embedded_qwen_hierarchy_v1" or self.longer_skips:
            raise ValueError("unsupported native schema or longer skips")
        integers = [
            self.canvas,
            self.initial_queries,
            self.latent_width,
            self.early_blocks,
            self.hierarchy_blocks,
            self.late_blocks,
            *self.resolutions,
        ]
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in integers):
            raise ValueError("positive integer geometry required")
        if (
            len(self.resolutions) != 2 * self.hierarchy_blocks
            or self.resolutions[0] != self.initial_queries
            or self.resolutions[-1] != self.canvas
            or sorted(self.resolutions) != list(self.resolutions)
        ):
            raise ValueError("invalid sequential hierarchy endpoints/count/order")
        if (
            self.initial_queries > self.canvas
            or not math.isfinite(self.gate_initial)
            or not 0 < self.gate_initial < 1
        ):
            raise ValueError("invalid initial query count or nonzero gate")

    @property
    def blocks(self):
        return self.early_blocks + self.hierarchy_blocks + self.late_blocks

    def to_dict(self):
        result = asdict(self)
        result["resolutions"] = list(self.resolutions)
        return result


def tiny_architecture(canvas=16, **kwargs):
    schedule = tuple(round(4 * (canvas / 4) ** (i / 31)) for i in range(32))
    return Architecture(canvas=canvas, latent_width=8, resolutions=schedule, **kwargs)
