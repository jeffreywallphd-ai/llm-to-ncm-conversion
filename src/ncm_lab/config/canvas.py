"""Production answer capacities. Labels never alter serialized architecture identity."""

from dataclasses import dataclass, replace
from ncm_lab.config.architecture import Architecture, RESOLUTIONS


@dataclass(frozen=True)
class Canvas:
    name: str
    positions: int

    @property
    def schedule(self):
        if self.positions == 1024:
            return RESOLUTIONS  # Preserve the original explicit architecture exactly.
        return tuple(round(4 * (self.positions / 4) ** (i / 31)) for i in range(32))

    def architecture(self, base=None):
        base = base or Architecture()
        if base.initial_queries != 4 or base.hierarchy_blocks != 16:
            raise ValueError("canvas presets require four queries and 32 hierarchy updates")
        return replace(base, canvas=self.positions, resolutions=self.schedule)


PRESETS = tuple(
    Canvas(n, c)
    for n, c in zip(("xsmall", "small", "medium", "large", "xlarge"), (8, 32, 64, 256, 1024), strict=True)
)
DEFAULT = "medium"
CHOICES = tuple(p.name for p in PRESETS) + tuple(str(p.positions) for p in PRESETS)


def resolve_canvas(value=DEFAULT):
    if type(value) not in {str, int}:
        raise ValueError("canvas size must be a preset name or exact integer alias")
    for preset in PRESETS:
        if value in (preset.name, preset.positions, str(preset.positions)):
            return preset
    raise ValueError("unknown canvas size; choose " + ", ".join(CHOICES))


def canvas_metadata(architecture):
    a = Architecture(**architecture)
    preset = next((p for p in PRESETS if p.positions == a.canvas and p.schedule == a.resolutions), None)
    return {
        "preset": preset.name if preset else "explicit",
        "positions": a.canvas,
        "resolutions": list(a.resolutions),
    }


def validate_production(architecture):
    a = Architecture(**architecture)
    preset = resolve_canvas(a.canvas)
    if (a.initial_queries, a.latent_width, a.early_blocks, a.hierarchy_blocks, a.late_blocks) != (
        4,
        384,
        8,
        16,
        12,
    ):
        raise ValueError("real presets require four queries, width 384 and 8/16/12 blocks")
    if a.resolutions != preset.schedule:
        raise ValueError("real preset schedule differs from the canonical canvas registry")


def add_canvas_argument(parser):
    parser.add_argument(
        "--canvas-size",
        choices=CHOICES,
        help="answer positions, not model size; names and exact numeric aliases",
    )
