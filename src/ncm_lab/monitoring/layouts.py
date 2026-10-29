"""Stable TensorBoard layouts and bounded missing-aware heatmap rendering."""

from collections import deque
import math


LAYOUT = {
    "Generator objective": {
        "Train / dev": [
            "Multiline",
            [
                r"generator/(train|dev)/loss/(text_ce|normalized_auxiliary|weighted_auxiliary|total_objective)$"
            ],
        ]
    },
    "Auxiliary groups": {
        "Normalized cumulative error": ["Multiline", [r"generator/train/aux/group_\d+/normalized_mse$"]]
    },
    "Hierarchy activity": {"32 effective gates": ["Multiline", [r"hierarchy/update_\d+/gate_effective$"]]},
    "Codec": {
        "Actual objective": [
            "Multiline",
            [
                r"codec/(train|dev)/loss/(total_objective|full_normalized_mse|sampled_cutoff_normalized_mse|head_kl)$"
            ],
        ]
    },
    "Resources": {"Step wall time": ["Multiline", [r"resources/step_wall_seconds$"]]},
}


class Heatmaps:
    def __init__(self, limit=2048):
        if not 1 <= limit <= 2048:
            raise ValueError("heatmap history limit must be 1..2048")
        self.history = {"aux": deque(maxlen=limit), "gates": deque(maxlen=limit)}

    def add(self, row):
        metrics = row.get("metrics", {})
        for key, count, pattern in (
            ("aux", 16, "generator/train/aux/group_{:02d}/normalized_mse"),
            ("gates", 32, "hierarchy/update_{:02d}/gate_effective"),
        ):
            values = [metrics.get(pattern.format(i), {}).get("value") for i in range(1, count + 1)]
            if any(v is not None for v in values):
                self.history[key].append((row["step"], values))

    def image(self, key):
        # Pillow is the only rendering dependency; no figure/history tensor queues.
        from PIL import Image, ImageDraw
        import numpy as np

        rows = self.history[key]
        if not rows:
            return None
        group_count = 16 if key == "aux" else 32
        width = min(1024, max(240, len(rows)))
        image = Image.new("RGB", (width, group_count * 8 + 36), (90, 90, 90))
        draw = ImageDraw.Draw(image)
        numbers = [v for _, values in rows for v in values if v is not None and math.isfinite(v)]
        low, high = (min(numbers), max(numbers)) if numbers else (0.0, 1.0)
        label = "16 cumulative normalized MSE" if key == "aux" else "32 effective gates"
        draw.text((2, 1), f"{label}; gray=missing", fill="white")
        draw.text((2, 14), f"steps {rows[0][0]}..{rows[-1][0]}; scale {low:.3g}..{high:.3g}", fill="white")
        for x in range(width):
            _, values = rows[min(len(rows) - 1, x * len(rows) // width)]
            for group, value in enumerate(values):
                if value is None or not math.isfinite(value):
                    continue
                ratio = (value - low) / (high - low) if high > low else 0.5
                color = (int(255 * ratio), int(180 * (1 - ratio)), int(255 * (1 - ratio)))
                draw.line((x, 36 + group * 8, x, 43 + group * 8), fill=color)
        return np.asarray(image)
