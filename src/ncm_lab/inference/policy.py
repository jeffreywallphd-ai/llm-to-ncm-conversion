"""Inference-only vocabulary projection allocation policy; training limits differ."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Inference:
    head_chunk_positions: int = 128
    max_head_temp_bytes: int = 268435456

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("positive integer inference head chunk/memory budget required")

    def resolve(self, canvas, vocabulary, dtype_bytes, batch=1):
        chunk = min(self.head_chunk_positions, canvas)
        temporary = 2 * batch * chunk * vocabulary * dtype_bytes + batch * canvas * 8
        if temporary > self.max_head_temp_bytes:
            raise ValueError(
                "inference vocabulary projection exceeds declared logits/workspace/ID memory budget"
            )
        return {
            **asdict(self),
            "resolved_head_chunk_positions": chunk,
            "estimated_temporary_bytes": temporary,
            "allocation_policy": "two logits-sized buffers plus full canvas int64 IDs; backend GEMM workspace is an additional measured hardware requirement",
        }
