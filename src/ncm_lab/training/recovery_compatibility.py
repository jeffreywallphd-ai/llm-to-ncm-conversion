"""Explicit reviewed clock-policy transition; never an arbitrary identity override."""

from ncm_lab.runtime.io import root, read, digest, implementation_hash, reference
from ncm_lab.training.checkpoints import native_identity

POLICY = "clock-policy-v1"
PREDECESSOR = "9404148f14f0496952e919c77119119b3c32e90b0c74b3947847f0cdce111ab6"
BASE_COMMIT = "c3032129d59e4a9c2b3d2414bf66f1c805202521"


def verify_recovery(reader, expected, compatibility=None):
    if compatibility not in {None, POLICY}:
        raise ValueError("unknown recovery compatibility policy")
    previous = reader.materialize(reader.field("metadata"))
    identity = reader.manifest["identity"]
    if native_identity(previous) != identity or reader.materialize(reader.field("identity")) != identity:
        raise ValueError("native recovery metadata/identity inconsistency")
    if reader.materialize(reader.field("scheduler")) != {"schema": "constant_separate_rates_v1"}:
        raise ValueError("native recovery scheduler contract mismatch")
    if reader.materialize(reader.field("scaler")) is not None:
        raise ValueError("native recovery scaler contract mismatch")
    if previous == expected:
        return None
    adjusted = {**previous, "implementation_sha256": expected["implementation_sha256"]}
    if adjusted != expected:
        raise ValueError("native recovery config/data/codec/optimizer/phase identity mismatch")
    if compatibility != POLICY or previous.get("implementation_sha256") != PREDECESSOR:
        raise ValueError(
            "unreviewed implementation transition; only the recorded clock-policy predecessor can "
            "use --recovery-compatibility clock-policy-v1 (no fresh-weight fallback)"
        )
    receipt_path = root() / "configs/recovery/clock_policy_v1.json"
    receipt = read(receipt_path)
    current = implementation_hash()
    if (
        receipt.get("schema") != "native_clock_policy_compatibility_v1"
        or receipt.get("policy") != POLICY
        or receipt.get("predecessor_implementation_sha256") != PREDECESSOR
        or receipt.get("reviewed_base_commit") != BASE_COMMIT
        or receipt.get("successor_implementation_sha256") != current
        or expected["implementation_sha256"] != current
    ):
        raise ValueError(
            "clock-policy compatibility receipt does not approve this exact source implementation"
        )
    return {
        "policy": POLICY,
        "parent_checkpoint": reference(reader.path),
        "predecessor_implementation_sha256": PREDECESSOR,
        "current_implementation_sha256": current,
        "receipt": {"path": str(receipt_path), "sha256": digest(receipt_path)},
        "reason": "Reviewed clock enforcement/accounting/monitoring and recovery finalization only; model math and training contract unchanged",
    }
