"""Explicit reviewed clock-policy transition; never an arbitrary identity override."""

from ncm_lab.runtime.io import root, read, digest, implementation_hash, reference
from ncm_lab.training.checkpoints import native_identity

POLICY = "clock-policy-v1"
PREDECESSOR = "9404148f14f0496952e919c77119119b3c32e90b0c74b3947847f0cdce111ab6"
BASE_COMMIT = "c3032129d59e4a9c2b3d2414bf66f1c805202521"
CHAT_POLICY = "pilot-chat-v1"
CHAT_PREDECESSOR = "e106a014d20dba62e209f7b2432b6789e1cd34e2cd57d6f9c03f152cb901a3aa"
CHAT_BASE = "0c9fdd50c637caa92a013c03a0f4a8a9eae23e99"
CLOCK_RECEIPT_SHA256 = "9ef0ad3991159ec0c662ef211fd92cdadd8c937c848844d4b1806d5edd1030ef"


def verify_recovery(reader, expected, compatibility=None):
    if compatibility not in {None, POLICY, CHAT_POLICY}:
        raise ValueError("unknown recovery compatibility policy")
    previous = reader.materialize(reader.field("metadata"))
    identity = reader.manifest["identity"]
    if native_identity(previous) != identity or reader.materialize(reader.field("identity")) != identity:
        raise ValueError("native recovery metadata/identity inconsistency")
    scheduler = reader.materialize(reader.field("scheduler"))
    expected_scheduler = {"schema": "constant_separate_rates_v1"}
    if previous.get("stage3_contract"):
        from ncm_lab.stage3.schedule import Schedule
        training = reader.materialize(reader.field("training"))
        peaks = [previous["training_contract"]["learning_rate"],
                 previous["training_contract"]["new_learning_rate"] if previous["component"] == "generator"
                 else previous["training_contract"]["learning_rate"]]
        expected_scheduler = Schedule(**previous["stage3_contract"]["schedule"]).state(peaks, training["optimizer_steps"])
    if scheduler != expected_scheduler:
        raise ValueError("native recovery scheduler contract mismatch")
    if reader.materialize(reader.field("scaler")) is not None:
        raise ValueError("native recovery scaler contract mismatch")
    if previous == expected:
        return None
    adjusted = {**previous, "implementation_sha256": expected["implementation_sha256"]}
    if adjusted != expected:
        raise ValueError("native recovery config/data/codec/optimizer/phase identity mismatch")
    old = previous.get("implementation_sha256")
    if old == PREDECESSOR and compatibility in {POLICY, CHAT_POLICY}:
        paths = [root() / "configs/recovery/clock_policy_v1.json"]
    elif old == CHAT_PREDECESSOR and compatibility == CHAT_POLICY:
        paths = []
    else:
        raise ValueError(
            "unreviewed implementation transition; use an explicitly approved clock-policy-v1 or "
            "pilot-chat-v1 predecessor (no fresh-weight fallback)"
        )
    current = implementation_hash()
    if paths:
        historical = read(paths[0])
        if (
            digest(paths[0]) != CLOCK_RECEIPT_SHA256
            or historical.get("predecessor_implementation_sha256") != PREDECESSOR
            or historical.get("successor_implementation_sha256") != CHAT_PREDECESSOR
            or historical.get("reviewed_base_commit") != BASE_COMMIT
        ):
            raise ValueError("clock-policy compatibility receipt does not approve this transition")
    receipt_path = root() / "configs/recovery/pilot_chat_v1.json"
    receipt = read(receipt_path)
    if (
        receipt.get("schema") != "native_pilot_chat_compatibility_v1"
        or receipt.get("policy") != CHAT_POLICY
        or receipt.get("predecessor_implementation_sha256") != CHAT_PREDECESSOR
        or receipt.get("reviewed_base_commit") != CHAT_BASE
        or receipt.get("clock_receipt_sha256") != CLOCK_RECEIPT_SHA256
        or receipt.get("successor_implementation_sha256") != current
        or expected["implementation_sha256"] != current
    ):
        raise ValueError("pilot-chat compatibility receipt does not approve this exact source implementation")
    return {
        "policy": compatibility,
        "parent_checkpoint": reference(reader.path),
        "predecessor_implementation_sha256": old,
        "current_implementation_sha256": current,
        "receipt": {"path": str(receipt_path), "sha256": digest(receipt_path)},
        "reviewed_chain": [{"path": str(path), "sha256": digest(path)} for path in paths + [receipt_path]],
        "reason": "Reviewed clock policy and pilot launcher/chat-only transitions; model math and complete training metadata unchanged",
    }
