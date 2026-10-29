"""Full-answer native records; no known answer anchor enters generator inputs."""

from dataclasses import dataclass, asdict
import torch
from ncm_lab.runtime.io import canonical_hash

SCHEMA = "native_answer_example_v1"
ALIGNMENT = "post_final_norm_at_prompt_length_minus_one_plus_answer_index"


@dataclass(frozen=True)
class Example:
    original_id: str
    group_id: str
    split: str
    source_type: str
    prompt_ids: list
    answer_ids: list
    eos_genuine: bool
    censored: bool
    provenance: dict
    schema: str = SCHEMA

    def __post_init__(self):
        if (
            self.schema != SCHEMA
            or self.split not in {"train", "dev"}
            or self.source_type not in {"reference", "teacher_generated", "software_fixture"}
        ):
            raise ValueError("native answer schema/split/source required; legacy anchors are incompatible")
        if not self.original_id or not self.group_id or not self.prompt_ids or not self.answer_ids:
            raise ValueError("nonempty native example identities/prompt/answer required")
        if any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.prompt_ids + self.answer_ids
        ):
            raise ValueError("token IDs must be nonnegative integers")
        if self.censored and self.eos_genuine:
            raise ValueError("censoring cannot invent genuine EOS")
        if not isinstance(self.censored, bool) or not isinstance(self.eos_genuine, bool):
            raise ValueError("EOS and censoring flags must be boolean")
        eos = self.provenance.get("eos_ids", [])
        if (
            self.eos_genuine
            and eos
            and (self.answer_ids[-1] not in eos or any(t in eos for t in self.answer_ids[:-1]))
        ):
            raise ValueError("genuine EOS must terminate the reference without post-EOS labels")
        if self.source_type != "software_fixture":
            required = {
                "model_id",
                "revision",
                "tokenizer_revision",
                "template_sha256",
                "thinking",
                "source_sha256",
                "dataset_revision",
                "license",
            }
            if not required <= self.provenance.keys() or self.provenance["thinking"] is not False:
                raise ValueError("pinned source/template/non-thinking provenance required")

    @property
    def identity(self):
        return canonical_hash(asdict(self))


def collate(examples, canvas, device="cpu", pad_id=0):
    if any(len(e.answer_ids) > canvas for e in examples):
        raise ValueError("overlength references require explicit preprocessing policy")
    prompt = max(len(e.prompt_ids) for e in examples)
    ids = torch.full((len(examples), prompt), pad_id, device=device, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    labels = torch.full((len(examples), canvas), -100, device=device, dtype=torch.long)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    for i, e in enumerate(examples):
        ids[i, : len(e.prompt_ids)] = torch.tensor(e.prompt_ids, device=device)
        mask[i, : len(e.prompt_ids)] = True
        labels[i, : len(e.answer_ids)] = torch.tensor(e.answer_ids, device=device)
        valid[i, : len(e.answer_ids)] = True
    # Loss-only labels/valid never become arguments to the generator.
    return ids, mask, labels, valid
