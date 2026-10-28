"""Pinned DCore released flat baseline; no remote Python loading or target changes."""
from ncm_lab.vendor.dcore.model import DFlashDraftModel

DCORE_ID='ernie-research/DCore-DFlash-Qwen3-8B-Budget16'
DCORE_REVISION='e2f43607fb2531341b09e41889f593c75456a3f2'


class DCoreReferenceDraft(DFlashDraftModel):
    def __init__(self,config):
        options=getattr(config,'dflash_config',{}) or {}
        if options.get('projector_type') is not None or options.get('pure_draft_prefix_len',0) or options.get('shift_label',False):
            raise ValueError('only the pinned plain DCore DFlash recipe is supported')
        super().__init__(config)

    def compute_logits(self,hidden_states,head):
        # The inspected released plain model has no Domino/logit projector.
        return head(hidden_states)
