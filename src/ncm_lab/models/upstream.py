import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ncm_lab.vendor.dflash.model import DFlashDraftModel, extract_context_feature


def freeze(module):
    module.requires_grad_(False)
    module.eval()
    return module


def validate_compatibility(target, draft):
    tc, dc = target.config, draft.config
    for field in ("hidden_size", "vocab_size"):
        if getattr(tc, field) != getattr(dc, field):
            raise ValueError(f"incompatible target/draft {field}")
    target_rope = getattr(tc, "rope_parameters", None) or {"rope_theta": getattr(tc, "rope_theta", None)}
    draft_rope = getattr(dc, "rope_parameters", None) or {"rope_theta": getattr(dc, "rope_theta", None)}
    if target_rope != draft_rope:
        raise ValueError("incompatible target/draft rotary configuration")
    if dc.num_target_layers != tc.num_hidden_layers:
        raise ValueError("target layer count differs from drafter recipe")
    if any(not 0 <= i < tc.num_hidden_layers for i in draft.target_layer_ids):
        raise ValueError("feature tap outside loaded target configuration")
    if dc.architectures and dc.architectures != ["DFlashDraftModel"]:
        raise ValueError("adapter supports original DFlash; DFlash2 selector/conv requires a separate protocol")
    return True


def load_pair(model_config, device="cpu", dtype=torch.float32):
    """Cache-only loading: no download, no remote model-code execution."""
    for key in ("target_revision", "draft_revision", "tokenizer_revision"):
        value = model_config[key]
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be an exact revision")
    common = dict(local_files_only=True, trust_remote_code=False)
    preflight_generation_metadata(model_config)
    target = AutoModelForCausalLM.from_pretrained(model_config.get("target_path",model_config["target_id"]), revision=model_config["target_revision"],
                                                dtype=dtype,attn_implementation='sdpa', **common).to(device)
    validate_generation_artifact(target,model_config)
    from ncm_lab.models.external_dcore import DCORE_ID,DCORE_REVISION,DCoreReferenceDraft
    draft_type=DFlashDraftModel
    if model_config['draft_id']==DCORE_ID:
        if model_config['draft_revision']!=DCORE_REVISION:
            raise ValueError('external baseline revision is not the audited release')
        draft_type=DCoreReferenceDraft
    draft,loading=draft_type.from_pretrained(model_config.get("draft_path",model_config["draft_id"]), revision=model_config["draft_revision"],
                                           dtype=dtype,attn_implementation='sdpa',output_loading_info=True, **common)
    if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('draft checkpoint does not exactly match the audited architecture')
    draft=draft.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_config.get("target_path",model_config["target_id"]), revision=model_config["tokenizer_revision"], **common)
    validate_compatibility(target, draft)
    return freeze(target), draft, tokenizer


def load_target(model_config, device="cpu", dtype=torch.float32):
    common = dict(local_files_only=True, trust_remote_code=False)
    preflight_generation_metadata(model_config)
    name = model_config.get("target_path", model_config["target_id"])
    target = AutoModelForCausalLM.from_pretrained(name, revision=model_config["target_revision"],
                                                 dtype=dtype,attn_implementation='sdpa', **common).to(device)
    validate_generation_artifact(target,model_config)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=model_config["tokenizer_revision"], **common)
    return freeze(target), tokenizer


def validate_generation_artifact(target,config):
    if config.get("target_id") == "Qwen/Qwen3-8B" and config.get("target_revision") == "b968826d9c46dd6066d109eabc6255188de91218":
        if target.generation_config.eos_token_id != [151645,151643]:
            raise ValueError("pinned Qwen generation_config.json missing/changed; retrieve the locked metadata asset")


def preflight_generation_metadata(config, overrides=None):
    from types import SimpleNamespace
    from transformers import GenerationConfig
    from ncm_lab.decoding.policy import resolve_policy
    if config.get('thinking',False):
        raise ValueError('the supported study policy disables thinking')
    if overrides and overrides.get('tokenizer_revision',config['tokenizer_revision'])!=config['tokenizer_revision']:
        raise ValueError('policy tokenizer revision contradicts the pinned artifact')
    name = config.get("target_path",config["target_id"])
    generation = GenerationConfig.from_pretrained(name,revision=config["target_revision"],local_files_only=True)
    target = SimpleNamespace(generation_config=generation)
    validate_generation_artifact(target,config)
    resolve_policy(target,overrides=overrides)
    return generation


class FeatureExtractor(nn.Module):
    def __init__(self, target, taps):
        super().__init__()
        self.target, self.taps = freeze(target), list(taps)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None):
        self.target.eval()
        output = self.target(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, use_cache=False)
        return extract_context_feature(output.hidden_states, self.taps).detach()


def checkpoint_id(path):
    from ncm_lab.runtime.io import digest
    return "sha256:" + digest(path)
