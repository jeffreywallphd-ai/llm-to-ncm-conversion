from dataclasses import dataclass, asdict
import torch
from transformers import DynamicCache
from ncm_lab.runtime.io import canonical_hash
from ncm_lab.runtime.budget import BudgetExceeded


@dataclass(frozen=True)
class GenerationPolicy:
    max_new_tokens: int
    eos_ids: tuple = ()
    stop_sequences: tuple = ()
    bos_id: int | None = None
    pad_id: int | None = None
    tokenizer_revision: str = "fixture"
    template_hash: str = "none"
    thinking: bool = False
    backend: str = "sdpa"
    precision: str = "float32"
    include_stop_tokens: bool = True
    mode: str = 'greedy'

    def __post_init__(self):
        if (isinstance(self.max_new_tokens,bool) or not isinstance(self.max_new_tokens,int)
                or self.max_new_tokens < 0 or not self.include_stop_tokens or self.thinking or self.mode!='greedy'):
            raise ValueError("unsupported generation policy")
        object.__setattr__(self,'eos_ids',tuple(self.eos_ids))
        object.__setattr__(self,'stop_sequences',tuple(tuple(s) for s in self.stop_sequences))
        ids=[*self.eos_ids,*(token for sequence in self.stop_sequences for token in sequence)]
        ids += [token for token in (self.bos_id,self.pad_id) if token is not None]
        if any(isinstance(token,bool) or not isinstance(token,int) or token<0 for token in ids):
            raise ValueError('policy token IDs must be nonnegative integers')
        if self.backend != "sdpa" or any(not s for s in self.stop_sequences):
            raise ValueError("unsupported backend or empty stop sequence")

    @property
    def digest(self):
        return canonical_hash(asdict(self))

    def stopped(self, tokens):
        return bool(tokens) and (tokens[-1] in self.eos_ids or any(
            len(tokens) >= len(s) and tokens[-len(s):] == list(s) for s in self.stop_sequences))


def resolve_policy(target, tokenizer=None, overrides=None):
    overrides = dict(overrides or {})
    allowed = set(GenerationPolicy.__dataclass_fields__)
    if set(overrides) - allowed:
        raise ValueError(f"unsupported generation options: {sorted(set(overrides) - allowed)}")
    config = target.generation_config
    for key, default in (("min_length", 0), ("min_new_tokens", None), ("forced_bos_token_id", None),
                         ("forced_eos_token_id", None), ("suppress_tokens", None),
                         ("begin_suppress_tokens", None), ("repetition_penalty", 1.0),
                         ("no_repeat_ngram_size", 0),('bad_words_ids',None),('sequence_bias',None),
                         ('forced_decoder_ids',None),('exponential_decay_length_penalty',None),
                         ('watermarking_config',None),('token_healing',False),('guidance_scale',None),
                         ('num_beams',1),('num_beam_groups',1),('stop_strings',None),('max_time',None),
                         ('encoder_repetition_penalty',1.0),('encoder_no_repeat_ngram_size',0),
                         ('remove_invalid_values',False),('renormalize_logits',False),('penalty_alpha',None)):
        value = getattr(config, key, default)
        if value not in (default, None) and not (key == "min_new_tokens" and value == 0):
            raise ValueError(f"generation processor {key} is unsupported")
    eos = config.eos_token_id
    defaults = {"max_new_tokens": 128, "eos_ids": tuple(eos if isinstance(eos, list) else [] if eos is None else [eos]),
                "bos_id": config.bos_token_id, "pad_id": config.pad_token_id}
    if 'eos_ids' in overrides and not set(defaults['eos_ids'])<=set(overrides['eos_ids']):
        raise ValueError('study policy may not suppress a released EOS ID')
    if hasattr(target,'get_input_embeddings'):
        actual_precision=str(target.get_input_embeddings().weight.dtype).removeprefix('torch.')
        if 'precision' in overrides and overrides['precision']!=actual_precision:
            raise ValueError('policy precision contradicts loaded model placement')
        defaults['precision']=actual_precision
    if tokenizer is not None:
        defaults["template_hash"] = canonical_hash(getattr(tokenizer, "chat_template", None))
        if 'template_hash' in overrides and overrides['template_hash']!=defaults['template_hash']:
            raise ValueError('policy template identity contradicts the loaded tokenizer')
    defaults.update(overrides)
    defaults["eos_ids"] = tuple(defaults["eos_ids"])
    defaults["stop_sequences"] = tuple(tuple(s) for s in defaults.get("stop_sequences", ()))
    return GenerationPolicy(**defaults)


@torch.no_grad()
def ar_greedy(target, input_ids, policy,guard=None):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
        raise ValueError("AR supports one nonempty unpadded request")
    tokens = []
    target.eval()
    cache = DynamicCache(config=target.config)
    current = input_ids
    for _ in range(policy.max_new_tokens):
        guard_decode(guard,tokens)
        out = target(input_ids=current, past_key_values=cache, use_cache=True)
        current = out.logits[:, -1].argmax(-1, keepdim=True)
        tokens.append(current.item())
        if policy.stopped(tokens):
            break
    return tokens


def guard_decode(guard,tokens,result=None):
    if guard is None:
        return
    try:
        guard()
    except BudgetExceeded as exc:
        exc.partial_tokens=list(tokens)
        exc.decode_result=result
        raise
