"""Small finite-vocabulary software fixtures, never research checkpoints."""
from transformers import Qwen3Config, Qwen3ForCausalLM
from ncm_lab.vendor.dflash.model import DFlashDraftModel
from ncm_lab.models.upstream import freeze


def tiny_pair():
    common = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=8, max_position_embeddings=256, rope_theta=10000.0,
                  attention_dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                  tie_word_embeddings=False)
    target = freeze(Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=4, **common)))
    config = Qwen3Config(num_hidden_layers=5, **common)
    config.num_target_layers = 4
    config.block_size = 16
    config.dflash_config = {"target_layer_ids": [0, 2], "mask_token_id": 3}
    config.architectures = ["DFlashDraftModel"]
    draft = DFlashDraftModel(config)
    return target, draft
