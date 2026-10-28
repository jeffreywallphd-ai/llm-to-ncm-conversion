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


def save_tiny_artifacts(directory):
    """Cache-like safetensors/tokenizer artifacts for production-executor tests only."""
    from pathlib import Path
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    directory=Path(directory)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        target,draft=tiny_pair()
    target.generation_config.eos_token_id=[30,31]
    target.save_pretrained(directory/'target',safe_serialization=True)
    draft.save_pretrained(directory/'draft',safe_serialization=True)
    names=['<pad>','<bos>','unused-eos','<mask>','train0','train1','dev0','dev1','answer']
    names += [f'tok{i}' for i in range(9,29)]+['<unk>','<eos>','<stop>']
    tokenizer=Tokenizer(WordLevel({name:i for i,name in enumerate(names)},unk_token='<unk>'))
    tokenizer.pre_tokenizer=Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,bos_token='<bos>',eos_token='<eos>',
        pad_token='<pad>',unk_token='<unk>',additional_special_tokens=['<mask>','<stop>'])
    fast.chat_template="{{ bos_token }} {{ messages[0]['content'] }} answer"
    fast.save_pretrained(directory/'target')
    return {'target_id':'software-fixture/qwen','draft_id':'software-fixture/dflash',
        'target_path':str(directory/'target'),'draft_path':str(directory/'draft'),
        'target_revision':'a'*40,'draft_revision':'b'*40,'tokenizer_revision':'a'*40,
        'scope':'test_fixture; never research evidence'}
