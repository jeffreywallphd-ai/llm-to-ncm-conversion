"""Evaluator-only labels; this module is not imported by deployed proposal code."""
import torch
from ncm_lab.models.codecs import pool
from ncm_lab.training.records import block_labels
from ncm_lab.runtime.io import canonical_hash,read,digest
from pathlib import Path


@torch.no_grad()
def teacher_block(target, row, anchor, horizon=15):
    device = target.get_input_embeddings().weight.device
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    out = target(input_ids=ids, output_hidden_states=True, use_cache=False)
    # Public final hidden states are already post-final-normalization.
    states = out.hidden_states[-1]
    block = states.new_zeros((1, horizon, states.shape[-1]))
    length = min(horizon, states.shape[1] - anchor)
    block[:, :length] = states[:, anchor:anchor + length]
    labels = block_labels(row, anchor, horizon, device)
    return block.detach(), labels.ne(-100), labels


def latent_loss(predictions, codes, valid, scale=None, weights=None):
    losses = []
    for level, prediction in predictions.items():
        mask = pool(valid[..., None].float(), level, valid)[1]
        norm = 1.0 if scale is None else scale[level].to(device=prediction.device,dtype=torch.float32).clamp_min(1e-6)
        error = ((prediction.float() - codes[level].detach().float()) / norm).square().mean(-1)
        losses.append((weights or {}).get(level, 1.0) * (error * mask).sum() / mask.sum().clamp_min(1))
    return sum(losses)


def cache_identity(row_hash, anchor, codec_hash, policy_hash,execution=None,levels=(4,8,15)):
    return canonical_hash({"schema": 2, "source": row_hash, "anchor": anchor, "codec": codec_hash,
                           "policy": policy_hash, "stage": "post_final_norm_prediction_position",
                           "partition": list(levels), "horizon": 15,'execution':execution})


def cache_execution(config,codec):
    from ncm_lab.runtime.provenance import implementation_hash
    return {'target_id':config['model']['target_id'],'target_revision':config['model']['target_revision'],
        'tokenizer_revision':config['model']['tokenizer_revision'],'precision':config['precision'],
        'device':config['device'],'codec_spec':codec.spec,'implementation_sha256':implementation_hash()}


class LatentCache:
    def __init__(self,path,codec_hash,execution):
        self.path=Path(path)
        self.manifest=read(self.path)
        if self.manifest['codec_sha256']!=codec_hash or self.manifest['execution']!=execution:
            raise ValueError('latent cache codec/teacher/execution identity mismatch')
        self.codec_hash,self.execution=codec_hash,execution
        self.shards={s['identity']:s for s in self.manifest['shards']}
        if len(self.shards)!=len(self.manifest['shards']):
            raise ValueError('duplicate latent cache identities')
        self.identity=digest(self.path)

    def get(self,row,anchor,levels,device):
        key=cache_identity(canonical_hash(row),anchor,self.codec_hash,row['policy_hash'],self.execution,levels)
        reference=self.shards.get(key)
        if reference is None:
            return None
        path=(self.path.parent/reference['path']).resolve()
        if not path.is_relative_to(self.path.parent.resolve()) or 'sha256:'+digest(path)!=reference['artifact']:
            raise ValueError('latent cache shard changed or escaped its directory')
        from ncm_lab.training.checkpoints import read_checkpoint
        payload=read_checkpoint(path,key)
        return {int(n):code.detach().to(device) for n,code in payload['codes'].items()},payload['valid'].to(device)
