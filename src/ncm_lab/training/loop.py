import random
from pathlib import Path
import time
import math
import torch
from ncm_lab.models.upstream import FeatureExtractor
from ncm_lab.runtime.budget import BudgetExceeded
from ncm_lab.runtime.io import root, read, digest, canonical_hash, write_json
from ncm_lab.training.records import load_records, normalize_record, anchors, block_labels, disjoint
from ncm_lab.training.losses import final_token_loss
from ncm_lab.training.checkpoints import (student_state, load_student, atomic_checkpoint, read_checkpoint,
    rng_state, restore_rng, identity, tensor_bytes, preflight, selected_payload)


def resolve_training_data(config):
    if config.get('training_data_reference'):
        reference = read(root() / config['training_data_reference'])
        path = Path(reference['path'])
        if not path.is_file() or digest(path) != reference['sha256']:
            raise ValueError('prepared teacher-data artifact is missing or changed')
        return path
    return root() / config['training_data']


def load_training(path, origin='target_distillation'):
    return load_records(path, 'train', origin)


def forward_block(target, drafter, row, anchor):
    device = target.get_input_embeddings().weight.device
    ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
    features = FeatureExtractor(target, drafter.backbone.target_layer_ids)(ids)
    positions = torch.arange(ids.shape[1], device=device)[None]
    logits = drafter(features, positions, torch.zeros_like(positions), torch.tensor([anchor], device=device),
                    ids[:, anchor], torch.zeros(1, dtype=torch.long, device=device))
    return logits, block_labels(row, anchor, drafter.horizon, device)


@torch.no_grad()
def evaluate_development(target, drafter, records, meter=None):
    if not records or any(r['split'] != 'dev' for r in records):
        raise ValueError('selection is development-only')
    prior = drafter.training
    drafter.eval()
    numerator, count, plan = 0.0, 0, []
    try:
        for row in records:
            candidates = anchors(row)
            if not candidates:
                continue
            anchor = candidates[0]
            if meter:
                meter.check(len(row['input_ids']), 0, step=False)
                meter.source_tokens += len(row['input_ids'])
            logits, labels = forward_block(target, drafter, row, anchor)
            valid = int(labels.ne(-100).sum())
            numerator += float(final_token_loss(logits, labels, decay_gamma=None)) * valid
            count += valid
            plan.append([row['group_id'], anchor])
    finally:
        drafter.train(prior)
        drafter.embeddings.eval()
        drafter.head.eval()
    if not count:
        raise ValueError('no usable development targets')
    return {'split': 'dev', 'dev_loss': numerator / count, 'loss_numerator': numerator,
            'valid_positions': count, 'anchor_plan_hash': canonical_hash(plan),
            'development_hash': canonical_hash(records), 'criterion': 'unweighted_valid_response_token_ce'}


def save_checkpoint(path, drafter, optimizer, counters, data_hash, best_dev_loss=None, sampler=None, meter=None):
    return atomic_checkpoint(path, {'schema_version': 2, 'identity': identity(drafter, data_hash),
        'architecture': getattr(drafter, 'compatibility', None), 'data_hash': data_hash,
        'training_contract':getattr(drafter,'training_contract',None),
        'model': student_state(drafter), 'optimizer': optimizer.state_dict(), 'counters': counters,
        'rng': rng_state(sampler or random.Random(0)), 'best_dev_loss': best_dev_loss}, meter)


def restore_checkpoint(path, drafter, optimizer, data_hash, sampler=None):
    payload = read_checkpoint(path, identity(drafter, data_hash))
    if payload.get('optimizer') is None or payload.get('rng') is None:
        raise ValueError('resume requires a recovery checkpoint; selected deployment weights omit optimizer/RNG')
    load_student(drafter, payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'], sampler or random.Random(0))
    return payload['counters']


def train(target, drafter, records, meter, data_hash, steps=2, seed=0, resume=None, learning_rate=1e-3,
          dev_records=None, evaluation_interval=1, codec=None, latent_weight=1.0, latent_scales=None,
          checkpoint_interval=1,latent_level_weights=None,latent_cache=None,fixed_codec_transfer=False):
    records = [normalize_record(r, 'train', r.get('origin', 'target_distillation')) for r in records]
    usable = [r for r in records if anchors(r)]
    if not usable:
        raise ValueError('no response anchors with an observed successor')
    if dev_records is not None:
        disjoint(records, dev_records)
    if codec and not fixed_codec_transfer and getattr(codec,'training_data_hash',None)!=canonical_hash(records):
        raise ValueError('total-data comparison requires codec and student to use the same admitted corpus')
    if evaluation_interval <= 0 or checkpoint_interval <= 0:
        raise ValueError('evaluation and recovery intervals must be positive')
    latent_scales = latent_scales or (getattr(codec, 'latent_scales', None) if codec else None)
    levels=getattr(getattr(drafter,'latent',None),'levels',())
    level_weights=({int(k):float(v) for k,v in latent_level_weights.items()} if latent_level_weights is not None
                   else {n:1/len(levels) for n in levels})
    if (set(level_weights)!=set(levels) or any(not math.isfinite(v) or v<0 for v in level_weights.values())
            or levels and sum(level_weights.values())<=0 or not math.isfinite(latent_weight) or latent_weight<0):
        raise ValueError('invalid explicit latent-loss weights')
    drafter.training_contract = {'train_hash': data_hash, 'dev_hash': canonical_hash(dev_records),
        'selection': 'unweighted_valid_response_token_ce', 'evaluation_interval': evaluation_interval,
        'learning_rate': learning_rate, 'data_seed': seed, 'latent_weight': latent_weight,
        'latent_level_weights':level_weights,
        'latent_cache':latent_cache.identity if latent_cache else 'on_demand',
        'data_curve':'fixed_codec_transfer' if fixed_codec_transfer else 'total_data',
        'latent_scales': canonical_hash({str(k): v.tolist() for k,v in (latent_scales or {}).items()})}
    sampler = random.Random(seed)
    optimizer = torch.optim.AdamW([p for p in drafter.parameters() if p.requires_grad], lr=learning_rate)
    counters = {'optimizer_steps': 0, 'source_token_passes': 0, 'supervised_future_positions': 0,
        'sampled_anchors': 0, 'unique_teacher_tokens': sum(len(r['input_ids']) for r in records),
        'generated_teacher_tokens': sum(r.get('generated_tokens', 0) for r in records),
        'losses': [], 'label_positions': [], 'sampling_order': [], 'encountered_sources': {},
        'distinct_source_positions': 0, 'distinct_eligible_response_positions': 0,
        'available_response_positions': sum(sum(r['response_mask']) for r in records),
        'feature_cache_replays': 0,
        'latent_cache_hits':0,'new_label_target_forwards':0,
        'skipped_unusable_records': len(records) - len(usable)}
    if resume:
        counters = restore_checkpoint(resume, drafter, optimizer, data_hash, sampler)
        meter.steps = counters['optimizer_steps']
        meter.source_tokens = counters['source_token_passes']
        meter.supervised_positions = counters['supervised_future_positions']
        parent_metrics=Path(resume).parent/'metrics.json'
        if parent_metrics.exists():
            parent=read(parent_metrics)
            meter.source_tokens=max(meter.source_tokens,parent.get('charged_source_positions',meter.source_tokens))
            meter.steps=max(meter.steps,parent.get('charged_optimizer_attempts',meter.steps))
            meter.supervised_positions=max(meter.supervised_positions,parent.get('charged_supervised_positions',meter.supervised_positions))
        meter.inherited_steps = meter.steps
        meter.inherited_source_tokens = meter.source_tokens
        meter.inherited_supervised_positions = meter.supervised_positions
        meter.started -= counters.get('cumulative_wall_seconds', 0)
    started, status, reason = time.monotonic(), 'completed', None
    inherited_steps = counters['optimizer_steps']
    inherited_positions = meter.inherited_source_tokens
    path = meter.artifact_dir / 'final.pt'
    observations, best = [], float('inf')
    if resume:
        best = read_checkpoint(resume).get('best_dev_loss') or best
        previous = Path(resume).parent / 'selection.json'
        if previous.exists():
            selection = read(previous)
            selected_path = previous.parent / selection['weights']
            if digest(selected_path) != selection['weights_sha256']:
                raise ValueError('resumed selection weights changed')
            observations = list(selection['candidates'])
            atomic_checkpoint(meter.artifact_dir / 'selected.pt', read_checkpoint(selected_path), meter)
    drafter.train()
    drafter.embeddings.eval()
    drafter.head.eval()
    if codec:
        codec.eval().requires_grad_(False)
    try:
        while counters['optimizer_steps'] < steps:
            row = usable[sampler.randrange(len(usable))]
            anchor = sampler.choice(anchors(row))
            valid = int(block_labels(row, anchor).ne(-100).sum())
            source = len(row['input_ids'])
            cached=(latent_cache.get(row,anchor,codec.levels,target.get_input_embeddings().weight.device)
                    if latent_cache and codec and latent_weight else None)
            extra = source if codec and latent_weight and cached is None else 0
            meter.check(source + extra, valid)
            preflight(meter, 3 * tensor_bytes(student_state(drafter)) + 2 * 1024 * 1024)
            meter.source_tokens+=source
            logits, labels = forward_block(target, drafter, row, anchor)
            loss = final_token_loss(logits, labels)
            if codec and latent_weight:
                from ncm_lab.training.latent_supervision import teacher_block, latent_loss
                if cached is not None:
                    codes,mask=cached
                    if not torch.equal(mask,labels.ne(-100)):
                        raise ValueError('cached latent validity differs from the response labels')
                    counters['latent_cache_hits']+=1
                else:
                    meter.source_tokens+=extra
                    block, mask, _ = teacher_block(target, row, anchor)
                    with torch.no_grad():
                        codes = codec.encode(block, mask)
                    counters['new_label_target_forwards']+=1
                loss = loss + latent_weight * latent_loss(drafter.latent.predictions, codes, mask, latent_scales,level_weights)
            if not torch.isfinite(loss):
                raise ValueError('non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            meter.record(0,valid)
            optimizer.step()
            counters['optimizer_steps'] += 1
            counters['source_token_passes'] = meter.source_tokens
            counters['supervised_future_positions'] += valid
            counters['sampled_anchors'] += 1
            counters['losses'].append(float(loss.detach()))
            counters['label_positions'].append([anchor + 1, min(anchor + 15, len(row['input_ids']) - 1)])
            source_id = row.get('original_id') or canonical_hash([row['group_id'],row['input_ids']])
            counters['sampling_order'].append([row['group_id'], source_id, anchor])
            counters['encountered_sources'][source_id] = [source, sum(row['response_mask'])]
            counters['distinct_source_positions'] = sum(v[0] for v in counters['encountered_sources'].values())
            counters['distinct_eligible_response_positions'] = sum(v[1] for v in counters['encountered_sources'].values())
            counters['distinct_sampled_blocks'] = len(set(map(tuple,counters['sampling_order'])))
            counters['cumulative_wall_seconds'] = time.monotonic() - meter.started
            observation = None
            if dev_records is not None and counters['optimizer_steps'] % evaluation_interval == 0:
                observation = evaluate_development(target, drafter, dev_records, meter)
                counters['source_token_passes'] = meter.source_tokens
            if not observation and counters['optimizer_steps'] % checkpoint_interval and counters['optimizer_steps'] != steps:
                continue
            artifact_id = save_checkpoint(path, drafter, optimizer, counters, data_hash,
                min(best, observation['dev_loss']) if observation else best, sampler, meter)
            if observation:
                observation.update(step=counters['optimizer_steps'], artifact_id=artifact_id)
                observation.update(source_token_passes=counters['source_token_passes'],
                                   distinct_source_positions=counters['distinct_source_positions'])
                write_json(meter.artifact_dir / ('dev-%06d.json' % observation['step']), observation, exclusive=True)
                observations.append(observation)
                if observation['dev_loss'] < best:
                    best = observation['dev_loss']
                    atomic_checkpoint(meter.artifact_dir / 'selected.pt', selected_payload({
                        'schema_version':2, 'identity':identity(drafter,data_hash),
                        'architecture':getattr(drafter,'compatibility',None), 'data_hash':data_hash,
                        'training_contract':getattr(drafter,'training_contract',None),
                        'model':student_state(drafter), 'counters':counters, 'source_checkpoint':artifact_id}),meter)
    except (BudgetExceeded, KeyboardInterrupt) as exc:
        status, reason = 'cancelled', str(exc) or 'interrupted; recover last completed checkpoint'
    if observations:
        chosen = select_checkpoint(observations)
        write_json(meter.artifact_dir / 'selection.json', {'schema_version': 2, 'criterion': chosen['criterion'],
            'tie_break': 'loss, earliest step, digest', 'candidates': observations, 'selected': chosen,
            'weights': 'selected.pt', 'weights_sha256': digest(meter.artifact_dir / 'selected.pt'),
            'development_hash': canonical_hash(dev_records)}, exclusive=True)
    return {'status': status, 'reason': reason, **counters, 'elapsed_seconds': time.monotonic() - started,
        'inherited_optimizer_steps':inherited_steps, 'inherited_target_positions':inherited_positions,
        'new_invocation_optimizer_steps':counters['optimizer_steps']-inherited_steps,
        'new_invocation_target_positions':meter.source_tokens-inherited_positions if status=='completed' else None,
        'charged_source_positions':meter.source_tokens,'charged_optimizer_attempts':meter.steps,
        'charged_supervised_positions':meter.supervised_positions,
        'incomplete_measurement_reason':None if status=='completed' else 'interrupted update; charged inputs/attempts conservatively bound incomplete work',
        'parameter_counts': drafter.parameter_counts(), 'final_checkpoint': path.name if path.exists() else None,
        'final_checkpoint_artifact_id': 'sha256:' + digest(path) if path.exists() else None,
        'selection': 'selection.json' if observations else 'recovery only; no held-out selection',
        'storage_bytes': sum(p.stat().st_size for p in meter.artifact_dir.rglob('*') if p.is_file())}


def select_checkpoint(development_records):
    valid = [r for r in development_records if r.get('split') == 'dev' and r.get('dev_loss') is not None]
    if not valid or len(valid) != len(development_records):
        raise ValueError('checkpoint selection requires development-only finite losses')
    if any(not torch.isfinite(torch.tensor(r['dev_loss'])) for r in valid):
        raise ValueError('non-finite checkpoint selection loss')
    return min(valid, key=lambda r: (r['dev_loss'], r['step'], r['artifact_id']))
