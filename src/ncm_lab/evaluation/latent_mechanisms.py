"""Oracle diagnostics and real interventions, excluded from deployed timing."""
import torch
from torch.nn import functional as F
from ncm_lab.training.loop import forward_block
from ncm_lab.training.latent_supervision import teacher_block
from ncm_lab.training.codec import codec_objective


@torch.no_grad()
def _diagnose(target, drafter, codec, row, anchor):
    drafter.eval()
    baseline, labels = forward_block(target, drafter, row, anchor)
    predictions = {n: z.detach().clone() for n, z in drafter.latent.predictions.items()}
    states, valid, _ = teacher_block(target, row, anchor)
    normalization=getattr(codec,'normalization',1.0)
    _, oracle = codec_objective(codec, target.get_output_embeddings(), states, valid, labels,normalization=normalization)
    predicted = {}
    for n, z in predictions.items():
        recovered = codec.reconstruct(z, n)
        target_states,recovered_states=states[valid],recovered[valid]
        kl,nll,agree=0.0,0.0,0
        for start in range(0,target_states.shape[0],8):
            teacher_logits=target.get_output_embeddings()(target_states[start:start+8]).float()
            reconstructed_logits=target.get_output_embeddings()(recovered_states[start:start+8]).float()
            kl+=float(F.kl_div(F.log_softmax(reconstructed_logits,-1),F.log_softmax(teacher_logits,-1),
                              log_target=True,reduction='sum'))
            nll+=float(F.cross_entropy(reconstructed_logits,labels[valid][start:start+8],reduction='sum'))
            agree+=int((reconstructed_logits.argmax(-1)==teacher_logits.argmax(-1)).sum())
        predicted[str(n)] = {"mse": float((recovered[valid].float() - states[valid].float()).square().mean()),
                            'normalized_mse':float((recovered[valid].float()-states[valid].float()).square().mean())/normalization,
                            "head_kl":kl/target_states.shape[0],"reference_nll":nll/target_states.shape[0],
                            "top_token_agreement":agree/target_states.shape[0],
                            "code_variance": float(z.float().var(unbiased=False)),
                            "effective_rank": int(torch.linalg.matrix_rank(z[0].float()))}
    interventions = {}
    for n in drafter.latent.levels:
        drafter.latent.disabled_levels = {n}
        changed, _ = forward_block(target, drafter, row, anchor)
        interventions[str(n)] = {"changed_proposals": int((changed.argmax(-1) != baseline.argmax(-1)).sum()),
                                 "max_logit_difference": float((changed - baseline).abs().max())}
    drafter.latent.disabled_levels = set()
    level_zeroing={}
    for n in drafter.latent.levels:
        drafter.latent.ablated_levels={n}
        changed,_=forward_block(target,drafter,row,anchor)
        level_zeroing[str(n)]={'changed_proposals':int((changed.argmax(-1)!=baseline.argmax(-1)).sum()),
                              'max_logit_difference':float((changed-baseline).abs().max())}
    drafter.latent.ablated_levels=set()
    skip_intervention=None
    if drafter.skip_enabled:
        gate=drafter.skip_gate.detach().clone()
        drafter.skip_gate.zero_()
        changed,_=forward_block(target,drafter,row,anchor)
        drafter.skip_gate.copy_(gate)
        skip_intervention={"gate":float(gate.tanh()),
            "changed_proposals":int((changed.argmax(-1)!=baseline.argmax(-1)).sum()),
            "max_logit_difference":float((changed-baseline).abs().max())}
    return {"oracle_codec": oracle, "causal_predictions": predicted, "interventions": interventions,
            "target_processed_positions":len(row['input_ids'])*(2+2*len(drafter.latent.levels)+int(drafter.skip_enabled)),
            'level_zeroing':level_zeroing,
            'intervention_definitions':{'interventions':'remove only this bridge; refinement retained',
                'level_zeroing':'zero predicted code before refinement/injection; frozen upsampler offsets retained'},
            "original_skip_intervention":skip_intervention,
            "draft_reference_nll":float(F.cross_entropy(baseline[valid].float(),labels[valid])),
            "positional_agreement":[bool(value) if bool(mask) else None for value,mask in
                                     zip((baseline.argmax(-1)==labels)[0],valid[0])],
            "gates": {n: float(g.tanh()) for n, g in drafter.latent.gates.items()},
            "scope": "held-out diagnostic; oracle unavailable during generation"}


@torch.no_grad()
def diagnose(target,drafter,codec,row,anchor):
    prior_training=drafter.training
    prior_diagnostic=drafter.latent.diagnostic
    prior_disabled=set(drafter.latent.disabled_levels)
    prior_ablated=set(drafter.latent.ablated_levels)
    skip=drafter.skip_gate.detach().clone() if drafter.skip_enabled else None
    drafter.latent.diagnostic=True
    drafter.latent.disabled_levels=set()
    drafter.latent.ablated_levels=set()
    try:
        return _diagnose(target,drafter,codec,row,anchor)
    finally:
        drafter.latent.diagnostic=prior_diagnostic
        drafter.latent.disabled_levels=prior_disabled
        drafter.latent.ablated_levels=prior_ablated
        if skip is not None:
            drafter.skip_gate.copy_(skip)
        drafter.train(prior_training)
        drafter.embeddings.eval()
        drafter.head.eval()
