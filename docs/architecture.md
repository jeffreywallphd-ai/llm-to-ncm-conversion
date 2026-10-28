# Architecture and alignment contract

Scope: NCM-inspired hierarchical speculative drafting, not native NCM conversion.
Qwen3-8B remains a frozen conventional autoregressive target; thinking is disabled.
The exact target/draft/tokenizer revisions are in `configs/models/qwen3_8b_dflash.yaml`.
The inspected released drafter has five layers, width 4096, 32 attention heads,
8 KV heads, head dimension 128, and target taps [1,9,17,25,33]. Those values are
read from loaded configurations and checked; they are not assumptions for new models.

The upstream model snapshot is unchanged. `ProgressiveDrafter` reuses its feature
projection, normalization, Qwen3 blocks, rotary interface and final vocabulary head.
A flat wrapper is checked against a direct upstream call with identical tiny weights.
This is implemented CPU parity, not reproduction of a released checkpoint benchmark.

## Anchor and nested queries

At anchor position a, token x_a is known; target KV contains positions <a.
`sample_from_anchor=false`, `anchor_slots=1`, `proposal_horizon=15` and full slots=16.
The anchor embedding is available, but its target hidden state is not yet available.
Clean target taps may come only from same-document positions <a. A coarse query's
representative position never sets the prefix cutoff. Tap i means hidden_states[i+1]
in the selected upstream helper (embedding output is hidden_states[0]).

Groups at resolution four are [1..4],[5..8],[9..12],[13..15]; at eight they are
[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15]; at fifteen they are singletons.
The anchor is never pooled. Each partition covers future offsets once, in order.
Parent maps are explicit registered buffers. Representative rotary position is
a+floor((first+last)/2); singleton positions restore a+1..a+15 exactly. The target's
positions are unchanged. This coarse encoding is an architectural choice to ablate.

Layers receive actual total query lengths [5,5,9,16,16], versus [16]*5 for flat.
The restricted work proxy decreases from 80 to 51 updates (36.25%). This omits
prefix work, head, verification, memory traffic and kernel utilization; no latency
prediction follows. Profiling returns timings without claiming a speedup.

## Main path, expansion, skip and gradients

Coarse initialization uses the frozen mask embedding plus a zero-initialized trainable
span offset; the known anchor uses its frozen token embedding. No gold future token
embedding enters the draft. Synthetic queries interact noncausally within their own
block and attend to the allowed clean prefix. Batch rows isolate anchor blocks.
The initial reference trains one source document per target call; packed target
extraction across documents is not enabled. Supplied feature blocks still apply
per-document and per-anchor visibility, including legitimate later-anchor prefixes.

After layers 2 and 3, expansion gathers each child's parent and adds zero-initialized
trainable child offsets. No full-resolution hidden transformer branch is computed.
Layer-2 output is lifted to leaves and projected by normally initialized low-rank
factors before layer 5, multiplied by tanh(g) with g=0 initially. The gate receives
a learning signal first; factors receive gradients once the gate moves. Zero gate
does not make the hierarchy equivalent to flat. Optional bridges are disabled.

Target parameters and extracted features are frozen/detached. Feature projection,
blocks, expansion and skips remain trainable. Embeddings/head are frozen, while the
head operation remains autograd-tracked so loss gradients reach drafter states.
Trainable and total parameter counts are recorded separately.

The inspected DCore CE recipe uses exponential positional decay, gamma=7 and no
anchor prediction. Here L=sum_valid exp(-j/7) CE(q_j,y_j)/N_valid, j=0..14.
Unlike upstream batch reduction this normalizes by supervised positions. This
declared reference adaptation is applied identically to paired flat/hierarchy
controls; it is not claimed as exact upstream training reproduction. Distillation
and auxiliary objectives are not silently added. Loss labels are future-only.

## Verification example

The target first emits x_a once. A cycle verifies [x_a,q_(a+1),...,q_(a+15)].
If k proposals match the target greedy choices, retain target state through a+k,
crop the rejected suffix, and carry correction x_(a+k+1) as the next anchor. If
all pass, carry the target bonus token. Cache length is old_length+1+k, where
old_length=a. The old anchor is not another newly emitted token in that cycle.
Synthetic draft states are never committed to target KV.

The single-request Transformers reference implements actual KV cropping. Its
proposal adapter recomputes permitted features, so it is for correctness and
not optimized speed claims. Padding is rejected there; callers batch as separate
unpadded requests. Empty prompts require explicit BOS. EOS, output limits and
tokenized stop sequences terminate without suppressing short outputs.

The finite-probability reference also supports deterministic-latent independent
categorical proposals with actual q: acceptance min(1,p/q), positive residual
max(p-q,0) at first rejection and target bonus after full acceptance. Its finite
distribution tests do not certify arbitrary model sampling, latent marginalization,
history-dependent processors or GPU sampling. The initial research protocol is greedy.

## Revision-2 implementation update — 2026-10-27

The current first-paper extension is the auxiliary frozen-target codec and causal latent transfer described in autoencoder_architecture.md. It is implemented in shared model/training modules; native target conversion remains future work. Production inference uses incremental confirmed-prefix target features and predicted codes only.
