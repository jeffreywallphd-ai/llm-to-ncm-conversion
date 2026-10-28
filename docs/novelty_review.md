# Preliminary novelty and upstream audit — 2026-10-25

No novelty claim is established. The testable candidate is **changing unknown future
query resolution across existing drafter layers**, with explicit expansion and
measured acceptance-versus-cycle-cost controls. Hierarchy, concepts, feature taps,
zero gates and fair comparisons are already prior art. Inspect further overlaps
before research-scale training; revise the contribution if an identical mechanism exists.

| Work | Compressed object / prefix or future | Resolution changes | Target modified / codec | Skips or gates | Exact verification / published evidence |
|---|---|---|---|---|---|
| [Original NCM](https://arxiv.org/abs/2610.14162v1) | Text into multiscale discrete codes; generated future | Codec/generator levels | Native generator; residual VQ codec | Multiscale architecture | No unchanged-AR-target verification; preprint and official released code |
| [DFlash](https://arxiv.org/abs/2602.06036v2) | Block-parallel future queries; target prefix features | Fixed full block in inspected layers | Frozen AR target; no codec | Standard block residuals | Exact speculative verification; paper reports ICML 2026 acceptance; released checkpoint |
| [DFlash2](https://inco.ai/blog/dflash2/) | Future candidates/dependencies | Fixed block with dynamic convolution and selector | AR verification; no required VQ codec | Local dependencies and path selector | Technical blog and released model/inference code; current recipe completeness not certified |
| [NCP-ArchPreview / concept-conditioned drafter](https://arxiv.org/pdf/2609.10715) | Quantized target concepts; completed prefix chunk conditions future proposals | Target token/concept hierarchy, not the proposed 4→8→15 drafter schedule | Target jointly pretrained with NCP; product-quantized concepts | Zero-initialized tanh concept gates in drafter layers | Report §5.3/Table 9 studies concept-conditioned block drafting with exact verification; modified target differs from frozen Qwen |
| [LongSpark](https://arxiv.org/abs/2609.37029v1) | Confirmed-prefix boundary/window/global views | Prefix interface compression | Target verification supplies views; no text VQ codec | Fixed-size context interfaces | Preprint and official code; quickstart requires two P2P GPUs |
| [DCore](https://arxiv.org/abs/2610.24993v2) | Shared full-resolution drafter comparisons/context/objective | No progressive future-query schedule established by inspected recipe | Frozen compatible target; no required VQ codec | Recipe-dependent components | Controlled training/inference study and releases; acceptance-versus-cost fairness is their contribution |
| [HiLP](https://arxiv.org/abs/2608.05806v1) | Higher-level future latent prediction | Auxiliary hierarchical latent rollout | LM pretraining changes; codec requirement needs full-code audit | Higher-level latent guidance | Preprint; latent prediction and speculative efficiency precede this project |
| [MuLo-SD](https://arxiv.org/abs/2601.05149v2) | Low-resolution image-token proposals | Upsampling into high-resolution proposals | Image AR verifier, not language | Local spatial resampling | CVPR 2026 noted in source; token upsampling and multiresolution speculation are prior art |
| This hypothesis | Unknown future states inside one drafter | Existing layers 4,4,8,15,15; anchor separate | Unchanged frozen Qwen; auxiliary continuous prediction-state codec | One lifted coarse-to-leaf low-rank skip | CPU implementation only; quality/runtime unmeasured |

NCP's report does not use the literal label “NCPFlash” in the inspected text. The
comparison above refers to its documented concept-conditioned speculative module,
not an assumed separately reproduced checkpoint. Its confirmed-prefix concept
injection with zero tanh gates overlaps any broad concept/skip claim directly.

## Backbone decision and availability

Exact commits, inspected file hashes, model cards/configs and licenses are recorded
in `third_party/sources.lock.yaml`. Initial model metadata is saved under
`third_party/configs`. No model weights were downloaded. The upstream DFlash model
snapshot is vendored unchanged under its MIT notice; the local adapter invokes its
actual layers. DCore CE/mask code and its five-layer gamma-7 recipe were inspected;
the local valid-token normalization is a documented adaptation, not exact reproduction.

| Baseline | Target / checkpoint | Backend and devices | License | Current status and exclusions |
|---|---|---|---|---|
| AR | Qwen/Qwen3-8B at locked target SHA | Transformers correctness reference, one device | Apache-2.0 model card; verify terms at weight acquisition | tiny-config CPU validated; real checkpoint not cached; optimized timing missing |
| FLAT_RELEASED | z-lab/Qwen3-8B-DFlash-b16 at locked SHA | Vendored DFlash layers; optimized SGLang/vLLM separately | MIT drafter/code | implemented_cpu_validated on tiny configurations; released-weight/GPU parity unverified |
| STRONG_EXTERNAL candidate | ernie-research/DCore-DFlash-Qwen3-8B-Budget16 at locked SHA | Pinned DCore source adapted to shared Transformers SDPA; published SGLang stack separate | MIT source; model-card weights license unspecified | direct-forward CPU parity tested; released-weight/GPU reproduction pending |
| DFlash2 | Current blog uses Qwen3.8-27B family | Current production-engine integrations | Check each checkpoint's terms | no verified drop-in original Qwen3-8B pair/training reproduction; exclude rather than substitute target |
| LongSpark | Qwen3-8B + Hehy/LongSpark-Qwen3-8B from its README | Two P2P GPUs in documented quickstart | MIT source; verify checkpoint | single-Spark adaptation absent; published multi-GPU speeds not comparable |
| NCP concept-conditioned | NCP-ArchPreview target and associated drafter | Modified pretrained target, not frozen Qwen3-8B | release-specific, not audited for reuse | architectural prior art; incompatible primary target, not a zero-throughput result |

Follow-up audit: inspect Hourglass/other shorten-and-upsample language architectures,
the full HiLP and MuLo implementation paths, current compatible optimized backends,
and DFlash2 training-release completeness before claiming a distinct mechanism.
Unknown or unavailable evidence stays explicit. This preliminary audit is sufficient
to avoid broad novelty claims, not to certify exhaustive literature coverage.


## Frozen-target codec transfer review — 2026-10-27

The earlier baseline-only hypothesis is superseded by research_plan.md. No novelty conclusion follows from implementation. NCM already combines multiscale coding and latent generation; NCP already conditions drafting on concept representations. The proposed distinction requires testing extraction from an unchanged ordinary target and the complete representation/prediction/runtime tradeoff. HiLP remains close prior art; full comparison must be refreshed before research-scale training.

[DFlare v2](https://arxiv.org/html/2606.02091v2) learns per-draft-layer combinations of target layers and studies deeper drafting and more training data. It is a stronger conditioning comparison, currently unavailable as a verified compatible local adapter. [EAGLE-3, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/c7b5a35ea98b62512a869c19ea7b03cb-Abstract-Conference.html) removes feature-prediction constraints in favor of token prediction and multi-layer fusion. Its mechanism motivates checking whether latent reconstruction hurts token prediction. [Matryoshka SAEs, ICML 2025](https://proceedings.mlr.press/v267/bussmann25a.html) independently reconstruct from nested dictionaries; our slot-count hierarchy is a different bottleneck, not a general claim to invent nested reconstruction. These primary sources were inspected during remediation. Reported paper speedups are their results and have not been reproduced here.
