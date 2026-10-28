"""Authoritative method resolution and homogeneous placement before optimization."""
from dataclasses import dataclass, asdict
import math
import torch
from ncm_lab.models.hierarchy import ProgressiveDrafter
from ncm_lab.models.latent_adapters import LatentAdapters
from ncm_lab.runtime.io import canonical_hash


SCHEDULES = {"EARLIER": (4, 8, 15, 15, 15), "LATER": (4, 4, 4, 15, 15),
             "LATE_NEAR_BUDGET": (8, 8, 8, 8, 15)}
BASE = {"AR", "FLAT_RELEASED", "FLAT_ADAPTED", "HIER_SKIP", "HIER_NO_SKIP", "FIXED_BROADCAST",
        "FLAT_COST", "STRONG_EXTERNAL", "HIER_POOLED_CONTEXT", *SCHEDULES}
LATENT = {"FLAT_MULTI_AE", "HIER_MULTI_AE", "HIER_MULTI_AE_NO_SKIP", "HIER_LATENT_CE_ONLY",
          "HIER_CODEC_CE_ONLY", "HIER_MULTI_AE_AUX_ONLY", "HIER_LINEAR_CODEC",
          "HIER_INDEPENDENT_AE", "HIER_SINGLE_AE"}


@dataclass(frozen=True)
class Method:
    name: str
    schedule: tuple
    skip: bool
    learned_expansion: bool
    initial_span: bool
    layer_indices: tuple
    device: str
    precision: str
    init_seed: int
    data_seed: int
    latent: bool
    codec_id: str | None
    rank: int = 64
    context_policy: str = "all_confirmed_features"
    backend: str = "transformers_incremental_sdpa"

    @property
    def signature(self):
        return canonical_hash(asdict(self))


def resolve_method(config, operation="diagnostic"):
    name = config.get("method", "HIER_SKIP")
    if name not in BASE | LATENT:
        raise ValueError(f"unknown method: {name}")
    if name == "STRONG_EXTERNAL":
        from ncm_lab.models.external_dcore import DCORE_ID,DCORE_REVISION
        if config.get('model',{}).get('draft_id')!=DCORE_ID or config.get('model',{}).get('draft_revision')!=DCORE_REVISION:
            raise ValueError('STRONG_EXTERNAL unavailable: configure the actual pinned DCore checkpoint, not ordinary DFlash')
        if operation=='train':
            raise ValueError('released external reference does not perform adaptation')
    for key, value in {"proposal_horizon": 15, "anchor_convention": "known_response_anchor",
                       "skip_source": 2, "skip_destination": 5, "sample_from_anchor": False}.items():
        if key in config and config[key] != value:
            raise ValueError(f"unsupported scientific option: {key}")
    if config.get("method_options"):
        raise ValueError("unknown method_options; use explicit schema fields")
    if config.get("attention_backend", "sdpa") != "sdpa" or config.get("compile", False):
        raise ValueError("only uncompiled SDPA execution is implemented")
    layers = () if name=='AR' else tuple(config.get("layer_indices", range(5)))
    if name == "FLAT_COST":
        if not config.get("cost_selection") or len(layers) >= 5:
            raise ValueError("FLAT_COST needs a development cost-selection receipt and reduced layer indices")
        if operation in {"train", "evaluate"}:
            from ncm_lab.runtime.io import read, digest
            selection = read(config["cost_selection"])
            if (digest(config["cost_selection"]) != config.get("cost_selection_sha256")
                    or selection.get("split") != "dev" or tuple(selection.get("layer_indices", [])) != layers
                    or selection.get("relative_cost_error", 1) > selection.get("tolerance", 0)):
                raise ValueError("invalid reduced-depth development cost selection")
    elif name!='AR' and layers != tuple(range(5)):
        raise ValueError("only FLAT_COST changes inherited depth")
    flat = name.startswith("FLAT") or name in {"AR","STRONG_EXTERNAL"}
    schedule = (15,) * len(layers) if flat else SCHEDULES.get(name, (4, 4, 8, 15, 15))
    skip = not flat and name not in {"HIER_NO_SKIP", "HIER_MULTI_AE_NO_SKIP"}
    for key, expected in (("schedule", schedule), ("skip", skip)):
        actual = config.get(key, expected)
        if (tuple(actual) if key == "schedule" else actual) != expected:
            raise ValueError(f"{name}: contradictory {key}")
    latent = name in LATENT
    if "latent_weight" in config:
        weight=config["latent_weight"]
        if isinstance(weight,bool) or not isinstance(weight,(float,int)) or not math.isfinite(weight):
            raise ValueError('latent_weight must be finite numeric')
        if not latent or name in {"HIER_CODEC_CE_ONLY","HIER_LATENT_CE_ONLY"} and weight!=0 or name not in {"HIER_CODEC_CE_ONLY","HIER_LATENT_CE_ONLY"} and weight<=0:
            raise ValueError("latent-loss override contradicts the named treatment")
    codec_id = config.get("codec_sha256")
    if latent and name != "HIER_LATENT_CE_ONLY" and not codec_id:
        raise ValueError("latent transfer requires selected frozen codec identity")
    if operation == "evaluate" and name not in {"AR", "FLAT_RELEASED","STRONG_EXTERNAL"} and not config.get("checkpoint_sha256"):
        raise ValueError("adapted evaluation requires a selected checkpoint")
    if name == "FLAT_RELEASED" and config.get("initial_span", False):
        raise ValueError("released flat recipe has zero fixed initial offsets")
    rank = config.get("skip_rank", 64)
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("skip_rank must be positive")
    return Method(name, schedule, skip, name not in {"FIXED_BROADCAST", "FLAT_RELEASED","STRONG_EXTERNAL"},
                  config.get("initial_span", name not in {"FLAT_RELEASED","STRONG_EXTERNAL"}), layers,
                  config.get("device", "cpu"), config.get("precision", "float32"),
                  config.get("init_seed", config.get("seed", 0)), config.get("data_seed", config.get("seed", 0)),
                  latent, codec_id, rank,
                  "mean_confirmed_features" if name == "HIER_POOLED_CONTEXT" else "all_confirmed_features")


def build_drafter(target, backbone, method, codec=None, latent_width=256):
    if method.name == "AR":
        raise ValueError("AR has no drafter; load target/tokenizer only")
    if method.name=='STRONG_EXTERNAL':
        from ncm_lab.models.external_dcore import DCoreReferenceDraft
        if not isinstance(backbone,DCoreReferenceDraft):
            raise ValueError('external reference requires its audited DCore implementation and weights')
    if codec is not None and codec.spec["hidden_size"] != target.config.hidden_size:
        raise ValueError("codec prediction-state width differs from the frozen target")
    device, dtype = torch.device(method.device), getattr(torch, method.precision)
    if device.type=='cuda' and device.index is None:
        if not torch.cuda.is_available():
            raise ValueError('requested CUDA placement is unavailable')
        device=torch.device('cuda',torch.cuda.current_device())
    elif device.type=='cpu':
        device=torch.device('cpu')
    elif device.type!='cuda':
        raise ValueError('only CPU and CUDA placement are supported')
    if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise ValueError("unsupported precision")
    if hasattr(target, "hf_device_map") and len(set(target.hf_device_map.values())) > 1:
        raise ValueError("sharded target placement is unsupported")
    target.to(device=device, dtype=dtype).requires_grad_(False).eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(method.init_seed)
        if method.layer_indices != tuple(range(len(backbone.layers))):
            backbone.layers = torch.nn.ModuleList([backbone.layers[i] for i in method.layer_indices])
        drafter = ProgressiveDrafter(backbone, target, method.schedule, method.skip,
                                    method.learned_expansion, rank=method.rank,
                                    initial_span_learned=method.initial_span)
        if method.latent:
            if method.name == "HIER_LATENT_CE_ONLY" and codec is not None:
                raise ValueError("capacity control must initialize its upsamplers without a fitted codec")
            if method.name != "HIER_LATENT_CE_ONLY" and codec is None:
                raise ValueError("selected frozen codec must be loaded before constructing transfer")
            expected_kind = {"HIER_LINEAR_CODEC": "linear", "HIER_INDEPENDENT_AE": "independent",
                             "HIER_SINGLE_AE": "independent"}.get(method.name,"nested")
            if method.name == "HIER_LATENT_CE_ONLY":
                expected_kind = None
            if expected_kind and codec.spec["kind"] != expected_kind:
                raise ValueError("control requires its declared distinct codec treatment")
            if expected_kind == "linear" and codec.spec.get("architecture_version",1)<2:
                raise ValueError("legacy linear codec does not implement the current nested linear control")
            if method.name == "HIER_SINGLE_AE" and codec.levels != (8,):
                raise ValueError("single-scale transfer requires exactly one eight-slot codec")
            if codec is not None and method.name != "HIER_SINGLE_AE" and codec.levels != (4,8,15):
                raise ValueError("multilevel transfer requires the three declared independent code levels")
            width = codec.spec["width"] if codec else latent_width
            drafter.latent = LatentAdapters(backbone.config.hidden_size, width, codec,
                inject=method.name != "HIER_MULTI_AE_AUX_ONLY", single=method.name == "HIER_SINGLE_AE",
                train_upsamplers=method.name == "HIER_LATENT_CE_ONLY")
    drafter.to(device=device, dtype=dtype)
    drafter.resolved_method = asdict(method)
    drafter.context_policy = method.context_policy
    drafter.compatibility = {"method": asdict(method), "target": target.config.to_dict(),
                             "draft": backbone.config.to_dict(), "taps": backbone.target_layer_ids}
    drafter.compatibility['execution'] = execution_settings()
    drafter.compatibility['placement']={'device':str(device),'dtype':str(dtype)}
    from ncm_lab.runtime.provenance import implementation_hash
    drafter.compatibility["implementation_sha256"] = implementation_hash()
    for param in drafter.parameters():
        if param.device != device or param.is_floating_point() and param.dtype != dtype:
            raise ValueError("heterogeneous model placement")
    assert drafter.embeddings is target.get_input_embeddings() and drafter.head is target.get_output_embeddings()
    return drafter


def execution_settings():
    return {'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'cuda_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cpu_threads':torch.get_num_threads(),
        'flash_sdp_enabled':torch.backends.cuda.flash_sdp_enabled(),
        'math_sdp_enabled':torch.backends.cuda.math_sdp_enabled(),
        'mem_efficient_sdp_enabled':torch.backends.cuda.mem_efficient_sdp_enabled()}
