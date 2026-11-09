"""One traversal of original Qwen blocks, with sequential embedded refinements."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ncm_lab.config.architecture import Architecture
from ncm_lab.embedded.geometry import resize, position_features, attention_geometry


class Refinement(nn.Module):
    def __init__(self, hidden, width):
        super().__init__()
        self.down = nn.Linear(hidden, width)
        self.position = nn.Linear(4, width)
        self.network = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width)
        )
        self.up = nn.Linear(width, hidden, bias=False)

    def forward(self, hidden, compact, features, embedding, gate, observer=None):
        delta = self.network(self.down(hidden) + compact + self.position(features) + embedding)
        residual = gate * delta
        bridge = self.up(residual)
        if observer is not None:
            observer(hidden, compact, residual, bridge, gate)
        return hidden + bridge, compact + residual


@dataclass
class NativeOutput:
    answer_states: torch.Tensor
    cumulative_codes: tuple
    trace: tuple


class EmbeddedGenerator(nn.Module):
    def __init__(self, backbone, architecture=None):
        super().__init__()
        self.architecture = architecture or Architecture()
        self.backbone = backbone
        config = backbone.config
        if config.model_type != "qwen3" or len(backbone.model.layers) != self.architecture.blocks:
            raise ValueError("native block allocation does not match loaded Qwen")
        if getattr(config, "tie_word_embeddings", False):
            raise ValueError("initial native path requires untied input/output matrices")
        config._attn_implementation = "sdpa"
        config.use_cache = False
        hidden = config.hidden_size
        width = self.architecture.latent_width
        self.queries = nn.Parameter(torch.empty(self.architecture.initial_queries, hidden))
        self.level_embeddings = nn.Parameter(torch.empty(len(self.architecture.resolutions), width))
        self.gates = nn.Parameter(
            torch.full((len(self.architecture.resolutions),), self.architecture.gate_initial)
        )
        self.refinements = nn.ModuleList(
            Refinement(hidden, width) for _ in range(self.architecture.hierarchy_blocks)
        )
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.level_embeddings, std=0.02)
        self.phase = "full"
        self.diagnostic_observer = None

    def set_phase(self, phase):
        if phase not in {"warmup", "full"}:
            raise ValueError("unknown training phase")
        self.phase = phase
        self.backbone.requires_grad_(phase == "full")
        for name, param in self.named_parameters():
            if not name.startswith("backbone."):
                param.requires_grad_(True)

    def forward(self, prompt_input_ids, prompt_attention_mask, *, return_aux=False):
        return self._forward(prompt_input_ids, prompt_attention_mask, return_aux=return_aux)

    def forward_cached_context(
        self, prompt_input_ids, prompt_attention_mask, prefix_cache, *, return_aux=False
    ):
        """Inference-only native traversal; transient hierarchy KVs never enter the text cache."""
        if (
            self.training
            or torch.is_grad_enabled()
            or prompt_input_ids.shape[0] != 1
            or not bool(prompt_attention_mask.all())
            or prefix_cache.get_seq_length() != prompt_input_ids.shape[1]
        ):
            raise ValueError("cached native forward needs an exact unpadded causal inference prefix")
        from ncm_lab.embedded.cache import ReadOnlyPrefix

        return self._forward(
            prompt_input_ids,
            prompt_attention_mask,
            return_aux=return_aux,
            prefix_cache=ReadOnlyPrefix(prefix_cache),
        )

    def _forward(self, prompt_input_ids, prompt_attention_mask, *, return_aux=False, prefix_cache=None):
        if prompt_input_ids.ndim != 2 or prompt_input_ids.shape != prompt_attention_mask.shape:
            raise ValueError("prompt IDs and mask must have equal [batch,prompt] shape")
        prompt_size = prompt_input_ids.shape[1]
        prompt = self.backbone.model.embed_tokens(prompt_input_ids) if prefix_cache is None else None
        answer = self.queries[None].expand(prompt_input_ids.shape[0], -1, -1)
        compact = answer.new_zeros(answer.shape[0], answer.shape[1], self.architecture.latent_width)
        lengths = prompt_attention_mask.sum(1)
        auxiliary = []
        trace = []
        for index, layer in enumerate(self.backbone.model.layers):
            if prefix_cache is None:
                hidden = torch.cat([prompt, answer], 1)
                mask, positions = attention_geometry(
                    prompt_attention_mask, answer.shape[1], self.architecture.canvas, hidden.dtype
                )
            else:
                from ncm_lab.embedded.geometry import intervals

                hidden = answer
                left, right = intervals(self.architecture.canvas, answer.shape[1], answer.device)
                positions = lengths[:, None] + ((left + right - 1) // 2)[None]
                mask = hidden.new_zeros(1, 1, answer.shape[1], prompt_size + answer.shape[1])
            rotary = self.backbone.model.rotary_emb(hidden, positions)

            def block(value, module=layer, attention=mask, ids=positions, rope=rotary):
                return module(
                    value,
                    attention_mask=attention,
                    position_ids=ids,
                    position_embeddings=rope,
                    use_cache=False,
                    past_key_values=prefix_cache,
                    is_causal=False,
                )

            if self.training and self.architecture.activation_checkpointing:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
            if prefix_cache is None:
                prompt = hidden[:, :prompt_size] * prompt_attention_mask[:, :, None]
                answer = hidden[:, prompt_size:]
            else:
                answer = hidden
            trace.append(
                {
                    "block": index + 1,
                    "answer_positions": answer.shape[1],
                    "sequence_positions": hidden.shape[1],
                }
            )
            group = index - self.architecture.early_blocks
            if 0 <= group < self.architecture.hierarchy_blocks:
                for offset in range(2):
                    level = 2 * group + offset
                    slots = self.architecture.resolutions[level]
                    answer = resize(answer, slots, self.architecture.canvas)
                    compact = resize(compact, slots, self.architecture.canvas)
                    features = position_features(
                        self.architecture.canvas,
                        slots,
                        level,
                        len(self.architecture.resolutions),
                        lengths,
                        answer.dtype,
                    )
                    answer, compact = self.refinements[group](
                        answer,
                        compact,
                        features,
                        self.level_embeddings[level],
                        self.gates[level],
                        (lambda *args, update=level: self.diagnostic_observer(update, slots, *args))
                        if self.diagnostic_observer is not None
                        else None,
                    )
                if return_aux:
                    auxiliary.append(compact)
        return NativeOutput(self.backbone.model.norm(answer), tuple(auxiliary), tuple(trace))

    def parameter_coverage(self):
        return {
            name: {"parameters": p.numel(), "trainable": p.requires_grad}
            for name, p in self.named_parameters()
        }


def from_metadata(metadata, device):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with torch.device("meta"):
        model = EmbeddedGenerator(
            Qwen3ForCausalLM(Qwen3Config(**metadata["backbone_config"])),
            Architecture(**metadata["architecture"]),
        )
    model.to(dtype=getattr(torch, metadata["dtype"]))
    model.to_empty(device=device)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    model.backbone.model.rotary_emb = Qwen3RotaryEmbedding(model.backbone.config).to(
        device=device, dtype=getattr(torch, metadata["dtype"])
    )
    return model
