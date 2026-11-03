"""One verified model per server; each prompt has normal finite run/phase accounting."""

from dataclasses import asdict, replace
from pathlib import Path
from ncm_lab.runtime.budget import Budget, BudgetStop
from ncm_lab.runtime.io import read, resolve_reference, reference
from ncm_lab.data.tokenization import validated_messages


def conversation(messages):
    return validated_messages(messages, max_messages=41, max_chars=16384)


class ModelBackend:
    def __init__(self, checkpoint, device="cpu", profile=None, *, run_root=None, phase_id="native-chat-v1"):
        from ncm_lab.runtime.provenance import Run
        from ncm_lab.runtime.doctor import preflight
        from ncm_lab.inference.export import load_export

        path = Path(checkpoint)
        path = resolve_reference(path / "manifest.json" if path.is_dir() else path)
        export = read(path)
        if export.get("schema") != "native_generator_export_v1":
            raise ValueError("chat requires a native generator export, not a training recovery checkpoint")
        self.budget = Budget(
            phase_id,
            max_wall_seconds=900,
            max_steps=1,
            max_source_tokens=512,
            max_answer_positions=1024,
            max_canvas_positions=1024,
            max_storage_bytes=20_000_000,
            max_phase_wall_seconds=86400,
            max_phase_steps=1000,
            max_phase_source_tokens=512000,
            max_phase_answer_positions=1024000,
            max_phase_canvas_positions=1024000,
            max_phase_storage_bytes=1_000_000_000,
        )
        spec = dict(export["model"])
        if spec.get("fixture"):
            spec["config"] = export["backbone_config"]
        self.value = {
            "stage": "chat_load",
            "model": spec,
            "architecture": export["architecture"],
            "device": device,
            "dtype": export["dtype"],
            "budget": asdict(self.budget),
            "hardware_profile": profile or ("cpu" if device == "cpu" else "cuda"),
            "monitoring": {"enabled": False},
            "inputs": {},
        }
        if run_root is not None:
            self.value["run_root"] = str(run_root)
        self.bound = {"export": reference(path)}
        run = Run(self.value, self.bound)
        try:
            run.start()
            report = preflight(self.value, run.path)
            if report["blockers"]:
                raise ValueError("; ".join(report["blockers"]))
            self.model, self.tokenizer, self.manifest = load_export(path, device, run.meter.check)
            run.meter.check_disk()
            run.close("completed", {"export": self.bound["export"], "model_loaded": True})
        except Exception as exc:
            run.close("failed", {"error": str(exc), "model_loaded": False})
            raise

    def describe(self):
        return {
            "label": self.manifest["model"]["model_id"],
            "identity": self.manifest["identity"],
            "canvas": self.model.architecture.canvas,
            "device": self.value["device"],
        }

    def respond(self, messages):
        import torch
        from ncm_lab.runtime.provenance import Run
        from ncm_lab.data.tokenization import prompt_tokens
        from ncm_lab.inference.export import generate_tokens

        messages = conversation(messages)
        ids = prompt_tokens(self.tokenizer, messages)
        if len(ids) > 512:
            raise ValueError(
                "Conversation exceeds 512 prompt tokens. Start a new conversation; history was not truncated."
            )
        value = {**self.value, "stage": "chat", "budget": asdict(replace(self.budget, max_wall_seconds=120))}
        run = Run(value, self.bound)
        try:
            run.start()
            run.meter.charge(source=len(ids), canvas=self.model.architecture.canvas, steps=1)
            tensor = torch.tensor([ids], device=self.value["device"])
            # Primitive checks preserve finite request limits without modifying model math.
            hooks = [
                layer.register_forward_pre_hook(lambda *args: run.meter.check())
                for layer in self.model.backbone.model.layers
            ]
            try:
                with torch.inference_mode():
                    result = generate_tokens(
                        self.model,
                        tensor,
                        torch.ones_like(tensor, dtype=torch.bool),
                        self.manifest["eos_ids"],
                        check=run.meter.check,
                    )
            finally:
                for hook in hooks:
                    hook.remove()
            text = self.tokenizer.decode(result["tokens"], skip_special_tokens=True)
            response = {
                "mode": "model",
                "text": text,
                "returned_tokens": len(result["tokens"]),
                "eos_found": result["eos_found"],
                "truncated": result["truncated"],
                "generation_seconds": result["forward_and_head_seconds"],
                "computed_canvas_positions": result["computed_answer_positions"],
                "export_identity": self.manifest["identity"],
                "backbone_traversals": result["backbone_traversals"],
            }
            run.meter.check_disk()
            run.close("completed", response)
            return response
        except Exception as exc:
            run.close("budget_stopped" if isinstance(exc, BudgetStop) else "failed", {"error": str(exc)})
            raise
