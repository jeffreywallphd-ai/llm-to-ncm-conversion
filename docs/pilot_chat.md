# Named pilots and local model playground

The **2 training step pilot** checks basic operations. The **100 step pilot** is an
explicit continuation for early learning inspection. Canvas-specific preparation binds matching targets/data/codec.
Full-backbone training, losses and optimizer precision are retained. Neither script
acquires assets or silently searches for a checkpoint.

## Setup and launch

See the root README's **Configurable canvas sizes and 100-step pilots** and
[canvas contracts](configurable_canvas.md). Fresh pilots default to medium/64,
with five real-model answer capacities. These change answer positions, never the
pinned Qwen3-8B backbone size. Run from the repository root in the installed active
environment, preserving CUDA Torch. Explicit asset acquisition remains separate.

```sh
python scripts/run_2_step_pilot.py --fresh --prepare --no-chat
python scripts/run_100_step_pilot.py --resume /ABS/2-STEP-RUN/checkpoints/current.json --no-chat
```

The checkpoint path is an explicit placeholder for the printed recovery point.
The continuation targets 100 TOTAL completed generator updates, 98 after two.
Do not combine resume with fresh or prepare. Different capacities require fresh
experiments. `--config` preserves explicit geometry unless `--canvas-size` overrides
it; the two named shipped defaults are medium. `generator_pilot.yaml` remains the
explicit legacy 1024 procedure. Direct canvas-dependent stage CLI commands accept
`--canvas-size` but do not execute prerequisites.

```sh
python scripts/run_100_step_pilot.py --fresh --canvas-size medium --prepare --no-chat --plan-only
```

`--prepare` qualifies locally available complete references, then runs teacher,
codec, targets and generator in separate children. It does not acquire weights or
data. Supply `--data-config PATH` with preparation for an approved source/bounds
configuration if the bounded slice cannot supply short-answer train/dev partitions.
Prepared data, normalization, codecs and targets are isolated by canvas and full
experiment fingerprint. Existing complete matching producers can be reused. The
codec preparation target remains its independently reported two updates.

| Setting | 2 training step pilot | 100 step pilot |
|---|---|---|
| Target | 2 total generator updates | 100 total generator updates |
| Validation/checkpoints | 1 | 25, plus terminal finalization |
| Held-out allowance | 32 examples | 32 examples |
| Fixed-prompt samples | Configured cadence | Up to four at 25/50/75/100 |
| Handoff basename | generator_2_step.json | generator_100_step.json |

Pointers live under `runs/handoffs/canvas/<preset>/<experiment-fingerprint>/`.
Attempts write `runs/pilots/<preset>/<id>/`, with a plan, per-stage configs,
launch/process/exit records and stdout/stderr, exact export and status. Stage runs
remain under `runs/<stage>/<id>/`. Completed stage handoffs are published only after
sealed run validation. Conflicting writers fail; retained failed attempts and ledgers
are never reset. Run/phase pilot training clocks and outer deadlines are disabled;
elapsed observations and non-time work/storage/memory/integrity caps remain active.

`--no-chat` needs no server dependency/free port and prints the immutable export and
standalone server command. Otherwise the training child must exit successfully at
the exact objective before chat loads that run's selected export, freeing optimizer
and gradient allocations first. Non-success never advances to another stage/chat.
The selected export labels its actual optimizer step, possibly before final update.

## Chatbot

The UI uses the already locked open-source **Werkzeug 3.1.9** HTTP/WSGI library,
with local HTML/CSS/JavaScript packaged in the wheel. No CDN, external AI API,
model downloads, generated-code execution or third-party frontend assets are used.
The optional `chat` extra adds no Torch dependency and does not replace CUDA Torch.
Existing monitoring installations already include Werkzeug.

```sh
# Honest interface-only preview, without importing Torch/Transformers:
ncm chat serve --preview
# An explicit independent generator export or hash-bound handoff:
ncm chat serve --checkpoint /ABS/EMITTED-EXPORT/manifest.json --device cuda --profile spark
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). Preview mode labels every
response as interface-only, with no fake tokens, timing or model answers. Model
mode verifies/loads the native export once and reuses it sequentially, without
optimizer, teacher or codec. Outputs remain the native parallel argmax/EOS path;
this interface does not add an AR correction. Generated text is rendered as text,
never HTML or executable Python. Each request has its own durable budget/run record
under `runs/chat/`; startup is recorded under `runs/chat_load/`.

History remains browser-session prompt context. Valid alternating text turns use
the existing non-thinking template. The 512-token prompt limit is checked after
tokenization; overlong history is rejected with a new-conversation instruction,
not silently truncated. Request bodies, message counts and text sizes are bounded.
Only one inference runs at a time. Local Host/Origin checks and a per-server token
protect the request API. No HTTP model/file upload or arbitrary path endpoint exists.

Chat is loopback-only. For a remote Spark, run the server there and tunnel from
your viewing computer:

```sh
ssh -L 7860:127.0.0.1:7860 spark@SPARK_HOST
```

Open the same localhost URL on the viewing computer. This is a local development
playground, not a public production hosting service. An occupied port is not taken
over; select another with `--port 7861`.

Server defaults: 7,200 seconds, 100 chat requests and 10,000 HTTP requests per
session. Persistent server-phase work/storage and wall accounting remain finite.
Model loading allows 900 seconds; individual inference requests allow 120 seconds,
512 prompt tokens and one configured answer canvas, with persistent ML-phase caps.
These inference/server controls are separate from the pilot's disabled training
clocks. Startup/closure/error records live under `runs/chat_servers/<id>/`.
`--max-seconds`, `--max-requests`, `--run-root` and `--ready-file` support explicit
standalone operation; pilot launchers expose `--chat-seconds`, `--chat-requests`.

## Checkpoint compatibility

Resume uses the exact original checkpoint layout and hash-verified launch config,
architecture and immutable input manifests. Conflicting canvas/configuration fails
before model allocation; normal source/optimizer/runtime/phase/RNG checks remain.
It does not consult a mutable handoff or scan for recent models.

The historical `clock-policy-v1` and `pilot-chat-v1` receipts remain immutable. They
approve only their recorded historical successors, not this scientific source
change. No legacy training transition is approved here. An old-source recovery
fails honestly: use a fresh experiment and rebuild matching prerequisites. Existing
independent 1024 inference exports still reconstruct their explicit 1024 geometry.
Same-source, same-size 2-to-100 continuation is tested with tiny CPU fixtures.

## Verification and interpretation

```sh
python -m pytest -q tests/unit/test_canvas_presets.py tests/integration/test_canvas_pipeline.py tests/unit/test_pilot_chat.py tests/integration/test_chat_and_pilots.py
python -m pytest -q
ruff check .
```

Tests include a real tiny-model 2→100 continuation, held-out selection, samples at
25/50/75/100, export/handoff integrity, released training subprocess before one
model load, repeated inference, malformed/overlong prompts, preview without model
libraries, server lifecycle and non-success launch gates. These are software
fixtures, not evidence of real-8B quality, speed or 100-step feasibility.
See [dated validation](pilot_chat_validation_2026-11-03.json) for actual commands,
results, package/browser checks and remaining hardware limits. No real model or
dataset acquisition or Spark training is started during implementation.

Primary library references: [Werkzeug serving](https://werkzeug.palletsprojects.com/en/stable/serving/)
and [WSGI testing](https://werkzeug.palletsprojects.com/en/stable/test/).

Current canvas verification and preserved failures are recorded in the
[2026-11-04 validation receipt](configurable_canvas_validation_2026-11-04.json).
