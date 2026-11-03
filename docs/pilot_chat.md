# Named pilots and local model playground

The **2 training step pilot** checks basic operations. The **100 step pilot** is an
explicit continuation for early learning inspection. Existing targets/data/codec,
full-backbone training, loss and optimizer precision are unchanged. Neither script
acquires assets or silently searches for a checkpoint.

## Setup and launch

In the existing CUDA environment on Spark:

```sh
git switch main
git pull --ff-only origin main
python -m pip install -e '.[chat]'
# First generator pilot, after the existing data/teacher/codec/targets stages:
python scripts/run_2_step_pilot.py
# Continue the explicit completed two-step recovery point to 100 TOTAL steps:
python scripts/run_100_step_pilot.py --resume /ABS/2-STEP-RUN/checkpoints/current.json
```

The shared implementation is `runtime/pilot.py`; scripts are thin entry points.
Equivalent installed commands are `ncm pilot train --steps 2` and
`ncm pilot train --steps 100 --resume /ABS/checkpoints/current.json`.
`--fresh` explicitly requests a fresh 100-step initialization. `--config` can name
an explicit typed config, but its generator target must match the selected pilot.
The old `generator_pilot.yaml` is preserved as the two-step legacy entry point.

| Setting | 2 training step pilot | 100 step pilot |
|---|---|---|
| Config | `generator_2_step_pilot.yaml` | `generator_100_step_pilot.yaml` |
| Target | 2 total optimizer steps | 100 total; 98 additional after step 2 |
| Validation/checkpoint interval | 1 | 25, plus forced terminal finalization |
| Held-out allowance | Existing 32 examples | Same 32; no changed selection identity |
| Automatic fixed-prompt samples | Existing default interval | Up to four at 25/50/75/100 |
| Handoff | `runs/handoffs/generator.json` | `runs/handoffs/generator_100_step.json` |

Both inherit the unchanged numeric work/storage caps and phase identity from
`real_pilot.yaml`, with run/phase wall enforcement disabled. Elapsed observations
remain active. The launcher has no additional execution deadline on training.
Controller work/log storage has separate persistent finite caps; training spending
still uses the original phase and never comes from a checkpoint or gets reset.

Each invocation writes `runs/pilots/<id>/`: resolved config, launch/process and exit
records, child stdout/stderr, exact export reference and `status.json`. The separate
training child must exit successfully, report `objective_attained` at the exact
target, seal its run and publish the intended handoff before chat starts. Its
optimizer/gradient allocations therefore end before inference model allocation.
The exported model remains the held-out selected state, which can be earlier than
the final update. Scheduled samples show current weights at each observation step;
the UI does not claim that the selected export necessarily contains step-100 weights.
Non-success never launches a UI with an unrelated model. If chat startup fails
after training, the completed training result/export are retained; launch chat
separately rather than repeating updates. An interrupted launcher retains the last
committed checkpoint; forced process termination can leave explicit incomplete
elapsed accounting, as described in [clock policy](pilot_clock_policy.md).

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
ncm chat serve --checkpoint runs/handoffs/generator_100_step.json --device cuda --profile spark
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

Adding launcher/chat Python changes the full package implementation hash. The new
explicit **`pilot-chat-v1`** receipt approves only the captured predecessor at
`0c9fdd50c637caa92a013c03a0f4a8a9eae23e99` (source
`e106a014d20dba62e209f7b2432b6789e1cd34e2cd57d6f9c03f152cb901a3aa`) and the exact
reviewed successor in `configs/recovery/pilot_chat_v1.json`. Named pilot launchers
supply this option on explicit resume. Direct CLI equivalent:

```sh
ncm generator train --config configs/validation/generator_100_step_pilot.yaml --resume /ABS/2-STEP-RUN/checkpoints/current.json --recovery-compatibility pilot-chat-v1
```

The original clock-policy receipt is immutable and hash-pinned. Its older reviewed
predecessor can use only the explicitly verified two-edge chain; there is no
arbitrary source identity override. Old metadata/manifest/payload identities,
every shard, model/data/codec/runtime/dtype/phase/training metadata, scheduler,
optimizer moments/groups, RNG and cursor still pass normal strict checks.
The source transition is recorded in the new attempt and never edits an original
checkpoint. Goal/interval changes are existing config-only continuation controls;
changing the held-out subset, data, learning rates or codec is not bypassed.

The actual Spark checkpoint is not present locally and is untested here. Tiny
pre-edit BF16 checkpoints were captured before source edits for genuine transition
verification. Unsupported predecessors/contracts fail rather than restart weights.

## Verification and interpretation

```sh
python -m pytest -q tests/unit/test_pilot_chat.py tests/integration/test_chat_and_pilots.py
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
