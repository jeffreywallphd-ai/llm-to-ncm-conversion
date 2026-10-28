# ACL-002 results

Not run; no empirical findings yet.

Raw runs live in `../runs/<immutable-run-id>/` (ignored). Use
`ncm results analyze --experiment ACL-002` to validate and derive the result index.
Retain failed/cancelled/blocked runs. Test fixtures never support publication claims.

2026-10-28 wiring: `configs/matrix.yaml` selects this experiment's actual bounded study jobs and reused dependency closure. The full study is `experiments/acl/ACL-005_multiscale_codec_and_transfer/configs/study.yaml`; it requires measured calibration. Raw runs retain configurations, input/dependency receipts and failures. `ncm results analyze --experiment ACL-002` produces per-seed observation analysis and codec/cost tables; claim eligibility and explicit selection are separate. CPU fixtures remain ineligible. See `docs/remediation_report.md` for executed validation and external prerequisites.
