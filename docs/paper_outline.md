# Paper outline and evidence map

1. Question and close prior art: frozen-target transfer, not native conversion or discovered human concepts.
2. Mechanism: prediction-position teacher states, independently reconstructed nested codec, causal refinement and gated runtime bridges.
3. Controls: skip-free four-cell design, original skip, capacity-only, frozen-codec loss removal, auxiliary-only, linear/independent/single-scale codecs, released flat and measured cost controls.
4. Protocol: train/development/test lineage, paired independent seed pairs, finite calibration, equal-data and total-cost comparisons, exact unchanged-target verification.
5. Results and limits: retain slower, failed and null outcomes; narrow claims where compatible external controls are unavailable.

| Table / figure | Source | Status |
|---|---|---|
| Rate versus MSE/head KL/top-token agreement/reference NLL | codec-selection observations and rate_distortion.csv | Research pending |
| Oracle versus causal predictor gap | latent-diagnostics and oracle_predicted.csv | Research pending |
| Generation latency, parity and accepted-prefix distribution | predictions.jsonl and per-seed analysis | Research pending |
| Data/exposure/total-cost learning curves | immutable development observations and cost receipts | Research pending |
| Level/skip interventions and gate/rank statistics | latent_mechanisms diagnostics | Research pending |
| Main paired contrasts/group bootstrap | evaluation.tables | Research pending |

Engineering-fixture tables and figures are separate acceptance artifacts and cannot fill these cells.
