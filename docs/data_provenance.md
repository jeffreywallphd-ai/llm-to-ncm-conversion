# Data provenance and leakage boundaries

Source URLs/revisions/licenses and split paths live in `data/manifests`. Full release
hashes start null until explicitly fetched and verified; acquisition writes receipts
and never labels an unverified release as checked. Original IDs and gold remain
unchanged. Counts are expected release claims, not fabricated downloaded counts.

Setup streamed two original FinQA training rows (32,768 bytes read) and two original
TAT-QA training contexts (16,384 bytes read). Their retained JSON hashes and exact
source URLs are in `tests/fixtures/*.provenance.json`. Both projects' MIT licenses
are preserved under `src/ncm_lab/vendor`. These are scorer/schema smoke inputs,
never headline model evaluation or new business labels.

The preparation command now also pins complete acquisition payload sizes and hashes
in `third_party/preparation_assets.lock.json`. A bounded acquisition check fetched the
official OASST train parquet and GSM8K train file, verified their locked identities,
and prepared 128 prompts with separate train/dev source receipts. This follows the
initial small financial fixture checks; it supplies no model result. See prepare.md.

FinQA/DocFinQA are one family. Content-derived groups help detect identical reports,
but cannot prove all paraphrased cross-dataset relations; audit mapped question/report
IDs before any split-dependent study. TAT-QA contexts preserve official splits;
company-disjoint claims require a separate justified design. Unknown company,
accession, geometry or period stays unknown. Preserve supplied table rows, headers,
units, periods and fiscal dates; never merge later restatements into historical gold.

The model sees only ModelInput.question/context/table/evidence. EvaluationGold holds
answers, programs, derivations and evidence labels. No raw-row serialization, answer-
conditioned retrieval, gold-directed repair or test-derived routing is permitted.
Chunk offsets and stable evidence hashes refer to original source text. DocFinQA
text does not imply recovered original table geometry; missing structure is explicit.

OASST prompt preparation filters English, undeleted, non-rejected prompter records,
splits whole message trees deterministically, and checks exact/near-duplicate overlap
before training. Target-generated continuations must be labeled target_distillation,
not financial ground truth. Pretrained-target contamination remains unknown even
after our new corpus is deduplicated. Lock test opening in the run record.

DocFinQA's multi-GB filing releases require a configured budget. An author scoring
implementation and exact normalization parity have not been established; this blocks
official-score claims. Its reference Python is never run on the host. Optional
FinanceBench/FinLFQA are not central endpoints; their evidence/terms need separate audits.
Errata require evidence, human adjudication and original-score sensitivity reporting.
