# Pinned sources and notices

`sources.lock.yaml` is JSON-form YAML containing exact repository/model revisions and
SHA256 values for inspected source files. Original DFlash model code, FinQA evaluator,
and TAT-QA utilities/metric are vendored with adjacent MIT license texts. TAT-QA has
one packaging-only change: its utility import is relative. DFlash is unchanged.
Run `scripts/audit_sources.py` only for an intentional fresh bounded source audit;
it is not invoked by install/plan. Locks record any failed source fetch explicitly.

The model config snapshots are metadata, not weights. Target/drafter cards require
Qwen/Qwen3-8B, not the Base variant. DCore's DFlash config has the same inspected
shape/tap/anchor interface, but its released weight and optimized-engine parity have
not been run. Do not label it reproduced. Full source locks are provenance, not a
claim that every upstream training/inference path is integrated.

Project GPL obligations and third-party MIT notices coexist. Model/data licenses and
underlying financial document rights are separately recorded in acquisition manifests.
