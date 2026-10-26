# Two studies, one shared research implementation

ACL working title: Progressive Query Expansion for Efficient Speculative Language
Generation. Question: can fewer future states early in existing drafter layers
improve proposal acceptance versus measured cycle cost without changing target
outputs? Test the progressive schedule, learned expansion, one default skip,
early/late/near-budget schedules and strong/cost-matched flat controls. ACL-001
is a bounded opportunity/pilot gate; ACL-002 is decisive quality/cost comparison;
ACL-003 isolates mechanisms; ACL-004 states model/context/load/hardware boundaries.

All necessary technical controls belong in ACL. Controlled additional adaptation
uses the same warm start, examples, token exposure, anchor policy and objective.
Parameter, inference-cost, training-compute and data matching are distinct controls.
Choose FLAT_COST using development profiling rather than unused parameter padding.
Decisive trained-method results need three independent seeds, with paired request
groups and timing repeats treated separately. General language, reasoning,
instructions, code and original long-context summarization are the evaluation scope.

IS working title: Designing Efficient and Verifiable Financial Analysis Systems with
Local Language Models. It studies evidence preparation, retrieval, calculation,
output contracts, one bounded repair, genuine multi-question workpapers, and a
development-trained pre-generation policy under controlled workload scenarios.
FinQA, TAT-QA and DocFinQA retain original financial gold. IS-001 supplies objective
answer controls; IS-002 tests full filings; IS-003 tests workpapers/validation; IS-004
tests operating policy. Numerical correctness, supporting evidence and structural
validity are separate endpoints. No investment-return, adoption or trust claims.

No hypothesis is established. Prior-art overlap is reviewed before scaling; see
novelty_review.md. Owner/reviewer assignments are pending the real team. A negative
pilot is a valid stopping point; a journal domain speed table alone does not justify
a new system-design contribution. See publication_relationship.md for snapshot and
disclosure rules, and claim maps for missing evidence.
