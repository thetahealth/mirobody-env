# Current product specification

This is the authoritative specification of HAEnv's current product behavior.
[Acceptance checklist](testing.md) mirrors its sections. Historical requirements and results
remain in immutable repository history; this document is neither a change log nor a test report.

“Required behavior” describes the implemented contract. Section 15 lists known limitations;
those limitations must remain visible when interpreting a run or publishing a result.
Clinical rules and thresholds come from the registered, versioned policies, not this document.

## 1. Product and task boundaries

HAEnv generates synthetic longitudinal patient records and evaluates an agent's decision at
an index time `T`. It supports evaluation, not patient care or clinical decision support.

The four current task contracts are:

| Pack | Decision | Required answer content |
|---|---|---|
| M2: differential diagnosis | Which explanation or explanations account for the record? | Differential, initial investigations, review/action fields and the applicable frame fields. |
| P2: acute triage | Where should the patient go now? | `disposition`, rule reference, evidence and reason. |
| P3: medication follow-up | Which current drug needs what action? | `target_drug`, `med_decision`, proposed dose and evidence, with clinician-review status. |
| P4: follow-up interpretation | What caused the observed change, and what next? | `change_source`, `factor`, `true_change_pct`, `next_step` and reason. |

P2 accepts `ed_now`, `within_24h`, `routine_followup` or `watchful_waiting`.
P3 accepts `uptitrate`, `downtitrate`, `maintain`, `switch` or
`check_adherence_or_adverse_effect`. P4 accepts `true_change`,
`analytic_biological_noise`, `preanalytical` or `method_difference` as its source.

Earlier diagnosis tracks, forecasting examples and multi-round review remain supported where
registered. Their geometries and scoring contracts do not become current four-pack requirements.
Prompts and clinical narrative are primarily Chinese; field names and answer enums are English.

## 2. Visible records and verifier-only truth

A solver receives the task instructions and authorized records through `T`, inclusive.
The verifier separately receives hidden control variables, derived gold, measurement truth,
answer sets and adjudication snapshots. These must not enter solver prompts or tool results.

Gold is determined from the generated world and the registered policy before judging the answer.
Observed laboratory values, true values and measurement perturbations have distinct roles.
A visible diagnosis already documented in the record is allowed; a hidden diagnostic answer is not.

Static prompt checks and dynamic result guards run before the corresponding content is sent.
A detected leak blocks delivery and is recorded as a generation refusal or `ABORT(leak)`.
Unavailable data is identified as unavailable; it is not a normal or negative observation.

Records may include decisive old events. Their dated source records must be accessible in the
same permitted view as recent evidence. History-reading profiles credit returned evidence,
not a requested section name. Gold annotations identifying decisive history remain private.

Public jobs and registries can contain answer-bearing source material. Runtime separation is
an evaluation boundary, not secrecy from an agent allowed to browse the distribution itself.

## 3. Deterministic generation and emission

With unchanged generation definitions, resources and job inputs, deterministic builds reproduce
the same clinical records without model calls. Case identifiers key independent field draws;
explicitly pinned facts remain pinned when identifiers change.

Pack generation additionally accepts a pack, size and seed. The same inputs reproduce the same
pack; the supported nested pack sizes retain their shared items. Jobs and batch metadata record
size and `seed_sha256`, not a private seed's plaintext. Published public seeds may be reused.
Historical private-seed packs require their original inputs or the subsequently released seed.

Every emitted case must pass its applicable premise, item-verification and leak checks.
Blocking findings withhold the case and retain reasons; an empty emitted batch is not success.
Pack audits check completeness, re-derived gold, declared coverage and shortcut controls.
Evaluation of audited pack items requires a matching passing audit marker.

History dependency and history quantity are separate settings. A zero history-dependency rate
preserves the corresponding clinical output; enabled dependencies use paired visible history
and derived gold. Invalid rates or contradictory premises fail explicitly.

Model-assisted generation is a separate, cached mode. Its outputs require saved artifacts and
input hashes for reproduction; deterministic-generation guarantees do not apply to new calls.

## 4. Query access and full context

`query.geometry` selects `query` by default or `single` for full context on supported tasks.
Query mode starts with the encounter header and provides read-only access to the permitted
profile, series, labs, medications, notes, record sections and hospital protocol library.
Full context presents the permitted record together; protocol-library delivery uses the same
scoped content as protocol retrieval. A task without a query adapter uses its registered view.

`query.max_rounds`, `query.max_calls` and `query.max_points` are recorded answer conditions.
Defaults are 32 tool rounds, 256 answered calls and 400 points or rows per result.
Long results disclose truncation and an earlier-page boundary. Date windows do not reveal
post-`T` records. Unknown sources and invalid tool arguments return explicit failures.

At a query round or call limit, a final answer turn closes tools. A further tool request on
that turn is unanswered. The stored transcript includes calls, returned results, usage and stop
reason. Lookup, duplication and evidence-seen profiles remain separate from outcome scoring.

M2 retains the purchase contract in section 5 under either query or legacy gated presentation;
selecting full context does not make withheld M2 evidence free.

## 5. M2 investigation and purchase budget

M2 is normally one diagnostic round. F0 bridge cases retain diagnosis gold; F1–F3 add urgency,
medication-direction or follow-up interpretation frames. Optional second-round revision is a
profile mode, disabled by default, rather than a requirement to answer every item twice.

The visible menu identifies purchasable targets, kinds and prices, alongside total budget,
spent amount and remaining balance. The registered workup budget covers the initial-stage
required investigations with its declared allowance; it is not permission to buy the whole menu.
The ordinary record-query call limit and this purchase allowance are separate constraints.

For a listed target, billing uses the menu kind and price, regardless of a self-reported kind.
A purchase exceeding the remaining allowance is refused before execution, with no charge or
revelation. A purchase exactly at the remaining allowance can execute.
Purchased test items count as investigations already ordered; `tests_to_order` adds tests not
listed in the menu. Irrelevant menu choices can still spend the purchase allowance.

An unsupported result explicitly indicates unavailability. It must not be treated as a negative
test. A cell exhausting its purchase budget without an answer retains
`ABORT(no_answer:budget)` and is terminal on ordinary resume.

The budget guarantee applies to listed targets. Off-menu execution is a known limitation
(section 15), not a claim that all arbitrary requested targets are refused or bounded.

## 6. Deterministic pack decisions and diagnostic readings

P2 derives an acceptable disposition set from the printed ordered rules and visible triage
record. Membership sets `p2_in_set`; accepted adjacent choices are equally correct.
Its separate pack diagnostics use ordinal under/over-triage cost, random-policy cost correction
and best-constant `cost_skill`. A missed sole emergency disposition is a pack hard gate.
Unparseable dispositions incur the item's worst disposition cost.

P3 checks the identified drug and membership in the accepted decision set together in
`p3_class_ok`. Decision-only agreement and adjacent-dose correctness are separate readings.
A structured dose/drug change without `clinician_review_required: true` trips its medication
gate. Accepted alternatives come from the saved gold, not a new clinical rule.

P4 checks source agreement and the saved acceptable next-step set independently.
Percentage error, factor agreement and numerical skill are profiles; the current standalone
pack composite uses source chance-corrected balanced accuracy. A missing percentage uses the
scorer's explicit zero estimate for that error profile; an invalid source is incorrect.

M2 diagnostic correctness uses the persisted diagnostic atoms: all required gold diagnoses
listed for unified/comorbid cases, or the independent-explanation judgement for independent
cases. M2 diagnosis matching can be semantic-judge derived; it is not wholly model-free.
Its standalone composite scores diagnosis listing and key-evidence coverage, with hard gates;
review utility, suggested-test quality and revision readings have their registered report roles.

Standalone pack composites are diagnostic outputs distinct from the overall score in section 9.
Undefined chance correction on fewer than two gold classes stays undefined; negative diagnostic
scores are not improved by a gate, and display floors do not change the underlying computation.

## 7. Semantic matching and uncertainty

Semantic diagnosis and test matching uses addressable atomic verdicts with saved reasons and
evidence. The default semantic policy uses two independent votes, a third for disagreement or
invalid output, and requires two agreeing valid votes. Otherwise the field remains unresolved.
The default judge is `gpt-6-luna` with high reasoning effort; saved policies bind actual runs.

Missing, unresolved, not applicable and false are distinct states. Coverage and failure reasons
must accompany affected readings. Code aggregates verdicts; the judge does not assign a free
continuous score. Offline stubs exercise the pipeline without establishing clinical validity.

Historical semantic conventions and saved votes retain their original identity. Reading a board
or regenerating a report does not buy new votes or silently replace unresolved fields.
Legacy scoring conventions that used a code substitute remain historical conventions.

## 8. Management quality Q and rubric v6.2

The current management policy is `management-rubric-v6.2`. It derives criteria from a saved
`adjudication.open_management` snapshot, source facts and registered adapter rules.
Its default is one quality judgement, separate from semantic matching's consensus policy.
The five criterion types are decisive history, executable management, safety, follow-up and
patient explanation. History is omitted when the snapshot supplies no decisive history.

Management asks for a concrete next step consistent with the answer's selected decision:
a drug and dose, named investigation, or disposition and timing, as applicable.
Follow-up identifies what and when to reassess and when to seek earlier care. Relative times
and ranges are allowed. A factually correct summary of decisive history or patient explanation
can satisfy its criterion; exact dates and protocol identifiers are not required.

Safety considers only registered prohibitions and their complete applicability conditions,
including explicitly registered required arrangements. It does not penalize an investigation
merely for its name or count, or invent an unstated clinical indication or interval.

Verdicts are `satisfied`, `unmet` or `contradicted`, with reason and zero or one answer quotation.
Required quotations must be locatable in the submitted answer; omitted material may have no
quotation. An unlocatable required quote demotes that assertion to `unmet`; malformed verdict
structure is an error. All submitted JSON fields can contribute, while instructions to the
judge inside them remain untrusted content.

Code computes `quality_score` (Q) from applicable positive quality items at registered equal
item weights. A satisfied negative safety item means a violation and sets the safety multiplier
to zero. The wrong-core-decision cap of 0.3 applies to `rubric_score`, not Q; both are retained.
Core correctness is taken from production decision atoms, not re-decided by the quality judge.

Missing snapshots, unknown required core atoms, invalid judge output and transport failures
remain explicit errors or unscored states. A wholly empty collected answer with known core
correctness follows the free zero-score path.
An empty `management_plan` with substantive other fields is still judged as a complete answer.
New rubric runs write independent outputs and preserve their source batches.

## 9. Overall score and reporting

The current overall item formula is `S = G * D * (1 + Q) / 2`.
`G` is zero for an abort or recorded hard-gate failure, otherwise one. The registered propensity
gates, including over-triage and missing clinician review, are reported outside this multiplier.
`D` reads P2 `p2_in_set`, P3 `p3_class_ok`, P4 both `p4_class_ok` and `p4_next_ok`, or M2's
persisted diagnostic-correctness rule. Board reading maps an aborted or unknown M2 D to zero;
the management preparation path instead preserves unknown core correctness as unscored.

If `G * D` is zero, S is exactly zero for any Q. Otherwise missing Q leaves S uncomputed.
The main `k3` result averages per-answer S across the saved repetitions for each common item;
the published convention uses three. `pk1` reports the first answer separately.
Each pack score averages item scores, and the overall result gives all four packs equal weight.
Missing required packs or uncomputed required item scores prevent an overall rank.

Paired item-cluster bootstrap draws resample the same items for each model within a pack.
Score intervals, rank intervals, pair differences, answer counts and Q coverage are reported.
D accuracy, pass-at-all-repetitions, hard/propensity gate rates and history groups are profiles.
Item-sampling intervals do not include judge uncertainty or establish clinical validity.

Historical boards keep their original formulas, populations, votes, Q and aggregation rules.
A report must label its scoring convention; earlier pack composites are not interchangeable
with S. A new rubric or code identity does not automatically revise a published board.

## 10. Saved artifacts and historical result identity

A batch retains its job identity, case and payload hashes, generation/world identity, solver
conditions, raw responses when available, evaluation rows, usage states and audit outcomes.
A paid attempt additionally retains its unmodified response, billing evidence and request ID.
Query and purchase traces retain enough first-hand observations to inspect the interaction.

Report rendering and archived-result replay read saved artifacts. They preserve original
clinical bytes, values and source identities. An explicit recomputation creates an identified
new measurement; it must not relabel old results as measured under the current code.
A changed schema, source label or dependency fingerprint must be disclosed even when clinical
output is unchanged. Certified definition-preserving moves may retain semantic identities.

A cell with no stored response cannot be newly judged from an imagined answer. Missing usage
retains its explicit status; measured subtotals remain lower bounds when costs are missing.
Generation cache hits and offline stubs are identified separately from measured paid usage.

## 11. Criterion provenance and publication

Results identify each measured criterion and the aggregate criterion set using
`judging_criteria` and `judging_total_sha16`, alongside retained legacy `judging_sha16` and
freeze metadata where present. World, generation, solver and execution identities are distinct.

A criterion's identity covers its registered scoring dependencies. An unchanged criterion
column remains comparable when another criterion changes; totals also require matching
aggregation, weights, task population and answer conditions. An execution route alone does
not establish a new scoring definition or equivalent judge verdicts.

Criterion-aware publication rejects stale, mixed or missing required criterion identities.
Legacy rows without per-criterion stamps remain explicitly unversioned for those columns.
Legacy-only publication uses its original whole-judging vintage gate. Applying a new measurement
updates only its measured criteria; it does not stamp every old
column with current identities. Historical inspection may render stale rows with a warning.

## 12. Configuration, transport and equivalent judge routes

Configuration resolves base settings, local settings, `HAENV_DEPLOY_CONFIG`,
`HAENV_CONFIG_OVERLAY` and `--set` overrides in order. Invalid typed settings and missing named
configuration files fail explicitly. Deployment owns concurrency; permitted experimental
concurrency changes carry their required reason. Effective settings and layer hashes are recorded.

The solver answer identity includes canonical model, upstream provider, output cap, reasoning,
sampling, response format and query limits. Endpoint, proxy, streaming and retry details are
route provenance. Final wire messages and answer fields must match the accounted request.
An incompatible answer-view change blocks resume; a permitted route-only change can reuse receipts.

Secrets are supplied through configured environment sources and masked in diagnostics.
Proxy choice can be per backend or judge route. Explicit proxies apply to that route's requests.
Connection retries are bounded; retries after a possibly sent request require accounted attempts.

Equivalent management routes must declare one canonical model and dated snapshot and retain
the sealed prompts and policy. OpenRouter verifies snapshot catalog evidence before dispatch;
other supported backends use returned model/date evidence. Contradictory evidence disables the
route and leaves the affected result unscored; its receipt and charge remain recorded.
Resume and reconstruction recheck saved evidence without a new live catalog lookup for replay.
`routes check` is offline by default; paid checks require the explicit paid option and budget.
`routes compare` reads matching saved inputs and reports Q, capped score and label differences.

## 13. Paid budgets, scheduling, stopping and resume

Billed evaluation requires an explicit shared ledger and approved USD cap. Each new paid attempt
is admitted against available budget, records its solver/judge role, and settles against billing
evidence. Missing or invalid cost is conservatively counted at a bound, never invented as zero;
a valid answer with unknown cost can still be scored. Estimates are not invoices.

Verified prices, output limits and supported request settings are checked before paid sends.
For models with measured output-length evidence, `max_tokens` below the derived floor is
refused before dispatch. Meeting the floor reduces truncation risk without guaranteeing completion.
Tariff expiry follows the declared `warn` or `refuse` policy. Shared paid lanes, backend limits,
model caps and judge reservations bound dispatch across processes using the same ledger.
At the cap, configured pause/refusal behavior stops new spending; in-flight work is accounted.
Operator drain stops new requests and retains finished rows. Model wall budgets stop new cells
for that model and record unstarted work; they do not fabricate completed answers.

Durable receipts precede derived result persistence. Resume replays matching saved completions
before any new purchase and reconciles interrupted attempts when evidence permits.
A confirmed unsent cancelled reservation can reopen; a possibly sent attempt retains its bound.
Only classified infrastructure/authentication failures or harness errors are eligible for resend,
within `paid.max_resends`. Empty output, malformed model JSON, output truncation, spent tool
budgets and deterministic leak verdicts remain terminal results under their original settings.

## 14. Installation, CLI and plugin compatibility

HAEnv supports Python 3.10+ and provides both `haenv` and `python -m haenv` entry points.
A non-editable installation includes runtime resources and the simulation kernel, so supported
build, offline run and report operations work outside a source checkout.
[Reproduction guide](../REPRODUCE.md) documents public commands and artifact requirements.

Installed judge entry-point groups and pack readouts remain discoverable, including
`haenv.m2`, `haenv.p2`, `haenv.p3`, `haenv.p4` and the legacy `haenv.pack2` group.
Repeated registration does not duplicate execution. Task plugins declare gold, visible probes,
answer contracts, supported geometries and emission checks through the shared registration
surface; unsupported geometries are explicit. Separate run contexts isolate mutable state.

With identical registered content and the same interpreter, source and wheel installations
produce matching semantic fingerprints. Resource/output overrides do not replace the loaded
code's provenance. Public packaging excludes secrets and private evaluation artifacts.

## 15. Known limitations and interpretation boundaries

Clinical policies and parts of the registries still await practising-clinician review.
The management rubric is provisional; passing deterministic controls is not clinical validation.
A single default judge supplies semantic matching and quality verdicts. Cross-vendor validity,
self-preference and judge-noise effects remain unestablished; historical board calibration and
vote counts must be disclosed rather than presented as independent validation of v6.2.

M2's off-menu purchase path is not fully rejected or budget-bounded. Some requested tests have
no interpretable simulated result. The protocol library is authored evaluation policy, and
FHIR-style responses do not imply complete FHIR-server conformance.

Emission and leak controls cover registered checks and finite mutation sets, not every possible
clinical contradiction or semantic leak. Synthetic records do not establish real-world validity;
P4 also has a material question-blind RCV baseline that limits interpretation of modest scores.

Published summaries do not supply all underlying model responses and judge receipts in a clean
checkout. Reproduction of exact historical numbers requires the corresponding saved artifacts.
The repository has inherited failing rubric assertions and artifact-dependent checks. Their
failures remain unresolved or separately reviewable; this specification does not certify a
green suite, complete safety coverage or a new clinical acceptance threshold.

## 16. Repository maintenance and artifact ownership

Source, versioned read-only resources and writable run artifacts have distinct ownership.
Changing an output or resource location must not change which installed code is executed.
Supported public commands and documented extension contracts remain usable after a source-layout
change; optional extensions do not become implicit dependencies of ordinary tasks.

Task registration declares supported behavior without replacing the shared workflow functions.
Generic request accounting, prices, recovery and scheduling are independent of a particular
judge implementation. Runtime capabilities do not require maintainer scripts or prose documents.

The current documentation has one index and distinguishes supported behavior, historical
observations and pending work. Historical evidence remains recoverable with its original
source and status; moving a file does not certify its clinical or empirical validity.

The current manuscript PDF is versioned alongside its source. A successful checked build updates
that artifact; a failed build preserves the previous validated copy. Build intermediates are
kept separately. Different manuscript variants retain separate source and publication states.

Maintenance changes preserve the meaning of existing questions, saved answers, scores and billing
receipts. Reproduction uses the original source where needed; new code never silently relabels
old experiments. Missing migration evidence remains an explicit unmet requirement.

Default run artifacts use `artifacts/` under the checkout, or under the working directory
for an installed distribution. `HAENV_OUTPUT_ROOT` explicitly selects another location. The
legacy combined `HAENV_DATA_ROOT` output redirect remains supported; it never selects code.
Bundled benchmark jobs use this artifact root, while externally supplied jobs retain their
explicit or established job-root contract. The default kernel names the loaded namespace;
an explicit comparison kernel path is never silently replaced with another tree.
