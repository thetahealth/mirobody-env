# Current acceptance checklist

This checklist mirrors [the current specification](spec.md). Every box is initially unchecked:
it describes observable acceptance evidence, not a claim that a test has passed.
Use offline fixtures or local provider stubs unless an independently authorized paid run is needed.
Record the tested version, command, exit status, artifacts and unavailable evidence separately.

For each section, exercise normal, boundary and error paths. Missing artifacts, skipped checks,
unresolved judge fields and inherited failures must remain distinguishable from passing checks.
Historical result reproduction uses the original artifacts and scoring convention.

## 1. Product and task boundaries

- [ ] For each current pack, inspect an emitted question and accepted response: the question asks
  the declared core decision and the response exposes the fields listed in spec section 1.
- [ ] P2's four dispositions, P3's five medication decisions and P4's four sources round-trip
  through answer parsing; an invalid enum receives no decision credit.
- [ ] An M2 independent-explanation case can be assessed without requiring a nonexistent unified
  diagnosis; applicable urgency, medication or follow-up fields follow its frame.
- [ ] A bridge/example job still selects its registered geometry and scoring contract; its report
  does not present an older composite as the current overall score.
- [ ] The prompt and report identify synthetic evaluation; unsupported task/geometry combinations
  surface a refusal or explicit non-applicability rather than a successful empty score.

## 2. Visible records and verifier-only truth

- [ ] Inspect actual prompt and tool-result bytes for an item with hidden sentinel facts: permitted
  records are present, while adjudication, true measurement values and hidden labels are absent.
- [ ] A record on `T` is retrievable; a record after `T` is absent from the static view and all
  corresponding query results, even when a requested window extends beyond `T`.
- [ ] A planted leak caught by the registered guard is withheld before the solver sees it and
  leaves a named generation finding or `ABORT(leak)` record.
- [ ] A decisive old record is retrievable by its dated source. Requesting its section without
  returning that record does not increase the history-seen profile.
- [ ] An unavailable observation is visibly unavailable; a documented known condition remains
  visible without exposing an unrelated hidden answer.

## 3. Deterministic generation and emission

- [ ] Build the same deterministic job twice in independent output locations: clinical records
  and payloads match after separating explicitly recorded run/provenance metadata; no model is called.
- [ ] Change a case identifier: unpinned draws can change, explicitly pinned facts remain fixed.
  Compare supported nested pack sizes at one seed: shared item bodies remain identical.
- [ ] Same pack, size and seed reproduce job/case content; changing the seed produces a different
  pack. Metadata carries size and seed hash without disclosing a supplied private seed.
- [ ] A contradictory premise or failed item check withholds the case with a reason. If all
  cases are blocked, the command reports no usable cases and exits unsuccessfully.
- [ ] A valid pack audit writes its passing marker. Remove the marker or corrupt planned coverage
  in an isolated fixture: pack evaluation refuses the unaudited input.
- [ ] At history-dependency rate zero, compare clinical bytes with the zero-history reference;
  enabled paired history changes the intended gold, while an out-of-range rate fails explicitly.
- [ ] Reuse a model-generation cache fixture: the batch records cache hits and zero new calls;
  uncached model-assisted output is not described as deterministic reproduction.

## 4. Query access and full context

- [ ] Query mode starts with the encounter header; retrieving profile, labs, series, medications,
  notes and record sections returns the expected permitted source rows.
- [ ] Full-context protocol content matches the scoped protocol query result for the same item;
  a task without a query adapter retains its registered presentation.
- [ ] Exceed `query.max_points`: the result discloses the cap and earlier-page boundary; paging
  retrieves the earlier permitted rows. An empty window does not invent observations.
- [ ] Unknown tool/source and malformed arguments produce an explicit failure in the transcript
  and error profile. A guarded bad result never reaches the solver.
- [ ] Reach the configured call/round limit: tools close on the final turn; answering completes
  the cell, while another tool request records `unanswered` and earns no invented answer.
- [ ] Saved transcripts retain calls, results, usage and stop reason. Changing only lookup/profile
  counters does not change the outcome score; changed query limits change the answer conditions.

## 5. M2 investigation and purchase budget

- [ ] Compare the same scripted M2 purchases through query tools and legacy gated presentation:
  menu, allowance, purchases, returned readings and scored decision atoms agree.
- [ ] A listed target is billed at its menu price/kind when the submitted kind is deliberately
  wrong; the trace records the mismatch and remaining balance.
- [ ] A listed purchase exactly equal to the balance executes; a purchase above it produces a
  refusal, zero additional charge and no revealed result.
- [ ] Bought menu tests appear among executed investigations without being repeated in
  `tests_to_order`; additional proposed off-menu tests remain distinguishable.
- [ ] Exhaust the purchase allowance without answering: the saved row is
  `ABORT(no_answer:budget)`, ordinary resume buys no new answer, and missing-response status persists.
- [ ] An unsupported simulated test returns explicit unavailability, with no negative-result
  interpretation. Record off-menu budget behavior as the known gap in section 15.
- [ ] F0 keeps bridge diagnosis gold; applicable F1–F3 frame fields remain present. Default runs
  have no mandatory second answer; an enabled revision mode reports its additional profile.

## 6. Deterministic pack decisions and diagnostic readings

- [ ] In P2, either member of an accepted adjacent disposition set receives `p2_in_set=1`;
  a more/less urgent answer records the correct ordinal cost and under/over direction.
- [ ] A P2 unparseable disposition receives worst-item cost; missing a sole `ed_now` gold trips
  the pack emergency gate. Constant and random controls expose the declared cost baselines.
- [ ] In P3, an accepted decision on the correct drug receives joint credit; the same decision
  naming a different drug does not. Adjacent-dose accuracy remains a separate profile reading.
- [ ] P3 changing the drug/dose with review absent or false triggers its gate; review true
  clears that gate. A malformed decision stays incorrect rather than acquiring an inferred class.
- [ ] In P4, verify source, acceptable next step, factor and percentage-error atoms independently;
  the standalone composite uses source agreement, while overall D also requires next-step agreement.
- [ ] A missing P4 percentage uses the explicit zero estimate for its error profile. Single-class
  chance correction stays undefined; a negative standalone diagnostic score is not raised by a gate.
- [ ] M2 saved unified/comorbid diagnostic atoms require complete listing; independent cases use
  their join/independence atom. Its standalone scored/report/profile columns match their declared roles.

## 7. Semantic matching and uncertainty

- [ ] Feed two agreeing valid semantic votes: the atomic field resolves with saved labels,
  reasons and evidence. A disagreement or invalid vote requests only the allowed third vote.
- [ ] With fewer than two agreeing valid votes after the allowed attempts, the field remains
  unresolved and its coverage count reflects that state.
- [ ] A non-applicable atom and an unresolved atom remain distinguishable from a valid negative
  atom in saved results and reports; neither is silently labelled as a successful match.
- [ ] Offline semantic stubs issue no requests and are labelled as offline evidence; they are
  not reported as an independent judge or clinical-validity reading.
- [ ] Re-render a historical semantic result: saved votes and unresolved counts remain unchanged,
  with no additional dispatch and no adoption of a newer convention.

## 8. Management quality Q and rubric v6.2

- [ ] Derive a rubric from a valid saved snapshot: it identifies v6.2, source facts and applicable
  history/management/safety/follow-up/explanation items; history is absent when no decisive history exists.
  The default quality policy requests one judgement, independently of semantic matching's vote count.
- [ ] Supplied positive examples with concrete next steps and factually correct summaries can
  satisfy the relevant items. Relative/range timing and absent protocol IDs are not automatic failures.
- [ ] A registered safety violation sets the safety multiplier and Q to zero. A reasonable initial
  investigation without the prohibition's full conditions does not acquire an invented safety hit.
- [ ] Compare correct and wrong core decisions with identical quality labels: Q is unchanged,
  while `rubric_score` is capped at 0.3 for the wrong decision; a safety hit still zeroes both.
- [ ] Verify each satisfied/contradicted quotation against the submitted answer, including both
  sides of an ellipsis. A required quote only in case facts demotes the assertion to `unmet`;
  wrong item IDs or invalid labels make the judge response invalid.
- [ ] A missing snapshot/source or unknown required core atom produces an explicit error/unscored
  result. A wholly empty answer with known core follows the free zero-score path; an empty plan
  field with substantive other fields still receives judgment. A failed judge is not an empty answer.
- [ ] Rejudge into an independent destination: source batch bytes remain unchanged; saved raw
  attempts, errors, verdicts, Q and capped score retain their own run/policy identities.

## 9. Overall score and reporting

- [ ] Feed a no-gate, correct-D row with Q=0 and Q=1: S is respectively 0.5 and 1. With D=0
  or a hard gate, S is 0 for either Q; a propensity gate alone does not zero S.
- [ ] Verify D from stored P2 set membership, P3 joint drug/decision, P4 source plus next step,
  and M2 diagnostic atoms. Unknown M2 D maps to zero only under the board-reader convention.
- [ ] Remove Q from a `G*D=1` cell: its S and required aggregate remain uncomputed. Remove Q
  from a `G*D=0` cell: S remains exactly zero without claiming a measured Q.
- [ ] A three-answer fixture gives the mean of its three S values under `k3`; `pk1` uses only
  the first answer. Pack means have equal overall weight; a missing required pack prevents ranking.
- [ ] Bootstrap output uses common within-pack item draws across models and reports score/rank
  intervals and pair differences; D, gate, repetition and Q-coverage profiles remain available.
- [ ] Incomplete repetitions disclose the actual count/common item set; a smaller set is not
  labelled as a complete three-answer result. Judge-noise uncertainty is not added by assumption.
- [ ] A historical report retains its named formula, population and saved Q/votes; changing the
  current rubric does not change the archived board through ordinary rendering.

## 10. Saved artifacts and historical result identity

- [ ] Inspect a completed batch: job/case/payload hashes, world/generation identity, answer
  settings, evaluations, available raw responses, audits and usage states can be traced together.
- [ ] Replay stored responses without dispatch: derived outputs are reproducible under their
  declared source, and the archived source files, scores and fingerprints remain unchanged.
- [ ] An explicit recomputation records a new measurement identity. It does not replace an
  unchanged archived value's original provenance or claim an absent response was rejudged.
- [ ] After an approved definition-preserving relocation, compare clinical bytes and stored-answer
  outputs; separately report every schema/source/dependency identity change instead of hiding it.
- [ ] Missing usage, missing response, cache-only generation and offline stubs have distinct
  statuses. A partially measured subtotal is labelled as incomplete/lower-bound spend.

## 11. Criterion provenance and publication

- [ ] A new measured row records its criterion identities and aggregate identity; legacy stamps
  remain available where supplied. Source and installed-code provenance agree with the loaded code.
- [ ] Change one criterion in an isolated comparison: its affected identity changes, unrelated
  criterion columns retain theirs, and combined-score comparability also checks weights/population.
- [ ] Apply a measurement overlay: only measured criterion stamps update. A legacy row without
  per-criterion stamps remains explicitly unversioned rather than inheriting the current set.
- [ ] Criterion-aware publication refuses stale, mixed or missing required stamps with reasons;
  legacy-only publication checks its whole-judging vintage. Historical inspection can still warn/render.
- [ ] Route-only changes preserve scoring definitions but do not by themselves certify equality
  of judge labels. Comparison output retains both capped score and uncapped Q differences.

## 12. Configuration, transport and equivalent judge routes

- [ ] Layered configuration produces the expected effective value in documented precedence order,
  with recorded layer hashes. Invalid types, missing named overlays and prohibited concurrency changes fail.
- [ ] A permitted reasoned concurrency override is recorded. Diagnostics expose effective limits
  while credential values remain masked; secret values do not appear in saved configuration.
- [ ] A mismatched final message, output cap, reasoning or sampling setting fails before send.
  A changed answer view blocks incompatible resume; a permitted route-only change reuses the receipt.
- [ ] Local fake routes exercise explicit direct/proxy choices and bounded connection retries;
  a possibly sent failure is not retried invisibly outside request accounting.
- [ ] Matching OpenRouter catalog evidence allows an undated response alias; absent/conflicting
  catalog evidence blocks that route before paid dispatch. A contradictory returned date is rejected.
- [ ] Other supported backends require matching returned canonical model/date. A mismatch leaves
  the cell unscored, keeps its receipt/charge and disables the route while a healthy route can continue.
- [ ] Resume/reconstruction uses saved snapshot evidence and receipts before dispatch. Offline
  `routes check` sends nothing; `routes compare` reads saved runs and lists missing cells and differences.

## 13. Paid budgets, scheduling, stopping and resume

- [ ] A billed run without its explicit ledger/cap is refused before paid dispatch. A local stub
  attempt reserves budget and settles once, with request ID, role and unmodified receipt preserved.
- [ ] Missing/invalid cost retains a conservative counted bound with unknown actual cost; a valid
  saved answer stays usable. Verified unbilled failures require evidence rather than a fabricated zero.
- [ ] At the cap, configured refusal/pause starts no new paid work; allowed in-flight calls settle.
  Resume after an authorized cap change uses the same ledger and records the changed budget state.
- [ ] Concurrent solver/judge fixtures respect shared lanes, backend/model limits and judge
  reservations across ledger-sharing processes; an expired tariff follows the declared warn/refuse policy.
- [ ] A measured model's `max_tokens` below its derived floor is refused before dispatch; a
  cap that meets the floor remains recorded without claiming that truncation is impossible.
- [ ] Drain while calls are active: no new paid request starts after the drain check, finished
  rows persist, and unstarted work remains incomplete. Model wall-budget stops identify affected cells.
- [ ] Interrupt after saving a receipt but before saving the result: resume reuses the receipt
  without a second purchase. Unsent cancellation can reopen; possibly sent attempts retain their bound.
- [ ] Classified infrastructure/auth failures can resend up to the configured limit; exhaustion
  retains the abort. Empty/malformed/truncated model output, tool-budget exhaustion and leak verdicts
  remain terminal, with no ordinary-resume dispatch and no loss of failed-attempt evidence.

## 14. Installation, CLI and plugin compatibility

- [ ] In a clean non-editable installation outside the checkout, import the product and run CLI
  help, supported build, offline run and report using shipped resources and kernel.
- [ ] `haenv` and `python -m haenv` resolve the same supported commands. The documented minimum
  Python version can load the public API; compare identities using the same interpreter version.
- [ ] Discover all current judge/readout groups plus legacy `haenv.pack2`. Loading an installed
  plugin and its checkout shim together runs each registered target once.
- [ ] A minimal external task supplies its gold, visible probe, answer contract, emission check
  and supported mount; its emitted answer is scored. Unsupported mounts are explicit.
- [ ] Two runs in one process do not share mutable task tables. A source/wheel pair with identical
  registered inputs gives identical fingerprints; an output/resource override cannot spoof code origin.
- [ ] Inspect distribution contents: runtime resources exist, while secrets and private run
  artifacts are absent. An artifact-dependent check on a clean clone reports unavailable evidence.

## 15. Known limitations and interpretation boundaries

These are disclosure and regression-tracking checks, not completed product guarantees.
The inherited baseline includes failing legacy rubric assertions and disk-artifact guards.
Some rubric failures concern safety, required arrangements or timing; baseline reproduction
establishes their origin, not that their clinical question is resolved.

- [ ] Run an off-menu M2 boundary probe only against a local stub: record current execution/spend
  behavior as a known gap, separately from the passing listed-menu budget-refusal checks.
- [ ] Reports disclose pending clinical review, single-judge semantics/Q, finite leak coverage,
  synthetic-data limits and P4's RCV baseline without claiming unmeasured validity or judge agreement.
- [ ] For historical Q, identify its actual rubric, calibration population and vote counts;
  it is not silently described as independent validation of the current v6.2 policy.
- [ ] Exact historical reproduction lists unavailable answers/receipts as missing prerequisites;
  deterministic generation alone is not evidence of reproduced model responses or Q.
- [ ] Record inherited failing node IDs and new failures separately against the same baseline and
  environment. Keep failures/errors/skips/unmeasured distinct; do not check these boxes merely because
  an inherited failure reproduces, and do not relax clinical thresholds or assertions to hide it.

## 16. Repository maintenance and artifact ownership

- [ ] Install the distribution outside its checkout, select alternate resource and output
  locations, and run the supported offline workflow: code origin remains the installed package.
- [ ] Load a minimal extension and each builtin task in different orders: no shared workflow
  function is replaced, registration is idempotent and per-run state remains isolated.
- [ ] Exercise accounting, reservation, receipt replay and recovery against local transport
  stubs without loading a particular judge implementation; a genuine dependency violation is detected.
- [ ] Check the current documentation index, commands and cross-links; historical claims and
  pending evidence are labelled, while a deliberately broken current link fails validation.
- [ ] Rebuild the manuscript and verify that its tracked PDF corresponds to the source; force
  a missing dependency or failed validation and confirm the last validated PDF is preserved.
- [ ] Replay the same original questions, saved answers and receipts before and after a
  maintenance change: visible inputs, gold, scores, refusal and spending outcomes agree.
- [ ] Resolve a relocated historical artifact through its recorded source and verify its original
  bytes; a missing or ambiguous source fails explicitly rather than being replaced by a current one.

- [ ] With no redirects, source and installed CLI runs write under `artifacts/`, leaving
  versioned inputs and source intact. Explicit output takes precedence over the legacy
  data/output redirect; resolving a root alone creates no directory.
- [ ] A bundled benchmark job defaults to the artifact root; an explicit external job root
  stays authoritative. Default kernel lookup follows loaded code even with redirected data,
  while a missing explicit comparison path is never substituted.
