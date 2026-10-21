# HAEnv: specifications and additional results

## Access layer and traces

HAEnv (Health Agent Environment) generates synthetic longitudinal patient records and evaluates clinical decisions against reference decisions derived from the same patient state. This supplement specifies access behavior, task scoring, generation checks and additional experimental results.

Tools return FHIR resources with case-relative days (Health Level Seven International, 2019, HL7 FHIR Release 4 specification). The record tools are `get_profile`, `get_series`, `get_labs`, `get_meds`, `get_notes`, `get_record`, `list_sources` and `get_protocol`; diagnosis additionally provides `get_menu` and `order_test`. Each call returns at most the latest 400 rows in its date window, the `total` number of rows in that window and, when truncated, the bound for the preceding page. A response permits at most 32 tool rounds and 256 calls, followed by a final turn without tools. A trace stores every call, its arguments and returned rows, the stop reason and the final answer.

Over HTTP, an extension preserves case-relative days beside ISO dates, allowing each result to be converted back to the function-call output. The replay check validates resources with the FHIR R4B models of fhir.resources 7.1.0. [Tool definitions](../src/haenv/interaction/query/tools.py) specify semantics, errors and execution limits; the [query loop](../src/haenv/interaction/query/loop.py) defines the trace format; and the [FHIR server](../src/haenv/serve/fhir_server.py) defines HTTP routes and their FHIR mapping. The [HTTP replay tool](../tools/release/replay_http.py) checks this correspondence.

The [offline example](../benchmarks/inputs/example-ew.job.yaml) generates and verifies a small question set, answers it with reference stubs and computes scores without model calls or API keys. The [evaluation scorer](../tools/evaluation/board_score.py) recomputes reported scores and intervals from stored responses and judgments without model calls.

Clinic policies, thresholds and accepted answers are defined in the registries listed under Generation validation and release inventory below. Editing them and regenerating from a seed produces a new question set. External packages can register additional judges, record streams and tasks through entry points without changing the environment code.

## Task specifications and scoring

A seed s fixes the patient state z, from which records, test results and the reference decision y* are computed. The evidence list E contains the decisive findings supporting y*; independent diagnosis questions have no evidence list. For a response trace τ, O(τ) is the set of items of the evidence list matched by a returned row before the final answer. Coverage is complete when E ⊆ O(τ), and the missed decisive findings are M = E ∖ O(τ). The decisive condition supplies the tool results B carrying those missed results. The placebo condition supplies an equal number of results from tests that the agent did not purchase and that carry no item of E: |P| = |B|. The resample condition supplies no additional results.

The evidence list contains the sections named by the applicable triage rules; relevant visits, refills, pill counts and symptoms in medication adjustment; class-dependent rows in lab-change attribution; and key findings of the reference diagnosis.

Decision correctness D is binary. Triage requires the normalized disposition to fall in the accepted interval. Medication adjustment requires the correct target drug and an accepted medication decision. Lab-change attribution requires the reference change-source class and an accepted next step. Diagnosis requires accepted aliases for all n reference diseases among the first n + 1 non-excluded differential entries, or recognition that no single disease unifies the findings on independent questions. Aborted responses are incorrect.

As a robustness check, semantic diagnosis matching also accepts a disease name or a more specific form when two model judges agree that it names the reference disease.

The rule-check indicator G equals 1 when every check passes. Policy checks require an approved action class, clinician review before medication changes, escalation of emergency red flags, referral when warranted and no escalation for a sensor artifact. The citation check accepts a record evidence ID, a unique abbreviation of that ID, a returned row identifier or index, or the name of a series or test that the response obtained. A missing plan receives quality Q = 0. The [management rubric](design/management-rubric.md) was developed on evaluation responses and is released with the judgments.

The response score is S = GD(1 + Q)/2, where Q ranges from 0 to 1. It is zero unless the answer is correct and passes all rule checks; otherwise it ranges from 0.5 to 1. GPT-6 Luna evaluates plan quality using history, protocol use, explanation, follow-up and testing; Q averages two judgments per response. Task scores weight questions equally, and the overall score weights tasks equally. The primary score averages three responses per question. The pre-specified first-response score uses the first response only. Two models are separated when the unadjusted paired interval of their score difference excludes zero; All intervals are 95% confidence intervals from 2,000 paired question-bootstrap draws, retaining all responses to each sampled question.

The within-model-and-question coverage gap is the Mantel–Haenszel risk difference: the weighted mean of complete-minus-incomplete coverage accuracy differences d_k over model-by-question strata k, with weights w_k = n₁k n₀k / (n₁k + n₀k). Here n₁k and n₀k count responses with complete and incomplete coverage, respectively.

## Generator and realism checks

Record-omission frequencies are measured from electronic medical records (EMRs). The table reports matching and eligible events; HF denotes heart failure, AF atrial fibrillation and ACEI an ACE inhibitor.

| Task | Record condition | Matching / eligible |
| --- | --- | --- |
| Triage | Established HF absent from recent notes | 2,929/4,136 |
|  | Established AF absent from recent notes | 1,398/2,420 |
| Medication adjustment | ACEI-switch reason repeated in recent notes | 0/3,098 |
|  | Reason near a statin dose reduction | 8/297 |
| Lab-change attribution | Prior creatinine more than 180 days earlier | 6,028/10,234 |
|  | Prior HbA1c more than 180 days earlier | 4,073/5,756 |
| Diagnosis | HF in narrative, omitted from diagnosis field | 76/228 |
|  | AF in narrative, omitted from diagnosis field | 60/222 |

Note-based rates are per visit in the coronary-disease and hypothyroidism cohorts. A diagnosis is established when first mentioned at least 365 days earlier, and absent from recent notes when none of the preceding 180 days mentions it. Switch reasons are searched in the 180 days of notes before visits at least 180 days after an ACE-inhibitor switch; relevant keywords occur near 21.0% of switches. Statin reasons are searched within 60 days of a dose reduction. Laboratory rates use same-patient, same-analyte reports at least 28 days apart, excluding urine and body-fluid creatinine. A diagnostic omission is a narrative condition absent from the diagnosis field.

The generator lists an established triage diagnosis with probability (1 − p_s)(1 − p_o), where p_s is its omission rate in recent notes and p_o is its diagnosis-field omission rate conditional on a recent mention. For heart failure these rates are 2,929/4,136 and 56/1,207; for AF they are 1,398/2,420 and 49/1,022. Class quotas and the seed determine generation and selection. Released parameter tables distinguish the source of each frequency.

Analytical and within-person biological variation are sampled independently from their coefficients of variation (Carobene et al., 2017, Clinical Chemistry, DOI: 10.1373/clinchem.2017.275115). The Jaffe–enzymatic creatinine bias follows an Abbott Architect comparison of 230 serum samples (Küme et al., 2018, Journal of Clinical Laboratory Analysis, DOI: 10.1002/jcla.22168). Creatinine changes after ACE-inhibitor use follow categories observed at first follow-up in a treated cohort (Schmidt et al., 2017, BMJ, DOI: 10.1136/bmj.j791). A cooked-meat meal one to four hours before a non-fasting draw adds an increment sampled uniformly within published bounds (Toffaletti et al., 2018, Clinical Biochemistry, DOI: 10.1016/j.clinbiochem.2018.05.016; Preiss et al., 2007, Annals of Clinical Biochemistry, DOI: 10.1258/000456307779595995).

Classifier two-sample tests compare lab trajectories visible to the agent with real trajectories from diabetes, hypothyroidism and coronary-disease cohorts having at least two blood-draw dates spanning 60 days or more. Each feature group reports the larger five-fold cross-validated area under the receiver operating characteristic curve (AUC) from logistic regression and gradient-boosted trees. Noise measures relative adjacent changes and lag-one correlation; Repeat measures consecutive repeats; Format measures decimal precision; Band measures values at reference-range boundaries; Coupling measures paired directions of change; All combines features without timing.

Classifier two-sample AUC for real versus synthetic lab trajectories. The limit is the largest AUC between two real cohorts plus 0.05; stars mark values above it, and a dash marks a feature group without eligible observations. The last row gives real / synthetic trajectory counts.

| Group | Limit | Triage | Medication adjustment | Lab-change attribution | Diagnosis |
| --- | --- | --- | --- | --- | --- |
| Noise | .8350 | .8110 | .7780 | .8370\* | .8352\* |
| Repeat | .7700 | .6500 | .5770 | .5830 | .5847 |
| Format | .6330 | .5110 | .5550 | .5840 | .5541 |
| Band | .7160 | .6380 | .6080 | .6820 | .7113 |
| Coupling | .8360 | --- | .4540 | .4890 | .5632 |
| All (no timing) | .8850 | .8350 | .7940 | .9110\* | .8876\* |
| Groups within limit |  | 5/5 | 6/6 | 4/6 | 4/6 |
| Real / synthetic trajectories (n) |  | 51/57 | 356/42 | 229/33 | 184/88 |

## Additional results: coverage and errors, re-asking, model differences, budget and scale, robustness

### Coverage and errors

Pooled over models, diagnosis accuracy is 84.8% with complete and 47.0% with incomplete coverage. Without Qwen3-32B, which never reaches complete coverage, 68.7% of wrong diagnoses lack some decisive findings.

Under semantic diagnosis matching, the within-model-and-question gap is 25.7 percentage points, accuracy is 92.9% with complete and 50.2% with incomplete coverage, and 86.2% of wrong diagnoses lack some decisive findings. Every model that reaches complete coverage still makes at least 4 wrong diagnoses with it.

The examples below illustrate wrong diagnoses with incomplete and complete coverage. Returned text is translated from Chinese; numbers and tool names retain the recorded values.

| Coverage and model | Reference, retrieved results and response |
| --- | --- |
| Incomplete; Gemini 3.8 Flash | Reference: depression. `get_menu` lists “clinical mental-state and function assessment” (cost 1.0 of a 128-unit budget). The agent purchases `scale_qc_flag` and `fasting_glucose`, then answers “finger crush injury with subungual hematoma”. The decisive assessment was not ordered. |
| Complete; GLM 5.3 Flash | Reference: primary adrenal insufficiency (Addison). `order_test` returns morning cortisol 1.28 μg/dL (low), ACTH 147.02 pg/mL (high), sodium 128 mmol/L (low) and potassium 5.65 mmol/L (high). The answers are “gastrointestinal adverse reaction to dulaglutide”, then “diabetic gastroparesis”. Primary adrenal insufficiency is missed despite the obtained results. |

### Re-asking

The decisive, placebo and resample conditions resume each model’s own conversation immediately before its final answer and permit one further answer without tools. The half condition supplies a seeded random half of B, rounded down. The comparison below is restricted to responses missing at least two results; Experimental settings below describes sampling and complete-coverage controls.

On the 79 responses missing at least two results, the correction rate is 6.3% when resampled, 41.8% with half of the missed results and 75.9% with all of the missed results. Half exceeds resampling by 35.4 percentage points [22.2, 48.5], and supplying all rather than half adds 34.2 percentage points [21.7, 47.6].

### Model differences

A seeded policy that answers uniformly at random scores S = .117 overall (.257, .089, .050 and .070 on triage, medication adjustment, lab-change attribution and diagnosis, respectively). The best constant answer, chosen with access to the references, reaches decision accuracy 52.0%, 25.0% and 20.0% in triage, lab-change attribution and diagnosis, where it gets every independent question right; medication adjustment has no such baseline, because its answer must name the patient's drug.

The pre-specified first-response score separates 5/12 adjacent pairs (primary score: 6/12). Near the ceiling, triage and medication adjustment still separate 2/12 and 5/12 adjacent pairs. The share of questions correct in all three responses ranges from .073 (Qwen3-32B) to .878 (GPT-6 Sol).

Where the header or one call supplies the history, eleven models are more accurate on history-dependent than on other lab-change questions, and nine models are more accurate on history-dependent medication-adjustment questions. The subgroup tables report pooled decision accuracy.

Decision accuracy (%) by question subgroup, pooling 39 responses per question (thirteen models, three each); aborted responses count as wrong, and parentheses give question counts. The last column counts models more accurate on other questions.

| Task | History-dependent | Other | Models more accurate on other |
| --- | --- | --- | --- |
| Triage | 83 (45) | 88 (55) | 12/13 |
| Medication adjustment | 75 (42) | 73 (58) | 4/13 |
| Lab-change attribution | 72 (38) | 70 (62) | 2/13 |
| Diagnosis | 55 (25) | 68 (75) | 9/13 |

| Diagnosis by type | One disease | Several | No unifying disease |
| --- | --- | --- | --- |
| Accuracy (%) and question count | 73 (50) | 53 (30) | 61 (20) |

### Budget and scale

Among the core models, GLM 5.3 Flash, GPT-6 Luna, Gemini 3.8 Flash and GPT-6 Sol form the cost–score frontier, and GPT-6 Sol costs 18 times as much per response as GPT-6 Luna. Costs are billed charges, or recorded tariffs with cached input at the cache rate for Gemini 3.8 Flash and Qwen 3.7 Flash.

Across the 100 diagnosis questions, first-response accuracy increases by 0.0 to 10.0 percentage points from half to twice the default test budget. The gain is clearly above zero only for GPT-6 Sol and DeepSeek V4 Flash, which spend most of the default budget. The 1× comparison uses leaderboard first responses.

Accuracy gains from 0.5× to 2× the default test budget, with paired 95% intervals:

| Model | Accuracy gain (percentage points) | Paired 95% interval |
| --- | --- | --- |
| GPT-6 Sol | 10.0 | [2.0, 18.0] |
| DeepSeek V4 Flash | 10.0 | [2.0, 18.0] |
| DeepSeek V4 Pro | 6.0 | [-1.0, 13.0] |
| Gemini 3.8 Flash | 4.0 | [-4.0, 12.0] |
| GPT-6 Luna | 3.0 | [-3.0, 9.0] |
| MiniMax M3 | 10.0 | [0.0, 19.0] |
| GLM 5.3 Flash | 8.0 | [-2.0, 18.0] |
| Qwen 3.7 Flash | 0.0 | [-9.0, 8.0] |

The gain tends to be larger for models that spend more of the default budget (Spearman ρ = .44, descriptive). Of GPT-6 Sol's 13 new correct answers at 2×, 10 are on questions where it exhausted the half budget without answering.

The 8B, 14B and 32B Qwen3 models end with an answer in 3,594 of 3,600 responses and average at most 3.4 tool calls per question across model–task combinations. Re-asked with the decisive results they had not obtained, they answer 38.8–55.0% of the 80 questions with an evidence list correctly, against 1.3–10.0% as run and 0.0–6.3% when re-asked unchanged. Qwen3-32B gains 37.5 percentage points over an unchanged re-ask and is not separated from Qwen3-8B. Each model keeps its own conversation, so the contexts differ beyond the supplied findings.

Qwen3.8-27B has an overall score of .789, not separated from MiMo V2.6 Flash and Gemini 3.8 Flash. DeepSeek V4.1 Flash reaches complete coverage in 67.4% of responses against 67.0% for V4 Flash, is correct with incomplete coverage in 73.0% against 44.9%, and scores .837 against .632.

### Robustness

On the core models, 4/7 adjacent pairs are separated with plan quality and 4/7 without it. Scoring with D alone reorders only models the primary score does not separate; either GPT-6 Luna judgment alone preserves the primary order. Adjusted for question, each judge model rates its own responses at most .02 higher in Q than the other judge does.

On fresh question sets generated without selection on model results, Gemini 3.8 Flash is the most accurate of the four tested core models on every set. The other tested models are GPT-6 Luna, DeepSeek V4 Flash and Qwen 3.7 Flash. Kendall’s τ relative to the public-set order is 1.00 on diagnosis and medication adjustment, .83 on lab-change attribution and .67 on triage.

## Experimental settings

The evaluated sets contain 100 questions per task, with three responses per question, and rank thirteen models. Qwen3-8B and Qwen3-14B appear only in the scale comparison. Robustness and cost analyses use eight core models: GPT-6 Sol, GPT-6 Luna, Gemini 3.8 Flash, DeepSeek V4 Pro, DeepSeek V4 Flash, GLM 5.3 Flash, MiniMax M3 and Qwen 3.7 Flash.

A response is aborted when it reaches a turn, call or budget limit without a final answer. Coverage analyses exclude aborted responses, while accuracy analyses count them as wrong.

The four local Qwen models use vLLM 0.19 in bfloat16. Qwen3-8B, Qwen3-14B and Qwen3-32B use YaRN and a 98,304-token context. Other models use provider-default top-p and seeds. The table gives requested model settings.

Requested model settings. Output caps are in tokens; Gemini's reasoning entry is its thinking-token budget. Default means that no temperature setting is accepted; \* indicates top-p .95 and top-k 20.

| Models | Temp. | Reasoning | Output cap (tokens) |
| --- | --- | --- | --- |
| GPT-6 Sol, GPT-6 Luna | default | medium | 72,000 |
| DeepSeek V4 Pro, V4 Flash and V4.1 Flash, GLM 5.3 Flash, MiniMax M3, MiMo V2.6 Flash | 0 | medium | 72,000 |
| Gemma 4 31B | 0 | medium | 16,384 |
| Gemini 3.8 Flash | 0 | 8,192 tokens | 65,536 |
| Qwen 3.7 Flash | 0 | none | 72,000 |
| Qwen3.8-27B, Qwen3-8B, 14B, 32B | 0.6\* | thinking | 32,768 |

The first half of each evaluated set served as development data. Generator versions were selected using responses from GPT-6 Luna, Qwen 3.7 Flash and, for triage, Gemini 3.8 Flash. A development version was accepted when at most 25 of its 50 questions were correct in every response by every development model, the two best development models were separated, and the evidence-consistency conditions described under Generation validation and release inventory below held. The leaderboard includes these development questions. Fresh-seed comparisons use sets never selected on model results.

The pre-specified re-asking analysis used a fixed seed and at most two responses per question, sampling up to 50 non-aborted wrong responses with incomplete coverage per model: 44 for GPT-6 Luna, 50 for GLM 5.3 Flash and 48 for Qwen3-32B. It also sampled up to 25 complete-coverage controls per model (30 in total, from GPT-6 Luna and GLM 5.3 Flash). A free turn supplied B in the decisive condition or P in the placebo condition, followed by one answer without tools. The resample condition and complete-coverage controls received no additional results. The half condition supplied a seeded random half of B, rounded down, to the 79 responses missing at least two results. The 63 responses missing one result received none and reused their resample answers.

In the pre-specified budget sweep, each core model answered the 100 diagnosis questions once at 0.5× and 2× the default budget with leaderboard settings; the 1× condition uses leaderboard first responses. MiniMax M3 used its own endpoint because the leaderboard provider no longer accepted an explicit tool choice.

## Response outcomes and access logs

The response-outcome table distinguishes decision errors from correct answers that fail rule checks.

Response outcomes over thirteen models, 3,900 responses per task; aborted responses count as wrong. Outcome percentages divide by 3,900. The last two percentages divide by n, the number of failed-check responses, including wrong answers; citation failure includes responses that also fail a policy check. Rounded outcome percentages may not sum to 100%.

| Task | Correct, checks pass (%) | Correct, check fails (%) | Wrong (%) | Failed-check responses (n) | Citation failure (% of n) | Policy only (% of n) |
| --- | --- | --- | --- | --- | --- | --- |
| Triage | 86.2 | 0.0 | 13.8 | 0 | --- | --- |
| Medication adjustment | 72.6 | 1.3 | 26.2 | 73 | 0.0 | 100.0 |
| Lab-change attribution | 70.8 | 0.0 | 29.2 | 0 | --- | --- |
| Diagnosis | 61.0 | 4.0 | 35.1 | 312 | 76.3 | 23.7 |

The access logs report calls, date filtering, pagination, execution limits and diagnosis-test expenditure. Denominators differ by measure, as specified with the table. A trace and its returned rows determine which decisive findings were obtained; see Access layer and traces above.

API use by eight core models. Unless labeled as counts or means, percentages use non-aborted responses; date-bounded calls use read calls, paging uses capped responses, and the final row uses all 2,400 responses. “A date bound excluded rows” is a lower bound.

| Measure | Triage | Medication adjustment | Lab-change attribution | Diagnosis |
| --- | --- | --- | --- | --- |
| Non-aborted responses | 2,400 | 2,398 | 2,400 | 2,370 |
| Calls per response (mean) | 17.1 | 17.7 | 12.9 | 21.5 |
| Date-bounded read calls (% of reads) | 41.0 | 42.3 | 41.8 | 45.8 |
| A date bound excluded rows | 20.5 | 35.9 | 30.3 | 8.4 |
| Received a capped result | 0.0 | 5.8 | 0.5 | 36.8 |
| requested an older page (% of capped) | --- | 0.0 | 18.2 | 71.8 |
| Turn or call limit reached (count) | 0 | 0 | 1 | 0 |
| Test budget spent (mean %) | --- | --- | --- | 51.2 |
| Spent ≥90% of the budget | --- | --- | --- | 13.5 |
| Had a purchase refused | --- | --- | --- | 6.3 |
| Budget exhausted, no answer (% of all) | --- | --- | --- | 1.1 |

## Generation validation and release inventory

The [generation entry point](../tools/generation/make_pack.py) constructs a candidate pool and selects the released question set. The [validator](../src/haenv/world/validation/invariants.py) applies registered conditions to both scopes. A violation of a blocking condition in either scope prevents output. Statistical screens remain diagnostic reports and do not establish clinical impossibility. The table describes acceptance conditions, rather than additional experimental measurements.

| Condition | Requirement | Prevents output on violation |
| --- | --- | --- |
| Chronic conditions | Every chronic condition in the shown record exists in the patient state that produces the laboratory results. | Yes |
| Active medications | A drug active on day T that affects a readable stream is visible, and every shown drug exists in the patient state. Current medication lists and the diagnosis scenario agree with active doses on T. | No |
| Laboratory conclusions | A laboratory conclusion in visible text, whether a number or direction, agrees with readable or purchasable values for the same analyte within 30 days. | Yes |
| Agreement across channels | An analyte read on the same day in a record stream, report, visit, purchase or follow-up has one value, within 2% or one rounding unit. | Yes |
| Accessible reference inputs | Every value used by a reference decision is readable or purchasable. This includes diagnosis medication-direction readings, medication-adjustment visit readings, adherence a(T) equal to the printed rule’s coverage of printed refills outside the gray band on target-missed items, and lab-change records. Every reference diagnosis test returns a result. | No |
| Body build | Body-build descriptions agree with visible weights. Obesity requires a weight reaching BMI 28 at a short adult’s height. “Always thin” permits no weight reaching BMI 28 at a tall adult’s height and no obesity in the record. | No |
| Critical values and urgency | A readable or purchasable critical value on day T—potassium below 2.5 or above 6.5, sodium below 120 or above 160, or glucose below 2.8 or above 25 mmol/L—requires red urgency in diagnosis or an accepted triage disposition reaching emergency care or care within 24 hours. | Yes |
| Composite laboratory values | Same-day composite values agree with their components: FIB-4 = age × AST / (PLT × √ALT) within 15% (Sterling et al., 2006, Hepatology; age at the age-band midpoint ±2.5 years). Purchased total cholesterol follows the declared direct-LDL residual model when triglycerides are below 4.5, rather than exact Friedewald equality. A FIB-4 stream with same-day AST and ALT implies platelets of 50–600. | No |
| Daily laboratory uniqueness | Each laboratory stream has at most one record per day. | No |
| Demographic agreement | The patient’s stated age and sex agree with the profile. A stated age falls in the age band with a tolerance of ±1 year. Menstruation text is absent for men and for women whose age band starts at 55, unless menopause is mentioned. | No |
| Medication statements | Drugs described as taken appear in the visible medication list or patient state, and a stated antihypertensive count equals the visible count. | Yes |
| Value boundaries | Readings at registered value-domain boundaries are screened. Display rounding may legitimately reach a boundary; clipping is diagnosed from the generating operation, rather than equality alone. | No |
| Plausible value ranges | Every readable or purchasable value lies within the population-plausible range defined by the findings registry’s lower limits and the validator’s upper limits. | Yes |
| Finding types | Qualitative findings return categories, and quantitative findings return numbers. | Yes |
| Glycemic agreement | Discordance among HbA1c, sparse fasting plasma glucose and continuous-glucose summaries is screened using the registered thresholds. ADAG relates HbA1c to average glucose, rather than sparse fasting measurements. Treatment transitions, sampling and assay interference require review; flags do not establish impossibility. | No |
| Drug-response screening | LDL increases after dose escalation and low glucose with drugs carrying low hypoglycemia risk are screened. Observed changes do not isolate drug effects; exposure timing, adherence, background disease and noise require review. Thresholds are for screening. | No |
| Diagnosis-marker screening | Diagnosis-marker discordance is screened. Missing type 2 diabetes labels may reflect undiagnosed disease; normal values with a diabetes history may reflect remission. Repeated displays of one reading are not independent confirmation samples. | No |
| Sex-specific intervals | Sex-specific analytes carry the patient’s reference interval: hemoglobin 130–175 g/L for men and 115–150 g/L for women (WS/T 405-2012). | Yes |
| Rates of change | Changes exceeding the registered weekly model rate plus 2.77 × coefficient of variation × value are screened. This statistical scale is not a universal physiological limit; acute changes and noise tails require review. | No |
| Decision-class coverage | Every printed rule and decision class has enough released items. Per 100 triage items, rules for care within 24 hours require at least 3 examples and other rules at least 5. Smaller triage sets use production integer quotas with at least one example per rule. Medication-adjustment decisions and lab-change classes require at least 5 examples. | Yes |
| Scoring-branch coverage | Every reference branch that can earn score has at least one released item: diagnosis urgency across 4 categories, medication-direction acceptance across 4 categories, and test cost within or above budget; medication-adjustment decision by subtype; and lab-change class by next step. | Yes |
| Patient diversity | Each item has an independently sampled patient. Complete benign-complaint bundles are not copied: the criterion is Jaccard similarity above 0.9 with at least 4 distinct complaints. | Yes |
| Dated source records | Non-acting factors have dated, retrievable source records; value fields contain no inline exemption. | Yes |
| Historical timing | Every history-dependent decisive record precedes the recent boundary by at least the registered gap and is retrievable. | Yes |
| Historical dependence | Deleting each decisive old source changes the reference decision. A registered treatment baseline may become indeterminate. For other cases, deleting every earlier date individually and together preserves the reference decision. | Yes |
| Historical source distributions | Acting and inactive archived sources share record types and date-age distributions. | Yes |
| Look-alike source distributions | Acting and inactive look-alike records have identical registered source distributions. | Yes |
| Lab-change shortcut screening | The effect of using only header amplitude to distinguish true change, method difference and preanalytical causes remains below the registered shortcut threshold. | Yes |
| Symptom-status shortcut screening | A reference stub reading only recorded symptom status has a chance-corrected effect below 0.20. | Yes |
| Medication-direction consistency | Named-drug directions respect dated adverse-effect reductions, dose feasibility and relevance to the hidden disease. | Yes |
| Diagnostic workup feasibility | The sourced initial required test set fits the actual menu budget with the registered margin. Post-diagnosis tests and duplicate alternatives are not mandatory. | Yes |
| Diagnostic test availability | Each test required by the reference diagnosis exactly matches a returned menu item. | Yes |
| Acquisition scope | CT pulmonary angiography, abdominal imaging, ambulatory ECG and elastography return only findings within their named acquisition scope. | Yes |
| Diagnostic urgency consistency | A component with identical key readings has the same urgency when it determines case urgency; escalation due to a companion disease or critical value is explicit. | Yes |
| Procedure presentation | Relevant and unrelated procedures have comparable lengths and step distributions and identical formats. | Yes |
| Automatic scores | The automatic score is a registered, fallible score report; the reference decision can be recomputed from completed observations. | Yes |
| Automatic-score error rates | Automatic-scale and missing-input error shares agree with registered populations and the limit over the full question set. | Yes |

The release includes [difficulty parameters](../registry/difficulty_r2.yaml), [triage scenarios](../registry/pack2_triage_scenarios.yaml), [medication-adjustment sources](../registry/p3_clinical_sources.yaml), [lab-change sources](../registry/p4_clinical_sources.yaml), the [management rubric definition](../registry/management_rubric.yaml) and its [documentation](design/management-rubric.md). These registries distinguish sourced frequencies from author-set choices. Stored responses and judgments support score recomputation with the evaluation scorer described under Access layer and traces above; the rubric documents plan-quality evaluation and semantic matching is described under Task specifications and scoring above.
