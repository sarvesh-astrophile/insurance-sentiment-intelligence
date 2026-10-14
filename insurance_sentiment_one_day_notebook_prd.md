# Product Requirements Document (PRD)

## Insurance Sentiment Intelligence — One-Day Jupyter Notebook MVP

**Version:** 2.0 (scope reduced from web-platform PRD)  
**Date:** 14 October 2026  
**Audience:** AI Engineer interview portfolio  
**Delivery format:** One well-documented Python Jupyter Notebook (`insurance_sentiment_analysis.ipynb`) plus a small README and reproducible outputs  
**Time budget:** ~8 focused hours (hard stop after one day)  
**Core comparison:** TypeSafe Jev vs a simple sentiment baseline; a pretrained transformer only if time permits  
**Decision to demonstrate:** Is Jev accurate and cost-efficient enough for classifying insurance complaints compared with an accessible baseline?

---

## 1. Summary

Build a **reproducible notebook** that takes a small collection of insurance customer complaints, cleans and labels them, predicts sentiment using at least two approaches, evaluates those predictions against human-labelled examples, and visualises recurring complaint categories. Jev additionally extracts insurance-specific issue type, urgency, and explicit cancellation intent. The project is **analysis and decision support**, not automated insurance claim handling.

**Interview elevator pitch:** “I built a reproducible insurance complaint-analysis notebook. I compared a rule-based baseline against Jev on the same labelled cases, measured accuracy, latency and API cost, inspected failure cases, and used structured classifications to surface complaints that a human should review.”

## 2. Problem and root cause

Insurance feedback often mixes multiple opinions: “The agent was polite, but my reimbursement is two months late.” A simple positive/negative score can obscure the actual operational problem. Teams also need to understand claim delays, billing issues, customer-service problems and explicit cancellation threats.

**Root cause:** Unstructured complaints have no consistent labels or evidence-backed measurement. This notebook turns text into repeatable classifications and measures whether the classification approach works.

## 3. One-day goals and exclusions

### P0 — Must ship today

1. Run end-to-end in one Jupyter Notebook on a fresh environment with documented installation instructions.
2. Load **at least 50 usable, non-duplicate feedback examples**, preferably real permitted public reviews; **aim for 80–150** if easy to source. Save the dataset locally as CSV. Do not collect restricted or identifiable customer data.
3. Manually label **at least 40–60 real examples** for a small *exploratory* evaluation set; strive for 80 if feasible. Label overall sentiment as `positive`, `negative`, or `neutral`. Record the labelling criteria.
4. Run **one basic baseline** (VADER or a similar deterministic rule-based sentiment tool) on the same evaluation rows.
5. Run **Jev** on each evaluation row using a typed **sentiment Choice** question. When available, also request issue category, urgency and explicit cancellation intent.
6. Evaluate both models on identical examples: accuracy, macro-F1, class-wise precision/recall, confusion matrix, latency and actual/estimated per-record API cost.
7. Show 2–4 notebook plots plus a table of wrongly classified examples.
8. Save outputs (`predictions.csv`, `benchmark_results.csv`) and write a brief conclusion based on measured evidence.

### P1 — Only after P0 works

- Add a pretrained Hugging Face sentiment model as a third comparison. No model fine-tuning required.
- Add a small set of annotated `mixed`-sentiment cases and show them separately; **do not silently map mixed to neutral**.
- Plot negative-review share per insurance issue type (issue classification is Jev-only unless also annotated and implemented for all models).
- Add a simple review flag for `high urgency`, `explicit cancellation`, or `unclassifiable` complaints.

### P2 — Not for the one-day MVP

No Next.js, Convex, FastAPI, SQL database, React dashboard, deployment, RAG, vector database, multi-agent system, orchestration framework, fine-tuning, real-time ingestion, speech-to-text, automated claim decisions, production privacy certification, or advanced calibration research.

## 4. Users and primary demonstration

**Primary user:** AI Engineer interviewer reviewing a self-contained experiment.  
**Secondary user:** Insurance CX analyst exploring basic customer feedback trends.

**Demo input:** “The support agent was nice, but my insurance claim has been pending for 40 days. If nothing changes, I will cancel.”

**Expected illustrative output (not a guaranteed model prediction):**

```json
{
  "sentiment": "negative",
  "issue": "claim_delay",
  "urgency": "high",
  "explicit_cancellation_intent": true,
  "needs_human_review": true
}
```

The notebook should show *which model predicted this*, whether the output is valid, and an excerpt of the original complaint. No automatic customer action.

## 5. Data and label design

### Required columns

| Field | Type | Meaning |
|---|---|---|
| `review_id` | string | Stable ID |
| `text` | string | Review/complaint text |
| `source` | string | Dataset source or `synthetic_demo` |
| `gold_sentiment` | string or blank | Human label: positive / negative / neutral |
| `is_test` | bool | Included in gold benchmark? |

### Additional output columns

`baseline_sentiment`, `jev_sentiment`, `jev_issue`, `jev_urgency`, `jev_cancellation`, `baseline_latency_ms`, `jev_latency_ms`, `jev_estimated_cost_usd`, `jev_status`.

### Data policy

- Choose the first **legally reusable** public insurance-review dataset you can download quickly. If the dataset is French, use an appropriate multilingual method or choose English samples; do not pretend a monolingual English classifier is fair on untranslated French text.
- Prefer genuine feedback for benchmarking. Synthetic examples may demonstrate edge cases, but must be marked `synthetic_demo` and **excluded from the headline evaluation**.
- Remove emails, phone numbers, policy IDs and other unnecessary identifiers before sending text to hosted models. Use mock/public feedback only.
- Do not represent manually labelled 40–60 examples as an authoritative industry-wide benchmark. Report the small sample size.
- **Important fairness rule:** All models are compared on the same **three common sentiment labels** (`positive`, `negative`, `neutral`). Store mixed reviews separately for exploratory analysis rather than biasing a model comparison whose baseline cannot represent `mixed`.

### Simple sentiment rubric

- **Positive:** Predominantly praise, satisfaction or approval.
- **Negative:** Predominantly dissatisfaction, complaint or criticism.
- **Neutral:** Factual/unclear with no clear positive or negative stance.
- **Mixed:** Both meaningful positive and negative opinions; mark as `mixed_demo` and analyse qualitatively if time permits.

The same review must not be used to tune prompts and evaluate the final reported results without disclosing that limitation. For a one-day notebook, freeze a small test set before experimenting.

## 6. Prediction methods

### A. Baseline: VADER (P0)

Run a lightweight lexicon-based sentiment classifier. Use documented, fixed polarity thresholds. Show its known shortcomings on insurance-domain language and mixed opinions. No training step.

### B. TypeSafe Jev System One (P0)

Use TypeSafe's current official SDK/API. Put credentials in environment variables or `.env` (never in a committed notebook). Define typed questions:

1. **Sentiment — `Choice` (required):** Positive / Negative / Neutral.
2. **Primary issue — `Choice` (preferred):** `claim_delay`, `claim_denial`, `billing`, `premium`, `coverage`, `customer_support`, `technical_issue`, `other`, `not_applicable`.
3. **Urgency — `Score` (preferred):** A clearly described scale such as 0 routine, 1 normal, 2 time-sensitive, 3 high priority. Treat as triage, not a final determination.
4. **Explicit cancellation threat — `Noul` (preferred):** Probability-like value associated with clearly expressed cancellation intent; do not call it a calibrated probability without measurement.

Evaluate each **individual complaint**; do not feed only the 25 most popular or 25 most recent records while claiming to measure sentiment over the full dataset. Use API caching by stable input hash to avoid duplicate charges. Validate and normalise Jev's outputs.

**If Jev credentials/API access are blocked:** Run the full pipeline with the baseline, keep the Jev adapter as clearly marked `not_run`, and document this honestly. Do not fabricate Jev predictions or performance.

### C. Pretrained transformer (P1)

Use a ready-made Hugging Face sentiment pipeline if easy to install. Map its supported classes to the same three common labels. Do not fine-tune or add infrastructure merely to make the comparison larger.

## 7. Jupyter Notebook sections (cell-by-cell specification)

| Section | Required content | Output |
|---|---|---|
| 0. Title and question | Objective, dataset assumptions, model list | Short markdown introduction |
| 1. Setup | Versions, imports, random seed, key loading | Environment summary |
| 2. Load data | CSV ingest or known sample file | DataFrame preview and row counts |
| 3. Data audit | Missing text, duplicates, lengths, basic cleaning, PII masking | Clean rows + quality table |
| 4. Annotation | Label definitions, gold examples and test-set count | Evaluation DataFrame |
| 5. Exploratory analysis | Label distribution, text lengths, example reviews | 1–2 plots |
| 6. Baseline | VADER predictions and elapsed time | Baseline columns |
| 7. Jev adapter | Typed per-review decisions, cache, errors and timing | Jev columns or explicit `not_run` |
| 8. Optional transformer | Pretrained inference | Optional prediction column |
| 9. Fair evaluation | Align row IDs and labels, calculate metrics | Comparison table, confusion matrices |
| 10. Error inspection | Wrong predictions, mixed/ambiguous examples | Failure table and 3+ observations |
| 11. Insurance insights | Issue distribution, escalation flags, examples | 1–2 charts + flagged DataFrame |
| 12. Conclusion/export | What worked, what failed, caveats, next steps | CSVs and markdown takeaways |

A notebook must be executable **top to bottom** without manually editing intermediate variables. Place optional/network-dependent cells behind clear configuration flags so a reader can reproduce baseline results without keys.

## 8. Output and visualisation requirements

Required visuals (Matplotlib/Plotly are both acceptable):

1. **Gold sentiment distribution** (bar chart; include `n`).
2. **Model accuracy and macro-F1 comparison** (side-by-side bars or simple table).
3. **Confusion matrix** for each model that actually ran.
4. **Insurance issue-category distribution** from Jev (optional if Jev isn't available).

Required tables:

- Preview of input reviews and manual labels.
- One row per analysed complaint with prediction, timing, issue, urgency and errors.
- Model comparison metrics, sample size, model version, mean/p95 latency, and cost assumptions.
- At least five error cases (or all errors if fewer) with notes about why the model may have failed.

## 9. Acceptance criteria / Definition of done

- [ ] Notebook starts with a clear business problem and hypothesis.
- [ ] Dataset source, licence/use assumptions and row counts are documented.
- [ ] At least 50 usable records loaded and at least 40–60 *real* examples labelled for an exploratory gold set.
- [ ] A baseline runs deterministically end-to-end.
- [ ] Jev has been executed on the same gold rows **or its inability to run is clearly documented**; no invented results.
- [ ] At least two models compared if Jev/API access works.
- [ ] At least accuracy, macro-F1 and confusion matrix calculated on common labels.
- [ ] Basic inference latency and stated cost/measurement assumptions included.
- [ ] At least two plots and an error-analysis table rendered.
- [ ] Predictions and benchmark CSV files exported.
- [ ] A 5-minute interview demonstration can be given from the notebook.
- [ ] No real policy/customer information is exposed or auto-actioned.

## 10. One-day build schedule (~8 hours)

| Time block | Activity | Stop condition |
|---|---|---|
| 0:00–0:45 | Create notebook, install libraries, acquire data | CSV loads correctly |
| 0:45–2:00 | Clean data, document rubric, hand-label 40–60 examples | Gold labels frozen |
| 2:00–2:45 | Implement and run VADER baseline | Baseline metrics computed |
| 2:45–4:15 | Integrate Jev on same rows, with caching/error handling | Typed predictions recorded |
| 4:15–5:15 | Model comparison, confusion matrices, per-record timings | Benchmark table rendered |
| 5:15–6:15 | Insurance issue breakdown, flagged examples, failure inspection | Insights visible |
| 6:15–7:15 | Conclusions, README, exports, rerun top-to-bottom | Reproducible notebook |
| 7:15–8:00 | Prepare 5-minute walkthrough; use remaining time for pretrained transformer | Demo-ready |

**Hard prioritisation:** If sourcing data or Jev integration takes longer, reduce dataset size to the documented minimum and skip the optional transformer. Do not add a frontend.

## 11. File deliverables

```text
insurance-sentiment-notebook/
├── insurance_sentiment_analysis.ipynb
├── README.md
├── requirements.txt
├── .env.example                  # TYPESAFE_API_KEY= (never commit .env)
├── data/
│   └── insurance_reviews.csv     # Public/de-identified and licence-appropriate
└── outputs/
    ├── predictions.csv
    └── benchmark_results.csv
```

Keep the repository small. Export plots to `outputs/` only if helpful for the README.

## 12. Five-minute interview storyline

1. **30 seconds:** Explain why positive/negative sentiment alone isn't actionable for insurers.
2. **60 seconds:** Show real review data and explain the annotation rubric.
3. **60 seconds:** Run one complaint through baseline and Jev; explain typed decisions.
4. **90 seconds:** Show the benchmark: macro-F1, confusion matrix, latency, cost and disagreements.
5. **60 seconds:** Show issue insights, high-priority cases and a difficult example.
6. **40 seconds:** Explain limitations: small test set, annotation uncertainty, domain shift, privacy and human review.

## 13. Interview-ready engineering decisions

- **Why Jev?** Typed, narrow decisions are a plausible cost/latency optimisation; measure rather than assume a benefit.
- **Why VADER?** A cheap, transparent baseline shows whether a more sophisticated system adds value.
- **Why a common three-class evaluation?** A fair comparison requires equivalent tasks; special mixed sentiment is assessed separately.
- **Why individual prediction?** Traceability, auditable failure analysis and representative aggregate results.
- **Why not a frontend or RAG?** Neither is required to test the primary hypothesis within one day.
- **Why not trust confidence scores?** A score is not an empirically calibrated probability without testing.
- **What happens next?** Collect larger, double-annotated insurance datasets; evaluate mixed/aspect-level tasks; implement explicit abstention, privacy controls and a human review service.

**Final success criterion:** One clean, honestly evaluated, reproducible notebook is better interview evidence than an unfinished full-stack AI platform.
