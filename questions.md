# Interview preparation: Insurance Sentiment Intelligence

Questions are ranked by preparation priority, then ordered to build understanding.
Start with **P0**, trace **P1** through the code, then practise **P2** follow-ups.
This is a question bank, not a claim that every capability is implemented.

## Ground rules for your answers

- Explain what is **implemented**, what was **measured**, and what is only proposed.
- Review text written by a human is not the same as a human-validated sentiment label.
- Disclose AI-assisted drafts, subsequent human review, sample size and class support.
- Never claim urgency/issue/cancellation accuracy without task-specific gold labels.
- Never call cached lookup latency API inference latency, or unknown cost zero.
- Use results from a single consistent run, not an old plot plus a newer CSV.

## Where to study

| Topic | Primary files / functions |
|---|---|
| Problem, hypothesis, acceptance criteria | `insurance_sentiment_one_day_notebook_prd.md`, `README.md` |
| Configuration and typed questions | `Config`, `QUESTION_SPEC` in `insurance_sentiment_analysis.py` |
| Data preparation | `scripts/create_daa_sample.py`, `load_reviews()`, `clean_text()` |
| Labels and holdout | `data/daa_annotations.csv`, `attach_annotations()`, `scripts/prepare_daa_annotations.py` |
| Human review UI | `scripts/review_annotations.py`, `review_app/` |
| VADER | `run_baseline()` |
| Jev, cache and cost | `run_jev()`, `validate_jev()`, `estimate_cost()` |
| Metrics and confusion matrices | `evaluate()` |
| Charts, insights and export | `main()`, `save_plot()` |
| Verification | `tests/test_pipeline.py`, `tests/test_review_app.py` |
| Measured evidence | Local `outputs/benchmark_results.csv`, `class_metrics.csv`, `error_analysis.csv`, `run_metadata.json` |

## P0 — Must answer confidently (rank 1–25)

**Target:** explain each in 30–90 seconds without reading the code.

1. **What business problem does this project solve, and who would use its results?**
2. **What is the precise hypothesis behind comparing Jev with VADER?**
3. **Walk through the entire system, from the input CSV to the final exports.**
4. **What did you implement yourself, and which capabilities come from external libraries or APIs?**
5. **What results can you honestly claim from the latest consistent paired run? What is the evaluation sample size?**
6. **Where did DAA come from, and what remains unverified about authenticity, licence and reuse permission?**
7. **Why use 100 reviews with 20 per star rating? How does this sampling change interpretation?**
8. **Why are star ratings not used as model inputs or automatically converted into gold labels?**
9. **How were the 50 evaluation IDs selected and frozen before model experimentation?**
10. **What is the difference between an AI draft label, a human-approved AI-assisted label and independently double-annotated gold?**
11. **How do you distinguish positive, negative, neutral and genuinely mixed reviews?**
12. **Why exclude mixed cases rather than silently map them to neutral?**
13. **What is VADER, how does it produce a compound score, and why is it a useful baseline?**
14. **What exact VADER thresholds does this project use? Were they tuned on the evaluation rows?**
15. **What is Jev doing here, and what do `Choice`, `Score` and `Noul` each represent?**
16. **How do you ensure that VADER and Jev are evaluated on exactly the same examples?**
17. **What do accuracy, precision, recall, F1 and macro-F1 measure? Why report more than accuracy?**
18. **Why could Jev show substantially higher accuracy but only a small macro-F1 improvement?**
19. **How do you read a confusion matrix, and which error pattern matters most to an insurance CX analyst?**
20. **What are the most important actual failure cases? Can you explain at least three without inventing causes?**
21. **How are inference latency, cache lookup latency and estimated API cost measured and separated?**
22. **What happens when Jev is disabled, blocked, unavailable or returns invalid output?**
23. **What sensitive information might remain after regex masking, and what must happen before hosted inference?**
24. **What can this experiment not prove, and why is it not ready to automate insurance decisions?**
25. **If you had one more week, what would you improve first and why?**

## P1A — Data, annotation and experimental validity (rank 26–42)

**Study:** data scripts, `clean_text()`, `load_reviews()`, `attach_annotations()`.

26. Why was the raw CSV read as Windows-1252 while the frozen sample is UTF-8?
27. How are title and body combined, and could a title change the apparent sentiment?
28. What is removed during sampling versus removed when the analysis loads the sample?
29. How is duplicate text detected? What near-duplicates would this method miss?
30. Why preserve negation, punctuation, capitalization and complaint details instead of aggressively normalizing text?
31. How are stable review IDs obtained, and why validate uniqueness before joining annotations?
32. Why keep annotations separate from the original sample rather than overwrite its label columns?
33. What is `text_sha256`, exactly which text does it hash, and what happens if masking changes?
34. Which fields must be present before an annotation becomes approved gold? How are invalid approvals rejected?
35. How does the local app reduce accidental approval and AI-label anchoring? What anchoring risk still remains?
36. How are neutral versus mildly negative requests handled? Give an ambiguous example and defend your choice.
37. How would you adjudicate a disagreement between two human reviewers? Would you measure Cohen's kappa?
38. Could seeing AI drafts, VADER predictions or Jev outputs influence human labels? How would you prevent that next time?
39. Have any prompts or decisions been changed after inspecting frozen-set results? What would need disclosure?
40. Why might predominantly travel-insurance reviews fail to generalize to health, home or motor insurance?
41. How do historical dates, pandemic-related complaints, English-only assumptions and no-claim purchase praise limit validity?
42. What would a larger, representative, privacy-reviewed and genuinely independent evaluation set look like?

## P1B — Baseline and Jev implementation (rank 43–61)

**Study:** `QUESTION_SPEC`, `run_baseline()`, `run_jev()`, `validate_jev()`.

43. Trace one row through VADER: input text, score, threshold, prediction and timing output.
44. How does a lexicon-based approach handle negation, intensifiers and contrastive wording? Where does it struggle?
45. Why can words describing a pleasant holiday or a frightening illness confuse sentiment toward the insurer?
46. Does VADER require training or downloading model weights at runtime in this implementation?
47. What exact state and question objects are sent to Jev? How do you prove that ratings and labels are not sent?
48. How are the primary issue categories defined, and how do `other` and `not_applicable` differ?
49. How is urgency defined on a 0–3 scale, and why can the returned score be fractional?
50. What counts as explicit policy cancellation? Why are trip cancellation and “never use again” insufficient alone?
51. Why is a Noul value not automatically a calibrated probability in this application?
52. Which Jev output fields are validated? What happens with an unknown choice, a missing answer, NaN or an out-of-range score?
53. What do `RUN_JEV`, `HOSTED_DATA_APPROVED`, `TYPESAFE_API_KEY` and `JEV_MAX_CALLS` each control?
54. Why prefer an explicit model ID such as `jev-1.14.0` over `jev-latest` for a reproducible benchmark?
55. What exactly is included in cache identity, and why include the questions, SDK version and adapter version?
56. Why cache invalid API responses too? What is the trade-off when correcting a bad response?
57. What happens to cached results if a model alias changes behind the same name?
58. Why are retries disabled here? What reliability versus charge-control trade-off does that create?
59. Why are calls sequential? How do the organization's request-per-second and input-token-per-second limits interact?
60. What happens on HTTP 429, authentication failure or timeout? How would production handling differ from this MVP?
61. What prompt-injection risk exists in untrusted review text? Why are instructions to ignore embedded commands not a complete security guarantee?

## P1C — Metrics, cost, charts and conclusions (rank 62–81)

**Study:** `evaluate()`, `main()`, local exports from the same run.

62. Write the accuracy, precision, recall and F1 formulas and calculate them for a small example.
63. How does macro-F1 differ from weighted-F1 and micro-F1? Which could hide poor minority-class performance?
64. What happens when a class has no true examples or no predictions? How does `zero_division=0` affect reporting?
65. What are the positive, negative and neutral support counts in the current approved cohort?
66. How many predictions did each model get wrong on the paired cohort? Which classes account for the difference?
67. What are `human_gold_baseline` and `paired_common_human_gold`, and why can VADER appear in both?
68. How is Jev coverage calculated? Could successful-only paired evaluation hide systematic API failures?
69. Are mixed, unapproved and synthetic rows excluded from headline metrics? Show the exact filtering logic.
70. What does `draft_disagreements.csv` mean, and why is it not a table of verified model mistakes?
71. Does the error-analysis export contain unique reviews or model/cohort-specific rows? How could naive counting double-count errors?
72. Where is the accuracy/macro-F1 chart generated, and why might it show more than one VADER group?
73. Where are the confusion matrices, sentiment distribution and issue-distribution plots saved? Which require gold labels or Jev results?
74. Why can `SHOW_PLOTS=false` still produce PNG files? How does a headless run differ from an interactive cell run?
75. What does mean versus p95 latency tell you? Which parts of initialization and preprocessing are excluded?
76. If Jev results are cached, can the benchmark's `jev_latency_ms` be used to claim live API speed? Which original-latency field should you inspect instead?
77. How do $42 per billion tokens and $0.042 per million tokens relate? Calculate the estimated cost for 50,000 input tokens.
78. How do original inference cost, current-run cache cost and actual invoiced cost differ? When must cost remain unknown?
79. Why does “zero API cost” for VADER not mean zero compute or operating cost?
80. Are the issue-distribution and human-review flags validated predictions or exploratory insights? What additional annotations would evaluate them?
81. Given the observed sample size and metrics, would you recommend Jev, VADER, a hybrid or further evaluation? State the decision criteria and uncertainty.

## P2A — Local app, testing and reproducibility (rank 82–96)

**Study:** `scripts/review_annotations.py`, `review_app/`, tests and README.

82. How do the HTML/JavaScript page, local HTTP endpoints and CSV store communicate?
83. Why must a save include an explicit reviewer, valid label and approval state? Why is there no approve-all button?
84. Why does changing the selected label clear the approval checkbox?
85. How are original AI drafts and notes preserved while human decisions are saved?
86. How do temporary-file replacement and backups reduce corruption/data-loss risk? What durability guarantees are still absent?
87. How does the CSV revision hash prevent stale browser tabs from silently overwriting newer saves?
88. What concurrency does the server lock protect? What remains unsafe if an external program edits the CSV during a save?
89. Why bind to loopback, validate Host, require a per-session save token and avoid permissive CORS?
90. Why render review content with `textContent` rather than `innerHTML`? What XSS risk does that avoid?
91. Why does the local app load no remote assets and avoid logging review text? Why is it still not a production multi-user app?
92. Which tests protect gold-label integrity, masking, deterministic inference, cache reuse and identical evaluation cohorts?
93. How do mocked Jev tests verify integration behavior without spending money or exposing real credentials?
94. How do annotation-app tests verify backups, invalid inputs, stale revisions and cross-site write protection without changing real approvals?
95. Why must tests use temporary unapproved fixtures after legitimate human approvals are added to the real dataset?
96. What do dependency pins, input hashes, question specifications and model versions capture? What reproducibility risks remain even with them?

## P2B — Hard follow-ups and next-stage design (rank 97–110)

These ask for reasoning or proposed extensions, not existing features.

97. How would you quantify uncertainty in the accuracy/F1 difference—paired bootstrap, confidence intervals or McNemar's test—and what are their limitations here?
98. What prevents you from claiming statistical significance or industry-wide superiority from 43 or 50 reviewed examples?
99. How would you evaluate a selective-classification/abstention strategy using coverage and error rates rather than raw confidence alone?
100. How would you choose human-review thresholds using false-positive/false-negative costs instead of the current heuristics?
101. How would you create separate gold labels for issue, urgency and explicit cancellation, and which metrics would suit each task?
102. How would aspect-level sentiment improve on one overall label for “polite agent, delayed claim”?
103. When would you add a pretrained transformer, and how would you avoid unfairly mapping a two-class model into a three-class benchmark?
104. What would change for multilingual reviews? How would translation errors affect evaluation and privacy?
105. How would a production pipeline implement bounded concurrency, rate-limit coordination, backoff, idempotency and auditability?
106. How would you detect model, data and concept drift without logging unnecessary personal information?
107. How would you version annotations and prompts, protect the holdout and organize future train/development/test splits?
108. How would you minimize, encrypt, retain and delete local text/cache artifacts under a real data-governance policy?
109. What additional fairness checks could be justified across insurance products or customer groups, and why is collecting sensitive attributes itself risky?
110. Why were a frontend platform, RAG, fine-tuning, database and deployment unnecessary for the original one-day hypothesis—and why was a small local annotation UI a justified later addition?

## Chart location and run-consistency checklist

Yes, the comparison chart is implemented by `plot_model_comparison()` in
`insurance_sentiment_analysis.py`, called inside `main()`:

```python
plot_model_comparison(benchmark, output, config)
```

- Chart: **`outputs/model_comparison.png`**.
- It compares **accuracy and macro-F1**, not latency or cost.
- Metrics come from `evaluate()`; `baseline` in the chart means **VADER**.
- For a fair side-by-side comparison, the chart selects only the two
  `paired_common_human_gold` rows. The full baseline cohort remains in the CSV,
  but is not duplicated in the chart.
- The two panels separate accuracy (percentage correct) and macro-F1 (0–1).
  Model names are horizontal and every bar has a numeric value.
- If Jev does not run, the chart explicitly says VADER only; if no gold metrics
  exist, it shows an unavailable message rather than fabricated zero scores.
- Existing conditional PNGs are not automatically deleted when a later run omits
  Jev or a gold cohort. An old paired confusion matrix is not proof the latest run
  performed a paired comparison.
- `SHOW_PLOTS=false` suppresses display windows but still saves plots. Inspect the
  PNG files directly after a headless run.

Before an interview, verify all of these together:

1. `outputs/benchmark_results.csv` contains both models in `paired_common_human_gold`.
2. Both have the same `n`; inspect coverage and per-class support.
3. `outputs/predictions.csv` shows approved label provenance and actual Jev statuses.
4. `outputs/run_metadata.json` identifies the input/annotation hashes and requested model.
5. Plots and CSVs were produced by the same run; compare file timestamps if needed.
6. Report whether latency came from fresh requests or cache hits.
7. State how many labels are approved, mixed and still pending—do not assume all 50 are gold.

**Results note:** a previous paired run reported n=43, VADER accuracy 72.1% /
macro-F1 0.611, and Jev accuracy 88.4% / macro-F1 0.625. These are historical
measurements, not promises about the current output files. At creation of this
guide, the inspected benchmark CSV contained a baseline-only run; old paired
PNGs still existed. Always present a freshly verified, consistent snapshot.

## Five-minute practice sequence

| Time | What to demonstrate | Questions to rehearse |
|---|---|---|
| 0:00–0:30 | Business problem and measurable hypothesis | 1–2 |
| 0:30–1:15 | Dataset, rubric and honest label provenance | 6–12 |
| 1:15–2:00 | One review through VADER and typed Jev decisions | 13–15, 47–51 |
| 2:00–3:30 | Paired chart, class support and two actual errors | 16–20, 65–73 |
| 3:30–4:15 | Cache, latency, cost and fail-closed behavior | 21–22, 55–60, 75–79 |
| 4:15–5:00 | Privacy, limitations and the next experiment | 23–25, 97–101 |

For every answer use: **decision → reason → evidence → limitation → next step**.
