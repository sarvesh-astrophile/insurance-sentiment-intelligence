# Insurance Sentiment Intelligence

A one-day Jupyter Notebook MVP to test whether **TypeSafe Jev adds enough accuracy and insurance-specific insight to justify its latency and API cost compared with VADER**.

**Status:** Planning and repository setup. The notebook, dataset, dependencies and benchmark outputs have not been implemented yet; no performance results are claimed.

See the [product requirements document](insurance_sentiment_one_day_notebook_prd.md) for the full scope and acceptance criteria.

## Planned experiment

- Collect at least 50 usable, non-duplicate, legally reusable insurance reviews; aim for 80–150.
- Manually label 40–60 real examples and freeze a small exploratory test set before prompt experimentation.
- Compare VADER and Jev on the same rows and three labels: `positive`, `negative`, `neutral`.
- Use Jev's typed decisions to extract issue category, urgency and explicit cancellation intent where available.
- Report accuracy, macro-F1, class-wise precision/recall, confusion matrices, mean/p95 latency and actual or estimated API cost.
- Inspect misclassifications, visualise complaint trends and export per-review predictions and benchmark results.

Mixed-sentiment examples are analysed separately, not silently mapped to neutral. Synthetic demonstrations are explicitly marked and excluded from headline evaluation. If Jev access is unavailable, its status must remain `not_run`; no predictions or metrics are fabricated.

## Local environment

The initial environment uses Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv venv .venv
source .venv/bin/activate
```

Dependency installation and notebook execution instructions will be added when those deliverables exist. Jev credentials will use the `TYPESAFE_API_KEY` environment variable; never commit credentials or a populated `.env` file. Baseline execution is intended to work without API keys.

## Planned deliverables

```text
insurance_sentiment_analysis.ipynb   # End-to-end experiment
requirements.txt                   # Reproducible dependencies
.env.example                       # Empty credential template
data/insurance_reviews.csv          # Permitted, de-identified reviews and labels
outputs/predictions.csv             # Per-review predictions, timing and status
outputs/benchmark_results.csv       # Metrics, sample sizes and cost assumptions
```

These are planned files, not currently available artifacts. The notebook will cover setup, data audit and annotation, exploratory analysis, baseline inference, cached Jev inference, fair evaluation, error inspection, insurance insights and exports. A pretrained transformer is optional only after the core experiment works.

## Responsible use and limitations

- Document dataset provenance, licence/use assumptions and row counts before publishing data.
- Remove unnecessary identifiers before sending text to hosted models; do not commit private customer data or raw API caches.
- Treat urgency and cancellation signals as human-review aids, not automated claim or customer decisions.
- Report annotation uncertainty and the limits of a small exploratory test set; do not claim industry-wide validity or calibrated confidence without evidence.
- Keep the scope to a reproducible notebook: no frontend, database, RAG, fine-tuning or deployment.
