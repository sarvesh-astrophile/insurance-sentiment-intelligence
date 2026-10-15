# Insurance Sentiment Intelligence

Exploratory insurance-feedback analysis comparing **VADER** with **TypeSafe Jev**.
Implemented in [`insurance_sentiment_analysis.py`](insurance_sentiment_analysis.py)
as notebook-style `# %%` cells. No frontend, training, or automated customer action.

## Run locally

Python 3.12, from the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python insurance_sentiment_analysis.py
```

Open the Python file in an editor supporting Jupyter Python cells to execute
interactively. For a noninteractive run:

```bash
MPLBACKEND=Agg SHOW_PLOTS=false RUN_JEV=false .venv/bin/python insurance_sentiment_analysis.py
```

No inference network access is needed for VADER after dependencies are installed.
Paths are resolved relative to the script, not your shell directory. Notebook
execution with no `__file__` assumes the repository root is the working directory.
The requested deliverable is currently the cell-based `.py`, not an `.ipynb`.

## Current status and data

- Loads `data/daa_reviews_sample.csv`: 100 unique English-assumed travel-insurance
  reviews, originally 20 per star rating, seed 42. This is **not** a representative
  estimate of insurer/customer sentiment. Stars never enter model inputs or labels.
- Source URLs indicate ProductReview.com.au. Authenticity, collection provenance,
  reuse permission and language coverage still require verification.
- Cleans whitespace and masks obvious emails, phone numbers, URLs, labelled IDs
  and some signoffs. Regex masking is **not complete de-identification**: manually
  inspect names, health details and other identifying context before hosted use.
- VADER runs on all usable rows with fixed compound thresholds: positive `>=0.05`,
  negative `<=-0.05`, otherwise neutral. Punctuation and negation are preserved.
- Jev runs only on the 50 frozen annotation IDs, including mixed qualitative cases.
- **No human-gold performance is claimed yet.** The source sample is unchanged.

## Review the 50 AI-assisted annotation drafts

### Local HTML review app (recommended)

```bash
.venv/bin/python scripts/review_annotations.py
```

Open **http://127.0.0.1:8765** in your browser. Enter your initials, read a review,
choose a sentiment, tick the personal-review confirmation, then **Save & next**.
AI draft labels/notes are hidden behind a disclosure to reduce anchoring; ratings
and model predictions are not shown. Nothing is bulk-approved or auto-approved.
Saving without the checkbox keeps a label unapproved and out of gold metrics.

The app updates `data/daa_annotations.csv` directly and atomically, preserving
drafts, notes and frozen hashes. Before each save, a copy is backed up under
`.cache/annotation_backups/`. Concurrent/stale-page edits are rejected rather than
silently overwritten. Unsaved navigation prompts before discarding edits.
If you change your label, you must check approval again before it counts as gold.
Use the pending filter to see remaining reviews. The app runs on loopback only,
loads no external assets, and makes **no Jev calls**. Do not expose this local
server to a network. Stop it with Ctrl+C; use `--port 8766` if the port is busy.

After reviewing, rerun `insurance_sentiment_analysis.py` to refresh benchmark
metrics (Jev reuses matching cached responses). To refresh locally without API
requests, run `RUN_JEV=false .venv/bin/python insurance_sentiment_analysis.py`;
that mode intentionally omits Jev results, including cached ones.

### Manual CSV editing (alternative)

`data/daa_annotations.csv` freezes the first 50 rows of the previously shuffled
sample. Drafts were assigned by reading text before model experimentation, not
inferred from ratings. `text_sha256` binds annotations to the masked input text.
The one-time bootstrap script refuses to overwrite existing annotations.

1. Match IDs with the review text in local `outputs/predictions.csv` or the sample.
2. Independently read each review. Use the rubric in the script/PRD:
   `positive`, `negative`, `neutral`; balanced meaningful opinions → `mixed_demo`.
3. Enter your final label in `human_sentiment`, your name/initials in `reviewer`,
   and `True` in `human_approved`. Preserve the draft for annotation provenance.
4. Rerun the pipeline. Only approved common-class labels from real-source rows
   enter headline metrics. Mixed cases are exported separately. Approve/correct
   at least 40–60 common-class reviews to satisfy the PRD minimum.

Changing only `draft_sentiment` never creates gold labels. AI assistance must be
disclosed even after human review. Do not tune prompts on the frozen holdout.
If input text is manually redacted later, the hash check intentionally fails:
review the changed text/label and update its hash explicitly before rerunning.

## Enable Jev (optional, paid inference)

Copy `.env.example` to local `.env` and set `TYPESAFE_API_KEY` there—never commit
or paste credentials. The adapter uses official `typesafe-sdk==0.7.5`, validated
against [official SDK docs](https://github.com/typesafe-docs/typesafeai).

Only after verifying reuse permission and manually reviewing/redacting all
selected texts, set:

```dotenv
RUN_JEV=true
HOSTED_DATA_APPROVED=true
JEV_MODEL=jev-latest
JEV_MAX_CALLS=50
```

Prefer an explicit model version for reproducibility. `jev-latest` can change;
cached entries under that alias continue to represent the earlier response.
To deliberately rerun an alias, archive its local cache or change to a pinned
model; this may incur new charges.

Set `JEV_INPUT_USD_PER_MILLION` and `JEV_OUTPUT_USD_PER_MILLION` only after checking
current rates for your chosen model. Explicitly set output price to `0` if free.
Missing prices or token usage mean **unknown cost**, not zero. These are usage-
based estimates, not reconciled invoices; failed requests may incur unknown cost.

Each review sends **only masked text**, never stars, labels, IDs or metadata.
Typed questions extract sentiment (`Choice`), issue (`Choice`), urgency (`Score`,
0–3), and explicit policy cancellation (`Noul`, 0–1). Trip cancellation and
"never use again" alone are excluded from explicit policy-cancellation intent.
Scores are not empirically calibrated here.

The adapter validates outputs, disables automatic retries, stops on an uncached
request failure, and caps new requests. Stable hashes include text, questions,
requested model, adapter version and SDK version. Local `.cache/jev/` is ignored
by Git; valid responses are reused without new API charges. Invalid responses
are also cached to avoid silently paying again. Never publish raw caches.

Disabled/blocked runs honestly report `not_run`. Missing keys, privacy approval,
API errors and call-limit skips cannot generate fake predictions or metrics.

## Outputs and evaluation

`outputs/` is generated locally and ignored by Git because it includes review
text. Review licence/privacy before selectively publishing anything.

| Output | Contents |
|---|---|
| `predictions.csv` | Per-review text, annotation provenance, predictions, latency, Jev errors and triage flags |
| `benchmark_results.csv` | Human-gold metrics, cohort sizes, versions, coverage and cost assumptions |
| `class_metrics.csv` | Class-wise precision/recall/F1/support |
| `error_analysis.csv` | All human-gold misclassifications and inspection prompts |
| `draft_disagreements.csv` | VADER disagreements with **unapproved AI drafts**, not benchmark errors |
| `mixed_cases.csv` | Mixed qualitative cases excluded from common-class metrics |
| `data_audit.csv` | Counts, cleaning and unresolved provenance/privacy checks |
| `run_metadata.json` | Package versions, input hashes, question definitions and pricing configuration |
| `conclusion.md`, `*.png` | Caveats, next steps and rendered plots |

VADER gets a full approved-gold cohort. If Jev succeeds, a **separate paired
cohort uses identical IDs for both models**, with Jev coverage reported. Partial
API success never silently changes only one model's denominator. Macro-F1 uses
all three fixed classes; missing classes get zero undefined metrics, so inspect
support counts before interpreting results.

Latency is wall-clock inference or cache lookup, excluding initialization.
Historical uncached Jev latency is stored separately; do not interpret cache
lookup as API latency. Current-run cache cost is zero; historical token-based
estimated cost is retained. Unknown/absent Jev signals do not mean "safe".
Review flags use heuristic urgency `>=2.5`, cancellation `>=0.5`, or invalid
output, and are human triage only.

## Tests

```bash
MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v
```

Tests run offline, including draft/gold separation, masking, deterministic
baseline results, output validation, privacy blocking and paired cohort sizes.

## Remaining PRD gates

Independent human approval of the draft labels; reuse/language/privacy checks;
live Jev verification; and evidence-based conclusions from the paired benchmark.
The optional transformer is intentionally omitted. See the
[PRD](insurance_sentiment_one_day_notebook_prd.md) for the full acceptance criteria.
