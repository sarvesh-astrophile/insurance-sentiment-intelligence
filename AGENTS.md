# Repository guidance

## Execution and verification
- Use Python 3.12: `uv venv .venv`, then `uv pip install --python .venv/bin/python -r requirements.txt` (exact dependency pins).
- Offline/headless smoke run: `MPLBACKEND=Agg SHOW_PLOTS=false RUN_JEV=false .venv/bin/python insurance_sentiment_analysis.py`. This rewrites local `outputs/`.
- Offline suite: `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -v`.
- Single test: `MPLBACKEND=Agg .venv/bin/python -m unittest discover -s tests -k test_changed_text_rejects_stale_annotation -v`. Tests mock Jev; no credentials or live API are needed.
- No repository lint, formatter, typecheck or CI configuration currently exists; do not assume those checks are wired up.

## Entrypoint and configuration
- `insurance_sentiment_analysis.py` is the actual notebook-style `# %%` deliverable, despite the PRD's `.ipynb` layout. `main()` wires loading → annotations → VADER → optional Jev → evaluation/exports; importing does not run the pipeline.
- Paths resolve from the script directory; interactive cells without `__file__` require the repository root as cwd.
- Import loads root `.env`; `Config.from_env()` reads it, while `Config()` keeps explicit offline defaults. Set `RUN_JEV=false` for smoke runs even if local `.env` enables inference.
- README references `.env.example`, but that file is currently absent. Do not read or expose local `.env` credentials to compensate.
- Use `insurance_sentiment_one_day_notebook_prd.md` for acceptance criteria, not current file paths. Scope is exploratory decision support, not a web app, training pipeline or automated insurance action.

## Frozen data and evaluation
- Pipeline input is `data/daa_reviews_sample.csv`, not the raw root CSV. `scripts/create_daa_sample.py` reads raw CP1252 data and overwrites the UTF-8 sample (seed 42, 20 per rating); do not casually regenerate the frozen sample.
- `data/daa_annotations.csv` owns frozen test membership and gold labels; `gold_sentiment`/`is_test` in the sample CSV are not used as labels. Ratings must never enter inference or label derivation.
- `scripts/prepare_daa_annotations.py` is a one-time bootstrap with hard-coded ID/order guards and refuses to overwrite annotations. Preserve AI drafts; only independent human review with `human_sentiment`, `human_approved=True` and nonblank `reviewer` creates gold.
- Annotation `text_sha256` hashes the output of `clean_text()`. Changing masking or sample text can invalidate the holdout: explicitly re-review annotations/hashes rather than bypassing validation.
- Tests read the real sample/annotation files; `test_frozen_drafts_are_not_gold` asserts 100 reviews, 50 frozen IDs and no approved gold. Legitimate human approvals require reconciling that fixture-dependent expectation, not deleting approvals.
- Keep `mixed_demo` separate from the three common classes. Compare Jev and VADER only on identical successful human-gold IDs (`paired_common_human_gold`); the full VADER gold cohort is separate. Blank metrics mean unavailable, not zero.

## Hosted inference and sensitive artifacts
- Live Jev requires `RUN_JEV=true`, `HOSTED_DATA_APPROVED=true` and `TYPESAFE_API_KEY`; do not enable it just to verify code. Source reuse permission is unresolved and regex masking is incomplete: manual privacy/reuse review must precede hosted use.
- Send only masked review text. Preserve output validation, request caps, disabled retries and fail-closed behavior; unavailable inference remains `not_run`, never fabricated predictions.
- `.cache/jev/` stores raw responses, including invalid ones to avoid repeat charges. Cache identity includes masked text, questions, requested model, SDK version and adapter version; changing these or removing caches can incur new charges. `jev-latest` does not automatically invalidate old cached responses.
- `outputs/`, caches and `.env` are ignored for privacy, not disposable publication artifacts. Never publish them wholesale. Missing pricing/token usage is unknown cost; cache lookup latency is not API latency.
