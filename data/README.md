# DAA annotation sample

`daa_reviews_sample.csv` contains 100 reviews: 20 randomly selected from each
star rating (1–5), using seed 42. Regenerate from the project root:

```bash
python3 scripts/create_daa_sample.py
```

The source CSV is read as Windows-1252; the sample is written as UTF-8.
Whitespace is normalised, empty bodies excluded, and case-insensitive duplicate
bodies removed before sampling. Titles and bodies are combined into `text`.
Review IDs come from the original review URLs. The original CSV is unchanged.

## Annotation and evaluation

The implemented pipeline uses the separate `daa_annotations.csv` rather than
editing the source sample. It freezes 50 review IDs and contains **unapproved
AI-assisted drafts**, notes and masked-text hashes. Enter `human_sentiment`,
`human_approved=True`, and `reviewer` after independent review. Only these approved
labels count as gold. `mixed_demo` is excluded from the common three-class cohort.
The source CSV's blank labels and `is_test=False` remain untouched; the pipeline
derives its evaluation selection from the frozen annotation file instead.
See the root README for the current workflow. The original guidance below applies
to the sample in isolation, not to the separate annotation overlay.

- `gold_sentiment` is deliberately blank. Read the text and manually label it
  `positive`, `negative`, or `neutral` using the PRD rubric. Do not derive labels
  from star ratings; three stars does not necessarily mean neutral.
- Keep genuinely mixed cases separate from the common three-class benchmark.
- `is_test` starts as `False`. Choose and label 40–60 benchmark rows, then freeze
  their IDs and set `is_test` to `True` before tuning prompts. Do not regenerate
  this file after annotating it without saving your annotations separately.
- Models should receive only `text`, not rating or annotation metadata.
- Rating-balanced sampling is for exploratory coverage, not an estimate of the
  original dataset's sentiment distribution. Report this sampling choice.

## Privacy and provenance

The source URLs point to ProductReview.com.au, but the original dataset's licence
and collection provenance have not been verified. Confirm reuse permission before
publishing or redistributing this sample.

This is **not a de-identified dataset**. Free-text reviews may contain personal
information even though identifier columns are not included. Manually inspect
and redact unnecessary identifiers before sending text to hosted models or
publishing it. Whitespace cleanup does not fix encoding artefacts or establish
that the content is safe to share.
