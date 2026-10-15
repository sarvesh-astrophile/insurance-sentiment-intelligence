"""Offline safeguards; no Jev credentials or API calls needed."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from insurance_sentiment_analysis import (
    Config, ROOT, attach_annotations, clean_text, estimate_cost, evaluate,
    load_reviews, plot_model_comparison, run_baseline, run_jev, validate_jev,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.reviews, _ = load_reviews(ROOT / "data/daa_reviews_sample.csv")
        # Test an unapproved fixture, not the user's changing approval state.
        # Never clear approvals in the real annotation CSV.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.annotations = Path(self.temp.name) / "annotations.csv"
        fixture = pd.read_csv(ROOT / "data/daa_annotations.csv", dtype=str, keep_default_na=False)
        fixture["human_sentiment"], fixture["human_approved"], fixture["reviewer"] = "", "False", ""
        fixture.to_csv(self.annotations, index=False)
        self.reviews = attach_annotations(self.reviews, self.annotations)

    def test_frozen_drafts_are_not_gold(self):
        self.assertEqual(len(self.reviews), 100)
        self.assertEqual(int(self.reviews.is_test.sum()), 50)
        self.assertFalse(self.reviews.gold_sentiment.ne("").any())

    def test_pii_masking_preserves_sentiment(self):
        text = clean_text("Not good! Email a@b.com; claim number 830549. Call +61 412 345 678")
        self.assertIn("Not good!", text)
        for private in ["a@b.com", "830549", "412"]:
            self.assertNotIn(private, text)

    def test_unknown_cost_is_not_zero(self):
        self.assertIsNone(estimate_cost({"input_tokens": 10}, Config()))
        self.assertAlmostEqual(estimate_cost({"input_tokens": 100, "output_tokens": 0},
                                            Config(input_price_per_million=1, output_price_per_million=0)), 0.0001)

    def test_jev_validation(self):
        payload = {"model": "test", "answers": {"sentiment": {"choice": "negative"},
                   "issue": {"choice": "claim_delay"}, "urgency": {"score": 2.7},
                   "cancellation": {"noul": 0.9}}}
        self.assertEqual(validate_jev(payload)["jev_sentiment"], "negative")
        payload["answers"]["urgency"]["score"] = float("nan")
        with self.assertRaises(ValueError):
            validate_jev(payload)

    def test_disabled_and_privacy_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            result = run_jev(self.reviews, Config(), cache)
            self.assertTrue(result.jev_status.eq("not_run").all())
            result = run_jev(self.reviews, Config(run_jev=True), cache)
            self.assertTrue(result.jev_error.str.startswith("blocked").all())
            self.assertEqual(list(cache.iterdir()), [])

    def test_jev_mocked_request_and_cache(self):
        payload = {"model": "mock-only", "usage": {"input_tokens": 100, "output_tokens": 0},
                   "answers": {"sentiment": {"choice": "negative"},
                               "issue": {"choice": "coverage"}, "urgency": {"score": 1.0},
                               "cancellation": {"noul": 0.1}}}
        response = Mock()
        response.model_dump.return_value = payload
        config = Config(run_jev=True, hosted_data_approved=True,
                        input_price_per_million=1, output_price_per_million=0)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"TYPESAFE_API_KEY": "fake-for-offline-test"}):
                with patch("typesafe_sdk.TypeSafeClient") as client:
                    client.return_value.system_one.return_value = response
                    first = run_jev(self.reviews.head(1), config, Path(directory))
                    second = run_jev(self.reviews.head(1), config, Path(directory))
                    self.assertEqual(client.return_value.system_one.call_count, 1)
                    self.assertEqual(first.jev_status.iloc[0], "ok")
                    self.assertEqual(second.jev_status.iloc[0], "cached")
                    self.assertEqual(second.jev_current_run_cost_usd.iloc[0], 0)
                    self.assertAlmostEqual(first.jev_estimated_cost_usd.iloc[0], 0.0001)
                    sent = client.return_value.system_one.call_args.kwargs
                    self.assertEqual(sent["state"], self.reviews.text.iloc[0])
                    self.assertNotIn("rating", sent)

    def test_changed_text_rejects_stale_annotation(self):
        self.reviews.loc[0, "text"] += " changed"
        with self.assertRaises(ValueError):
            attach_annotations(self.reviews[["review_id", "text", "source"]],
                               ROOT / "data/daa_annotations.csv")

    def test_baseline_determinism_and_no_fake_metrics(self):
        first, second = run_baseline(self.reviews), run_baseline(self.reviews)
        self.assertTrue(first.baseline_sentiment.equals(second.baseline_sentiment))
        with tempfile.TemporaryDirectory() as directory:
            df = run_jev(first, Config(), Path(directory))
            metrics, _, errors = evaluate(df, Path(directory), Config(show_plots=False))
            self.assertTrue(metrics.accuracy.isna().all())
            self.assertEqual(len(errors), 0)

    def test_paired_models_have_identical_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            df = run_jev(run_baseline(self.reviews), Config(), Path(directory))
            df.loc[:2, "gold_sentiment"] = ["negative", "positive", "negative"]
            df.loc[:1, "jev_status"] = "ok"
            df.loc[:1, "jev_sentiment"] = ["negative", "positive"]
            df.loc[:1, "jev_model"] = "mock-only"
            df.loc[:1, "jev_latency_ms"] = 1.0
            with patch("insurance_sentiment_analysis.save_plot"):
                metrics, _, _ = evaluate(df, Path(directory), Config(show_plots=False))
            paired = metrics.loc[metrics.cohort.eq("paired_common_human_gold")]
            self.assertEqual(paired.n.tolist(), [2, 2])
            self.assertEqual(metrics.loc[metrics.cohort.eq("human_gold_baseline"), "n"].tolist(), [3])

    def test_comparison_chart_uses_only_paired_models(self):
        import matplotlib.pyplot as plt
        frame = pd.DataFrame([
            {"model": "baseline", "cohort": "human_gold_baseline", "n": 45,
             "status": "evaluated", "accuracy": 0.7, "macro_f1": 0.6},
            {"model": "baseline", "cohort": "paired_common_human_gold", "n": 43,
             "status": "evaluated", "accuracy": 0.72, "macro_f1": 0.61},
            {"model": "jev", "cohort": "paired_common_human_gold", "n": 43,
             "status": "evaluated", "accuracy": 0.88, "macro_f1": 0.625},
        ])
        with tempfile.TemporaryDirectory() as directory:
            with patch("insurance_sentiment_analysis.plt.close"):
                plot_model_comparison(frame, Path(directory), Config(show_plots=False))
                fig = plt.gcf()
                self.assertEqual([tick.get_text() for tick in fig.axes[0].get_xticklabels()], ["VADER", "Jev"])
                self.assertEqual(len(fig.axes[0].patches), 2)
                self.assertTrue(any("same 43" in text.get_text() for text in fig.texts))
            plt.close(fig)
            self.assertTrue((Path(directory) / "model_comparison.png").exists())

    def test_comparison_chart_marks_missing_jev_and_missing_gold(self):
        import matplotlib.pyplot as plt
        frame = pd.DataFrame([{"model": "baseline", "cohort": "human_gold_baseline", "n": 43,
                               "status": "evaluated", "accuracy": 0.72, "macro_f1": 0.61}])
        with tempfile.TemporaryDirectory() as directory:
            for has_gold in [True, False]:
                frame.loc[0, "status"] = "evaluated" if has_gold else "awaiting_human_labels"
                with patch("insurance_sentiment_analysis.plt.close"):
                    plot_model_comparison(frame, Path(directory), Config(show_plots=False))
                    fig = plt.gcf()
                    expected = "No paired Jev" if has_gold else "No evaluated human-gold"
                    self.assertTrue(any(expected in text.get_text() for text in fig.texts))
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
