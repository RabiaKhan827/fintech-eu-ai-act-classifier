import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

import fintech_risk_pipeline as pipe  # noqa: E402
from fintech_ai_act import EUAIActRiskClassifier  # noqa: E402


class TestRuleBaseline(unittest.TestCase):
    def test_tiers(self):
        clf = EUAIActRiskClassifier()
        self.assertEqual(clf.predict_tier("AI creditworthiness model for loan approval"), "High-Risk")
        self.assertEqual(clf.predict_tier("Customer support virtual assistant chatbot"), "Limited-Risk")
        self.assertEqual(clf.predict_tier("Sort monthly expenses by date"), "Minimal-Risk")
        self.assertEqual(clf.predict_tier("Citizen social scoring engine"), "Unacceptable")


class TestPipelineHelpers(unittest.TestCase):
    def test_label_canonicalisation(self):
        self.assertEqual(pipe.canonical_label("HIGH-RISK"), "High-Risk")
        self.assertEqual(pipe.canonical_label(" high risk "), "High-Risk")
        self.assertEqual(pipe.canonical_label("Unacceptable/Blocked"), "Unacceptable")
        self.assertEqual(pipe.canonical_label("Limited Risk"), "Limited-Risk")
        self.assertEqual(pipe.canonical_label("minimal"), "Minimal-Risk")
        self.assertIsNone(pipe.canonical_label("banana"))
        self.assertIsNone(pipe.canonical_label(float("nan")))

    def test_clean_text(self):
        self.assertEqual(pipe.clean_text("As a User, I want 3 loans!  "), "as a user i want loans")

    def test_cleaning_removes_bad_rows(self):
        df = pd.DataFrame({
            "story": ["a story one", "a story one", None, "conflict", "conflict", "other"] + ["x%d item" % i for i in range(30)],
            "risk": ["High-Risk", "high risk", "High-Risk", "High-Risk", "Minimal-Risk", "Limited-Risk"] + ["Minimal-Risk"] * 15 + ["High-Risk"] * 15,
        })
        out, log, fig = pipe.clean_data(df, ["story"], "risk", None, min_class_size=1)
        pipe.plt.close(fig)
        self.assertEqual(log["dropped_missing_text_or_label"], 1)
        self.assertEqual(log["dropped_conflicting_duplicates"], 2)
        self.assertEqual(log["dropped_duplicates"], 1)
        self.assertNotIn("conflict", set(out["text_clean"]))


class TestEndToEnd(unittest.TestCase):
    def test_demo_run_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipe.main(["--demo", "--out", tmp, "--cv-folds", "3"])
            out = Path(tmp)
            for name in ("model_comparison.csv", "summary.json", "test_predictions.csv", "best_model.joblib"):
                self.assertTrue((out / name).exists(), name)
            self.assertGreaterEqual(len(list((out / "figures").glob("*.png"))), 10)
            res = pd.read_csv(out / "model_comparison.csv")
            self.assertTrue(res["test_accuracy"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
