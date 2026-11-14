import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

import fintech_risk_classification as frc  # noqa: E402


class TestHelpers(unittest.TestCase):
    def test_label_canonicalisation(self):
        self.assertEqual(frc.canonical_label("HIGH-RISK"), "High-Risk")
        self.assertEqual(frc.canonical_label(" high risk "), "High-Risk")
        self.assertEqual(frc.canonical_label("Unacceptable/Blocked"), "Unacceptable")
        self.assertEqual(frc.canonical_label("Limited Risk"), "Limited-Risk")
        self.assertEqual(frc.canonical_label("minimal"), "Minimal-Risk")
        self.assertIsNone(frc.canonical_label("banana"))
        self.assertIsNone(frc.canonical_label(float("nan")))

    def test_clean_text(self):
        self.assertEqual(frc.clean_text("As a User, I want 3 loans!  "), "as a user i want loans")

    def test_keyword_baseline(self):
        rules = frc.KeywordRuleBaseline()
        self.assertEqual(rules.predict_tier("AI creditworthiness model for loan approval"), "High-Risk")
        self.assertEqual(rules.predict_tier("Customer support virtual assistant chatbot"), "Limited-Risk")
        self.assertEqual(rules.predict_tier("Sort monthly expenses by date"), "Minimal-Risk")
        self.assertEqual(rules.predict_tier("Citizen social scoring engine"), "Unacceptable")

    def test_cleaning_removes_bad_rows(self):
        df = pd.DataFrame({
            "story": ["a story one", "a story one", None, "conflict", "conflict", "other"] + [f"x{i} item" for i in range(30)],
            "risk": ["High-Risk", "high risk", "High-Risk", "High-Risk", "Minimal-Risk", "Limited-Risk"] + ["Minimal-Risk"] * 15 + ["High-Risk"] * 15,
        })
        out, log = frc.clean_data(df, ["story"], "risk", min_class_size=1)
        self.assertEqual(log["dropped_missing_text_or_label"], 1)
        self.assertEqual(log["dropped_conflicting_duplicates"], 2)
        self.assertEqual(log["dropped_duplicates"], 1)
        self.assertNotIn("conflict", set(out["text_clean"]))

    def test_mcnemar_and_bootstrap(self):
        import numpy as np
        y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        good, bad = y.copy(), 1 - y
        self.assertLess(frc.mcnemar_test(y, good, bad)["p_value"], 0.05)
        self.assertEqual(frc.mcnemar_test(y, good, good)["p_value"], 1.0)
        ci = frc.bootstrap_ci(y, good, 50)
        self.assertEqual(ci["accuracy"], (1.0, 1.0))


class TestEndToEnd(unittest.TestCase):
    def test_demo_run_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            frc.main(["--demo", "--out", tmp, "--cv-folds", "3"])
            out = Path(tmp)
            for name in ("RESULTS.md", "model_comparison.csv", "summary.json", "test_predictions.csv",
                         "misclassified_stories.csv", "near_duplicate_pairs.csv", "run_log.txt", "best_model.joblib"):
                self.assertTrue((out / name).exists(), name)
            self.assertGreaterEqual(len(list((out / "figures").glob("*.png"))), 14)
            res = pd.read_csv(out / "model_comparison.csv")
            self.assertTrue(res["test_accuracy"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
