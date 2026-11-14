import unittest
from fintech_ai_act import EUAIActRiskClassifier

class TestFinTechAIAct(unittest.TestCase):
    def test_tiers(self):
        clf = EUAIActRiskClassifier()
        self.assertEqual(clf.predict_tier("AI creditworthiness model for loan approval"), "High-Risk")
        self.assertEqual(clf.predict_tier("Customer support virtual assistant chatbot"), "Limited-Risk")
        self.assertEqual(clf.predict_tier("Sort monthly expenses by date"), "Minimal-Risk")
        self.assertEqual(clf.predict_tier("Citizen social scoring engine"), "Unacceptable")

    def test_benchmark(self):
        rep = EUAIActRiskClassifier().run_benchmark()
        self.assertEqual(rep.classification_accuracy, 1.0)
        self.assertEqual(rep.high_risk_recall, 1.0)

if __name__ == "__main__":
    unittest.main()
