"""FinTech Agile User Stories & EU AI Act Risk Classification Toolkit (DOI: 10.17632/ww3wmkw466.1).

``EUAIActRiskClassifier`` is a transparent keyword-rule *baseline*. It is not a
trained model and its accuracy is never assumed: measure it on the dataset with
``fintech_risk_pipeline.py`` at the repository root, which compares it against
trained TF-IDF models on the same split.
"""

__version__ = "1.1.0"
__author__ = "Rabia Khan"
__doi__ = "10.17632/ww3wmkw466.1"


class EUAIActRiskClassifier:
    HIGH_RISK_CUES = {"creditworthiness", "credit score", "loan approval", "lending", "insurance underwriting",
                      "risk pricing", "biometric", "fraud freeze"}
    LIMITED_RISK_CUES = {"chatbot", "virtual assistant", "conversational", "generated"}
    UNACCEPTABLE_CUES = {"social scoring", "subliminal manipulation"}

    def predict_tier(self, user_story: str) -> str:
        low = user_story.lower()
        if any(c in low for c in self.UNACCEPTABLE_CUES):
            return "Unacceptable"
        if any(c in low for c in self.HIGH_RISK_CUES):
            return "High-Risk"
        if any(c in low for c in self.LIMITED_RISK_CUES):
            return "Limited-Risk"
        return "Minimal-Risk"
