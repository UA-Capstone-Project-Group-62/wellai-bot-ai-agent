from types import SimpleNamespace
import unittest

from src.services.sentiment_monitor import (
    DetectionSource,
    SentimentCategory,
    SentimentMonitor,
    detect_language,
)


class FakeLlm:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content or '{"category":"safe","language":"english","reason":"safe"}'
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


class SentimentMonitorTests(unittest.TestCase):
    def test_rules_detect_english_self_harm_without_llm(self):
        llm = FakeLlm()
        result = SentimentMonitor(llm).evaluate("I want to kill myself")

        self.assertEqual(result.category, SentimentCategory.SELF_HARM)
        self.assertEqual(result.source, DetectionSource.RULES)
        self.assertEqual(result.language, "english")
        self.assertEqual(llm.calls, [])

    def test_rules_detect_malay_profanity_without_llm(self):
        llm = FakeLlm()
        result = SentimentMonitor(llm).evaluate("Awak memang sial")

        self.assertEqual(result.category, SentimentCategory.PROFANITY)
        self.assertEqual(result.source, DetectionSource.RULES)
        self.assertEqual(result.language, "malay")
        self.assertEqual(llm.calls, [])

    def test_rules_detect_mandarin_distress_without_llm(self):
        llm = FakeLlm()
        result = SentimentMonitor(llm).evaluate("我觉得很绝望")

        self.assertEqual(result.category, SentimentCategory.DISTRESS)
        self.assertEqual(result.source, DetectionSource.RULES)
        self.assertEqual(result.language, "mandarin")
        self.assertEqual(llm.calls, [])

    def test_llm_fallback_classifies_nuanced_distress(self):
        llm = FakeLlm(
            '{"category":"distress","language":"english","reason":"hopeless crisis language"}',
        )
        result = SentimentMonitor(llm).evaluate("I cannot keep doing this anymore")

        self.assertEqual(result.category, SentimentCategory.DISTRESS)
        self.assertEqual(result.source, DetectionSource.LLM)
        self.assertEqual(result.language, "english")
        self.assertEqual(len(llm.calls), 1)

    def test_depression_medication_booking_context_is_safe_via_llm(self):
        # Fail-safe: a benign booking that mentions a diagnosis/medication is
        # deferred to the LLM (no keyword short-circuit). A correctly behaving
        # classifier returns safe, per the instruction in _CLASSIFIER_PROMPT not
        # to escalate purely because a condition or medication is mentioned.
        llm = FakeLlm(
            '{"category":"safe","language":"english","reason":"routine booking"}',
        )
        result = SentimentMonitor(llm).evaluate(
            "I need to book a short appoinment to refill my depression medication",
        )

        self.assertEqual(result.category, SentimentCategory.SAFE)
        self.assertEqual(result.source, DetectionSource.LLM)
        self.assertEqual(len(llm.calls), 1)

    def test_hostile_doctor_booking_context_uses_llm_and_escalates(self):
        llm = FakeLlm(
            '{"category":"harmful_intent","language":"english","reason":"hostile intent toward doctor"}',
        )
        result = SentimentMonitor(llm).evaluate(
            "I wanna book an appoinment to make my doctor take his own medicine",
        )

        self.assertEqual(result.category, SentimentCategory.HARMFUL_INTENT)
        self.assertEqual(result.source, DetectionSource.LLM)
        self.assertEqual(result.language, "english")
        self.assertEqual(len(llm.calls), 1)

    def test_self_harm_still_escalates_with_booking_context(self):
        llm = FakeLlm()
        result = SentimentMonitor(llm).evaluate(
            "I want to kill myself and need an appointment",
        )

        self.assertEqual(result.category, SentimentCategory.SELF_HARM)
        self.assertEqual(result.source, DetectionSource.RULES)
        self.assertEqual(llm.calls, [])

    def test_od_on_depression_medication_escalates(self):
        llm = FakeLlm()
        result = SentimentMonitor(llm).evaluate(
            "I wanna OD on my depression medication",
        )

        self.assertEqual(result.category, SentimentCategory.SELF_HARM)
        self.assertEqual(result.source, DetectionSource.RULES)
        self.assertEqual(llm.calls, [])

    def test_overdose_on_depression_medication_escalates(self):
        llm = FakeLlm()
        result = SentimentMonitor(llm).evaluate(
            "I wanna Overdose in my depression medication",
        )

        self.assertEqual(result.category, SentimentCategory.SELF_HARM)
        self.assertEqual(result.source, DetectionSource.RULES)
        self.assertEqual(llm.calls, [])

    def test_llm_failure_falls_back_to_safe_when_no_rule_matches(self):
        llm = FakeLlm(error=RuntimeError("model unavailable"))
        result = SentimentMonitor(llm).evaluate("The weather is nice today")

        self.assertEqual(result.category, SentimentCategory.SAFE)
        self.assertEqual(result.source, DetectionSource.LLM_ERROR)
        self.assertEqual(result.language, "english")

    def test_language_detection_covers_required_languages(self):
        self.assertEqual(detect_language("What are your hours?"), "english")
        self.assertEqual(detect_language("Saya nak buat temu janji"), "malay")
        self.assertEqual(detect_language("你们的营业时间是什么时候？"), "mandarin")


if __name__ == "__main__":
    unittest.main()
