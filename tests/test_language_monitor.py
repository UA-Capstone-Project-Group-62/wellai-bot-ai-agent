from types import SimpleNamespace
import unittest

from src.services.language_monitor import (
    LanguageCategory,
    LanguageDetectionSource,
    LanguageMonitor,
)


class FakeLlm:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content or '{"language":"english","reason":"English detected"}'
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


class LanguageMonitorTests(unittest.TestCase):
    def test_rules_detect_supported_english_without_llm(self):
        llm = FakeLlm()
        result = LanguageMonitor(llm).evaluate("Hello, I want to book")

        self.assertEqual(result.category, LanguageCategory.ENGLISH)
        self.assertEqual(result.source, LanguageDetectionSource.RULES)
        self.assertEqual(llm.calls, [])

    def test_rules_detect_supported_malay_without_llm(self):
        llm = FakeLlm()
        result = LanguageMonitor(llm).evaluate("Saya nak buat temu janji")

        self.assertEqual(result.category, LanguageCategory.MALAY)
        self.assertEqual(result.source, LanguageDetectionSource.RULES)
        self.assertEqual(llm.calls, [])

    def test_rules_detect_supported_mandarin_without_llm(self):
        llm = FakeLlm()
        result = LanguageMonitor(llm).evaluate("我想预约")

        self.assertEqual(result.category, LanguageCategory.MANDARIN)
        self.assertEqual(result.source, LanguageDetectionSource.RULES)
        self.assertEqual(llm.calls, [])

    def test_llm_detects_unsupported_language(self):
        llm = FakeLlm('{"language":"unsupported","reason":"French detected"}')
        result = LanguageMonitor(llm).evaluate("Bonjour")

        self.assertEqual(result.category, LanguageCategory.UNSUPPORTED)
        self.assertEqual(result.source, LanguageDetectionSource.LLM)
        self.assertFalse(result.is_supported)
        self.assertEqual(len(llm.calls), 1)

    def test_unsupported_reply_mentions_optimised_languages(self):
        reply = LanguageMonitor(FakeLlm()).unsupported_reply()

        self.assertIn("current language", reply)
        self.assertIn("English", reply)
        self.assertIn("Bahasa Melayu", reply)
        self.assertIn("Mandarin Chinese", reply)


if __name__ == "__main__":
    unittest.main()
