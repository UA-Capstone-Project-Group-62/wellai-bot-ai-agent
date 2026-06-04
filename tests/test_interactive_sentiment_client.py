import unittest

from mocks.interactive_sentiment_test import is_escalation_reply


class InteractiveSentimentClientTests(unittest.TestCase):
    def test_detects_english_escalation_reply(self):
        self.assertTrue(
            is_escalation_reply(
                "I recognise that you may be distressed. I have requested a human operator to take over this conversation and support you.",
            ),
        )

    def test_detects_malay_escalation_reply(self):
        self.assertTrue(
            is_escalation_reply(
                "Saya telah meminta operator manusia untuk mengambil alih perbualan ini.",
            ),
        )

    def test_detects_mandarin_escalation_reply(self):
        self.assertTrue(
            is_escalation_reply("我已经请求人工客服接手这段对话并为你提供帮助。"),
        )

    def test_does_not_detect_normal_reply(self):
        self.assertFalse(is_escalation_reply("Sure, I can help you book an appointment."))


if __name__ == "__main__":
    unittest.main()
