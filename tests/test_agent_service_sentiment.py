from types import SimpleNamespace
from unittest.mock import patch
import unittest

from proto.common import common_pb2

from src.services.agent_service import AgentService
from src.services.sentiment_monitor import (
    DetectionSource,
    SentimentCategory,
    SentimentResult,
)


class FakeBotClient:
    target_addr = "localhost:50052"

    def __init__(self):
        self.sent_messages = []
        self.history_requested = False

    def send(self, user_id: str, content: str):
        self.sent_messages.append((user_id, content))
        return common_pb2.Response(success=True, message="")

    def get_messages(self, user_id: str, count: int = 50):
        self.history_requested = True
        return SimpleNamespace(messages=[])


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class AgentServiceSentimentTests(unittest.TestCase):
    def test_flagged_message_sends_escalation_and_skips_intent_graph(self):
        bot_client = FakeBotClient()
        service = AgentService(bot_client)
        request = SimpleNamespace(user_id="user-1", content="I want to die")

        with (
            patch("src.services.agent_service.sentiment_monitor") as monitor,
            patch("src.services.agent_service.language_monitor"),
            patch("src.services.agent_service.intent_graph") as graph,
        ):
            monitor.evaluate.return_value = SentimentResult(
                category=SentimentCategory.SELF_HARM,
                source=DetectionSource.RULES,
                reason="direct self-harm phrase",
                language="english",
            )
            monitor.escalation_reply.return_value = "Human operator requested."

            response = service.Receive(request, FakeContext())

        self.assertTrue(response.success)
        self.assertEqual(bot_client.sent_messages, [("user-1", "Human operator requested.")])
        self.assertFalse(bot_client.history_requested)
        graph.invoke.assert_not_called()

    def test_escalated_conversation_stays_locked_on_follow_up(self):
        bot_client = FakeBotClient()
        service = AgentService(bot_client)

        with (
            patch("src.services.agent_service.sentiment_monitor") as monitor,
            patch("src.services.agent_service.language_monitor"),
            patch("src.services.agent_service.intent_graph") as graph,
        ):
            monitor.evaluate.return_value = SentimentResult(
                category=SentimentCategory.DISTRESS,
                source=DetectionSource.RULES,
                reason="severe distress phrase",
                language="english",
            )
            monitor.escalation_reply.return_value = "Human operator requested."

            first_response = service.Receive(
                SimpleNamespace(user_id="user-locked", content="I feel hopeless"),
                FakeContext(),
            )

            monitor.evaluate.return_value = SentimentResult(
                category=SentimentCategory.SAFE,
                source=DetectionSource.LLM,
                reason="safe follow up",
                language="english",
            )
            second_response = service.Receive(
                SimpleNamespace(user_id="user-locked", content="Can I book now?"),
                FakeContext(),
            )

        self.assertTrue(first_response.success)
        self.assertTrue(second_response.success)
        self.assertEqual(
            bot_client.sent_messages,
            [
                ("user-locked", "Human operator requested."),
                ("user-locked", "Human operator requested."),
            ],
        )
        self.assertFalse(bot_client.history_requested)
        graph.invoke.assert_not_called()
        self.assertEqual(monitor.evaluate.call_count, 1)

    def test_safe_message_fetches_history_and_runs_intent_graph(self):
        bot_client = FakeBotClient()
        service = AgentService(bot_client)
        request = SimpleNamespace(user_id="user-2", content="I want to book")

        with (
            patch("src.services.agent_service.sentiment_monitor") as monitor,
            patch("src.services.agent_service.language_monitor") as language_monitor,
            patch("src.services.agent_service.intent_graph") as graph,
        ):
            monitor.evaluate.return_value = SentimentResult(
                category=SentimentCategory.SAFE,
                source=DetectionSource.LLM,
                reason="safe booking request",
                language="english",
            )
            language_monitor.evaluate.return_value.is_supported = True
            graph.invoke.return_value = {
                "messages": [SimpleNamespace(content="Sure, I can help you book.")],
            }

            response = service.Receive(request, FakeContext())

        self.assertTrue(response.success)
        self.assertTrue(bot_client.history_requested)
        self.assertEqual(bot_client.sent_messages, [("user-2", "Sure, I can help you book.")])
        graph.invoke.assert_called_once()

    def test_unsupported_language_sends_capability_notice_and_skips_intent_graph(self):
        bot_client = FakeBotClient()
        service = AgentService(bot_client)
        request = SimpleNamespace(user_id="user-3", content="Bonjour")

        with (
            patch("src.services.agent_service.sentiment_monitor") as monitor,
            patch("src.services.agent_service.language_monitor") as language_monitor,
            patch("src.services.agent_service.intent_graph") as graph,
        ):
            monitor.evaluate.return_value = SentimentResult(
                category=SentimentCategory.SAFE,
                source=DetectionSource.LLM,
                reason="safe",
                language="english",
            )
            language_monitor.evaluate.return_value.is_supported = False
            language_monitor.evaluate.return_value.source.value = "llm"
            language_monitor.evaluate.return_value.reason = "French detected"
            language_monitor.unsupported_reply.return_value = "Optimised language notice."

            response = service.Receive(request, FakeContext())

        self.assertTrue(response.success)
        self.assertEqual(bot_client.sent_messages, [("user-3", "Optimised language notice.")])
        self.assertFalse(bot_client.history_requested)
        graph.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
