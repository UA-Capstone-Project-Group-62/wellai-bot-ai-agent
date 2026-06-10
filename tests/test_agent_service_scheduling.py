import unittest
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace

from src.services.agent_service import AgentService

# run as PYTHONPATH=proto/gen/py:. python -m unittest tests/test_agent_service_scheduling.py

class TestAgentServiceScheduling(unittest.TestCase):

    def setUp(self):
        self.bot_client = Mock()
        self.scheduling_client = Mock()

        # BotService send/get_messages stubs
        self.bot_client.send.return_value = SimpleNamespace(success=True)
        self.bot_client.get_messages.return_value = SimpleNamespace(messages=[])

        self.service = AgentService(
            bot_client=self.bot_client,
            scheduling_client=self.scheduling_client,
        )

        # Mock grpc context
        self.context = Mock()

    def _build_request(self, text="hello", user_id="user_1"):
        return SimpleNamespace(user_id=user_id, content=text)

    @patch("src.services.agent_service.intent_graph.invoke")
    @patch("src.services.agent_service.sentiment_monitor.evaluate")
    @patch("src.services.agent_service.language_monitor.evaluate")

    def test_book_appointment_success(
        self,
        mock_lang,
        mock_sentiment,
        mock_intent,
    ):
        mock_sentiment.return_value.should_escalate = False
        mock_lang.return_value.is_supported = True

        mock_intent.return_value = {
            "intent": "book_app",
            "messages": [SimpleNamespace(content="AI reply base")],
        }

        # Patch extraction + scheduling flow
        with patch.object(
            self.service,
            "_extract_booking_details",
            return_value={
                "preferred_date": "2026-06-10",
                "preferred_time": "14:00",
                "clinic_id": "clinic_001",
                "user_name": "John",
            },
        ):
            self.scheduling_client.schedule.return_value = SimpleNamespace(
                success=True,
                message="Booked successfully",
            )

            request = self._build_request("I want to book an appointment")

            self.service.Receive(request, self.context)

            self.scheduling_client.schedule.assert_called_once()

            # Ensure reply was augmented
            sent_reply = self.bot_client.send.call_args[0][1]
            self.assertIn("Appointment confirmed", sent_reply)

    @patch("src.services.agent_service.intent_graph.invoke")
    @patch("src.services.agent_service.sentiment_monitor.evaluate")
    @patch("src.services.agent_service.language_monitor.evaluate")

    def test_cancel_appointment_success(
        self,
        mock_lang,
        mock_sentiment,
        mock_intent,
    ):
        mock_sentiment.return_value.should_escalate = False
        mock_lang.return_value.is_supported = True

        mock_intent.return_value = {
            "intent": "cancel_app",
            "messages": [SimpleNamespace(content="AI reply base")],
        }

        self.scheduling_client.cancel.return_value = SimpleNamespace(
            success=True,
            message="Cancelled successfully",
        )

        request = self._build_request("Cancel my appointment")

        self.service.Receive(request, self.context)

        self.scheduling_client.cancel.assert_called_once_with("user_1")

        sent_reply = self.bot_client.send.call_args[0][1]
        self.assertIn("Appointment cancelled", sent_reply)

    @patch("src.services.agent_service.intent_graph.invoke")
    @patch("src.services.agent_service.sentiment_monitor.evaluate")
    @patch("src.services.agent_service.language_monitor.evaluate")

    def test_reschedule_success(
        self,
        mock_lang,
        mock_sentiment,
        mock_intent,
    ):
        mock_sentiment.return_value.should_escalate = False
        mock_lang.return_value.is_supported = True

        mock_intent.return_value = {
            "intent": "reschedule_app",
            "messages": [SimpleNamespace(content="AI reply base")],
        }

        with patch.object(
            self.service,
            "_extract_booking_details",
            return_value={
                "preferred_date": "2026-06-11",
                "preferred_time": "10:00",
                "clinic_id": "clinic_001",
                "user_name": "John",
            },
        ):
            self.scheduling_client.cancel.return_value = SimpleNamespace(
                success=True,
                message="Cancelled old appointment",
            )

            self.scheduling_client.schedule.return_value = SimpleNamespace(
                success=True,
                message="Rescheduled successfully",
            )

            request = self._build_request("Reschedule my appointment")

            self.service.Receive(request, self.context)

            self.scheduling_client.cancel.assert_called_once()
            self.scheduling_client.schedule.assert_called_once()

            sent_reply = self.bot_client.send.call_args[0][1]
            self.assertIn("rescheduled", sent_reply.lower())

if __name__ == "__main__":
    unittest.main()