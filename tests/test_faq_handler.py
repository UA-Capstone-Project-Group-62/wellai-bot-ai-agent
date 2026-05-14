import pytest
from src.services.faq_handler import FAQHandler


class TestFAQHandler:
    @pytest.fixture
    def handler(self):
        return FAQHandler()

    def test_get_answer_report_info(self, handler):
        answer = handler.get_answer("How can I know more about my report?")
        assert answer is not None
        assert "online consultation" in answer.lower()
        assert "health report" in answer.lower()

    def test_get_answer_appointment(self, handler):
        answer = handler.get_answer("I want to make an appointment")
        assert answer is not None
        assert "online appointment" in answer.lower()

    def test_get_answer_cost(self, handler):
        answer = handler.get_answer("how much does the consultation cost?")
        assert answer is not None
        assert "free" in answer.lower()

    def test_get_answer_free(self, handler):
        answer = handler.get_answer("is it free?")
        assert answer is not None
        assert "free" in answer.lower()

    def test_get_answer_hours(self, handler):
        answer = handler.get_answer("what are the consultation hours?")
        assert answer is not None
        assert "appointment" in answer.lower()

    def test_get_answer_location(self, handler):
        answer = handler.get_answer("where is the consultation?")
        assert answer is not None
        assert "virtually" in answer.lower()

    def test_get_answer_platform(self, handler):
        answer = handler.get_answer("what platform do you use?")
        assert answer is not None
        assert "google meet" in answer.lower() or "zoom" in answer.lower()

    def test_get_answer_duration(self, handler):
        answer = handler.get_answer("how long is the session?")
        assert answer is not None
        assert "one hour" in answer.lower()

    def test_get_answer_reschedule(self, handler):
        answer = handler.get_answer("can I reschedule my appointment?")
        assert answer is not None

    def test_get_answer_cancel(self, handler):
        answer = handler.get_answer("I want to cancel my appointment")
        assert answer is not None

    def test_get_answer_data_security(self, handler):
        answer = handler.get_answer("is my data secure?")
        assert answer is not None
        assert "secure" in answer.lower()

    def test_get_answer_no_match(self, handler):
        answer = handler.get_answer("what is the meaning of life?")
        assert answer is None

    def test_format_response(self, handler):
        answer = "Test answer"
        response = handler.format_response(answer)
        assert "Test answer" in response
        assert "anything else" in response.lower()

    def test_all_faq_keys_covered(self, handler):
        from src.services.faq_knowledge_base import FAQ_KNOWLEDGE_BASE
        assert len(FAQ_KNOWLEDGE_BASE) == 10