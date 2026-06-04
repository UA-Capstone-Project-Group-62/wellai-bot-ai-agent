from unittest.mock import patch
import unittest

from langchain_core.messages import HumanMessage

from src.services.intent_graph import (
    LANGUAGE_SUPPORT_RESPONSE,
    intent_classifier,
    language_support_node,
    question_node,
)


class LanguageSupportResponseTests(unittest.TestCase):
    def test_language_question_routes_to_language_support_without_llm_classifier(self):
        state = {
            "messages": [HumanMessage(content="what languages do you know?")],
            "history": "",
            "intent": "unrelated_to_your_job",
        }

        with patch("src.services.intent_graph._invoke_with_retry") as invoke:
            result = intent_classifier(state)

        self.assertEqual(result, {"intent": "language_support"})
        invoke.assert_not_called()

    def test_language_support_node_returns_fixed_supported_languages(self):
        state = {
            "messages": [HumanMessage(content="how multilingual is this assistant?")],
            "history": "",
            "intent": "language_support",
        }

        with patch("src.services.intent_graph._invoke_with_retry") as invoke:
            result = language_support_node(state)

        self.assertEqual(result["messages"][-1].content, LANGUAGE_SUPPORT_RESPONSE)
        self.assertIn("English", result["messages"][-1].content)
        self.assertIn("Bahasa Melayu", result["messages"][-1].content)
        self.assertIn("Mandarin Chinese", result["messages"][-1].content)
        self.assertNotIn("Spanish", result["messages"][-1].content)
        invoke.assert_not_called()

    def test_language_follow_up_returns_fixed_supported_languages(self):
        state = {
            "messages": [HumanMessage(content="what are the other languages")],
            "history": "",
            "intent": "ask_question",
        }

        with patch("src.services.intent_graph._invoke_with_retry") as invoke:
            result = question_node(state)

        self.assertEqual(result["messages"][-1].content, LANGUAGE_SUPPORT_RESPONSE)
        invoke.assert_not_called()

    def test_language_question_with_typo_routes_to_language_support_without_llm_classifier(self):
        state = {
            "messages": [HumanMessage(content="what laguages do you kn ow")],
            "history": "",
            "intent": "unrelated_to_your_job",
        }

        with patch("src.services.intent_graph._invoke_with_retry") as invoke:
            result = intent_classifier(state)

        self.assertEqual(result, {"intent": "language_support"})
        invoke.assert_not_called()

    def test_llm_can_classify_language_support_intent_for_non_hardcoded_wording(self):
        state = {
            "messages": [HumanMessage(content="how multilingual is this assistant?")],
            "history": "",
            "intent": "unrelated_to_your_job",
        }

        with patch("src.services.intent_graph._invoke_with_retry") as invoke:
            invoke.return_value.content = "language_support"
            result = intent_classifier(state)

        self.assertEqual(result, {"intent": "language_support"})
        invoke.assert_called_once()

    def test_speak_question_returns_fixed_supported_languages(self):
        state = {
            "messages": [HumanMessage(content="what do you know how to speak?")],
            "history": "",
            "intent": "ask_question",
        }

        with patch("src.services.intent_graph._invoke_with_retry") as invoke:
            result = question_node(state)

        self.assertEqual(result["messages"][-1].content, LANGUAGE_SUPPORT_RESPONSE)
        self.assertNotIn("Spanish", result["messages"][-1].content)
        invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
