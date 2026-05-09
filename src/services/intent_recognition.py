from openai import OpenAI
from loguru import logger

from src.env import env
from src.services.faq_knowledge_base import FAQ_KNOWLEDGE_BASE

VALID_INTENTS = ["booking", "cancel", "reschedule", "faq", "unknown"]


class IntentRecognitionService:
    def __init__(self):
        self._client = OpenAI(
            api_key=env["OPENAI_API_KEY"],
            base_url=env["OPENAI_BASE_URL"]
        )

    def _build_faq_context(self) -> str:
        faq_lines = []
        for key, value in FAQ_KNOWLEDGE_BASE.items():
            faq_lines.append(f"- {key}: {value}")
        return "\n".join(faq_lines)

    def _build_message_context(self, messages: list) -> str:
        if not messages:
            return "No previous messages."
        return "\n".join(
            f"User: {msg.content}" for msg in messages
        )

    def recognize_intent(self, current_message: str, recent_messages: list) -> str:
        faq_context = self._build_faq_context()
        message_context = self._build_message_context(recent_messages)

        prompt = f"""You are an intent classifier for a medical clinic booking chatbot.

Available intents:
- booking: User wants to book, reschedule, or schedule an appointment
- cancel: User wants to cancel an existing appointment
- reschedule: User wants to change the time/date of an existing appointment
- faq: User is asking a general question (opening hours, location, payment, documents, etc.)
- unknown: User intent is unclear or doesn't fit the above categories

FAQ Knowledge Base:
{faq_context}

Recent conversation (last 10 messages):
{message_context}

Current user message: {current_message}

Classify the intent of the current user message. Return ONLY the intent word (booking, cancel, reschedule, faq, or unknown)."""

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful intent classifier. Return only one word."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0
            )
            intent = response.choices[0].message.content.strip().lower()
            if intent not in VALID_INTENTS:
                logger.warning(f"Invalid intent from LLM: {intent}, defaulting to unknown")
                return "unknown"
            logger.info(f"Recognized intent: {intent}")
            return intent
        except Exception as e:
            logger.error(f"Error recognizing intent: {e}")
            return "unknown"