import grpc
import os
from loguru import logger
from groq import Groq
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful, polite, and professional WhatsApp booking assistant for WellAI clinics in Malaysia.
You support English, Malay, and Mandarin.
Always be respectful, clear, and neutral. Never give medical advice.
Help patients book, reschedule, cancel appointments, or answer FAQs."""

class AgentService(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self, bot_client: BotClient):
        self.bot_client = bot_client

    def Receive(self, request, context):
        logger.info(
            "Received message from user. user_id={}, content_length={}",
            request.user_id,
            len(request.content),
        )

        user_message = request.content.lower()

        # Simple intent recognition (what teammate wanted)
        if any(word in user_message for word in ["book", "appointment", "jadual", "temu janji", "nak jumpa", "buat tempahan"]):
            intent = "booking"
        elif any(word in user_message for word in ["cancel", "batal", "cancelkan"]):
            intent = "cancel"
        elif any(word in user_message for word in ["reschedule", "ubah", "pindah"]):
            intent = "reschedule"
        elif any(word in user_message for word in ["faq", "soalan", "tanya", "question", "jam", "bila", "buka"]):
            intent = "faq"
        else:
            intent = "unknown"

        logger.info(f"Detected intent: {intent}")

        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + f"\nUser intent detected: {intent}"},
                    {"role": "user", "content": request.content}
                ],
                temperature=0.7,
                max_tokens=300
            )
            ai_reply = completion.choices[0].message.content.strip()
            logger.info("AI replied successfully")
        except Exception as e:
            logger.error(f"Groq error: {e}")
            ai_reply = "Sorry, I'm having trouble right now. Please try again."

        return common_pb2.Response(
            success=True,
            message=ai_reply
        )