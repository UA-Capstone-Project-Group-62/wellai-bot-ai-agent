import grpc
from loguru import logger
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient

# Step 1: Import from our new FAQ file
from src.services.faq_knowledge_base import get_system_prompt_with_faq


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

        # Intent recognition
        if any(word in user_message for word in ["book", "appointment", "jadual", "temu janji", "nak jumpa", "buat tempahan"]):
            intent = "booking"
        elif any(word in user_message for word in ["cancel", "batal", "cancelkan"]):
            intent = "cancel"
        elif any(word in user_message for word in ["reschedule", "ubah", "pindah"]):
            intent = "reschedule"
        elif any(word in user_message for word in ["faq", "soalan", "tanya", "question", "jam", "bila", "buka", "hour", "location", "where"]):
            intent = "faq"
        else:
            intent = "unknown"

        logger.info(f"Detected intent: {intent}")

        try:
            if intent == "faq":
                faq_answer = get_faq_answer(request.content)
                if faq_answer:
                    ai_reply = faq_answer
                else:
                    # fallback to LLM
                    result = graph.invoke({"messages": [HumanMessage(content=request.content)]})
                    ai_reply = result["messages"][-1].content.strip()
            else:
                # normal LLM flow
                result = graph.invoke({"messages": [HumanMessage(content=request.content)]})
                ai_reply = result["messages"][-1].content.strip()

            logger.info("AI replied successfully")
        except Exception as e:
            logger.error(f"Error: {e}")
            ai_reply = "Sorry, I'm having trouble right now. Please try again."

        return common_pb2.Response(
            success=True,
            message=ai_reply
        )