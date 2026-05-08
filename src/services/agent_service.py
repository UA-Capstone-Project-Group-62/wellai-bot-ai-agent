import grpc
from loguru import logger
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient

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

        try:

            intent_prompt = f"""
Classify the user intent in one word. Choose only from: booking, cancel, reschedule, faq, unknown.

User message: {request.content}
"""

       
            bot_response = self.bot_client.send(
                user_id=request.user_id,
                content=request.content,
            )
            ai_reply = bot_response.message

    

            logger.info("AI replied successfully")
        except Exception as e:
            logger.error(f"Error: {e}")
            ai_reply = "Sorry, I'm having trouble right now. Please try again."

        return common_pb2.Response(
            success=True,
            message=ai_reply
        )