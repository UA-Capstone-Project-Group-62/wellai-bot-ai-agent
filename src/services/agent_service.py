import grpc
from loguru import logger
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient

from src.services.intent_recognition import IntentRecognitionService


class AgentService(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self, bot_client: BotClient):
        self.bot_client = bot_client
        self.intent_service = IntentRecognitionService()

    def Receive(self, request, context):
        logger.info(
            "Received message from user. user_id={}, content_length={}",
            request.user_id,
            len(request.content),
        )

        try:
            recent_messages_response = self.bot_client.get_messages(
                user_id=request.user_id,
                count=10
            )
            recent_messages = list(recent_messages_response.messages)

            intent = self.intent_service.recognize_intent(
                current_message=request.content,
                recent_messages=recent_messages
            )
            logger.info(f"User {request.user_id} - Intent: {intent}")

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