import grpc
from loguru import logger
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient


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
            bot_response = self.bot_client.send(
                user_id=request.user_id,
                content=request.content,
            )
        except grpc.RpcError as error:
            logger.error(
                "Failed to send message to bot service. destination={}, code={}, details={}",
                self.bot_client.target_addr,
                error.code(),
                error.details(),
            )
            context.set_code(error.code())
            context.set_details(
                f"Failed to send message to bot service: {error.details()}"
            )
            return common_pb2.Response(
                success=False, message="Failed to forward message to bot service"
            )

        return common_pb2.Response(
            success=bot_response.success,
            message=bot_response.message,
        )
