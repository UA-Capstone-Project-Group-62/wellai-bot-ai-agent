from loguru import logger
from langchain_core.messages import HumanMessage
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient
from src.services.intent_graph import graph as intent_graph


class AgentService(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self, bot_client: BotClient):
        self.bot_client = bot_client

    def _fetch_history(self, user_id: str) -> str:
        """Fetch conversation history from the bot service via GetMessages."""
        try:
            response = self.bot_client.get_messages(user_id, count=50)
            if not response.messages:
                return ""
            lines = []
            for msg in response.messages:
                lines.append(f"- {msg.content}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("Failed to fetch history from bot service: {}", e)
            return ""

    def Receive(self, request, context):
        user_id = request.user_id
        content = request.content

        logger.info(
            "Received message from user. user_id={}, content_length={}",
            user_id,
            len(content),
        )

        history_text = self._fetch_history(user_id)

        try:
            result = intent_graph.invoke({
                "messages": [HumanMessage(content=content)],
                "history": history_text,
            })
            ai_reply = result["messages"][-1].content.strip()
            logger.info("AI replied successfully")
        except Exception as e:
            logger.error("LangGraph error: {}", e)
            ai_reply = "Sorry, I'm having trouble right now. Please try again."

        # Forward the AI reply back to the user via the BotService
        try:
            send_response = self.bot_client.send(user_id, ai_reply)
            if not send_response.success:
                logger.warning(
                    "BotService.Send returned failure. user_id={}, error={}",
                    user_id,
                    send_response.message,
                )
        except Exception as e:
            logger.error("Failed to send reply via BotService: {}", e)

        return common_pb2.Response(success=True, message=ai_reply)
