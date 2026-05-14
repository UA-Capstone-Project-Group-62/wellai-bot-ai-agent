import grpc
from concurrent import futures
import logging
import sys
import os

# Ensure project root is on path so `import src` and `import proto` work
# regardless of where this script is launched from.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from proto.agent import agent_pb2_grpc

from src.services.agent_service import AgentService as RealAgentService
from src.clients.bot_client import BotClient


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # BotClient is required by AgentService.__init__. We pass a dummy address
    # here; BotService.Send / GetMessages will fail and log warnings, but
    # the AI logic (intent classification + reply generation) still runs.
    bot_client = BotClient("localhost:50052")
    agent_service = RealAgentService(bot_client)

    agent_pb2_grpc.add_AgentServiceServicer_to_server(agent_service, server)

    server.add_insecure_port("[::]:50053")
    logger.info("AI Agent Mock Server running on port 50053")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
