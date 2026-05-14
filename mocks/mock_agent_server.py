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

from proto.agent import agent_pb2
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.services.agent_service import AgentService as RealAgentService
from src.clients.bot_client import BotClient


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # BotClient is required by AgentService.__init__ but is never actually
    # used inside Receive / ReceiveAndRespond, so a dummy address is fine.
    bot_client = BotClient("localhost:50052")
    agent_service = RealAgentService(bot_client)

    # NOTE: We register via generic RPC handlers instead of
    # add_AgentServiceServicer_to_server because grpcio 1.80.0 has a bug with
    # registered methods for stream_unary handlers (it passes a single Message
    # object instead of an iterator).
    rpc_method_handlers = {
        "Receive": grpc.stream_unary_rpc_method_handler(
            agent_service.Receive,
            request_deserializer=agent_pb2.Message.FromString,
            response_serializer=common_pb2.Response.SerializeToString,
        ),
        "ReceiveAndRespond": grpc.stream_stream_rpc_method_handler(
            agent_service.ReceiveAndRespond,
            request_deserializer=agent_pb2.Message.FromString,
            response_serializer=common_pb2.Response.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "wellai_bot.agent.AgentService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))

    server.add_insecure_port("[::]:50053")
    logger.info("AI Agent Mock Server running on port 50053")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
