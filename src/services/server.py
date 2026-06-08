from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
from loguru import logger
from proto.agent import agent_pb2, agent_pb2_grpc

from src.clients.bot_client import BotClient
from src.clients.scheduling_client import SchedulingClient
from src.env import env
from src.services.agent_service import AgentService


def run_server() -> None:
    """Start and run the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    bot_client = BotClient(env["BOT_SERVICE_ADDR"])
    scheduling_client = SchedulingClient(env["SCHEDULING_SERVICE_ADDR"])
    
    agent_pb2_grpc.add_AgentServiceServicer_to_server(
        AgentService(bot_client, scheduling_client),
        server,
    )

    service_names = (
        agent_pb2.DESCRIPTOR.services_by_name["AgentService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    bind_addr = f"[::]:{env['PORT']}"
    server.add_insecure_port(bind_addr)
    server.start()
    logger.info("gRPC server listening on {}", bind_addr)

    try:
        server.wait_for_termination()
    finally:
        bot_client.close()
        scheduling_client.close()
