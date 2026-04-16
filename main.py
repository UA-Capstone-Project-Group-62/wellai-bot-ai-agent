import os
from concurrent import futures

import grpc
from dotenv import load_dotenv
from grpc_reflection.v1alpha import reflection
from proto.agent import agent_pb2, agent_pb2_grpc
from proto.bot import bot_pb2, bot_pb2_grpc
from proto.common import common_pb2

# Initialize environment variables
load_dotenv()
PORT = os.getenv("PORT", "50051")
BOT_SERVICE_ADDR = os.getenv("BOT_SERVICE_ADDR", "localhost:50052")


class AgentService(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self):
        self.bot_channel = grpc.insecure_channel(BOT_SERVICE_ADDR)
        self.bot_client = bot_pb2_grpc.BotServiceStub(self.bot_channel)

    def Receive(self, request, context):
        print(f"Received message from {request.user_id}: {request.content}")
        try:
            bot_response = self.bot_client.Send(
                bot_pb2.Message(user_id=request.user_id, content=request.content)
            )
        except grpc.RpcError as error:
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


def main():
    # Initialize gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentService(), server)

    # Enable reflection for debugging and testing (Postman)
    SERVICE_NAMES = (
        agent_pb2.DESCRIPTOR.services_by_name["AgentService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start the server
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"gRPC server listening on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
