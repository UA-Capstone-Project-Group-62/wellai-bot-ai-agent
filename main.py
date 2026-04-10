from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
from proto.agent import agent_pb2, agent_pb2_grpc
from proto.common import common_pb2


class AgentService(agent_pb2_grpc.AgentServiceServicer):
    def Receive(self, request, context):
        print(f"Received message from {request.user_id}: {request.content}")
        return common_pb2.Response(success=True, message="Agent received message")


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
    server.add_insecure_port("[::]:50051")
    server.start()
    print("gRPC server listening on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
