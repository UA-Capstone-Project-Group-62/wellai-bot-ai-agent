import grpc
from concurrent import futures
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

proto_path = "/app/proto"
sys.path.insert(0, proto_path)

from proto.agent import agent_pb2
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

class AgentServiceServicer(agent_pb2_grpc.AgentServiceServicer):
    def Receive(self, request_iterator, context):
        messages = []
        for message in request_iterator:
            messages.append(message)
            logger.info(f"Received message from user {message.user_id}: {message.content}")

        logger.info(f"Total messages received: {len(messages)}")

        return common_pb2.Response(
            success=True,
            message=f"Processed {len(messages)} messages"
        )

    def ReceiveAndRespond(self, request_iterator, context):
        def generate_responses():
            message_count = 0
            for message in request_iterator:
                message_count += 1
                logger.info(f"Received message {message_count} from user {message.user_id}: {message.content}")
                yield common_pb2.Response(
                    success=True,
                    message=f"Echo {message_count}: {message.content}"
                )
            logger.info(f"Total messages processed: {message_count}")
            yield common_pb2.Response(
                success=True,
                message=f"Completed processing {message_count} messages"
            )

        return generate_responses()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(
        AgentServiceServicer(), server
    )
    server.add_insecure_port('[::]:50053')
    logger.info("Mock AgentService running on port 50053")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()