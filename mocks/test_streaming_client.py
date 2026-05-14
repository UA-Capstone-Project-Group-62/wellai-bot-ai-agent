import grpc
import sys
import os

proto_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "proto", "gen", "py")
sys.path.insert(0, proto_path)

from proto.agent import agent_pb2, agent_pb2_grpc

def generate_messages():
    messages = [
        agent_pb2.Message(user_id="user123", content="Hello, I want to book an appointment"),
        agent_pb2.Message(user_id="user123", content="I prefer morning time slots"),
        agent_pb2.Message(user_id="user123", content="For next week if possible"),
    ]
    for msg in messages:
        yield msg

def call_unary_streaming_endpoint(addr: str):
    channel = grpc.insecure_channel(addr)
    stub = agent_pb2_grpc.AgentServiceStub(channel)

    print("=== Testing Receive (client streaming -> unary response) ===")
    print("Sending streaming messages to AgentService...")
    response = stub.Receive(generate_messages())

    print(f"Response: success={response.success}, message={response.message}")
    channel.close()

def call_bidirectional_streaming(addr: str):
    channel = grpc.insecure_channel(addr)
    stub = agent_pb2_grpc.AgentServiceStub(channel)

    print("\n=== Testing ReceiveAndRespond (bidirectional streaming) ===")
    print("Sending streaming messages and receiving streaming responses...")

    responses = stub.ReceiveAndRespond(generate_messages())
    for response in responses:
        print(f"Response received: success={response.success}, message={response.message}")

    channel.close()

if __name__ == '__main__':
    addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50053"
    call_unary_streaming_endpoint(addr)
    call_bidirectional_streaming(addr)