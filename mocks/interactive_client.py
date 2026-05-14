import grpc
import sys
import os

proto_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "proto", "gen", "py")
sys.path.insert(0, proto_path)

from proto.agent import agent_pb2, agent_pb2_grpc


def chat(addr: str, user_id: str):
    channel = grpc.insecure_channel(addr)
    stub = agent_pb2_grpc.AgentServiceStub(channel)

    print(f"Connected to {addr}")
    print("Type your message and press Enter. Press Ctrl+C or type 'quit' to exit.\n")

    try:
        while True:
            text = input("> ").strip()
            if not text or text.lower() in ("quit", "exit"):
                break

            msg = agent_pb2.Message(user_id=user_id, content=text)
            response = stub.Receive(msg)
            print(f"Bot: {response.message}\n")
    except KeyboardInterrupt:
        pass
    finally:
        channel.close()
        print("Goodbye!")


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
    user_id = sys.argv[2] if len(sys.argv) > 2 else "user123"
    chat(addr, user_id)
