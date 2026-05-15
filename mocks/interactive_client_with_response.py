import grpc
import sys
import os

proto_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "proto", "gen", "py")
if not os.path.isdir(proto_path):
    raise SystemExit(
        "Missing generated protobuf Python stubs. Expected directory: "
        f"{proto_path}. Generate the files under 'proto/gen/py' before "
        "running this client."
    )
sys.path.insert(0, proto_path)

from proto.agent import agent_pb2, agent_pb2_grpc
from proto.bot import bot_pb2, bot_pb2_grpc


def chat(addr: str, bot_addr: str, user_id: str):
    agent_channel = grpc.insecure_channel(addr)
    agent_stub = agent_pb2_grpc.AgentServiceStub(agent_channel)

    bot_channel = grpc.insecure_channel(bot_addr)
    bot_stub = bot_pb2_grpc.BotServiceStub(bot_channel)

    print(f"Connected to AI Agent at {addr}")
    print(f"Connected to BotService at {bot_addr}")
    print("Type your message and press Enter. Press Ctrl+C or type 'quit' to exit.\n")

    try:
        while True:
            text = input("> ").strip()
            if not text or text.lower() in ("quit", "exit"):
                break

            msg = agent_pb2.Message(user_id=user_id, content=text)
            response = agent_stub.Receive(msg)

            if response.success:
                msgs = bot_stub.GetMessages(bot_pb2.GetMessagesRequest(user_id=user_id, count=1))
                if msgs.messages:
                    print(f"\n🤖 AI: {msgs.messages[-1].content}\n")
                else:
                    print("(Message forwarded to BotService for delivery - no response yet)\n")
            else:
                print(f"Error: {response.message}\n")
    except KeyboardInterrupt:
        pass
    finally:
        agent_channel.close()
        bot_channel.close()
        print("Goodbye!")


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
    bot_addr = sys.argv[2] if len(sys.argv) > 2 else "localhost:50052"
    user_id = sys.argv[3] if len(sys.argv) > 3 else "user123"
    chat(addr, bot_addr, user_id)