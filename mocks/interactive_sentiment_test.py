import os
import sys
import time
from uuid import uuid4

import grpc
from dotenv import load_dotenv


proto_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "proto",
    "gen",
    "py",
)
if not os.path.isdir(proto_path):
    raise SystemExit(
        "Missing generated protobuf Python stubs. Expected directory: "
        f"{proto_path}. Initialize the proto submodule before running this client."
    )
sys.path.insert(0, proto_path)

from proto.agent import agent_pb2, agent_pb2_grpc
from proto.bot import bot_pb2, bot_pb2_grpc


EXAMPLES = [
    ("Safe English booking", "I want to book an appointment"),
    ("English distress", "I feel extremely depressed and hopeless"),
    ("Malay profanity", "Awak memang sial"),
    ("Mandarin self-harm", "我想死"),
]

ESCALATION_REPLY_MARKERS = [
    "requested a human operator",
    "operator manusia",
    "人工客服接手",
]


def _message_count(bot_stub, user_id: str) -> int:
    response = bot_stub.GetMessages(
        bot_pb2.GetMessagesRequest(user_id=user_id, count=100),
    )
    return len(response.messages)


def _latest_new_reply(bot_stub, user_id: str, previous_count: int) -> str | None:
    response = bot_stub.GetMessages(
        bot_pb2.GetMessagesRequest(user_id=user_id, count=100),
    )
    if len(response.messages) <= previous_count:
        return None
    return response.messages[-1].content


def _wait_for_reply(bot_stub, user_id: str, previous_count: int) -> str | None:
    for _ in range(10):
        reply = _latest_new_reply(bot_stub, user_id, previous_count)
        if reply is not None:
            return reply
        time.sleep(0.2)
    return None


def is_escalation_reply(reply: str) -> bool:
    lower_reply = reply.lower()
    return any(marker in lower_reply for marker in ESCALATION_REPLY_MARKERS)


def print_examples() -> None:
    print("Try these sentiment-monitoring examples:")
    for label, message in EXAMPLES:
        print(f"- {label}: {message}")
    print()


def print_environment_note() -> None:
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        return

    print("Note: GROQ_API_KEY is not set in your environment.")
    print("Safe messages like 'Hello' continue into the normal AI graph and need Groq.")
    print("Rule-triggered sentiment messages can still show the human-operator reply.")
    print()


def chat(addr: str, bot_addr: str, user_id: str) -> None:
    agent_channel = grpc.insecure_channel(addr)
    agent_stub = agent_pb2_grpc.AgentServiceStub(agent_channel)

    bot_channel = grpc.insecure_channel(bot_addr)
    bot_stub = bot_pb2_grpc.BotServiceStub(bot_channel)

    print(f"Connected to AI Agent at {addr}")
    print(f"Connected to BotService at {bot_addr}")
    print(f"Using user_id={user_id}")
    print_environment_note()
    print_examples()
    print("Type your message and press Enter.")
    print("Commands: /examples, /user <user_id>, /fresh, /quit")
    print()

    try:
        while True:
            text = input("> ").strip()
            if not text:
                continue
            if text.lower() in ("/quit", "quit", "exit"):
                break
            if text.lower() == "/examples":
                print_examples()
                continue
            if text.lower().startswith("/user "):
                user_id = text.split(maxsplit=1)[1].strip()
                if not user_id:
                    print("Usage: /user <user_id>\n")
                    continue
                print(f"Switched to user_id={user_id}\n")
                continue
            if text.lower() == "/fresh":
                user_id = f"sentiment_test_{uuid4().hex[:8]}"
                print(f"Switched to fresh user_id={user_id}\n")
                continue

            before_count = _message_count(bot_stub, user_id)
            response = agent_stub.Receive(
                agent_pb2.Message(user_id=user_id, content=text),
            )

            if not response.success:
                print(f"AgentService error: {response.message}\n")
                continue

            reply = _wait_for_reply(bot_stub, user_id, before_count)
            if reply is None:
                print("(Message accepted, but no BotService reply was found yet.)\n")
            else:
                print(f"\nAI: {reply}\n")
                if is_escalation_reply(reply):
                    print("Sentiment escalation detected. Logging out of this test chat.")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        agent_channel.close()
        bot_channel.close()
        print("Goodbye!")


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
    bot_addr = sys.argv[2] if len(sys.argv) > 2 else "localhost:50052"
    user_id = sys.argv[3] if len(sys.argv) > 3 else f"sentiment_test_{uuid4().hex[:8]}"
    chat(addr, bot_addr, user_id)
