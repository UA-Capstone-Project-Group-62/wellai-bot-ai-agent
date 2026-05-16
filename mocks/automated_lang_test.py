import grpc
import sys
import os
import time

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


def send_and_get_response(agent_stub, bot_stub, user_id: str, content: str, lang: str) -> str:
    msg = agent_pb2.Message(user_id=user_id, content=content)
    response = agent_stub.Receive(msg)

    if not response.success:
        return f"ERROR: {response.message}"

    time.sleep(0.5)
    msgs = bot_stub.GetMessages(bot_pb2.GetMessagesRequest(user_id=user_id, count=10))
    for m in msgs.messages:
        if m.content != content:
            return m.content

    return "(No response found)"


def run_tests(addr: str, bot_addr: str):
    agent_channel = grpc.insecure_channel(addr)
    agent_stub = agent_pb2_grpc.AgentServiceStub(agent_channel)

    bot_channel = grpc.insecure_channel(bot_addr)
    bot_stub = bot_pb2_grpc.BotServiceStub(bot_channel)

    print(f"Testing AI Agent at {addr}")
    print(f"BotService at {bot_addr}")
    print("=" * 60)

    test_cases = [
        ("English", "Hello, I want to book an appointment"),
        ("English", "What are your working hours?"),
        ("English", "Can I reschedule my appointment?"),
        ("English", "Where is your clinic located?"),
        ("English", "How much is the consultation fee?"),
        ("English", "What languages do you know?"),
        ("Malay", "Saya nak buat temu janji"),
        ("Malay", "Apakah waktu operasi anda?"),
        ("Malay", "Boleh saya tukar tarikh temu janji?"),
        ("Malay", "Di mana klinik anda?"),
        ("Malay", "Berapa yuran konsultasi?"),
        ("Malay", "Apakah bahasa yang anda faham?"),
        ("Mandarin", "我想预约看诊"),
        ("Mandarin", "你们的营业时间是什么时候？"),
        ("Mandarin", "我可以改预约吗？"),
        ("Mandarin", "你们的诊所在哪里？"),
        ("Mandarin", "看诊费用是多少？"),
        ("Mandarin", "你们支持什么语言？"),
    ]

    results = {"English": {"passed": 0, "failed": 0}, "Malay": {"passed": 0, "failed": 0}, "Mandarin": {"passed": 0, "failed": 0}}

    counter = 0

    for lang, message in test_cases:
        counter += 1
        user_id = f"test_{lang.lower()}_{counter}"
        print(f"\n[{lang}] {message}")
        response = send_and_get_response(agent_stub, bot_stub, user_id, message, lang)
        print(f"  → Response: {response}")

        if response and not response.startswith("ERROR") and response != "(No response found)":
            results[lang]["passed"] += 1
            print(f"  ✓ PASS")
        else:
            results[lang]["failed"] += 1
            print(f"  ✗ FAIL")

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for lang in results:
        total = results[lang]["passed"] + results[lang]["failed"]
        pct = (results[lang]["passed"] / total * 100) if total > 0 else 0
        print(f"{lang}: {results[lang]['passed']}/{total} passed ({pct:.0f}%)")

    total_all = sum(r["passed"] for r in results.values())
    total_tests = sum(r["passed"] + r["failed"] for r in results.values())
    print(f"\nOverall: {total_all}/{total_tests} passed ({(total_all/total_tests*100):.0f}%)")

    agent_channel.close()
    bot_channel.close()


if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else "localhost:50051"
    bot_addr = sys.argv[2] if len(sys.argv) > 2 else "localhost:50052"
    run_tests(addr, bot_addr)