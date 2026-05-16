from concurrent import futures
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto', 'gen', 'py'))

import grpc
from proto.bot import bot_pb2, bot_pb2_grpc
from proto.common import common_pb2


class BotService(bot_pb2_grpc.BotServiceServicer):
    _messages: dict[str, list[dict]] = {}

    def Send(self, request, context):
        user_id = request.user_id
        content = request.content
        if user_id not in self._messages:
            self._messages[user_id] = []
        self._messages[user_id].append({"content": content})
        return common_pb2.Response(success=True, message="")

    def GetMessages(self, request, context):
        user_id = request.user_id
        count = request.count
        msgs = self._messages.get(user_id, [])[-count:]
        return bot_pb2.GetMessagesResponse(
            messages=[
                bot_pb2.Message(user_id=user_id, content=m["content"])
                for m in msgs
            ]
        )


def run_bot_service(port: int = 50052):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    bot_pb2_grpc.add_BotServiceServicer_to_server(BotService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Mock BotService listening on [::]:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50052
    run_bot_service(port)