import grpc
from loguru import logger
from proto.bot import bot_pb2, bot_pb2_grpc


class BotClient:
    def __init__(self, target_addr: str):
        self._target_addr = target_addr
        self._channel = grpc.insecure_channel(target_addr)
        self._stub = bot_pb2_grpc.BotServiceStub(self._channel)

    @property
    def target_addr(self) -> str:
        return self._target_addr

    def send(self, user_id: str, content: str):
        logger.info(
            "Forwarding message to bot service. destination={}, user_id={}",
            self._target_addr,
            user_id,
        )
        return self._stub.Send(bot_pb2.Message(user_id=user_id, content=content))

    def close(self) -> None:
        self._channel.close()
