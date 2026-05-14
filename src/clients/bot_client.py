import grpc
from loguru import logger
from proto.bot import bot_pb2, bot_pb2_grpc


# Default per-RPC timeout in seconds.
_DEFAULT_TIMEOUT = 5.0


class BotClient:
    def __init__(self, target_addr: str):
        self._target_addr = target_addr
        self._channel = grpc.insecure_channel(target_addr)
        self._stub = bot_pb2_grpc.BotServiceStub(self._channel)

    @property
    def target_addr(self) -> str:
        return self._target_addr

    def send(self, user_id: str, content: str, timeout: float = _DEFAULT_TIMEOUT):
        logger.info(
            "Forwarding message to bot service. destination={}, user_id={}",
            self._target_addr,
            user_id,
        )
        try:
            return self._stub.Send(
                bot_pb2.Message(user_id=user_id, content=content),
                timeout=timeout,
            )
        except grpc.RpcError as error:
            logger.error(
                "BotService.Send failed. destination={}, code={}, details={}",
                self._target_addr,
                error.code(),
                error.details(),
            )
            raise

    def get_messages(self, user_id: str, count: int = 50, timeout: float = _DEFAULT_TIMEOUT):
        logger.info(
            "Fetching messages from bot service. destination={}, user_id={}, count={}",
            self._target_addr,
            user_id,
            count,
        )
        request = bot_pb2.GetMessagesRequest(user_id=user_id, count=count)
        try:
            return self._stub.GetMessages(request, timeout=timeout)
        except grpc.RpcError as error:
            logger.error(
                "BotService.GetMessages failed. destination={}, code={}, details={}",
                self._target_addr,
                error.code(),
                error.details(),
            )
            raise

    def close(self) -> None:
        self._channel.close()
