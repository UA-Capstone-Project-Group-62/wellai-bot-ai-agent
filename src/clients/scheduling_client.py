import grpc
from loguru import logger
from proto.scheduling import scheduling_pb2, scheduling_pb2_grpc
from google.protobuf.empty_pb2 import Empty


# Default per-RPC timeout in seconds.
_DEFAULT_TIMEOUT = 10.0


class SchedulingClient:
    def __init__(self, target_addr: str):
        self._target_addr = target_addr
        self._channel = grpc.insecure_channel(target_addr)
        self._stub = scheduling_pb2_grpc.SchedulingServiceStub(self._channel)

    @property
    def target_addr(self) -> str:
        return self._target_addr

    def schedule(
        self,
        user_id: str,
        user_name: str,
        clinic_id: str,
        start_time,
        end_time,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        """Schedule an appointment."""
        logger.info(
            "Scheduling appointment. destination={}, user_id={}, clinic_id={}",
            self._target_addr,
            user_id,
            clinic_id,
        )
        try:
            time_range = scheduling_pb2.TimeRange(
                start_time=start_time,
                end_time=end_time,
            )
            request = scheduling_pb2.ScheduleRequest(
                user_id=user_id,
                user_name=user_name,
                clinic_id=clinic_id,
                time=time_range,
            )
            return self._stub.Schedule(request, timeout=timeout)
        except grpc.RpcError as error:
            logger.error(
                "Scheduling service error. destination={}, code={}, details={}",
                self._target_addr,
                error.code(),
                error.details(),
            )
            raise

    def list_clinics(self, timeout: float = _DEFAULT_TIMEOUT):
        """List all available clinics."""
        logger.info("Fetching clinic list. destination={}", self._target_addr)
        try:
            return self._stub.ListClinics(Empty(), timeout=timeout)
        except grpc.RpcError as error:
            logger.error(
                "Scheduling service error. destination={}, code={}, details={}",
                self._target_addr,
                error.code(),
                error.details(),
            )
            raise

    def query_available_slots(
        self, clinic_id: str, timeout: float = _DEFAULT_TIMEOUT
    ):
        """Query available time slots for a clinic."""
        logger.info(
            "Querying available slots. destination={}, clinic_id={}",
            self._target_addr,
            clinic_id,
        )
        try:
            request = scheduling_pb2.QueryRequest(clinic_id=clinic_id)
            return self._stub.Query(request, timeout=timeout)
        except grpc.RpcError as error:
            logger.error(
                "Scheduling service error. destination={}, code={}, details={}",
                self._target_addr,
                error.code(),
                error.details(),
            )
            raise

    def cancel(self, user_id: str, timeout: float = _DEFAULT_TIMEOUT):
        """Cancel an appointment."""
        logger.info(
            "Cancelling appointment. destination={}, user_id={}",
            self._target_addr,
            user_id,
        )
        try:
            request = scheduling_pb2.CancelRequest(user_id=user_id)
            return self._stub.Cancel(request, timeout=timeout)
        except grpc.RpcError as error:
            logger.error(
                "Scheduling service error. destination={}, code={}, details={}",
                self._target_addr,
                error.code(),
                error.details(),
            )
            raise

    def close(self):
        """Close the gRPC channel."""
        self._channel.close()
