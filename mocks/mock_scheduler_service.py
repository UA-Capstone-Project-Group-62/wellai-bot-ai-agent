from concurrent import futures
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto', 'gen', 'py'))

import grpc
from proto.scheduling import scheduling_pb2, scheduling_pb2_grpc
from proto.common import common_pb2
from google.protobuf.empty_pb2 import Empty


class MockSchedulingService(scheduling_pb2_grpc.SchedulingServiceServicer):
    """Mock scheduling service for testing without the real scheduler."""
    
    _appointments = {}  # user_id -> appointment details
    _clinics = [
        scheduling_pb2.Clinic(
            clinic_id="clinic_001",
            clinic_info='{"name": "Central Clinic", "address": "123 Main St", "working_hours": "9am-5pm"}'
        ),
        scheduling_pb2.Clinic(
            clinic_id="clinic_002",
            clinic_info='{"name": "Downtown Clinic", "address": "456 Oak Ave", "working_hours": "10am-6pm"}'
        ),
    ]

    def Schedule(self, request, context):
        """Mock Schedule RPC - stores appointment and returns success."""
        user_id = request.user_id
        clinic_id = request.clinic_id
        
        # Store appointment
        self._appointments[user_id] = {
            "user_id": user_id,
            "user_name": request.user_name,
            "clinic_id": clinic_id,
            "start_time": request.time.start_time,
            "end_time": request.time.end_time,
        }
        
        return common_pb2.Response(
            success=True,
            message=f"Appointment scheduled at {clinic_id} for user {request.user_name}"
        )

    def ListClinics(self, request, context):
        """Mock ListClinics RPC - returns mock clinic list."""
        return scheduling_pb2.ListClinicsResponse(clinics=self._clinics)

    def Query(self, request, context):
        """Mock Query RPC - returns available time slots."""
        # Return mock available slots (e.g., 9am-12pm, 2pm-5pm)
        from google.protobuf.timestamp_pb2 import Timestamp
        from datetime import datetime, timedelta
        
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        
        slots = []
        # Morning slot: 9am-12pm
        start = Timestamp()
        start.FromDatetime(tomorrow.replace(hour=9, minute=0, second=0))
        end = Timestamp()
        end.FromDatetime(tomorrow.replace(hour=12, minute=0, second=0))
        slots.append(scheduling_pb2.TimeRange(start_time=start, end_time=end))
        
        # Afternoon slot: 2pm-5pm
        start = Timestamp()
        start.FromDatetime(tomorrow.replace(hour=14, minute=0, second=0))
        end = Timestamp()
        end.FromDatetime(tomorrow.replace(hour=17, minute=0, second=0))
        slots.append(scheduling_pb2.TimeRange(start_time=start, end_time=end))
        
        return scheduling_pb2.QueryResponse(available_slots=slots)

    def Cancel(self, request, context):
        """Mock Cancel RPC - cancels user's appointment."""
        user_id = request.user_id
        
        if user_id in self._appointments:
            del self._appointments[user_id]
            return common_pb2.Response(
                success=True,
                message=f"Appointment cancelled for user {user_id}"
            )
        else:
            return common_pb2.Response(
                success=False,
                message=f"No appointment found for user {user_id}"
            )


def run_mock_scheduler(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    scheduling_pb2_grpc.add_SchedulingServiceServicer_to_server(
        MockSchedulingService(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Mock Scheduling Service listening on [::]:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 50051
    run_mock_scheduler(port)
