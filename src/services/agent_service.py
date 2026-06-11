import json
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import grpc
from dateutil import parser as date_parser
from google.protobuf.timestamp_pb2 import Timestamp
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient
from src.clients.scheduling_client import SchedulingClient
from src.services.intent_graph import graph as intent_graph
from src.services.language_monitor import language_monitor
from src.services.sentiment_monitor import sentiment_monitor


class AgentService(agent_pb2_grpc.AgentServiceServicer):
    def __init__(
        self,
        bot_client: BotClient,
        scheduling_client: Optional[SchedulingClient] = None,
    ):
        self.bot_client = bot_client
        self.scheduling_client = scheduling_client
        self._escalated_user_languages: dict[str, str] = {}
        self._escalation_lock = threading.Lock()
        self._llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.1,
            max_tokens=500,
        )

    def _parse_time_to_hhmm(self, time_str: str) -> str:
        """Convert various time formats to HH:MM format."""
        if not time_str:
            return "09:00"

        time_str = time_str.strip().lower()

        try:
            # Use midnight as default so only the time component from the string
            # is used — avoids inheriting the current minute/second from datetime.now().
            midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            parsed_time = date_parser.parse(time_str, default=midnight)
            return parsed_time.strftime("%H:%M")
        except Exception as e:
            logger.warning("Failed to parse time '{}': {}", time_str, e)
            return "09:00"

    def _fetch_clinics(self) -> list[dict]:
        """Fetch available clinics from the scheduling service."""
        if self.scheduling_client is None:
            return []
        try:
            response = self.scheduling_client.list_clinics()
            clinics = []
            for c in response.clinics:
                info = json.loads(c.clinic_info)
                clinics.append(
                    {
                        "id": c.clinic_id,
                        "name": info.get("name", c.clinic_id),
                        "address": info.get("address", ""),
                        "phone": info.get("phone", ""),
                        "email": info.get("email", ""),
                    }
                )
            return clinics
        except Exception as e:
            logger.warning("Failed to fetch clinics: {}", e)
            return []

    def _extract_booking_details(
        self, conversation: str, user_message: str, clinics: list[dict]
    ) -> dict:
        """Extract booking details from conversation using LLM."""
        clinic_list_text = "\n".join(
            f'- id: "{c["id"]}", name: "{c["name"]}"' for c in clinics
        )
        default_clinic_instruction = (
            f'if not mentioned, use the first clinic id: "{clinics[0]["id"]}"'
            if clinics
            else "if not mentioned, use null"
        )

        extraction_prompt = f"""Extract booking details for the current booking request. Return a JSON object with these fields (use null for missing values):
- clinic_id: must be one of the clinic IDs listed below ({default_clinic_instruction})
- preferred_date: Relative date expression (e.g., "tomorrow", "today", "next week", or YYYY-MM-DD) - check latest message first, then history - null if not found
- preferred_time: Time in HH:MM format (e.g., "13:00", "1pm") - check latest message first, then history - null if not found
- user_name: patient name - check latest message first, then conversation history (the user may have given their name in a previous reply) - null if not found anywhere

Available clinics:
{clinic_list_text}

IMPORTANT: For clinic_id, preferred_date, and preferred_time — prefer the latest message but fall back to history if not in the latest message. For user_name — check both latest message and history. Use relative date expressions as-is. The clinic_id must exactly match one of the IDs listed above.

Conversation history:
{conversation}

Latest message:
{user_message}

Return ONLY valid JSON, no other text."""

        response = self._llm.invoke(
            [
                SystemMessage(
                    content="You are a booking details extractor. Extract structured information from the LATEST USER MESSAGE ONLY. Do not use information from older messages."
                ),
                HumanMessage(content=extraction_prompt),
            ]
        )

        try:
            # Strip markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)

            details = json.loads(content.strip())

            # Convert relative dates to ISO format
            if details.get("preferred_date"):
                date_str = details["preferred_date"].lower().strip()
                try:
                    if "tomorrow" in date_str:
                        target_date = datetime.now() + timedelta(days=1)
                    elif "today" in date_str:
                        target_date = datetime.now()
                    elif "next week" in date_str:
                        target_date = datetime.now() + timedelta(days=7)
                    else:
                        target_date = date_parser.parse(date_str)

                    details["preferred_date"] = target_date.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.warning("Failed to parse date '{}': {}", date_str, e)
                    details["preferred_date"] = None

            # Validate clinic_id against the known list; fall back to first clinic
            valid_ids = {c["id"] for c in clinics}
            if not details.get("clinic_id") or details["clinic_id"] not in valid_ids:
                details["clinic_id"] = clinics[0]["id"] if clinics else None

            logger.info("Extracted booking details: {}", details)
            return details
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction response: {}", response.content)
            return {}

    def _query_available_slots(
        self, clinic_id: str, target_date: str | None = None, days: int = 7
    ) -> list[dict]:
        """Query available time slots for a clinic.

        Returns list of {"date": "YYYY-MM-DD", "time": "HH:MM"}.
        If target_date is given, only slots on that date are returned.
        """
        if self.scheduling_client is None:
            return []
        try:
            response = self.scheduling_client.query_available_slots(
                clinic_id, days=days
            )
            slots = []
            for slot in response.available_slots:
                # ToDatetime() returns naive UTC — make it aware then convert to local time
                start_dt = slot.start_time.ToDatetime(tzinfo=timezone.utc).astimezone()
                slot_date = start_dt.strftime("%Y-%m-%d")
                if target_date is None or slot_date == target_date:
                    slots.append(
                        {"date": slot_date, "time": start_dt.strftime("%H:%M")}
                    )
            return slots
        except Exception as e:
            logger.warning("Failed to query available slots: {}", e)
            return []

    def _resolve_clinic_id(self, message: str, clinics: list[dict]) -> str | None:
        """Use LLM to match a clinic name/reference in the message to a clinic ID."""
        if not clinics:
            return None
        if len(clinics) == 1:
            return clinics[0]["id"]
        clinic_list = "\n".join(
            f'- id: "{c["id"]}", name: "{c["name"]}"' for c in clinics
        )
        prompt = f"""Given this message: "{message}"

And these available clinics:
{clinic_list}

Which clinic is the user referring to? Return ONLY the clinic id exactly as listed, or "unknown" if unclear."""
        response = self._llm.invoke(
            [
                SystemMessage(
                    content="You are a clinic name resolver. Return only the clinic id or 'unknown'."
                ),
                HumanMessage(content=prompt),
            ]
        )
        resolved = response.content.strip().strip('"')
        valid_ids = {c["id"] for c in clinics}
        return (
            resolved
            if resolved in valid_ids
            else (clinics[0]["id"] if clinics else None)
        )

    def _attempt_schedule_appointment(
        self,
        user_id: str,
        user_name: str,
        clinic_id: str,
        start_time_str: str,
    ) -> dict:
        """Attempt to schedule an appointment with the scheduling service."""

        if self.scheduling_client is None:
            return {
                "success": False,
                "message": "Scheduling service unavailable",
            }

        try:
            # Parse ISO datetime string and localise to the system timezone so
            # Timestamp.FromDatetime (which expects an aware datetime) converts
            # correctly instead of treating the naive value as UTC.
            dt = datetime.fromisoformat(start_time_str).astimezone()
            timestamp = Timestamp()
            timestamp.FromDatetime(dt)

            end_dt = dt + timedelta(hours=1)
            end_timestamp = Timestamp()
            end_timestamp.FromDatetime(end_dt)

            response = self.scheduling_client.schedule(
                user_id=user_id,
                user_name=user_name or user_id,
                clinic_id=clinic_id,
                start_time=timestamp,
                end_time=end_timestamp,
            )

            logger.info(
                "Scheduling service response: success={}, message={}",
                response.success,
                response.message,
            )

            return {
                "success": response.success,
                "message": response.message,
            }
        except grpc.RpcError as error:
            logger.error(
                "Scheduling service error. code={}, details={}",
                error.code(),
                error.details(),
            )
            return {
                "success": False,
                "message": f"Scheduling service error: {error.details()}",
            }
        except Exception as e:
            logger.error("Failed to schedule appointment: {}", e)
            return {
                "success": False,
                "message": f"Failed to process scheduling: {str(e)}",
            }

    def _cancel_appointment(self, user_id: str) -> dict:
        """Cancel an appointment for the user."""

        if self.scheduling_client is None:
            return {
                "success": False,
                "message": "Scheduling service unavailable",
            }

        try:
            response = self.scheduling_client.cancel(user_id)
            logger.info(
                "Cancel response: success={}, message={}",
                response.success,
                response.message,
            )
            return {
                "success": response.success,
                "message": response.message,
            }
        except grpc.RpcError as error:
            logger.error(
                "Scheduling service error. code={}, details={}",
                error.code(),
                error.details(),
            )
            return {
                "success": False,
                "message": f"Scheduling service error: {error.details()}",
            }
        except Exception as e:
            logger.error("Failed to cancel appointment: {}", e)
            return {
                "success": False,
                "message": f"Failed to cancel appointment: {str(e)}",
            }

    def _fetch_history(self, user_id: str) -> str:
        """Fetch conversation history from the bot service via GetMessages."""
        try:
            response = self.bot_client.get_messages(user_id, count=8)
            if not response.messages:
                return ""
            lines = []
            for msg in response.messages:
                lines.append(f"- {msg.content}")
            return "\n".join(lines)
        except grpc.RpcError as error:
            logger.warning(
                "Failed to fetch history from bot service. destination={}, code={}, details={}",
                self.bot_client.target_addr,
                error.code(),
                error.details(),
            )
            return ""
        except Exception as e:
            logger.warning("Unexpected error fetching history: {}", e)
            return ""

    def _send_reply(self, user_id: str, ai_reply: str, context):
        try:
            send_response = self.bot_client.send(user_id, ai_reply)
            if not send_response.success:
                logger.warning(
                    "BotService.Send returned failure. user_id={}, error={}",
                    user_id,
                    send_response.message,
                )
                return common_pb2.Response(
                    success=False,
                    message=f"BotService rejected the message: {send_response.message}",
                )
        except grpc.RpcError as error:
            logger.error(
                "Failed to send reply to bot service. destination={}, code={}, details={}",
                self.bot_client.target_addr,
                error.code(),
                error.details(),
            )
            context.set_code(error.code())
            context.set_details(
                f"Failed to send reply to bot service: {error.details()}"
            )
            return common_pb2.Response(
                success=False,
                message="Failed to deliver reply to user",
            )
        except Exception as e:
            logger.error("Unexpected error sending reply: {}", e)
            return common_pb2.Response(
                success=False,
                message="Failed to deliver reply to user",
            )

        return common_pb2.Response(success=True, message="")

    def Receive(self, request, context):
        user_id = request.user_id
        content = request.content

        logger.info(
            "Received message from user. user_id={}, content_length={}",
            user_id,
            len(content),
        )

        with self._escalation_lock:
            escalated_language = self._escalated_user_languages.get(user_id)
        if escalated_language is not None:
            logger.warning(
                "Conversation already escalated. user_id={}, content_length={}",
                user_id,
                len(content),
            )
            return self._send_reply(
                user_id,
                sentiment_monitor.escalation_reply(escalated_language),
                context,
            )

        sentiment_result = sentiment_monitor.evaluate(content)
        if sentiment_result.should_escalate:
            with self._escalation_lock:
                self._escalated_user_languages[user_id] = sentiment_result.language
            logger.warning(
                "Sentiment escalation triggered. user_id={}, category={}, source={}, language={}, reason={}, content_length={}",
                user_id,
                sentiment_result.category.value,
                sentiment_result.source.value,
                sentiment_result.language,
                sentiment_result.reason,
                len(content),
            )
            escalation_reply = sentiment_monitor.escalation_reply(
                sentiment_result.language,
            )
            return self._send_reply(user_id, escalation_reply, context)

        language_result = language_monitor.evaluate(content)
        if not language_result.is_supported:
            logger.info(
                "Unsupported language detected. user_id={}, source={}, reason={}, content_length={}",
                user_id,
                language_result.source.value,
                language_result.reason,
                len(content),
            )
            return self._send_reply(
                user_id,
                language_monitor.unsupported_reply(),
                context,
            )

        history_text = self._fetch_history(user_id)

        # --- Step 1: Classify intent and generate AI reply ---
        try:
            intent_result = intent_graph.invoke(
                {
                    "messages": [HumanMessage(content=content)],
                    "history": history_text,
                    "intent": "unrelated_to_your_job",
                }
            )
            detected_intent = intent_result.get("intent", "unrelated_to_your_job")
            ai_reply = intent_result["messages"][-1].content.strip()

            logger.info(
                "Intent detected: {}, AI reply length: {}",
                detected_intent,
                len(ai_reply),
            )
        except Exception as e:
            logger.error("LangGraph error: {}", e)

            return common_pb2.Response(
                success=False,
                message="Loading... Please wait.",
            )

        # --- Step 2: Handle scheduling intents ---
        try:
            if detected_intent == "list_clinics":
                clinics = self._fetch_clinics()
                if clinics:
                    lines = []
                    for c in clinics:
                        lines.append(
                            f"🏥 *{c['name']}*\n"
                            f"  📍 {c['address']}\n"
                            f"  📞 {c['phone']}\n"
                            f"  ✉️ {c['email']}"
                        )
                    ai_reply += "\n\n" + "\n\n".join(lines)
                else:
                    ai_reply += "\n\nSorry, I couldn't retrieve the clinic list at this time. Please try again later."
                logger.info("Listed {} clinics for user {}", len(clinics), user_id)

            elif detected_intent == "query_availability":
                clinics = self._fetch_clinics()
                clinic_id = self._resolve_clinic_id(content, clinics)
                if clinic_id:
                    available = self._query_available_slots(clinic_id, days=7)
                    clinic_name = next(
                        (c["name"] for c in clinics if c["id"] == clinic_id), clinic_id
                    )
                    if available:
                        by_date: dict[str, list[str]] = {}
                        for s in available:
                            by_date.setdefault(s["date"], []).append(s["time"])
                        lines = [
                            f"*{date}*: " + ", ".join(times)
                            for date, times in sorted(by_date.items())
                        ]
                        ai_reply += (
                            f"\n\nAvailable slots at *{clinic_name}* (next 7 days):\n"
                            + "\n".join(lines)
                        )
                    else:
                        ai_reply += f"\n\nThere are no available slots at *{clinic_name}* in the next 7 days."
                    logger.info(
                        "Queried availability for clinic {} for user {}",
                        clinic_id,
                        user_id,
                    )
                else:
                    ai_reply += "\n\nI couldn't determine which clinic you meant. Please specify a clinic name."

            elif detected_intent in ["book_app", "cancel_app", "reschedule_app"]:
                if self.scheduling_client is None:
                    logger.warning(
                        "Scheduling intent detected but scheduling client is not configured. user_id={}, intent={}",
                        user_id,
                        detected_intent,
                    )
                    return self._send_reply(user_id, ai_reply, context)

                full_conversation = history_text + f"\nUser: {content}"
                clinics = self._fetch_clinics()
                if detected_intent == "book_app":
                    details = self._extract_booking_details(
                        full_conversation, content, clinics
                    )
                    if details.get("preferred_date") and not details.get(
                        "preferred_time"
                    ):
                        # Date known but no time — show available slots so user can pick
                        available = self._query_available_slots(
                            details["clinic_id"], target_date=details["preferred_date"]
                        )
                        if available:
                            slots_text = "\n".join(f"• {s['time']}" for s in available)
                            ai_reply += f"\n\nHere are the available slots on {details['preferred_date']}:\n{slots_text}\n\nWhich time would you prefer?"
                        else:
                            ai_reply += f"\n\nUnfortunately there are no available slots on {details['preferred_date']}. Would you like to try a different date?"
                        logger.info(
                            "Showed available slots for user {} on {}",
                            user_id,
                            details["preferred_date"],
                        )
                    elif details.get("preferred_date") and details.get("preferred_time") and details.get("user_name"):
                        time_str = self._parse_time_to_hhmm(details["preferred_time"])
                        datetime_str = f"{details['preferred_date']}T{time_str}"
                        schedule_result = self._attempt_schedule_appointment(
                            user_id=user_id,
                            user_name=details["user_name"],
                            clinic_id=details["clinic_id"],
                            start_time_str=datetime_str,
                        )
                        if schedule_result["success"]:
                            ai_reply = f"✅ Appointment confirmed! {schedule_result['message']}"
                            logger.info("Appointment scheduled successfully for user {}", user_id)
                        else:
                            ai_reply += f"\n\n⚠️ Could not complete booking: {schedule_result['message']}. Please try again or contact us directly."
                            logger.warning("Scheduling failed for user {}: {}", user_id, schedule_result["message"])
                    else:
                        logger.info("Incomplete booking details. Waiting for more information from user.")

                elif detected_intent == "cancel_app":
                    # Try to cancel appointment
                    cancel_result = self._cancel_appointment(user_id)
                    if cancel_result["success"]:
                        ai_reply = f"✅ Appointment cancelled! {cancel_result['message']}"
                        logger.info("Appointment cancelled for user {}", user_id)
                    else:
                        ai_reply = f"⚠️ Could not cancel appointment: {cancel_result['message']}"
                        logger.warning(
                            "Cancellation failed for user {}: {}",
                            user_id,
                            cancel_result["message"],
                        )

                elif detected_intent == "reschedule_app":
                    details = self._extract_booking_details(
                        full_conversation, content, clinics
                    )
                    if (
                        details.get("clinic_id")
                        and details.get("preferred_date")
                        and not details.get("preferred_time")
                    ):
                        # Date known but no time — show available slots
                        available = self._query_available_slots(
                            details["clinic_id"], target_date=details["preferred_date"]
                        )
                        if available:
                            slots_text = "\n".join(f"• {s['time']}" for s in available)
                            ai_reply += f"\n\nHere are the available slots on {details['preferred_date']}:\n{slots_text}\n\nWhich time would you prefer for rescheduling?"
                        else:
                            ai_reply += f"\n\nUnfortunately there are no available slots on {details['preferred_date']}. Would you like to try a different date?"
                        logger.info(
                            "Showed available slots for reschedule user {} on {}",
                            user_id,
                            details["preferred_date"],
                        )
                    elif (
                        details.get("clinic_id")
                        and details.get("preferred_date")
                        and details.get("preferred_time")
                        and details.get("user_name")
                    ):
                        # First cancel old appointment, then create new one
                        cancel_result = self._cancel_appointment(user_id)
                        if cancel_result["success"]:
                            time_str = self._parse_time_to_hhmm(
                                details.get("preferred_time")
                            )
                            datetime_str = f"{details['preferred_date']}T{time_str}"
                            schedule_result = self._attempt_schedule_appointment(
                                user_id=user_id,
                                user_name=details.get("user_name"),
                                clinic_id=details["clinic_id"],
                                start_time_str=datetime_str,
                            )

                            if schedule_result["success"]:
                                ai_reply = f"✅ Appointment rescheduled! {schedule_result['message']}"
                                logger.info(
                                    "Appointment rescheduled for user {}", user_id
                                )
                            else:
                                ai_reply += (
                                    "\n\n⚠️ Failed to reschedule. Please try again."
                                )
                        else:
                            ai_reply += f"\n\n⚠️ Could not process rescheduling: {cancel_result['message']}"
                    else:
                        logger.info(
                            "Incomplete reschedule details. Waiting for more information from user."
                        )
        except Exception as e:
            logger.error("Error handling scheduling intent: {}", e)
            # Continue with just the AI reply, don't fail

        # --- Step 3: Forward the AI reply back to the user via BotService ---
        return self._send_reply(user_id, ai_reply, context)
