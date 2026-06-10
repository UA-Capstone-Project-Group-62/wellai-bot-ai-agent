import threading
import json
import re
from datetime import datetime, timedelta
from dateutil import parser as date_parser

from loguru import logger
import grpc
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from src.clients.bot_client import BotClient
from src.clients.scheduling_client import SchedulingClient
from src.services.intent_graph import graph as intent_graph
from src.services.language_monitor import language_monitor
from src.services.sentiment_monitor import sentiment_monitor

from typing import Optional

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
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=500,
        )

    def _parse_time_to_hhmm(self, time_str: str) -> str:
        """Convert various time formats to HH:MM format."""
        if not time_str:
            return "09:00"
        
        time_str = time_str.strip().lower()
        
        try:
            # Try parsing with dateutil parser (handles "4pm", "4:30pm", "13:00", etc.)
            parsed_time = date_parser.parse(time_str, default=datetime.now())
            return parsed_time.strftime("%H:%M")
        except Exception as e:
            logger.warning("Failed to parse time '{}': {}", time_str, e)
            return "09:00"  # Default fallback

    def _extract_booking_details(self, conversation: str, user_message: str) -> dict:
        """Extract booking details from conversation using LLM."""
        extraction_prompt = f"""Extract booking details from the LATEST USER MESSAGE. Return a JSON object with these fields (use null for missing values):
- clinic_id: clinic identifier or name (e.g., "clinic_001" or "Central Clinic") - if not mentioned, use default "clinic_001"
- preferred_date: Relative date expression from latest message (e.g., "tomorrow", "today", "next week", or YYYY-MM-DD if specific date mentioned) - null if not mentioned
- preferred_time: Time in HH:MM format (e.g., "13:00", "1pm") or null if not mentioned
- user_name: patient name or null

IMPORTANT: Extract ONLY from the latest user message, NOT from conversation history. Use relative date expressions as-is (e.g., "tomorrow" not a specific date).

Conversation history:
{conversation}

Latest message:
{user_message}

Return ONLY valid JSON, no other text."""

        if self._llm is None:
            self._llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=500,
            )
        response = self._llm.invoke([
            SystemMessage(content="You are a booking details extractor. Extract structured information from the LATEST USER MESSAGE ONLY. Do not use information from older messages."),
            HumanMessage(content=extraction_prompt),
        ])

        try:
            # Strip markdown code blocks if present
            content = response.content.strip()
            if content.startswith("```"):
                # Remove markdown code block markers
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
                        # Try parsing as actual date
                        target_date = date_parser.parse(date_str)
                    
                    details["preferred_date"] = target_date.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.warning("Failed to parse date '{}': {}", date_str, e)
                    details["preferred_date"] = None
            
            # Set default clinic_id if not provided
            if not details.get("clinic_id"):
                details["clinic_id"] = "clinic_001"
            
            logger.info("Extracted booking details: {}", details)
            return details
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse extraction response: {}", response.content)
            return {}

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
            # Parse ISO datetime string to Timestamp
            dt = datetime.fromisoformat(start_time_str)
            timestamp = Timestamp()
            timestamp.FromDatetime(dt)

            # Calculate end time (assume 1 hour appointment)
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
            response = self.bot_client.get_messages(user_id, count=50)
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
            intent_result = intent_graph.invoke({
                "messages": [HumanMessage(content=content)],
                "history": history_text,
                "intent": "unrelated_to_your_job",
            })
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
                message="Sorry, I'm having trouble right now. Please try again.",
            )

        # --- Step 2: Handle scheduling intents ---
        try:
            if detected_intent in ["book_app", "cancel_app", "reschedule_app"]:
                if self.scheduling_client is None:
                    logger.warning(
                        "Scheduling intent detected but scheduling client is not configured. user_id={}, intent={}",
                        user_id,
                        detected_intent,
                    )
                    return self._send_reply(user_id, ai_reply, context)

                full_conversation = history_text + f"\nUser: {content}"
                if detected_intent == "book_app":
                    # Try to extract booking details and schedule
                    details = self._extract_booking_details(full_conversation, content)
                    if details.get("preferred_date"):
                        # Attempt scheduling with extracted details
                        # Parse time to HH:MM format
                        time_str = self._parse_time_to_hhmm(details.get('preferred_time'))
                        datetime_str = f"{details['preferred_date']}T{time_str}"
                        schedule_result = self._attempt_schedule_appointment(
                            user_id=user_id,
                            user_name=details.get("user_name"),
                            clinic_id=details["clinic_id"],
                            start_time_str=datetime_str,
                        )

                        if schedule_result["success"]:
                            # Append success message to AI reply
                            ai_reply += f"\n\n✅ Appointment confirmed! {schedule_result['message']}"
                            logger.info("Appointment scheduled successfully for user {}", user_id)
                        else:
                            # Append error and continue conversation
                            ai_reply += f"\n\n⚠️ Could not complete booking: {schedule_result['message']}. Please try again or contact us directly."
                            logger.warning(
                                "Scheduling failed for user {}: {}",
                                user_id,
                                schedule_result["message"],
                            )
                    else:
                        logger.info("Incomplete booking details. Waiting for more information from user.")

                elif detected_intent == "cancel_app":
                    # Try to cancel appointment
                    cancel_result = self._cancel_appointment(user_id)
                    if cancel_result["success"]:
                        ai_reply += f"\n\n✅ Appointment cancelled! {cancel_result['message']}"
                        logger.info("Appointment cancelled for user {}", user_id)
                    else:
                        ai_reply += f"\n\n⚠️ Could not cancel appointment: {cancel_result['message']}"
                        logger.warning(
                            "Cancellation failed for user {}: {}",
                            user_id,
                            cancel_result["message"],
                        )

                elif detected_intent == "reschedule_app":
                    # For reschedule, extract new details and attempt to update
                    details = self._extract_booking_details(full_conversation, content)
                    if details.get("clinic_id") and details.get("preferred_date"):
                        # First cancel old appointment, then create new one
                        cancel_result = self._cancel_appointment(user_id)
                        if cancel_result["success"]:
                            time_str = self._parse_time_to_hhmm(details.get("preferred_time"))
                            datetime_str = f"{details['preferred_date']}T{time_str}"
                            schedule_result = self._attempt_schedule_appointment(
                                user_id=user_id,
                                user_name=details.get("user_name"),
                                clinic_id=details["clinic_id"],
                                start_time_str=datetime_str,
                            )

                            if schedule_result["success"]:
                                ai_reply += f"\n\n✅ Appointment rescheduled! {schedule_result['message']}"
                                logger.info("Appointment rescheduled for user {}", user_id)
                            else:
                                ai_reply += f"\n\n⚠️ Failed to reschedule. Please try again."
                        else:
                            ai_reply += f"\n\n⚠️ Could not process rescheduling: {cancel_result['message']}"
                    else:
                        logger.info("Incomplete reschedule details. Waiting for more information from user.")
        except Exception as e:
            logger.error("Error handling scheduling intent: {}", e)
            # Continue with just the AI reply, don't fail

        # --- Step 3: Forward the AI reply back to the user via BotService ---
        return self._send_reply(user_id, ai_reply, context)
