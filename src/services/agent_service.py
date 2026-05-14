import grpc
import os
from loguru import logger
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient

# LangGraph + LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, Literal
import operator

# LangChain Groq wrapper
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=300
)

SYSTEM_PROMPT = """
YOUR JOB DESCRIPTION: You are a professional and polite booking assistant for health clinics. 
You are happy to help users with booking, cancelling, and rescheduling appointments. 
You can also answer questions about the clinic, but you will never give medical advice or answers not related to your job.
"""

possible_intents = ["book_app", "cancel_app", "reschedule_app", "ask_question", "unrelated_to_your_job"]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str

def _format_conversation(messages: list[BaseMessage]) -> str:
    if len(messages) <= 1:
        return "No previous messages."
    
    formatted = []
    for msg in messages[:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

def intent_classifier(state: AgentState):
    user_message = state["messages"][-1].content

    prompt = f"""
        {SYSTEM_PROMPT}

        Please classify the user message into one of these intents: {possible_intents}

        The intent is "book_app" if the user wants to make a booking, make a new appointment, or
        other similar requests.

        The intent is "cancel_app" if the user wants to cancel a booking, cancel an appointment, or
        other similar requests.

        The intent is "reschedule_app" if the user wants to change the time of an appointment they already have,
        change the date of their booking, or change the medical professional they are seeing. 

        The intent is "ask_question" if the user is asking for more information about the clinic, booking process,
        or similar topics. This DOES NOT INCLUDE questions about topics that are unrelated to the medical clinic,
        you (the booking assistant), or the medical professionals they are able to book appointments with.

        The intent is "unrelated_to_your_job" if the user has a request that is anything else.

        Return ONLY the intent label.

        Message: {user_message}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    intent = response.content.strip()

    logger.info("Detected intent: {}", intent)

    return {"intent": intent}

def agent_node(state: AgentState):
    messages = state["messages"]
    conversation = _format_conversation(messages)
    user_message = messages[-1].content

    prompt = f"""Continue the conversation naturally, addressing the user's latest message while considering the conversation history.

{SYSTEM_PROMPT}

Conversation history:
{conversation}

Latest message from user: {user_message}

Provide a helpful response that continues the conversation naturally."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def book_node(state: AgentState):
    messages = state["messages"]
    conversation = _format_conversation(messages)
    user_message = messages[-1].content

    prompt = f"""You are helping the user book an appointment. Consider the conversation history to understand what information has already been provided.

{SYSTEM_PROMPT}

Conversation history:
{conversation}

Latest user message: {user_message}

Respond as a helpful booking assistant. If the user has already provided information (like preferred time or date), acknowledge it and ask for any missing details. Do not ask for information they have already given."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def cancel_node(state: AgentState):
    messages = state["messages"]
    conversation = _format_conversation(messages)
    user_message = messages[-1].content

    prompt = f"""You are helping the user cancel an appointment. Consider the conversation history to understand what information has already been provided.

{SYSTEM_PROMPT}

Conversation history:
{conversation}

Latest user message: {user_message}

Respond as a helpful booking assistant. Acknowledge any details they have provided and ask for only the missing information needed to process the cancellation."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def reschedule_node(state: AgentState):
    messages = state["messages"]
    conversation = _format_conversation(messages)
    user_message = messages[-1].content

    prompt = f"""You are helping the user reschedule an appointment. Consider the conversation history to understand what information has already been provided.

{SYSTEM_PROMPT}

Conversation history:
{conversation}

Latest user message: {user_message}

Respond as a helpful booking assistant. Acknowledge any details they have provided and ask for only the missing information needed to process the rescheduling."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def question_node(state: AgentState):
    messages = state["messages"]
    conversation = _format_conversation(messages)
    user_message = messages[-1].content

    prompt = f"""You are answering the user's question about the clinic. Consider the conversation history for context.

{SYSTEM_PROMPT}

Conversation history:
{conversation}

Latest user message: {user_message}

Provide a helpful and informative response about the clinic."""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

# langgraph workflow begins here
workflow = StateGraph(AgentState)
workflow.add_node("intent", intent_classifier)
workflow.add_node("agent", agent_node)
workflow.add_node("book", book_node)
workflow.add_node("cancel", cancel_node)
workflow.add_node("reschedule", reschedule_node)
workflow.add_node("question", question_node)

def route(state: AgentState):
    intent = state["intent"]

    if intent == "book_app":
        return "book"
    elif intent == "cancel_app":
        return "cancel"
    elif intent == "reschedule_app":
        return "reschedule"
    elif intent == "ask_question":
        return "question"
    else:
        return "agent"

workflow.set_entry_point("intent")

workflow.add_conditional_edges(
    "intent",
    route,
    {
        "agent": "agent"
    }
)

workflow.add_edge("agent", END)

graph = workflow.compile()

class AgentService(agent_pb2_grpc.AgentServiceServicer):
    def __init__(self, bot_client: BotClient):
        self.bot_client = bot_client
        self.conversation_history: dict[str, list[BaseMessage]] = {}

    def _get_history(self, user_id: str) -> list[BaseMessage]:
        return self.conversation_history.get(user_id, [])

    def _update_history(self, user_id: str, message: BaseMessage):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append(message)

    def Receive(self, request_iterator, context):
        user_id = None
        messages = []
        for request in request_iterator:
            user_id = request.user_id
            logger.info(
                "Received message from user. user_id={}, content_length={}",
                request.user_id,
                len(request.content),
            )
            messages.append(request.content)

        if not user_id:
            return common_pb2.Response(success=False, message="No user_id provided")

        combined_content = "\n".join(messages)
        history = self._get_history(user_id)

        try:
            result = graph.invoke({
                "messages": history + [HumanMessage(content=combined_content)]
            })
            ai_reply = result["messages"][-1].content.strip()

            self._update_history(user_id, HumanMessage(content=combined_content))
            self._update_history(user_id, HumanMessage(content=ai_reply))

            logger.info("AI replied successfully")

        except Exception as e:
            logger.error(f"LangGraph error: {e}")
            ai_reply = "Sorry, I'm having trouble right now. Please try again."

        return common_pb2.Response(
            success=True,
            message=ai_reply
        )

    def ReceiveAndRespond(self, request_iterator, context):
        def generate_responses():
            for request in request_iterator:
                user_id = request.user_id
                logger.info(
                    "Received message from user. user_id={}, content={}",
                    request.user_id,
                    request.content,
                )

                history = self._get_history(user_id)

                try:
                    result = graph.invoke({
                        "messages": history + [HumanMessage(content=request.content)]
                    })
                    ai_reply = result["messages"][-1].content.strip()

                    self._update_history(user_id, HumanMessage(content=request.content))
                    self._update_history(user_id, HumanMessage(content=ai_reply))

                    logger.info("AI replied successfully")
                except Exception as e:
                    logger.error(f"LangGraph error: {e}")
                    ai_reply = "Sorry, I'm having trouble right now. Please try again."

                yield common_pb2.Response(success=True, message=ai_reply)

        return generate_responses()