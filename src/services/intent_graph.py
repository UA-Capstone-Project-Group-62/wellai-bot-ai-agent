from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_groq import ChatGroq
from loguru import logger


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

POSSIBLE_INTENTS = [
    "book_app",
    "cancel_app",
    "reschedule_app",
    "ask_question",
    "unrelated_to_your_job",
]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str
    history: str


def format_conversation(messages: list[BaseMessage]) -> str:
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

        Please classify the user message into one of these intents: {POSSIBLE_INTENTS}

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
    conversation = state.get("history", "") or format_conversation(messages)
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
    conversation = state.get("history", "") or format_conversation(messages)
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
    conversation = state.get("history", "") or format_conversation(messages)
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
    conversation = state.get("history", "") or format_conversation(messages)
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
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    prompt = f"""You are answering the user's question about the clinic. Consider the conversation history for context.

{SYSTEM_PROMPT}

Conversation history:
{conversation}

Latest user message: {user_message}

Provide a helpful and informative response about the clinic."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}


def _route(state: AgentState):
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


# ---------------------------------------------------------------------------
# Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------
_workflow = StateGraph(AgentState)
_workflow.add_node("intent", intent_classifier)
_workflow.add_node("agent", agent_node)
_workflow.add_node("book", book_node)
_workflow.add_node("cancel", cancel_node)
_workflow.add_node("reschedule", reschedule_node)
_workflow.add_node("question", question_node)

_workflow.set_entry_point("intent")
_workflow.add_conditional_edges(
    "intent",
    _route,
    {
        "book": "book",
        "cancel": "cancel",
        "reschedule": "reschedule",
        "question": "question",
        "agent": "agent",
    },
)
_workflow.add_edge("book", END)
_workflow.add_edge("cancel", END)
_workflow.add_edge("reschedule", END)
_workflow.add_edge("question", END)
_workflow.add_edge("agent", END)

graph = _workflow.compile()
