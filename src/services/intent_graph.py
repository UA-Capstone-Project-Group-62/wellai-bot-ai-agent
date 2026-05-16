from typing import TypedDict, Annotated, NotRequired
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_groq import ChatGroq
from loguru import logger

from src.services.faq_handler import FAQHandler
from src.services.faq_knowledge_base import FAQ_KNOWLEDGE_BASE


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

    system_content = f"""{SYSTEM_PROMPT}

Please classify the user message into one of these intents ONLY: {POSSIBLE_INTENTS}

IMPORTANT RULES:
- "book_app" = User explicitly wants to CREATE/MAKE a new appointment (e.g., "I want to book", "I need an appointment", "Saya nak buat temu janji", "我想预约")
- "cancel_app" = User wants to CANCEL an existing appointment
- "reschedule_app" = User wants to CHANGE an existing appointment time/date
- "ask_question" = User is asking for INFORMATION about the clinic, hours, location, fees, services, etc. (e.g., "What are your hours?", "Where is your clinic?", "How much?", "Berapa yuran", "营业时间")
- "unrelated_to_your_job" = Everything else

EXAMPLES:
- "Hello, I want to book an appointment" -> book_app
- "What are your working hours?" -> ask_question
- "Where is your clinic located?" -> ask_question
- "How much is the consultation fee?" -> ask_question
- "Can I reschedule my appointment?" -> reschedule_app
- "I want to cancel my booking" -> cancel_app
- "Saya nak buat temu janji" -> book_app
- "Apakah waktu operasi anda?" -> ask_question
- "Berapa yuran konsultasi?" -> ask_question
- "我可以改预约吗?" -> reschedule_app
- "我想预约看诊" -> book_app

Return ONLY the intent label, nothing else."""

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=f"Message: {user_message}"),
    ])
    raw_intent = response.content.strip()
    intent = _parse_intent(raw_intent)
    logger.info("Detected intent: {} (raw: {})", intent, raw_intent)
    return {"intent": intent}


def _parse_intent(raw: str) -> str:
    """Extract and validate an intent label from LLM output.

    Handles surrounding quotes, extra punctuation, trailing explanations, etc.
    Falls back to 'unrelated_to_your_job' only when no known intent is found.
    """
    cleaned = raw.strip().strip('"').strip("'").lower()

    # If the cleaned string exactly matches a known intent, return it.
    for valid in POSSIBLE_INTENTS:
        if cleaned == valid.lower():
            return valid

    # Otherwise try substring matching — the model sometimes answers with a
    # full sentence that still contains the label.
    for valid in POSSIBLE_INTENTS:
        if valid.lower() in cleaned:
            return valid

    # No known intent found — default to the catch-all so routing still works.
    return "unrelated_to_your_job"


def agent_node(state: AgentState):
    messages = state["messages"]
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    system_content = f"""{SYSTEM_PROMPT}

Continue the conversation naturally, addressing the user's latest message while considering the conversation history.

Conversation history:
{conversation}

Provide a helpful response that continues the conversation naturally."""

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def book_node(state: AgentState):
    messages = state["messages"]
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    system_content = f"""{SYSTEM_PROMPT}

You are helping the user book an appointment. Consider the conversation history to understand what information has already been provided.

Conversation history:
{conversation}

Respond as a helpful booking assistant. If the user has already provided information (like preferred time or date), acknowledge it and ask for any missing details. Do not ask for information they have already given."""

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def cancel_node(state: AgentState):
    messages = state["messages"]
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    system_content = f"""{SYSTEM_PROMPT}

You are helping the user cancel an appointment. Consider the conversation history to understand what information has already been provided.

Conversation history:
{conversation}

Respond as a helpful booking assistant. Acknowledge any details they have provided and ask for only the missing information needed to process the cancellation."""

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def reschedule_node(state: AgentState):
    messages = state["messages"]
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    system_content = f"""{SYSTEM_PROMPT}

You are helping the user reschedule an appointment. Consider the conversation history to understand what information has already been provided.

Conversation history:
{conversation}

Respond as a helpful booking assistant. Acknowledge any details they have provided and ask for only the missing information needed to process the rescheduling."""

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def question_node(state: AgentState):
    messages = state["messages"]
    user_message = messages[-1].content

    language_keywords = ["language", "languages", "bahasa", "语言", "malay", "english", "mandarin", "chinese", "tahu", "faham", "support"]
    if any(kw in user_message.lower() for kw in language_keywords):
        logger.info("Language question detected: {}", user_message)
        return {"messages": [AIMessage(content="Our health clinic is committed to providing excellent care to patients from diverse backgrounds. Our staff, including our doctors and nurses, are multilingual and proficient in three languages: English, Malay (Bahasa Melayu), and Mandarin.")]}

    faq_handler = FAQHandler()
    faq_answer = faq_handler.get_answer(user_message)

    if faq_answer:
        logger.info("FAQ match found for question: {}", user_message)
        return {"messages": [AIMessage(content=faq_answer)]}

    conversation = state.get("history", "") or format_conversation(messages)

    system_content = f"""{SYSTEM_PROMPT}

You are answering the user's question about the clinic. Consider the conversation history for context.

Available FAQ information:
{_format_faq_context()}

Conversation history:
{conversation}

Provide a helpful and informative response about the clinic."""

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def _format_faq_context() -> str:
    faq_lines = []
    for key, value in FAQ_KNOWLEDGE_BASE.items():
        faq_lines.append(f"- {key}: {value}")
    return "\n".join(faq_lines)


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
