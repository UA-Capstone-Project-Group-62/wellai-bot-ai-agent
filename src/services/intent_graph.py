import time
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_groq import ChatGroq
from loguru import logger

from src.services.faq_knowledge_base import FAQ_KNOWLEDGE_BASE


# LangChain Groq wrapper
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=300
)


def _invoke_with_retry(messages, max_retries=5, base_delay=2):
    """Invoke LLM with exponential backoff retry for rate limits."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            if "rate_limit" in error_str.lower() or "429" in error_str:
                delay = base_delay * (2 ** attempt)
                logger.warning("Rate limited, retrying in {}s (attempt {}/{})", delay, attempt + 1, max_retries)
                time.sleep(delay)
            else:
                raise
    raise Exception(f"LLM rate limit exceeded after {max_retries} retries")

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
        return ""

    formatted = []
    for msg in messages[:-1]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)


def _language_instruction(user_message: str) -> str:
    """Return a strict language-matching instruction based on the user's message."""
    msg = user_message.strip()
    # Mandarin detection: CJK characters
    if any("\u4e00" <= ch <= "\u9fff" for ch in msg):
        return "CRITICAL: The user's message is in Mandarin Chinese. You MUST reply in Mandarin Chinese ONLY. Do NOT use any English or Malay words, phrases, or sentences in your response. Every word must be in Mandarin Chinese."
    # Malay detection: common Malay words / Latin script with Malay character
    malay_markers = ["saya", "awak", "kamu", "anda", "nak", "mahu", "boleh", "tak", "tidak", "berapa", "di mana", "apa", "yang", "untuk", "dengan", "dari", "ini", "itu", "kami", "kita", "mereka", "sini", "sana", "bila", "mana", "macam", "temu janji", "konsultasi", "yuran", "waktu", "operasi", "klinik", "bahasa"]
    lower_msg = msg.lower()
    import re
    if any(re.search(rf'\b{re.escape(m)}\b', lower_msg) for m in malay_markers):
        return "CRITICAL: The user's message is in Malay (Bahasa Melayu). You MUST reply in Malay ONLY. Do NOT use any English or Mandarin words, phrases, or sentences in your response. Every word must be in Malay."
    # Default to English — frame as identity/capability, not a rule
    return "CRITICAL: The user's message is in English. You MUST reply in English ONLY. Do NOT use any Malay or Mandarin words, phrases, or sentences in your response. Every word must be in English."


def intent_classifier(state: AgentState):
    user_message = state["messages"][-1].content

    system_content = f"""{SYSTEM_PROMPT}

Please classify the user message into one of these intents ONLY: {POSSIBLE_INTENTS}

IMPORTANT RULES:
- Classify based on the USER'S INTENT/ACTION, not the language they use.
- "book_app" = User explicitly wants to CREATE/MAKE a new appointment (e.g., "I want to book", "I need an appointment", "Saya nak buat temu janji", "我想预约", "Saya mahu temujanji")
- "cancel_app" = User wants to CANCEL an existing appointment (e.g., "I want to cancel", "Batal temu janji", "取消预约")
- "reschedule_app" = User wants to CHANGE/MOVE an existing appointment time/date (e.g., "Can I reschedule", "Boleh saya tukar tarikh temu janji", "我可以改预约吗", "tukar jadual", "ubah masa")
- "ask_question" = User is asking for INFORMATION about the clinic, hours, location, fees, services, etc. (e.g., "What are your hours?", "Where is your clinic?", "How much?", "Berapa yuran", "营业时间", "Di mana klinik", "berapa harga")
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
- "Boleh saya tukar tarikh temu janji?" -> reschedule_app
- "Saya mahu batal temu janji" -> cancel_app
- "我可以改预约吗?" -> reschedule_app
- "我想预约看诊" -> book_app
- "你们诊所在哪里？" -> ask_question
- "看诊费用是多少？" -> ask_question
- "Berapa harga konsultasi?" -> ask_question

Return ONLY the intent label, nothing else."""

    response = _invoke_with_retry([
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

    history_section = f"There is no previous conversation history." if not conversation else f"Conversation history:\n{conversation}"

    system_content = f"""{SYSTEM_PROMPT}

Continue the conversation naturally, addressing the user's latest message.

{history_section}

Provide a helpful response that continues the conversation naturally.

{_language_instruction(user_message)}"""

    response = _invoke_with_retry([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def book_node(state: AgentState):
    messages = state["messages"]
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    history_section = f"There is no previous conversation history." if not conversation else f"Conversation history:\n{conversation}"

    system_content = f"""{SYSTEM_PROMPT}

You are helping the user book an appointment.

{history_section}

Respond as a helpful booking assistant. If the user has already provided information (like preferred time or date), acknowledge it and ask for any missing details. Do not ask for information they have already given.

{_language_instruction(user_message)}"""

    response = _invoke_with_retry([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def cancel_node(state: AgentState):
    messages = state["messages"]
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    history_section = f"There is no previous conversation history." if not conversation else f"Conversation history:\n{conversation}"

    system_content = f"""{SYSTEM_PROMPT}

You are helping the user cancel an appointment.

{history_section}

Respond as a helpful booking assistant. Acknowledge any details they have provided and ask for only the missing information needed to process the cancellation.

{_language_instruction(user_message)}"""

    response = _invoke_with_retry([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def reschedule_node(state: AgentState):
    messages = state["messages"]
    conversation = state.get("history", "") or format_conversation(messages)
    user_message = messages[-1].content

    history_section = f"There is no previous conversation history." if not conversation else f"Conversation history:\n{conversation}"

    system_content = f"""{SYSTEM_PROMPT}

You are helping the user reschedule an appointment.

{history_section}

Respond as a helpful booking assistant. Acknowledge any details they have provided and ask for only the missing information needed to process the rescheduling.

{_language_instruction(user_message)}"""

    response = _invoke_with_retry([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])
    return {"messages": [response]}


def question_node(state: AgentState):
    messages = state["messages"]
    user_message = messages[-1].content

    language_keywords = [
        "what language", "what languages", "languages do you", "do you support",
        "language support", "can speak", "can communicate",
        "bahasa apa", "apa bahasa", "bahasa yang", "tahu bahasa",
        "什么语言", "支持什么语言", "会说", "会说中文"
    ]
    if any(kw in user_message.lower() for kw in language_keywords):
        logger.info("Language question detected: {}", user_message)
        return {"messages": [AIMessage(content="Our health clinic is committed to providing excellent care to patients from diverse backgrounds. Our staff, including our doctors and nurses, are multilingual and proficient in three languages:\n\n🇬🇧 English / 英文: We can communicate in English.\n🇲🇾 Bahasa Melayu / 马来文: Kami boleh berkomunikasi dalam Bahasa Melayu.\n🇨🇳 Mandarin Chinese / 中文 普通话: 我们可以用普通话交流。")]}

    conversation = state.get("history", "") or format_conversation(messages)

    history_section = f"There is no previous conversation history." if not conversation else f"Conversation history:\n{conversation}"

    faq_context = _format_faq_context()

    lang_instr = _language_instruction(user_message)

    system_content = f"""{SYSTEM_PROMPT}

You are answering the user's question about the clinic. Use the FAQ information below to provide accurate answers.

Available FAQ information:
{faq_context}

{history_section}

Provide a helpful and informative response about the clinic.

Example Q&A (follow the language of the question):
- "Where is your clinic located?" -> Answer in the same language as the question, referencing virtual consultations from the FAQ.

{lang_instr}"""

    response = _invoke_with_retry([
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
