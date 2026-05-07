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

def intent_classifier(state: AgentState):
    # classify what the user's intent is

    user_message = state["messages"][-1].content

    prompt = f"""
        {SYSTEM_PROMPT}

        Please classify the user message into one of these intents: {possible_intents}

        The intent is "book_app" if the user wants to make a booking, make a new appointment, or
        other similar requests.

        The intent is "cancel_app" if the user wants to cancel a booking, cancel an appointment, or
        other similar requests.

        The intent is "reschedule_app" if the user says they want to change the time of an appointment they already have,
        change the date of their booking, or change the medical professional they are seeing. 

        The intent is "ask_question" if the user is asking for more information about the clinic, booking process,
        or similar topics. This DOES NOT include questions about topics that are unrelated to the medical clinic,
        you (the booking assistant), or the medical professionals they are able to book appointments with.

        The intent is "unrelated_to_your_job" if the user has a request that is anything else.

        Return ONLY the intent label.

        Message: {user_message}
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    intent = response.content.strip()

    logger.info("Detected intent: {}", intent) # for debugging purposes

    return {"intent": intent}

def agent_node(state: AgentState):
    # reroutes according to updated user intent 

    intent = state["intent"]
    messages = state["messages"]
    prompt = f"{SYSTEM_PROMPT}. The user's intent is {intent}. Please respond appropriately."
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def book_node(state: AgentState):
    messages = state["messages"]
    prompt = """
        The user wants to book an appointment with a clinic. 
    """
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def cancel_node(state: AgentState):
    messages = state["messages"]
    prompt = """
        The user wants to cancel an appointment with a clinic. 
    """
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}


def reschedule_node(state: AgentState):
    messages = state["messages"]
    prompt = """
        The user wants to reschedule an appointment with a clinic. 
    """
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [response]}

def question_node(state: AgentState):
    messages = state["messages"]
    prompt = """
        The user wants to ask a question about the clinic. 
    """
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
    # route the agent depending on what the user's intent is

    if state["intent"] is "book_app":
        return "book"

    elif state["intent"] is "cancel_app":
        return "cancel"

    elif state["intent"] is "reschedule_app":
        return "reschedule"
    
    elif state["intent"] is "ask_question":
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

    def Receive(self, request, context):
        logger.info(
            "Received message from user. user_id={}, content_length={}",
            request.user_id,
            len(request.content),
        )

        try:
            result = graph.invoke({"messages": [HumanMessage(content=request.content)]})
            ai_reply = result["messages"][-1].content.strip()
            logger.info("AI replied successfully")

        except Exception as e:
            logger.error(f"LangGraph error: {e}")
            ai_reply = "Sorry, I'm having trouble right now. Please try again."

        return common_pb2.Response(
            success=True,
            message=ai_reply
        )