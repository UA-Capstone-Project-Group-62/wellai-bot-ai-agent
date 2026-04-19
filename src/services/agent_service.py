import grpc
import os
from loguru import logger
from proto.agent import agent_pb2_grpc
from proto.common import common_pb2

from src.clients.bot_client import BotClient

# LangGraph + LangChain imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
import operator

# LangChain Groq wrapper
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=300
)

SYSTEM_PROMPT = """You are a helpful, polite, and professional WhatsApp booking assistant for WellAI clinics in Malaysia.
You support English, Malay, and Mandarin.
Always be respectful, clear, and neutral. Never give medical advice.
Help patients book, reschedule, cancel appointments, or answer FAQs."""

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm.invoke([HumanMessage(content=SYSTEM_PROMPT)] + messages)
    return {"messages": [response]}

# Build simple LangGraph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.set_entry_point("agent")
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