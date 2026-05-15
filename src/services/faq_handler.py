from loguru import logger

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.services.faq_knowledge_base import FAQ_KNOWLEDGE_BASE


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=50
)


class FAQHandler:
    """Handles FAQ queries using LLM-based semantic matching against knowledge base."""
    def __init__(self):
        self.knowledge_base = FAQ_KNOWLEDGE_BASE

    def get_answer(self, query: str) -> str | None:
        faq_topics = "\n".join(
            f"- {key}: {value}" for key, value in self.knowledge_base.items()
        )

        system_prompt = f"""You are a FAQ classifier. Given a user query and a list of FAQ topics, determine if the query matches any topic.

FAQ Topics:
{faq_topics}

Respond with ONLY the topic key if matched, or "NONE" if no topic matches.
Examples:
- Query: "How much does it cost?" -> consultation_cost
- Query: "Is it free?" -> consultation_cost
- Query: "What time do you open?" -> consultation_hours
- Query: "Where is the clinic?" -> consultation_location
- Query: "What's the meaning of life?" -> NONE"""

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {query}"),
            ])
            topic = response.content.strip().lower()

            if topic == "none" or topic not in self.knowledge_base:
                logger.debug("No FAQ match for query: {}", query)
                return None

            answer = self.knowledge_base.get(topic)
            logger.info("FAQ matched: topic={} for query: {}", topic, query)
            return answer

        except Exception as e:
            logger.error("FAQ LLM matching failed: {}", e)
            return None

    def format_response(self, answer: str) -> str:
        return f"Here's what I found:\n\n{answer}\n\nIs there anything else I can help you with?"