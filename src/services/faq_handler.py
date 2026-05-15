from loguru import logger

from src.services.faq_knowledge_base import FAQ_KNOWLEDGE_BASE


class FAQHandler:
    """Handles FAQ queries with keyword-based matching against knowledge base."""
    def __init__(self):
        self.knowledge_base = FAQ_KNOWLEDGE_BASE

    def get_answer(self, query: str) -> str | None:
        query_lower = query.lower()
        
        keyword_phrases = [
            (["health report", "my report", "view report", "check report"], "report_info"),
            (["reschedule"], "reschedule_cancel"),
            (["cancel appointment", "cancel my appointment", "change appointment", "want to cancel"], "reschedule_cancel"),
            (["make an appointment", "make appointment", "i want to book"], "make_appointment"),
            (["schedule appointment", "consultation appointment"], "make_appointment"),
            (["book appointment"], "make_appointment"),
            (["consultation cost", "much does it cost", "cost of", "price", "charge", "is it free", "free"], "consultation_cost"),
            (["consultation hours", "appointment hours", "opening hours", "when do you open", "when are you open", "what time do you open"], "consultation_hours"),
            (["consultation virtual", "virtual consultation", "online consultation"], "consultation_method"),
            (["consultation location", "where are you located", "clinic location", "where is the consultation", "consultation location"], "consultation_location"),
            (["consultation platform", "platform", "google meet", "zoom"], "consultation_platform"),
            (["consultation duration", "how long is", "session length"], "consultation_duration"),
            (["data secure", "privacy", "data safe", "secure"], "data_security"),
        ]
        
        matches = []
        for phrases, key in keyword_phrases:
            for phrase in phrases:
                if phrase in query_lower:
                    score = len(phrase)
                    matches.append((score, key))
        
        if matches:
            best_match = max(matches, key=lambda x: x[0])
            logger.debug("FAQ matched: key={}, score={}", best_match[1], best_match[0])
            return self.knowledge_base.get(best_match[1])
        
        return None

    def format_response(self, answer: str) -> str:
        return f"Here's what I found:\n\n{answer}\n\nIs there anything else I can help you with?"