from src.services.faq_knowledge_base import FAQ_KNOWLEDGE_BASE


class FAQHandler:
    def __init__(self):
        self.knowledge_base = FAQ_KNOWLEDGE_BASE

    def get_answer(self, query: str) -> str | None:
        query_lower = query.lower()
        
        keyword_phrases = [
            (["health report", "report"], "report_info"),
            (["book", "make", "schedule", "consultation appointment"], "make_appointment"),
            (["consultation cost", "cost", "free", "price", "charge"], "consultation_cost"),
            (["consultation hours", "appointment hours", "available", "hours"], "consultation_hours"),
            (["consultation virtual", "virtual consultation", "online consultation"], "consultation_method"),
            (["consultation location", "where", "location"], "consultation_location"),
            (["consultation platform", "platform", "google meet", "zoom"], "consultation_platform"),
            (["consultation duration", "how long", "session length", "hour"], "consultation_duration"),
            (["reschedule", "cancel appointment", "cancel", "change appointment"], "reschedule_cancel"),
            (["data secure", "privacy", "data safe", "secure"], "data_security"),
        ]
        
        for phrases, key in keyword_phrases:
            for phrase in phrases:
                if phrase in query_lower:
                    return self.knowledge_base.get(key)
        
        return None

    def format_response(self, answer: str) -> str:
        return f"Here's what I found:\n\n{answer}\n\nIs there anything else I can help you with?"