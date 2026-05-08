# src/services/faq_knowledge_base.py
# Quick and dirty FAQ stuff for now - will improve after Wai Kok replies

FAQ_KNOWLEDGE_BASE = {
    # General stuff
    "opening_hours": "Our clinics are usually open Mon-Fri 9AM-6PM, Sat 9AM-1PM. Closed Sunday & public holidays.",
    
    "location": "We have a few clinics around Malaysia. Which one are you looking for? (Damansara, KL, Penang etc)",
    
    "appointment_duration": "Normal appointment is about 15-30 minutes.",
    
    # TODO: add more when we get info from client
}

def get_faq_answer(query):
    """rough keyword search"""
    q = query.lower()
    
    for key, answer in FAQ_KNOWLEDGE_BASE.items():
        if any(word in q for word in key.replace("_", " ").split()):
            return answer
            
    return None


def get_system_prompt_with_faq():
    """adds faq instructions to the prompt"""
    return """You are a helpful WhatsApp booking assistant for WellAI clinics in Malaysia.
You can speak English, Malay, and Mandarin. Be polite and professional. Never give medical advice.

You have some basic FAQ knowledge. Use it if the user asks about opening hours, location, appointment length, etc.
If you don't know the answer, just say you don't know and offer to book an appointment or talk to a human."""