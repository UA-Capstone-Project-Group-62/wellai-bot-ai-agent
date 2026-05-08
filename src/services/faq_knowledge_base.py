# src/services/faq_knowledge_base.py
# Quick and dirty FAQ stuff for now - will improve after Wai Kok replies

FAQ_KNOWLEDGE_BASE = {
    # src/services/faq_knowledge_base.py
# Quick and dirty FAQ stuff for now - will improve after Wai Kok replies

FAQ_KNOWLEDGE_BASE = {
    # General Clinic Info ALL OF THIS BTW IS JUST TEMPLATES BC THE CLIENT REFUSES TO ACTUALLY READ MY EMAILS TO GIVE ME REAL INFO
    "opening_hours": "Our clinics are usually open Mon-Fri 9AM-6PM, Sat 9AM-1PM. Closed Sunday & public holidays.",
    
    "location": "We have a few clinics around Malaysia. Which one are you looking for? (Damansara, KL, Penang etc)",
    
    "appointment_duration": "Normal appointment is about 15-30 minutes.",
    
    # New Patients & Documents
    "new_patient": "New patients are welcome! Just tell me which clinic and we can help book your first appointment.",
    "bring_documents": "Please bring your IC/passport and any previous medical records if you have them.",
    
    # Payment & Insurance
    "payment": "We accept cash, card, and most major insurance. Please check with the clinic for details.",
    "insurance": "We work with many insurance providers. Please tell me your insurance company and I can help check.",
    
    # Booking & Cancellation
    "cancellation": "You can cancel or reschedule up to 24 hours before your appointment without charge.",
    "telemedicine": "Yes, some doctors offer online/video consultations. Would you like me to check availability?",
    
    # Other Common Questions
    "wait_time": "Wait times vary but we try to keep them as short as possible. Would you like me to book an appointment for you?",
    "doctor_list": "We have many specialists. Please tell me what kind of doctor you need (e.g. general, dermatologist, etc).",
    "child": "Yes, we see children. Please mention the child's age when booking.",
    "emergency": "For emergencies please go to the nearest hospital. We are not an emergency clinic.",
}
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

You have some basic FAQ knowledge. Use it if the user asks about opening hours, location, appointment length, documents, payment, cancellation, telemedicine, insurance, etc.
If you don't know the answer, just say you don't know and offer to book an appointment or talk to a human."""