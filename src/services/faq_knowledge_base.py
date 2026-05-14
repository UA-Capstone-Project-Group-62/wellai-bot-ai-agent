# src/services/faq_knowledge_base.py

# Note: Currently using a common knowledge base.
# TODO: Make this clinic-specific once client gives us per-clinic FAQs

FAQ_KNOWLEDGE_BASE = {
    "report_info": "You can make an online consultation appointment with our health consultation team in order to receive further analysis on your health report.",

    "make_appointment": "You can make appointment using the online appointment option with us here. You can select or inform us your preferred appointment date and time. One day before the appointment, our consultant will provide you the link for the consultation.",

    "consultation_cost": "The cost of consultation is free.",

    "consultation_hours": "The hours are by appointment only. Customers can indicate their availability or preferred appointment time, and the consultant can contact them for appointment.",

    "consultation_method": "Virtually.",

    "consultation_location": "All consultations are performed virtually.",

    "consultation_platform": "Google Meet or Zoom.",

    "consultation_duration": "Each consultation session is one hour. Should there be further questions from the customer, the consultation session may be extended for another thirty minutes.",

    "reschedule_cancel": "Yes. You can do that with us here.",

    "data_security": "Yes, your data is completely secure. We use the high-level protection to ensure your information stays private. We will never share your personal information with any third party except our own health consultant team.",
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

You have some basic FAQ knowledge. Use it if the user asks about reports, appointments, consultation cost, hours, location, platforms (Google Meet, Zoom), session duration, rescheduling, cancellation, or data security.
If you don't know the answer, just say you don't know and offer to book an appointment or talk to a human."""



def set_clinic_faqs(clinic_name: str):
    """Placeholder for future clinic-specific FAQs"""
    print(f"TODO: Load FAQ for clinic: {clinic_name}")
    # Will be expanded when client replies