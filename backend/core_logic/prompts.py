# backend/core_logic/prompts.py
"""
Safety and regulatory response templates for the pharmaceutical assistant.
"""

AE_ROUTE_TEMPLATE = (
    "I cannot provide individualized medical guidance. Because you mentioned a possible adverse event, "
    "the safest next step is to contact your healthcare provider. You can also report side effects to FDA MedWatch "
    "(1-800-332-1088) or at www.fda.gov/medwatch. If this is an emergency, call 911."
)

PHI_SANITIZE_NOTE = (
    "For your privacy, please avoid sharing personal information such as your address, phone number, or email."
)

ILLICIT_REFUSAL = "I cannot provide information about illegal or violent activities."

SELF_HARM_REFUSAL = (
    "I'm really concerned for your safety. If you are thinking about harming yourself, "
    "please call 988 in the U.S. or your local emergency number right away."
)

JAILBREAK_REFUSAL = "I cannot ignore safety rules or pretend to act outside approved medical guidance."

OFFLABEL_REFUSAL = (
    "I can only provide information about approved uses of this medicine, "
    "not unapproved or investigational uses."
)

MEDICAL_ADVICE_REFUSAL = (
    "I can't provide individualized medical advice. "
    "For guidance specific to you, please contact your healthcare provider."
)

NO_CONTEXT_FALLBACK = (
    "I apologize, I don't seem to have that information in the approved documentation. "
    "Can I assist you with something else?"
)