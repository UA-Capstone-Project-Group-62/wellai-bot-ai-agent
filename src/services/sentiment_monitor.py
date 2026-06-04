from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger


class SentimentCategory(str, Enum):
    SAFE = "safe"
    PROFANITY = "profanity"
    DISTRESS = "distress"
    SELF_HARM = "self_harm"
    HARMFUL_INTENT = "harmful_intent"


class DetectionSource(str, Enum):
    RULES = "rules"
    LLM = "llm"
    LLM_ERROR = "llm_error"


@dataclass(frozen=True)
class SentimentResult:
    category: SentimentCategory
    source: DetectionSource
    reason: str
    language: str

    @property
    def should_escalate(self) -> bool:
        return self.category != SentimentCategory.SAFE


@dataclass(frozen=True)
class _Rule:
    language: str
    phrase: str
    reason: str


_VALID_LANGUAGES = {"english", "malay", "mandarin"}

_MALAY_MARKERS = [
    "saya",
    "awak",
    "kamu",
    "anda",
    "nak",
    "mahu",
    "boleh",
    "tak",
    "tidak",
    "berapa",
    "di mana",
    "apa",
    "yang",
    "untuk",
    "dengan",
    "dari",
    "ini",
    "itu",
    "kami",
    "kita",
    "mereka",
    "bila",
    "temu janji",
    "konsultasi",
    "klinik",
    "bahasa",
    "tertekan",
    "putus asa",
]

_RULES_BY_CATEGORY: dict[SentimentCategory, list[_Rule]] = {
    SentimentCategory.SELF_HARM: [
        _Rule("english", "kill myself", "direct self-harm phrase"),
        _Rule("english", "end my life", "direct self-harm phrase"),
        _Rule("english", "want to die", "direct self-harm phrase"),
        _Rule("english", "i want to die", "direct self-harm phrase"),
        _Rule("english", "hurt myself", "self-harm phrase"),
        _Rule("english", "self harm", "self-harm phrase"),
        _Rule("english", "suicide", "suicide-related phrase"),
        _Rule("english", "suicidal", "suicide-related phrase"),
        _Rule("english", "overdose", "overdose self-harm phrase"),
        _Rule("english", "wanna overdose", "overdose self-harm phrase"),
        _Rule("english", "want to overdose", "overdose self-harm phrase"),
        _Rule("english", "od on", "overdose self-harm phrase"),
        _Rule("english", "wanna od", "overdose self-harm phrase"),
        _Rule("english", "want to od", "overdose self-harm phrase"),
        _Rule("malay", "bunuh diri", "suicide-related phrase"),
        _Rule("malay", "nak mati", "direct self-harm phrase"),
        _Rule("malay", "mahu mati", "direct self-harm phrase"),
        _Rule("malay", "ingin mati", "direct self-harm phrase"),
        _Rule("malay", "saya mahu mati", "direct self-harm phrase"),
        _Rule("malay", "cederakan diri", "self-harm phrase"),
        _Rule("malay", "mencederakan diri", "self-harm phrase"),
        _Rule("malay", "sakiti diri", "self-harm phrase"),
        _Rule("mandarin", "自杀", "suicide-related phrase"),
        _Rule("mandarin", "想死", "direct self-harm phrase"),
        _Rule("mandarin", "我想死", "direct self-harm phrase"),
        _Rule("mandarin", "不想活", "direct self-harm phrase"),
        _Rule("mandarin", "结束生命", "direct self-harm phrase"),
        _Rule("mandarin", "伤害自己", "self-harm phrase"),
        _Rule("mandarin", "活不下去", "direct self-harm phrase"),
    ],
    SentimentCategory.DISTRESS: [
        _Rule("english", "extremely depressed", "severe distress phrase"),
        _Rule("english", "severely depressed", "severe distress phrase"),
        _Rule("english", "deeply depressed", "severe distress phrase"),
        _Rule("english", "hopeless", "severe distress phrase"),
        _Rule("english", "no reason to live", "severe distress phrase"),
        _Rule("english", "can't go on", "severe distress phrase"),
        _Rule("english", "cannot go on", "severe distress phrase"),
        _Rule("english", "give up on life", "severe distress phrase"),
        _Rule("malay", "sangat tertekan", "severe distress phrase"),
        _Rule("malay", "terlalu tertekan", "severe distress phrase"),
        _Rule("malay", "sangat murung", "severe distress phrase"),
        _Rule("malay", "putus asa", "severe distress phrase"),
        _Rule("malay", "tiada harapan", "severe distress phrase"),
        _Rule("malay", "tak sanggup lagi", "severe distress phrase"),
        _Rule("mandarin", "非常抑郁", "severe distress phrase"),
        _Rule("mandarin", "很抑郁", "severe distress phrase"),
        _Rule("mandarin", "绝望", "severe distress phrase"),
        _Rule("mandarin", "没有希望", "severe distress phrase"),
        _Rule("mandarin", "撑不下去", "severe distress phrase"),
        _Rule("mandarin", "崩溃", "severe distress phrase"),
    ],
    SentimentCategory.PROFANITY: [
        _Rule("english", "fuck", "profanity"),
        _Rule("english", "shit", "profanity"),
        _Rule("english", "bullshit", "profanity"),
        _Rule("english", "bitch", "profanity"),
        _Rule("english", "asshole", "profanity"),
        _Rule("english", "bastard", "profanity"),
        _Rule("malay", "sial", "profanity"),
        _Rule("malay", "babi", "profanity"),
        _Rule("malay", "bangsat", "profanity"),
        _Rule("malay", "celaka", "profanity"),
        _Rule("malay", "pukimak", "profanity"),
        _Rule("mandarin", "他妈的", "profanity"),
        _Rule("mandarin", "操你", "profanity"),
        _Rule("mandarin", "傻逼", "profanity"),
        _Rule("mandarin", "混蛋", "profanity"),
        _Rule("mandarin", "王八蛋", "profanity"),
        _Rule("mandarin", "贱人", "profanity"),
    ],
}

_CLINICAL_BOOKING_MARKERS = [
    "appointment",
    "appoinment",
    "appoint",
    "book",
    "booking",
    "refill",
    "prescription",
    "medication",
    "medicine",
    "doctor",
    "consultation",
    "temu janji",
    "ubat",
    "konsultasi",
    "预约",
    "药",
    "药物",
    "处方",
    "复诊",
]

_SEMANTIC_REVIEW_MARKERS = [
    "make",
    "force",
    "harm",
    "hurt",
    "attack",
    "threat",
    "threaten",
    "revenge",
    "retaliate",
    "punish",
    "take his own medicine",
    "take her own medicine",
    "take their own medicine",
    "teach him a lesson",
    "teach her a lesson",
    "teach them a lesson",
]

_ESCALATION_REPLIES = {
    "english": (
        "I recognise that you may be distressed. I have requested a human "
        "operator to take over this conversation and support you."
    ),
    "malay": (
        "Saya faham bahawa anda mungkin sedang tertekan. Saya telah meminta "
        "operator manusia untuk mengambil alih perbualan ini dan membantu anda."
    ),
    "mandarin": "我能感受到你可能正在困扰中。我已经请求人工客服接手这段对话并为你提供帮助。",
}

_CLASSIFIER_PROMPT = """
You are a safety and sentiment monitoring classifier for a multilingual health clinic WhatsApp bot.

Classify the user's latest message into exactly one category:
- safe: no escalation needed.
- profanity: swear words, abusive insults, or hostile profanity.
- distress: severe depression, extreme emotional distress, hopelessness, or crisis language that is not clearly self-harm.
- self_harm: suicidal thoughts, wanting to die, self-harm intent, or direct risk to the user's life.
- harmful_intent: threats, coercion, retaliation, or hostile intent toward another person, including doctors, staff, clinic workers, or other patients.

Do not classify a message as distress only because it mentions a diagnosis, condition, medication, or treatment. If the user is asking to book, refill medication, discuss treatment, or arrange care without crisis language, classify it as safe.
If a booking or appointment request includes hostile, coercive, threatening, or retaliatory meaning, classify it as harmful_intent. In a medical context, phrases like "make my doctor take his own medicine" can indicate hostile intent toward the doctor and should not be treated as a normal booking request.

Supported languages are English, Malay/Bahasa Melayu, and Mandarin Chinese. Detect meaning across these languages.

Return JSON only with this exact shape:
{"category":"safe|profanity|distress|self_harm|harmful_intent","language":"english|malay|mandarin","reason":"short reason"}
"""


class SentimentMonitor:
    def __init__(self, llm_client=None):
        self._llm_client = llm_client

    def evaluate(self, message: str) -> SentimentResult:
        content = (message or "").strip()
        language = detect_language(content)
        if not content:
            return SentimentResult(
                category=SentimentCategory.SAFE,
                source=DetectionSource.RULES,
                reason="empty message",
                language=language,
            )

        rule_result = self._evaluate_rules(content)
        if rule_result.should_escalate:
            return rule_result
        if _is_clinical_booking_context(content):
            if _needs_semantic_safety_review(content):
                return self._evaluate_with_llm(content, language)
            return SentimentResult(
                category=SentimentCategory.SAFE,
                source=DetectionSource.RULES,
                reason="clinical booking or medication context",
                language=language,
            )

        return self._evaluate_with_llm(content, language)

    def escalation_reply(self, language: str) -> str:
        return _ESCALATION_REPLIES.get(language, _ESCALATION_REPLIES["english"])

    def _evaluate_rules(self, message: str) -> SentimentResult:
        language = detect_language(message)
        for category in (
            SentimentCategory.SELF_HARM,
            SentimentCategory.DISTRESS,
            SentimentCategory.PROFANITY,
        ):
            for rule in _RULES_BY_CATEGORY[category]:
                if _matches_rule(message, rule):
                    return SentimentResult(
                        category=category,
                        source=DetectionSource.RULES,
                        reason=f"{rule.reason}: {rule.phrase}",
                        language=rule.language,
                    )

        return SentimentResult(
            category=SentimentCategory.SAFE,
            source=DetectionSource.RULES,
            reason="no rule matched",
            language=language,
        )

    def _evaluate_with_llm(self, message: str, fallback_language: str) -> SentimentResult:
        try:
            response = self._invoke_with_retry(
                [
                    SystemMessage(content=_CLASSIFIER_PROMPT),
                    HumanMessage(content=f"Message: {message}"),
                ],
            )
        except Exception as error:
            logger.warning("Sentiment LLM classification failed: {}", error)
            return SentimentResult(
                category=SentimentCategory.SAFE,
                source=DetectionSource.LLM_ERROR,
                reason="LLM classification failed; no rule matched",
                language=fallback_language,
            )

        return _parse_llm_response(response.content, fallback_language)

    def _invoke_with_retry(self, messages, max_retries: int = 3, base_delay: int = 1):
        for attempt in range(max_retries):
            try:
                return self._get_llm().invoke(messages)
            except Exception as error:
                error_str = str(error)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Sentiment LLM rate limited, retrying in {}s (attempt {}/{})",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise
        raise Exception(f"Sentiment LLM rate limit exceeded after {max_retries} retries")

    def _get_llm(self):
        if self._llm_client is None:
            self._llm_client = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=120,
            )
        return self._llm_client


def detect_language(message: str) -> str:
    content = (message or "").strip()
    if any("\u4e00" <= ch <= "\u9fff" for ch in content):
        return "mandarin"

    lower_content = content.lower()
    if any(
        re.search(rf"\b{re.escape(marker)}\b", lower_content)
        for marker in _MALAY_MARKERS
    ):
        return "malay"

    return "english"


def _matches_rule(message: str, rule: _Rule) -> bool:
    if rule.language == "mandarin":
        return rule.phrase in message

    normalized = re.sub(r"\s+", " ", message.lower())
    if rule.phrase == "od on":
        return re.search(r"(?<![a-z0-9_])o\.?d\.?\s+on(?![a-z0-9_])", normalized) is not None
    if rule.phrase == "wanna od":
        return re.search(r"(?<![a-z0-9_])wanna\s+o\.?d\.?(?![a-z0-9_])", normalized) is not None
    if rule.phrase == "want to od":
        return re.search(r"(?<![a-z0-9_])want\s+to\s+o\.?d\.?(?![a-z0-9_])", normalized) is not None

    phrase_parts = [re.escape(part) for part in rule.phrase.lower().split()]
    phrase_pattern = r"\s+".join(phrase_parts)
    pattern = rf"(?<![a-z0-9_]){phrase_pattern}(?![a-z0-9_])"
    return re.search(pattern, normalized) is not None


def _is_clinical_booking_context(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.lower())
    return any(marker in normalized for marker in _CLINICAL_BOOKING_MARKERS)


def _needs_semantic_safety_review(message: str) -> bool:
    normalized = re.sub(r"\s+", " ", message.lower())
    return any(marker in normalized for marker in _SEMANTIC_REVIEW_MARKERS)


def _parse_llm_response(raw_content: str, fallback_language: str) -> SentimentResult:
    payload = _loads_json_object(raw_content)

    raw_category = str(payload.get("category", "safe")).strip().lower().replace("-", "_")
    category = (
        SentimentCategory(raw_category)
        if raw_category in {item.value for item in SentimentCategory}
        else SentimentCategory.SAFE
    )

    language = str(payload.get("language", fallback_language)).strip().lower()
    if language not in _VALID_LANGUAGES:
        language = fallback_language

    reason = str(payload.get("reason", "LLM classification")).strip()
    if not reason:
        reason = "LLM classification"

    return SentimentResult(
        category=category,
        source=DetectionSource.LLM,
        reason=reason,
        language=language,
    )


def _loads_json_object(raw_content: str) -> dict:
    content = (raw_content or "").strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}


sentiment_monitor = SentimentMonitor()
