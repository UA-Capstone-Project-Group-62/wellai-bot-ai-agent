from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from loguru import logger



class LanguageCategory(str, Enum):
    ENGLISH = "english"
    MALAY = "malay"
    MANDARIN = "mandarin"
    UNSUPPORTED = "unsupported"


class LanguageDetectionSource(str, Enum):
    RULES = "rules"
    LLM = "llm"
    LLM_ERROR = "llm_error"


@dataclass(frozen=True)
class LanguageResult:
    category: LanguageCategory
    source: LanguageDetectionSource
    reason: str

    @property
    def is_supported(self) -> bool:
        return self.category != LanguageCategory.UNSUPPORTED


UNSUPPORTED_LANGUAGE_REPLY = (
    "I can keep communicating in your current language, but I am optimised for "
    "English, Bahasa Melayu, and Mandarin Chinese. Our clinic staff is "
    "proficient in these three main languages."
)

_MALAY_MARKERS = [
    "saya",
    "awak",
    "kamu",
    "anda",
    "nak",
    "mahu",
    "boleh",
    "tidak",
    "berapa",
    "di mana",
    "temu janji",
    "konsultasi",
    "klinik",
    "bahasa",
]

_COMMON_ENGLISH_MARKERS = [
    "hello",
    "hi",
    "what",
    "where",
    "when",
    "how",
    "can",
    "could",
    "would",
    "appointment",
    "book",
    "cancel",
    "reschedule",
    "clinic",
    "language",
    "speak",
    "support",
    "know",
    "hours",
    "fee",
    "cost",
]

_LANGUAGE_PROMPT = """
You are a language detector for a multilingual health clinic WhatsApp bot.

Classify the user's latest message into exactly one category:
- english
- malay
- mandarin
- unsupported

Use "unsupported" for any language other than English, Bahasa Melayu/Malay, or Mandarin Chinese.

Return JSON only with this exact shape:
{"language":"english|malay|mandarin|unsupported","reason":"short reason"}
"""


class LanguageMonitor:
    def __init__(self, llm_client=None):
        self._llm_client = llm_client

    def evaluate(self, message: str) -> LanguageResult:
        content = (message or "").strip()
        if not content:
            return LanguageResult(
                category=LanguageCategory.ENGLISH,
                source=LanguageDetectionSource.RULES,
                reason="empty message defaults to english",
            )

        rule_result = self._evaluate_rules(content)
        if rule_result is not None:
            return rule_result

        return self._evaluate_with_llm(content)

    def unsupported_reply(self) -> str:
        return UNSUPPORTED_LANGUAGE_REPLY

    def _evaluate_rules(self, message: str) -> LanguageResult | None:
        if any("\u4e00" <= ch <= "\u9fff" for ch in message):
            return LanguageResult(
                category=LanguageCategory.MANDARIN,
                source=LanguageDetectionSource.RULES,
                reason="CJK character detected",
            )

        lower_message = message.lower()
        if any(
            re.search(rf"\b{re.escape(marker)}\b", lower_message)
            for marker in _MALAY_MARKERS
        ):
            return LanguageResult(
                category=LanguageCategory.MALAY,
                source=LanguageDetectionSource.RULES,
                reason="Malay marker detected",
            )

        if any(
            re.search(rf"\b{re.escape(marker)}\b", lower_message)
            for marker in _COMMON_ENGLISH_MARKERS
        ):
            return LanguageResult(
                category=LanguageCategory.ENGLISH,
                source=LanguageDetectionSource.RULES,
                reason="English marker detected",
            )

        return None

    def _evaluate_with_llm(self, message: str) -> LanguageResult:
        try:
            response = self._invoke_with_retry(
                [
                    SystemMessage(content=_LANGUAGE_PROMPT),
                    HumanMessage(content=f"Message: {message}"),
                ],
            )
        except Exception as error:
            logger.warning("Language LLM classification failed: {}", error)
            return LanguageResult(
                category=LanguageCategory.ENGLISH,
                source=LanguageDetectionSource.LLM_ERROR,
                reason="LLM classification failed; defaulting to english",
            )

        return _parse_llm_response(response.content)

    def _invoke_with_retry(self, messages, max_retries: int = 3, base_delay: int = 1):
        for attempt in range(max_retries):
            try:
                return self._get_llm().invoke(messages)
            except Exception as error:
                error_str = str(error)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Language LLM rate limited, retrying in {}s (attempt {}/{})",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                else:
                    raise
        raise Exception(f"Language LLM rate limit exceeded after {max_retries} retries")

    def _get_llm(self):
        if self._llm_client is None:
            self._llm_client = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=80,
            )
        return self._llm_client


def _parse_llm_response(raw_content: str) -> LanguageResult:
    payload = _loads_json_object(raw_content)
    raw_language = str(payload.get("language", "english")).strip().lower()
    language = (
        LanguageCategory(raw_language)
        if raw_language in {item.value for item in LanguageCategory}
        else LanguageCategory.ENGLISH
    )

    reason = str(payload.get("reason", "LLM language classification")).strip()
    if not reason:
        reason = "LLM language classification"

    return LanguageResult(
        category=language,
        source=LanguageDetectionSource.LLM,
        reason=reason,
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


language_monitor = LanguageMonitor()
