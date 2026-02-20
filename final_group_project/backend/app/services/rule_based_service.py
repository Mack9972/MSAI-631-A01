from __future__ import annotations

import datetime as dt
import re
from typing import Optional


INTENTS = {
    "greeting": [
        r"\bhello\b",
        r"\bhi\b",
        r"\bhey\b",
        r"\bgood morning\b",
        r"\bgood afternoon\b",
        r"\bgood evening\b",
    ],
    "goodbye": [
        r"\bbye\b",
        r"\bgoodbye\b",
        r"\bsee you\b",
        r"\bexit\b",
        r"\bquit\b",
    ],
    "thanks": [
        r"\bthanks\b",
        r"\bthank you\b",
        r"\bappreciate it\b",
    ],
    "help": [
        r"\bhelp\b",
        r"\bwhat can you do\b",
        r"\boptions\b",
        r"\bmenu\b",
    ],
    "capabilities": [
        r"\bfeatures\b",
        r"\bcapabilities\b",
        r"\bfunctions\b",
    ],
    "topic_definition": [
        r"\bdefine\b",
        r"\bwhat is\b",
        r"\bexplain\b",
    ],
    "time": [
        r"\btime\b",
        r"\bdate\b",
        r"\btoday\b",
    ],
}

TOPIC_KB = {
    "chatbot": "A chatbot is a software program that simulates conversation using rules, patterns, or machine learning.",
    "rule-based": "Rule-based systems use predefined patterns and responses to handle user inputs.",
    "nlp": "Natural language processing (NLP) helps computers work with human language.",
    "intent": "An intent is the user's goal, inferred from the text they input.",
    "entity": "An entity is a key piece of information extracted from a user message, like a name or date.",
}

TOPIC_ALIASES = {
    "chatbot": ["chat bot", "assistant", "virtual assistant"],
    "rule-based": ["rule based", "rules engine", "pattern matching"],
    "nlp": ["natural language processing", "language processing"],
    "intent": ["user intent", "intent classification"],
    "entity": ["named entity", "entity extraction"],
}

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "have", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please", "the",
    "this", "to", "what", "when", "where", "who", "why", "you", "your",
}


class RuleBasedService:
    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in TOKEN_RE.findall(text.lower()) if token]

    def _match_intent(self, text: str) -> Optional[str]:
        for intent, patterns in INTENTS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        return None

    def _detect_topic(self, text: str) -> Optional[str]:
        normalized = self._normalize(text)
        tokens = set(self._tokenize(normalized)) - STOPWORDS
        best_topic = None
        best_score = 0.0

        for topic in TOPIC_KB:
            if re.search(rf"\b{re.escape(topic)}\b", normalized):
                return topic

            alias_tokens = []
            for alias in TOPIC_ALIASES.get(topic, []):
                alias_tokens.extend(self._tokenize(alias))
            candidate_tokens = set(self._tokenize(topic)) | set(alias_tokens)
            if not candidate_tokens:
                continue
            score = len(tokens & candidate_tokens) / max(1, len(candidate_tokens))
            if score > best_score:
                best_score = score
                best_topic = topic

        if best_score >= 0.55:
            return best_topic
        return None

    def respond(self, text: str) -> Optional[str]:
        normalized = self._normalize(text)

        intent = self._match_intent(normalized)
        if intent == "greeting":
            return "Hello! How can I help today?"
        if intent == "goodbye":
            return "Goodbye! Have a great day."
        if intent == "thanks":
            return "You're welcome! Anything else?"
        if intent == "help":
            return (
                "I can define AI terms, answer FAQs with semantic search, and share today's date/time. "
                "Try: 'define intent', 'what is a chatbot', or 'what time is it?'."
            )
        if intent == "capabilities":
            return "Capabilities: rule-based intents plus NLP similarity, topic definitions, and a short FAQ."
        if intent == "time":
            now = dt.datetime.now()
            return f"It is {now.strftime('%A, %B %d, %Y at %I:%M %p')}."

        if intent == "topic_definition":
            topic = self._detect_topic(normalized)
            if topic:
                return TOPIC_KB[topic]
            return None

        topic = self._detect_topic(normalized)
        if topic:
            return TOPIC_KB[topic]

        return None
