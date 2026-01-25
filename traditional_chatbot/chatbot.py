#!/usr/bin/env python3
"""
Traditional, rule-based chatbot (no LLMs).
Run: python3 chatbot.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


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

CORPUS_PATH = Path(__file__).with_name("corpus.json")
CORPUS_MIN_SCORE = 0.55


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def match_intent(text: str) -> str | None:
    for intent, patterns in INTENTS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return intent
    return None


def extract_name(text: str) -> str | None:
    match = re.search(r"\bmy name is ([a-zA-Z\-']{2,})\b", text)
    if match:
        return match.group(1).title()
    return None


def detect_topic(text: str) -> str | None:
    for topic in TOPIC_KB:
        if re.search(rf"\b{re.escape(topic)}\b", text):
            return topic
    return None


def build_retriever(corpus_path: Path) -> dict | None:
    if SentenceTransformer is None or np is None or not corpus_path.exists():
        return None
    try:
        # Use a local model if available; avoid network access.
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        return None

    entries = load_corpus(corpus_path)
    if not entries:
        return None

    questions = [item["question"] for item in entries]
    embeddings = model.encode(questions, normalize_embeddings=True)
    return {"model": model, "entries": entries, "embeddings": embeddings}


def load_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        return []
    if corpus_path.suffix.lower() == ".json":
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("qa", [])
        return normalize_corpus_entries(data)
    if corpus_path.suffix.lower() == ".csv":
        with corpus_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            return normalize_corpus_entries(list(rows))
    if corpus_path.suffix.lower() == ".txt":
        entries = []
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            question, answer = line.split("\t", 1)
            entries.append({"question": question.strip(), "answer": answer.strip()})
        return normalize_corpus_entries(entries)
    return []


def normalize_corpus_entries(entries: list[dict]) -> list[dict]:
    normalized = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("q")
        answer = item.get("answer") or item.get("a")
        if question and answer:
            normalized.append({"question": str(question), "answer": str(answer)})
    return normalized


RETRIEVER = build_retriever(CORPUS_PATH)


def retrieve_answer(text: str) -> str | None:
    if RETRIEVER is None:
        return None
    model = RETRIEVER["model"]
    entries = RETRIEVER["entries"]
    embeddings = RETRIEVER["embeddings"]
    query_vec = model.encode([text], normalize_embeddings=True)[0]
    scores = np.dot(embeddings, query_vec)
    best_idx = int(np.argmax(scores))
    if scores[best_idx] >= CORPUS_MIN_SCORE:
        return entries[best_idx]["answer"]
    return None


def respond(text: str, state: dict) -> tuple[str, bool]:
    normalized = normalize(text)

    name = extract_name(normalized)
    if name:
        state["name"] = name
        return f"Nice to meet you, {name}. How can I help?", False

    intent = match_intent(normalized)
    if intent == "greeting":
        if state.get("name"):
            return f"Hi {state['name']}! What would you like to talk about?", False
        return "Hello! How can I help today?", False
    if intent == "goodbye":
        return "Goodbye! Have a great day.", True
    if intent == "thanks":
        return "You're welcome! Anything else?", False
    if intent == "help":
        return (
            "I can define AI terms, explain basic chatbot concepts, and share today's date/time. "
            "Try: 'define intent', 'what is a chatbot', or 'what time is it?'.",
            False,
        )
    if intent == "capabilities":
        return (
            "Capabilities: rule-based intent matching, simple topic definitions, and a short FAQ.",
            False,
        )
    if intent == "time":
        now = dt.datetime.now()
        return f"It is {now.strftime('%A, %B %d, %Y at %I:%M %p')}.", False

    if intent == "topic_definition":
        topic = detect_topic(normalized)
        if topic:
            return TOPIC_KB[topic], False
        return (
            "I can define: chatbot, rule-based, nlp, intent, entity. "
            "Try asking about one of those.",
            False,
        )

    topic = detect_topic(normalized)
    if topic:
        return TOPIC_KB[topic], False

    retrieved = retrieve_answer(normalized)
    if retrieved:
        return retrieved, False

    return (
        "I did not catch that. Try 'help' for options or ask me to define a term.",
        False,
    )


def main() -> None:
    print("Traditional Chatbot (rule-based). Type 'exit' to quit.")
    state: dict[str, str] = {}
    while True:
        try:
            user_input = input("> ")
        except EOFError:
            print("\nGoodbye!")
            break

        response, should_exit = respond(user_input, state)
        print(response)
        if should_exit:
            break


if __name__ == "__main__":
    main()
