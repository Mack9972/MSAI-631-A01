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
import sys
import urllib.error
import urllib.request
from pathlib import Path
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None


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

INTENT_SAMPLES = {
    "greeting": ["hello", "hi", "hey there", "good morning", "good afternoon"],
    "goodbye": ["bye", "goodbye", "see you later", "exit", "quit"],
    "thanks": ["thanks", "thank you", "appreciate it"],
    "help": ["help", "what can you do", "show me options", "menu"],
    "capabilities": ["features", "capabilities", "what can you do"],
    "topic_definition": ["define", "what is", "explain", "tell me about"],
    "time": ["what time is it", "what's the date", "today's date", "current time"],
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

CORPUS_PATH = Path(__file__).with_name("corpus.json")
CORPUS_MIN_SCORE = 0.55
CORPUS_TFIDF_MIN_SCORE = 0.22
INTENT_SIM_MIN_SCORE = 0.32
TOPIC_SIM_MIN_SCORE = 0.55
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "have", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please", "the",
    "this", "to", "what", "when", "where", "who", "why", "you", "your",
}

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)

AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_ENDPOINT.rstrip("/")
AZURE_OPENAI_MAX_TURNS = 6


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token]


def normalize_for_tfidf(text: str) -> str:
    tokens = [token for token in tokenize(text) if token not in STOPWORDS]
    return " ".join(tokens)


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
    normalized = normalize(text)
    tokens = set(tokenize(normalized)) - STOPWORDS
    best_topic = None
    best_score = 0.0

    for topic in TOPIC_KB:
        if re.search(rf"\b{re.escape(topic)}\b", normalized):
            return topic

        alias_tokens = []
        for alias in TOPIC_ALIASES.get(topic, []):
            alias_tokens.extend(tokenize(alias))
        candidate_tokens = set(tokenize(topic)) | set(alias_tokens)
        if not candidate_tokens:
            continue
        score = len(tokens & candidate_tokens) / max(1, len(candidate_tokens))
        if score > best_score:
            best_score = score
            best_topic = topic

    if best_score >= TOPIC_SIM_MIN_SCORE:
        return best_topic
    return None


def build_retriever(corpus_path: Path) -> dict | None:
    if not corpus_path.exists():
        return None

    entries = load_corpus(corpus_path)
    if not entries:
        return None

    questions = [item["question"] for item in entries]
    if SentenceTransformer is not None and np is not None:
        try:
            # Use a local model if available; avoid network access.
            model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            model = None
        if model is not None:
            embeddings = model.encode(questions, normalize_embeddings=True)
            return {
                "type": "sbert",
                "model": model,
                "entries": entries,
                "embeddings": embeddings,
            }

    if TfidfVectorizer is None:
        return None

    vectorizer = TfidfVectorizer(
        preprocessor=normalize_for_tfidf,
        tokenizer=str.split,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(questions)
    return {
        "type": "tfidf",
        "vectorizer": vectorizer,
        "entries": entries,
        "matrix": matrix,
    }


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

    if RETRIEVER["type"] == "sbert":
        model = RETRIEVER["model"]
        entries = RETRIEVER["entries"]
        embeddings = RETRIEVER["embeddings"]
        query_vec = model.encode([text], normalize_embeddings=True)[0]
        scores = np.dot(embeddings, query_vec)
        best_idx = int(np.argmax(scores))
        if scores[best_idx] >= CORPUS_MIN_SCORE:
            return entries[best_idx]["answer"]
        return None

    if RETRIEVER["type"] == "tfidf":
        vectorizer = RETRIEVER["vectorizer"]
        entries = RETRIEVER["entries"]
        matrix = RETRIEVER["matrix"]
        query_vec = vectorizer.transform([text])
        scores = (matrix @ query_vec.T).toarray().ravel()
        best_idx = int(np.argmax(scores))
        if scores[best_idx] >= CORPUS_TFIDF_MIN_SCORE:
            return entries[best_idx]["answer"]
        return None

    return None


def azure_openai_available() -> bool:
    return bool(
        AZURE_OPENAI_ENDPOINT
        and AZURE_OPENAI_API_KEY
        and AZURE_OPENAI_DEPLOYMENT
    )


def build_llm_messages(text: str, state: dict) -> list[dict]:
    now = dt.datetime.now()
    system_prompt = (
        "You are a helpful assistant for a university AI services demo. "
        "Be concise, friendly, and accurate. If you do not know, say so. "
        f"Current local date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}."
    )
    if state.get("name"):
        system_prompt += f" The user's name is {state['name']}."

    messages = [{"role": "system", "content": system_prompt}]
    history = state.get("history", [])
    messages.extend(history[-AZURE_OPENAI_MAX_TURNS :])
    messages.append({"role": "user", "content": text})
    return messages


def append_history(state: dict, role: str, content: str) -> None:
    history = state.setdefault("history", [])
    history.append({"role": role, "content": content})
    if len(history) > AZURE_OPENAI_MAX_TURNS:
        del history[:-AZURE_OPENAI_MAX_TURNS]


def call_azure_openai(text: str, state: dict) -> str | None:
    if not azure_openai_available():
        return None

    url = (
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/"
        f"{AZURE_OPENAI_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    payload = {
        "messages": build_llm_messages(text, state),
        "temperature": 0.2,
        "max_tokens": 300,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
        response = json.loads(body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(
            f"[azure-openai] HTTP {exc.code}: {error_body}",
            file=sys.stderr,
        )
        return None
    except urllib.error.URLError as exc:
        print(f"[azure-openai] Connection error: {exc}", file=sys.stderr)
        return None
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[azure-openai] Invalid response: {exc}", file=sys.stderr)
        return None

    choices = response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        return None
    append_history(state, "user", text)
    append_history(state, "assistant", content)
    return content.strip()


def build_intent_classifier() -> dict | None:
    if TfidfVectorizer is None:
        return None
    samples = []
    labels = []
    for intent, phrases in INTENT_SAMPLES.items():
        for phrase in phrases:
            samples.append(phrase)
            labels.append(intent)

    if not samples:
        return None
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_for_tfidf,
        tokenizer=str.split,
        lowercase=False,
    )
    matrix = vectorizer.fit_transform(samples)
    return {"vectorizer": vectorizer, "labels": labels, "matrix": matrix}


INTENT_CLASSIFIER = build_intent_classifier()


def classify_intent(text: str) -> str | None:
    if INTENT_CLASSIFIER is None:
        return None
    vectorizer = INTENT_CLASSIFIER["vectorizer"]
    labels = INTENT_CLASSIFIER["labels"]
    matrix = INTENT_CLASSIFIER["matrix"]
    query_vec = vectorizer.transform([text])
    scores = (matrix @ query_vec.T).toarray().ravel()
    best_idx = int(np.argmax(scores))
    if scores[best_idx] >= INTENT_SIM_MIN_SCORE:
        return labels[best_idx]
    return None


def respond(text: str, state: dict) -> tuple[str, bool]:
    normalized = normalize(text)

    name = extract_name(normalized)
    if name:
        state["name"] = name
        return f"Nice to meet you, {name}. How can I help?", False

    intent = match_intent(normalized)
    if intent is None:
        intent = classify_intent(text)
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
            "I can define AI terms, answer FAQs with semantic search, and share today's date/time. "
            "Try: 'define intent', 'what is a chatbot', or 'what time is it?'.",
            False,
        )
    if intent == "capabilities":
        return (
            "Capabilities: rule-based intents plus NLP similarity, topic definitions, and a short FAQ.",
            False,
        )
    if intent == "time":
        now = dt.datetime.now()
        return f"It is {now.strftime('%A, %B %d, %Y at %I:%M %p')}.", False

    if intent == "topic_definition":
        topic = detect_topic(normalized)
        if topic:
            return TOPIC_KB[topic], False
        llm_answer = call_azure_openai(text, state)
        if llm_answer:
            return llm_answer, False
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

    llm_answer = call_azure_openai(text, state)
    if llm_answer:
        return llm_answer, False

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
