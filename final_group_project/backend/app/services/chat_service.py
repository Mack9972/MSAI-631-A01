from __future__ import annotations

import re
from typing import Optional

from app.repositories.corpus_repository import CorpusRepository
from app.services.llm_service import chat_completion, azure_openai_available
from app.services.rule_based_service import RuleBasedService
from app.services.translation_service import TranslationService


class ChatService:
    def __init__(self, corpus_repo: CorpusRepository) -> None:
        self.corpus_repo = corpus_repo
        self.rule_based = RuleBasedService()
        self.translator = TranslationService()

    def _build_rag_messages(self, question: str, contexts: list[dict[str, str]]) -> list[dict[str, str]]:
        context_lines = []
        for item in contexts:
            source = item.get("source", "unknown")
            chunk_text = item.get("chunk", "")
            if chunk_text:
                context_lines.append(f"Source: {source}\nContent: {chunk_text[:1200]}")
        context_block = "\n\n".join(context_lines)

        system = (
            "You are a helpful assistant for a university AI services demo. "
            "Answer in English using only the provided context. "
            "If the context does not contain the answer, say you do not know. "
            "Keep responses concise (max 2 sentences) and answer only what was asked. "
            "Do not add background unless the user asks for it. "
            "When possible, mention source file names used."
        )
        user = f"Context:\n{context_block}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_general_messages(self, question: str) -> list[dict[str, str]]:
        system = (
            "You are a helpful assistant for a university AI services demo. "
            "Answer in English, concisely (max 2 sentences), and only what was asked."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    def _is_rag_relevant(self, question: str, contexts: list[dict[str, str]]) -> bool:
        if not contexts:
            return False

        top = contexts[0]
        top_chunk = top.get("chunk", "")
        try:
            top_score = float(top.get("score", "0"))
        except (TypeError, ValueError):
            top_score = 0.0

        # Strong semantic hit (works well for normalized embedding retrievers).
        if top_score >= 0.42:
            return True

        # Lexical safety net for tf-idf style retrieval.
        question_tokens = {
            token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 2
        }
        chunk_tokens = {
            token for token in re.findall(r"[a-z0-9]+", top_chunk.lower()) if len(token) > 2
        }
        overlap = len(question_tokens & chunk_tokens)
        return top_score >= 0.08 and overlap >= 2

    def _extractive_fallback(self, question: str, contexts: list[dict[str, str]]) -> Optional[str]:
        question_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if len(token) > 2
        }
        best_line = ""
        best_source = ""
        best_score = -1

        for item in contexts:
            source = item.get("source", "unknown")
            chunk = item.get("chunk", "")
            for line in re.split(r"(?<=[.!?])\s+|\n+", chunk):
                line = line.strip()
                if len(line) < 20:
                    continue
                line_tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
                overlap = len(question_tokens & line_tokens)
                density = overlap / max(1, len(line_tokens))
                score = (overlap * 3) + density
                if score > best_score:
                    best_score = score
                    best_line = line
                    best_source = source

        if best_score <= 0 or not best_line:
            return None
        return f"{best_line} (source: {best_source})"

    def generate_response(self, text: str, include_english: bool) -> dict[str, Optional[str]]:
        english = self.rule_based.respond(text)

        if english is None:
            contexts = self.corpus_repo.retrieve_context(text, k=4)
            rag_relevant = self._is_rag_relevant(text, contexts)
            if azure_openai_available():
                messages = (
                    self._build_rag_messages(text, contexts)
                    if rag_relevant
                    else self._build_general_messages(text)
                )
                english = chat_completion(messages, temperature=0.1, max_tokens=140)
            if not english and rag_relevant:
                english = self._extractive_fallback(text, contexts)

        if not english:
            english = "I did not catch that. Try asking about a term or an FAQ question."

        spanish = self.translator.translate_to_spanish(english)

        return {
            "english": english if include_english else None,
            "spanish": spanish,
        }
