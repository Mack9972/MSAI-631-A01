from __future__ import annotations

from app.services.llm_service import chat_completion, azure_openai_available


class TranslationService:
    def translate_to_spanish(self, english_text: str) -> str:
        if not azure_openai_available():
            return "Traducción no disponible. Configure Azure OpenAI."

        messages = [
            {
                "role": "system",
                "content": "You are a professional translator. Translate English to neutral Latin American Spanish. Return only the Spanish translation.",
            },
            {"role": "user", "content": english_text},
        ]
        result = chat_completion(messages, temperature=0.1, max_tokens=400)
        if not result:
            return "Traducción no disponible. Configure Azure OpenAI."
        return result
