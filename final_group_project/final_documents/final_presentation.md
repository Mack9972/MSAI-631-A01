% Final LLM Chatbot Project
% Design, Scope, Challenges, Learnings
% February 20, 2026

# Problem Addressed

- General-purpose chatbots can sound fluent but still hallucinate.
- Academic/project users need grounded answers with transparent sources.
- The project targets a bilingual assistant: English input, Spanish-first output.

# Scope and Objectives

- Build a working chatbot with web UI + API backend.
- Route intelligently between grounded retrieval and general LLM response.
- Provide Spanish output with optional English for transparency.
- Support fallback responses when confidence is low.
- Keep architecture modular for future extension.

# System Design

- Frontend: React chat interface (message thread + loading state).
- Backend: FastAPI with MVC-style layers (controllers/services/repositories/models).
- Retrieval: SentenceTransformer (preferred) with TF-IDF fallback.
- Generation: Azure OpenAI for RAG completion, general QA, and translation when configured.
- Router: If retrieved context is relevant, use RAG prompt; otherwise use general LLM prompt.

![Architecture](assets/screenshot_architecture.png)

# App Screenshot: Chat Experience

- Sequential conversation flow in a messaging layout.
- User prompt + assistant bilingual response.
- Include-English toggle for response transparency.

![Chat UI](assets/screenshot_chat_ui.png)

# App Screenshot: Response Payload

- API route: `POST /api/translate`
- Request: `{ text, include_english }`
- Response: `{ spanish, english }` (english optional)

![API Response](assets/screenshot_api_response.png)

# Approach from Proposal to Build

- Proposal goal: RAG-based bilingual assistant with citations.
- Implemented practical hybrid strategy:
  - Rule-based first
  - Retrieval relevance check second
  - RAG answer path for in-domain queries
  - General LLM path for out-of-domain/general queries
- Reason: better reliability, lower cost/latency, graceful degradation.

# Runtime Notes (Latest)

- Frontend `dev` script is under `final_group_project/frontend` (not project root).
- Start backend and frontend in separate terminals:
  - Backend: `uvicorn app.main:app --reload --app-dir backend`
  - Frontend: `npm run dev` from `final_group_project/frontend`
- For older macOS LibreSSL Python builds, pinning `urllib3<2` avoids `NotOpenSSLWarning`.

# Challenges Encountered

- Environment/dependency availability varied by runtime.
- Retrieval tuning required trial-and-error (chunk size, similarity thresholds).
- Maintaining concise bilingual quality required prompt calibration.
- Coverage limitations in knowledge base can still produce uncertainty.

# Key Learnings

- Hybrid architecture outperforms pure-LLM in reliability for scoped domains.
- Clean layer separation improves maintainability and debugging speed.
- RAG quality depends heavily on data chunking and retrieval thresholds.
- Fallback behavior is critical for user trust.

# Conclusion and Next Steps

- Final chatbot now supports both grounded RAG answers and general LLM Q&A.
- Future upgrades:
  - Session memory for multi-turn continuity
  - Better observability and evaluation metrics
  - Expanded multilingual and guardrail support
