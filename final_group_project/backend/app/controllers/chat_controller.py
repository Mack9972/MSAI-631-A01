from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import RAG_KB_DIR, RAG_KB_PATHS
from app.models.schemas import ChatRequest, ChatResponse
from app.repositories.corpus_repository import CorpusRepository
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api", tags=["chat"])

_corpus_repo = CorpusRepository(kb_paths=RAG_KB_PATHS, kb_dir=RAG_KB_DIR)
_chat_service = ChatService(_corpus_repo)


def get_chat_service() -> ChatService:
    return _chat_service


@router.post("/translate", response_model=ChatResponse)
def translate(request: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    data = service.generate_response(request.text, request.include_english)
    return ChatResponse(**data)
