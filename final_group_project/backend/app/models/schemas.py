from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    include_english: bool = True


class ChatResponse(BaseModel):
    spanish: str
    english: Optional[str] = None
