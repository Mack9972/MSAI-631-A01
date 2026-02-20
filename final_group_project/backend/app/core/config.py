from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[3]

load_dotenv(BASE_DIR / ".env")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "chatbot-gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

RAG_KB_DIR = Path(os.getenv("RAG_KB_DIR", BASE_DIR / "knowledge_base"))

_default_kb_paths = [
    BASE_DIR / "LLM_Chatbot_Design_Report.docx",
    BASE_DIR / "Group_Project_Proposal_Full_RAG_Bilingual_Assistant.docx",
]
_raw_kb_paths = os.getenv("RAG_KB_PATHS", ",".join(str(path) for path in _default_kb_paths))
RAG_KB_PATHS = [
    Path(path.strip())
    for path in _raw_kb_paths.split(",")
    if path.strip()
]

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
