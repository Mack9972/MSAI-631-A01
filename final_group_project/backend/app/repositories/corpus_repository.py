from __future__ import annotations

import csv
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

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


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "have", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please", "the",
    "this", "to", "what", "when", "where", "who", "why", "you", "your",
}
SUPPORTED_EXTENSIONS = {".docx", ".txt", ".md", ".json", ".csv"}


@dataclass
class Retriever:
    type: str
    chunks: list[dict[str, str]]
    model: Optional[Any] = None
    embeddings: Optional[Any] = None
    vectorizer: Optional[Any] = None
    matrix: Optional[Any] = None


class CorpusRepository:
    def __init__(
        self,
        kb_paths: list[Path],
        kb_dir: Optional[Path] = None,
        min_score: float = 0.42,
        tfidf_min_score: float = 0.08,
    ) -> None:
        self.kb_paths = kb_paths
        self.kb_dir = kb_dir
        self.min_score = min_score
        self.tfidf_min_score = tfidf_min_score
        self._source_signature: tuple[tuple[str, float], ...] = tuple()
        self.retriever = self._build_retriever()

    def _tokenize(self, text: str) -> list[str]:
        return [token for token in TOKEN_RE.findall(text.lower()) if token]

    def _normalize_for_tfidf(self, text: str) -> str:
        tokens = [token for token in self._tokenize(text) if token not in STOPWORDS]
        return " ".join(tokens)

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _chunk_text(self, text: str, source: str, chunk_size: int = 220, overlap: int = 40) -> list[dict[str, str]]:
        words = text.split()
        if not words:
            return []
        chunks: list[dict[str, str]] = []
        step = max(1, chunk_size - overlap)
        for start in range(0, len(words), step):
            segment = words[start: start + chunk_size]
            if len(segment) < 30:
                continue
            chunks.append({
                "source": source,
                "chunk": " ".join(segment),
            })
            if start + chunk_size >= len(words):
                break
        return chunks

    def _extract_docx_text(self, path: Path) -> str:
        try:
            with zipfile.ZipFile(path) as archive:
                raw_xml = archive.read("word/document.xml")
        except Exception:
            return ""

        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError:
            return ""

        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs: list[str] = []
        for paragraph in root.findall(".//w:p", namespace):
            text_nodes = paragraph.findall(".//w:t", namespace)
            line = "".join(node.text or "" for node in text_nodes).strip()
            if line:
                paragraphs.append(line)
        return "\n".join(paragraphs)

    def _load_qa_entries(self, path: Path) -> list[dict[str, str]]:
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return []
            if isinstance(data, dict):
                data = data.get("qa", [])
            if not isinstance(data, list):
                return []
            rows = data
        elif path.suffix.lower() == ".csv":
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except Exception:
                return []
        else:
            return []

        normalized: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            question = row.get("question") or row.get("q")
            answer = row.get("answer") or row.get("a")
            if not question or not answer:
                continue
            normalized.append({
                "source": path.name,
                "chunk": f"Question: {str(question).strip()}\nAnswer: {str(answer).strip()}",
            })
        return normalized

    def _load_text_chunks(self, path: Path) -> list[dict[str, str]]:
        suffix = path.suffix.lower()
        if suffix in {".json", ".csv"}:
            return self._load_qa_entries(path)
        if suffix == ".docx":
            content = self._extract_docx_text(path)
        else:
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                content = ""
        content = self._normalize_text(content)
        if not content:
            return []
        return self._chunk_text(content, source=path.name)

    def _discover_sources(self) -> list[Path]:
        files: list[Path] = []
        for path in self.kb_paths:
            resolved = path if path.is_absolute() else path.resolve()
            if resolved.exists() and resolved.is_file() and resolved.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(resolved)

        if self.kb_dir is not None and self.kb_dir.exists():
            for child in sorted(self.kb_dir.iterdir()):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(child.resolve())

        deduped: list[Path] = []
        seen: set[str] = set()
        for path in files:
            key = str(path)
            if key not in seen:
                seen.add(key)
                deduped.append(path)
        return deduped

    def _make_signature(self, files: list[Path]) -> tuple[tuple[str, float], ...]:
        signature: list[tuple[str, float]] = []
        for path in files:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            signature.append((str(path), mtime))
        return tuple(signature)

    def _load_chunks(self) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        files = self._discover_sources()
        self._source_signature = self._make_signature(files)
        for path in files:
            chunks.extend(self._load_text_chunks(path))
        return chunks

    def _build_retriever(self) -> Optional[Retriever]:
        chunks = self._load_chunks()
        if not chunks:
            return None

        texts = [item["chunk"] for item in chunks]
        if SentenceTransformer is not None and np is not None:
            try:
                model = SentenceTransformer(
                    "all-MiniLM-L6-v2",
                    model_kwargs={"local_files_only": True},
                )
            except Exception:
                model = None
            if model is not None:
                embeddings = model.encode(texts, normalize_embeddings=True)
                return Retriever(type="sbert", model=model, chunks=chunks, embeddings=embeddings)

        if TfidfVectorizer is None:
            return None

        vectorizer = TfidfVectorizer(
            preprocessor=self._normalize_for_tfidf,
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(texts)
        return Retriever(type="tfidf", vectorizer=vectorizer, chunks=chunks, matrix=matrix)

    def _refresh_if_needed(self) -> None:
        files = self._discover_sources()
        current = self._make_signature(files)
        if current != self._source_signature:
            self.retriever = self._build_retriever()

    def _score_chunks(self, text: str) -> tuple[list[dict[str, str]], list[float]]:
        self._refresh_if_needed()
        if self.retriever is None:
            return [], []

        if self.retriever.type == "sbert":
            model = self.retriever.model
            chunks = self.retriever.chunks
            embeddings = self.retriever.embeddings
            if model is None or embeddings is None or np is None:
                return [], []
            query_vec = model.encode([text], normalize_embeddings=True)[0]
            scores = np.dot(embeddings, query_vec).tolist()
            return chunks, [float(score) for score in scores]

        if self.retriever.type == "tfidf":
            vectorizer = self.retriever.vectorizer
            chunks = self.retriever.chunks
            matrix = self.retriever.matrix
            if vectorizer is None or matrix is None or np is None:
                return [], []
            query_vec = vectorizer.transform([text])
            scores = (matrix @ query_vec.T).toarray().ravel().tolist()
            return chunks, [float(score) for score in scores]

        return [], []

    def retrieve_answer(self, text: str) -> Optional[str]:
        chunks, scores = self._score_chunks(text)
        if not chunks or not scores:
            return None
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        threshold = self.min_score if self.retriever and self.retriever.type == "sbert" else self.tfidf_min_score
        if scores[best_idx] < threshold:
            return None
        source = chunks[best_idx]["source"]
        snippet = chunks[best_idx]["chunk"]
        return f"From {source}:\n{snippet}"

    def retrieve_context(self, text: str, k: int = 3) -> list[dict[str, str]]:
        chunks, scores = self._score_chunks(text)
        if not chunks or not scores:
            return []
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        selected: list[dict[str, str]] = []
        for idx in ranked[: max(1, k)]:
            selected.append({
                "source": chunks[idx]["source"],
                "chunk": chunks[idx]["chunk"],
                "score": f"{scores[idx]:.4f}",
            })
        return selected
