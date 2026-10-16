"""ChromaDB-backed vector index for syllabus chunks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import chromadb
import numpy as np
from chromadb.config import Settings

from src.config import config


def _bypass_proxy_for_loopback(host: str) -> None:
    """Keep local Chroma traffic out of system HTTP proxies."""
    normalized_host = host.strip().strip("[]").lower()
    if normalized_host not in {"127.0.0.1", "localhost", "::1"}:
        return
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy", "")
    entries = [entry.strip() for entry in existing.split(",") if entry.strip()]
    if normalized_host not in {entry.lower() for entry in entries}:
        entries.append(normalized_host)
    os.environ["NO_PROXY"] = ",".join(entries)


def _hash_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    """Deterministic embedding used by tests and explicit hash mode."""
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class TextEncoder:
    """Encode text with SentenceTransformer, an OpenAI-compatible API, or hash mode.

    ``model_name == "hash"`` selects the deterministic encoder; otherwise the
    backend follows ``EMBEDDING_PROVIDER`` (``local`` or ``api``).
    """

    API_BATCH_SIZE = 64

    def __init__(self, model_name: str, dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None
        self._api_client = None
        if model_name == "hash":
            return
        if config.embedding_provider == "api":
            self._init_api()
        else:
            self._init_local(model_name)

    def _init_local(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            if hasattr(self._model, "get_embedding_dimension"):
                self.dimensions = self._model.get_embedding_dimension()
            else:
                self.dimensions = (
                    self._model.get_sentence_embedding_dimension()
                )
        except Exception as exc:
            raise RuntimeError(
                f"embedding model unavailable: {model_name}"
            ) from exc

    def _init_api(self) -> None:
        if not config.embedding_api_key:
            raise RuntimeError(
                "EMBEDDING_API_KEY is required when EMBEDDING_PROVIDER=api"
            )
        try:
            from openai import OpenAI

            self._api_client = OpenAI(
                api_key=config.embedding_api_key,
                base_url=config.embedding_base_url or None,
            )
            # Probe once so the index metadata records the real dimension.
            self.dimensions = int(self._api_encode(["dimension probe"]).shape[1])
        except Exception as exc:
            raise RuntimeError(
                f"embedding API unavailable: {self.model_name}"
            ) from exc

    def _api_encode(self, values: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(values), self.API_BATCH_SIZE):
            batch = values[start : start + self.API_BATCH_SIZE]
            response = self._api_client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            rows.extend(item.embedding for item in response.data)
        matrix = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        if self._model is not None:
            return np.asarray(
                self._model.encode(values, normalize_embeddings=True),
                dtype=np.float32,
            )
        if self._api_client is not None:
            return self._api_encode(values)
        return np.vstack(
            [_hash_embedding(text, self.dimensions) for text in values]
        )


def create_chroma_client():
    """Create the configured local persistent or HTTP Chroma client."""
    settings = Settings(anonymized_telemetry=False)
    if config.chroma_mode == "http":
        _bypass_proxy_for_loopback(config.chroma_host)
        return chromadb.HttpClient(
            host=config.chroma_host,
            port=config.chroma_port,
            ssl=config.chroma_ssl,
            settings=settings,
        )
    if config.chroma_mode != "local":
        raise ValueError("CHROMA_MODE must be 'local' or 'http'")
    config.vector_db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=config.vector_db_path,
        settings=settings,
    )


def _chunk_id(chunk: dict[str, Any]) -> str:
    value = str(chunk.get("chunk_id", "")).strip()
    if value:
        return value
    payload = "|".join(
        [
            str(chunk.get("source_file", "")),
            str(chunk.get("section", "")),
            str(chunk.get("text", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _breadcrumb_enabled() -> bool:
    """判定本次索引是否需要把面包屑注入向量文本。

    取值来自 ``config.embedding_breadcrumb``：``on`` 始终注入，``off`` 不注入，
    ``auto`` 仅在向量模型不是 MiniLM 时注入（MiniLM 的 128 token 窗口较小，
    面包屑会挤占正文，风险见改造方案 §8）。
    """
    strategy = str(getattr(config, "embedding_breadcrumb", "auto") or "auto")
    strategy = strategy.strip().lower()
    if strategy == "on":
        return True
    if strategy == "off":
        return False
    model_name = str(getattr(config, "embedding_model", "") or "").lower()
    return "minilm" not in model_name


def build_embedding_text(chunk: dict[str, Any], *, breadcrumb_enabled: bool) -> str:
    """构造用于向量编码的文本，可选在正文前叠加结构面包屑前缀。

    面包屑只影响 ``embedding_text``；``chunk["text"]`` 始终保留原文，供引用
    卡片与 NLI 上下文使用。当正文已以该面包屑开头时不重复叠加。
    """
    text = str(chunk.get("text", ""))
    breadcrumb = str(chunk.get("breadcrumb") or "").strip()
    if not breadcrumb_enabled or not breadcrumb:
        return text
    if text.startswith(breadcrumb):
        return text
    return f"{breadcrumb}\n{text}"


def _chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    domain_metadata = chunk.get("metadata") or {}

    # 列表 / 长文本字段不适合放进 Chroma 标量 metadata，统一并入 metadata_json
    metadata_payload = dict(domain_metadata)
    for field in ("heading_path", "breadcrumb"):
        value = chunk.get(field)
        if value:
            metadata_payload[field] = value

    metadata: dict[str, Any] = {
        "chunk_id": _chunk_id(chunk),
        "course_code": domain_metadata.get("course_code")
        or chunk.get("course_code"),
        "course_name": domain_metadata.get("course_name"),
        "source_type": domain_metadata.get("source_type"),
        "program_name": domain_metadata.get("program_name"),
        "program_type": domain_metadata.get("program_type"),
        "course_category": domain_metadata.get("course_category"),
        "course_subcategory": domain_metadata.get("course_subcategory"),
        "credits": domain_metadata.get("credits"),
        "hours": domain_metadata.get("hours"),
        "semester": domain_metadata.get("semester"),
        "instructor": domain_metadata.get("instructor"),
        "section_type": domain_metadata.get("section_type"),
        "source_year": domain_metadata.get("source_year"),
        "parent_document": domain_metadata.get("parent_document"),
        "parent_section": domain_metadata.get("parent_section"),
        "chunk_part": domain_metadata.get("chunk_part"),
        "section": chunk.get("section")
        or domain_metadata.get("syllabus_section"),
        "source_file": chunk.get("source_file"),
        # 结构感知新增的标量字段；None 由末尾的标量过滤剔除
        "chunk_index": chunk.get("chunk_index"),
        "chapter_no": chunk.get("chapter_no"),
        "is_header": chunk.get("is_header"),
        "prev_chunk_id": chunk.get("prev_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
        "parent_id": chunk.get("parent_id"),
        "metadata_json": json.dumps(
            metadata_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) and value != ""
    }


class ChromaVectorIndex:
    """Build a fresh Chroma collection from parsed chunks."""

    def __init__(
        self,
        model_name: str,
        dimensions: int = 384,
        *,
        client=None,
        collection_name: str | None = None,
    ):
        self.client = client or create_chroma_client()
        self.collection_name = collection_name or config.chroma_collection
        self.encoder = TextEncoder(model_name, dimensions)

    def build(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        existing = {collection.name for collection in self.client.list_collections()}
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.encoder.model_name,
                "dimensions": self.encoder.dimensions,
            },
        )

        breadcrumb_enabled = _breadcrumb_enabled()
        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [str(chunk.get("text", "")) for chunk in batch]
            # 概念阶段已运行：embedding_text 已含（面包屑 + 概念增强），直接用；
            # 概念阶段未运行：此处补上面包屑前缀，保证两条路径语义一致。
            embedding_inputs = []
            for chunk, document in zip(batch, documents):
                existing = chunk.get("embedding_text")
                if existing:
                    embedding_inputs.append(str(existing))
                else:
                    embedding_inputs.append(
                        build_embedding_text(
                            chunk,
                            breadcrumb_enabled=breadcrumb_enabled,
                        )
                    )
            embeddings = self.encoder.encode(embedding_inputs)
            collection.upsert(
                ids=[_chunk_id(chunk) for chunk in batch],
                embeddings=embeddings,
                documents=documents,
                metadatas=[_chroma_metadata(chunk) for chunk in batch],
            )
        return {
            "collection": self.collection_name,
            "count": collection.count(),
            "model": self.encoder.model_name,
            "dimensions": self.encoder.dimensions,
        }


def build_index(
    chunks: list[dict[str, Any]],
    _output_dir: str | Path | None = None,
    *,
    model_name: str,
    dimensions: int = 384,
    client=None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """Compatibility entry point used by the pipeline and maintenance task."""
    return ChromaVectorIndex(
        model_name,
        dimensions,
        client=client,
        collection_name=collection_name,
    ).build(chunks)
