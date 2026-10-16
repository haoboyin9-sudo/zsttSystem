"""ChromaDB retrieval for the online service."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from chromadb.errors import ChromaError, NotFoundError

from src.config import config
from src.data_processing.chroma_index import TextEncoder, create_chroma_client
from src.data_processing.collection_registry import resolve_active_collection
from src.data_processing.lexical_stats import load_lexical_stats
from src.utils.lexical import (
    BM25Scorer,
    InvertedIndex,
    overlap_score,
    term_set,
    tokenize,
)

#: RRF 常数：``1 / (RRF_K + rank)``，rank 从 1 开始（改造方案 M5.2）。
RRF_K = 60.0

#: 进程内倒排索引缓存，键为 (chunks 戳, 统计戳, k1, b)。
#: ``chunks.json`` 与 ``lexical_stats.json`` 没变时跨实例复用，避免热重建 /
#: 多个检索器实例重复解析 17MB 语料；有上限，防止消融扫参时无限增长。
_LEXICAL_INDEX_CACHE: dict[tuple[Any, ...], InvertedIndex | None] = {}
_LEXICAL_INDEX_CACHE_LIMIT = 8


def _artifact_stamp(path: Any) -> tuple[str, float | None]:
    """文件路径 + mtime 戳；文件缺失时 mtime 为 ``None``（用于旧化检测）。"""
    try:
        resolved = Path(str(path))
        return (str(resolved), float(resolved.stat().st_mtime))
    except (OSError, TypeError, ValueError):
        return (str(path), None)


_CHROMA_SCALAR_FIELDS: tuple[str, ...] = (
    "course_code",
    "course_name",
    "source_type",
    "program_name",
    "program_type",
    "course_category",
    "course_subcategory",
    "credits",
    "hours",
    "semester",
    "instructor",
    "section_type",
    "source_year",
    "parent_document",
    "parent_section",
    "chunk_part",
    "chunk_index",
    "chapter_no",
    "is_header",
    "prev_chunk_id",
    "next_chunk_id",
    "parent_id",
)


def _scalar_metadata_from_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """从原始 chunk 还原 Chroma 标量 metadata（含 ``metadata_json``）。

    优先复用 :mod:`src.data_processing.chroma_index` 里的 ``_chroma_metadata``，
    保证 BM25-only 命中与稠密命中携带同一套字段；该私有函数万一被重构掉，
    这里有一份同口径的兜底实现，不至于让检索直接抛错。
    """
    try:
        from src.data_processing import chroma_index

        builder = getattr(chroma_index, "_chroma_metadata", None)
        if callable(builder):
            return dict(builder(chunk))
    except Exception:  # noqa: BLE001 - 兜底路径不允许影响检索
        pass
    return _fallback_scalar_metadata(chunk)


def _fallback_scalar_metadata(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """与 ``chroma_index._chroma_metadata`` 同口径的最小兜底实现。"""
    domain = chunk.get("metadata") or {}
    payload = dict(domain)
    for field in ("heading_path", "breadcrumb"):
        value = chunk.get(field)
        if value:
            payload[field] = value
    metadata: dict[str, Any] = {
        "chunk_id": str(chunk.get("chunk_id", "")),
        "section": chunk.get("section") or domain.get("syllabus_section"),
        "source_file": chunk.get("source_file"),
        "metadata_json": json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    for field in _CHROMA_SCALAR_FIELDS:
        metadata[field] = domain.get(field)
    for field in (
        "chunk_index",
        "chapter_no",
        "is_header",
        "prev_chunk_id",
        "next_chunk_id",
        "parent_id",
    ):
        value = chunk.get(field)
        if value is not None:
            metadata[field] = value
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool)) and value != ""
    }

class RerankWeights:
    """Linear rerank coefficients, resolved from config unless overridden.

    Held in one object so that ablation runs (``eval/ablation.py``) and the
    weight sweep (``eval/tune_rerank.py``) can vary them without touching the
    environment of a running service.
    """

    __slots__ = ("vector", "lexical", "section_boost", "lexical_scheme", "k1", "b")

    def __init__(
        self,
        *,
        vector: float | None = None,
        lexical: float | None = None,
        section_boost: float | None = None,
        lexical_scheme: str | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> None:
        self.vector = config.rerank_weight_vector if vector is None else float(vector)
        self.lexical = (
            config.rerank_weight_lexical if lexical is None else float(lexical)
        )
        self.section_boost = (
            config.rerank_section_boost
            if section_boost is None
            else float(section_boost)
        )
        self.lexical_scheme = (
            config.rerank_lexical_scheme
            if lexical_scheme is None
            else str(lexical_scheme).strip().lower()
        )
        self.k1 = config.rerank_bm25_k1 if k1 is None else float(k1)
        self.b = config.rerank_bm25_b if b is None else float(b)

    def as_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector,
            "lexical": self.lexical,
            "section_boost": self.section_boost,
            "lexical_scheme": self.lexical_scheme,
            "bm25_k1": self.k1,
            "bm25_b": self.b,
        }


class ChromaRetriever:
    # Class-level defaults so that a bare ``__new__`` instance -- the shape
    # used by the retrieval tests -- can still score without a full __init__.
    _weights: RerankWeights | None = None
    _bm25: BM25Scorer | None = None
    _bm25_loaded: bool = False
    _lexical_stats_path: Any = None
    _lexical_index: InvertedIndex | None = None
    _lexical_index_loaded: bool = False
    _lexical_index_pinned: bool = False
    _lexical_index_stamp: tuple[Any, ...] | None = None
    _lexical_index_path: Any = None
    _fusion: str | None = None

    def __init__(
        self,
        model_name: str | None = None,
        *,
        client=None,
        collection_name: str | None = None,
        weights: RerankWeights | None = None,
        lexical_stats_path: str | None = None,
        lexical_index: InvertedIndex | None = None,
        fusion: str | None = None,
    ):
        self.model_name = model_name or config.local_embedding_model
        self._weights = weights
        self._lexical_stats_path = lexical_stats_path
        self._bm25 = None
        self._bm25_loaded = False
        self._fusion = fusion
        # 外部（如 ``main.py`` 的重建回调）可以直接塞一个现成的倒排索引进来，
        # 此时不再惰性构建、也不做 mtime 旧化检查。
        self._lexical_index = lexical_index
        self._lexical_index_pinned = lexical_index is not None
        self._lexical_index_loaded = lexical_index is not None
        self._lexical_index_stamp = None
        self._lexical_index_path = None
        self.collection_name = collection_name or config.chroma_collection
        self.client = None
        self.collection = None
        self.encoder = None
        self.connected = False
        try:
            self.client = client or create_chroma_client()
            self.collection_name = collection_name or self._resolve_collection_name()
            self.collection = self.client.get_collection(
                self.collection_name,
                embedding_function=None,
            )
            indexed_model = str(
                (self.collection.metadata or {}).get("embedding_model", "")
            )
            if indexed_model and indexed_model != self.model_name:
                raise ValueError(
                    "embedding model mismatch: "
                    f"index={indexed_model}, query={self.model_name}"
                )
            dimensions = int(
                (self.collection.metadata or {}).get("dimensions", 384)
            )
            self.encoder = TextEncoder(self.model_name, dimensions)
            self.connected = True
        except NotFoundError:
            self.connected = True
        except (ChromaError, OSError, ValueError):
            self.connected = False

    def _resolve_collection_name(self) -> str:
        """Follow the alias pointer written by the retraining updater."""
        try:
            available = [
                collection.name for collection in self.client.list_collections()
            ]
        except (ChromaError, OSError, AttributeError, TypeError):
            available = None
        return resolve_active_collection(available=available)

    @property
    def count(self) -> int:
        if self.collection is None:
            return 0
        return self.collection.count()

    @property
    def weights(self) -> RerankWeights:
        """Rerank coefficients, resolved from config on first use."""
        if self._weights is None:
            self._weights = RerankWeights()
        return self._weights

    @property
    def bm25(self) -> BM25Scorer | None:
        """Lazily loaded corpus statistics; ``None`` when unavailable."""
        if not self._bm25_loaded:
            self._bm25_loaded = True
            stats = load_lexical_stats(
                self._lexical_stats_path or config.lexical_stats_path
            )
            if stats:
                scorer = BM25Scorer(
                    stats,
                    k1=self.weights.k1,
                    b=self.weights.b,
                )
                self._bm25 = scorer if scorer.usable else None
        return self._bm25

    @property
    def fusion(self) -> str:
        """融合模式：``linear`` / ``rrf`` / ``dense`` / ``bm25``（见 ``config``）。"""
        value = self._fusion or config.retrieval_fusion
        value = str(value or "").strip().lower()
        return value if value in ("linear", "rrf", "dense", "bm25") else "linear"

    @property
    def lexical_index(self) -> InvertedIndex | None:
        """惰性构建的全库 BM25 倒排索引；语料缺失 / 异常时返回 ``None``。

        ``outputs/chunks.json`` 不存在（测试环境、全新 checkout）时静默降级为
        ``None``，``search()`` 随即退回改造前的稠密 + 候选内加权行为。
        ``chunks.json`` / ``lexical_stats.json`` 的 mtime 变化会触发重建，
        因此热重建后不需要调用方额外通知（见改造方案 §8 的一致性风险）。
        """
        if self._lexical_index_pinned:
            return self._lexical_index
        if not self._lexical_index_loaded:
            self._lexical_index_loaded = True
            self._load_lexical_index()
            return self._lexical_index
        current = self._current_lexical_index_stamp()
        if current != self._lexical_index_stamp:
            self._load_lexical_index()
        return self._lexical_index

    @lexical_index.setter
    def lexical_index(self, index: InvertedIndex | None) -> None:
        """注入外部构建好的倒排索引（此后不再自动重建）。"""
        self._lexical_index = index
        self._lexical_index_loaded = True
        self._lexical_index_pinned = True
        self._lexical_index_stamp = None

    def refresh_lexical_index(self) -> InvertedIndex | None:
        """按当前 ``chunks.json`` 强制重建倒排；被外部注入时保持不动。"""
        if self._lexical_index_pinned:
            return self._lexical_index
        self._lexical_index_loaded = True
        self._load_lexical_index()
        return self._lexical_index

    def _current_lexical_index_stamp(self) -> tuple[Any, ...]:
        path = self._lexical_index_path or config.chunks_output_path
        stats_path = self._lexical_stats_path or config.lexical_stats_path
        return (
            _artifact_stamp(path),
            _artifact_stamp(stats_path),
            self.weights.k1,
            self.weights.b,
        )

    def _load_lexical_index(self) -> None:
        """解析 ``chunks.json`` 构建倒排；任何异常都静默降级为 ``None``。"""
        path = config.chunks_output_path
        self._lexical_index_path = path
        stamp = self._current_lexical_index_stamp()
        self._lexical_index_stamp = stamp
        if stamp[0][1] is None:  # chunks.json 不存在
            self._lexical_index = None
            return
        cached = _LEXICAL_INDEX_CACHE.get(stamp)
        if cached is not None or stamp in _LEXICAL_INDEX_CACHE:
            self._lexical_index = cached
            return
        try:
            payload = json.loads(Path(str(path)).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            self._lexical_index = None
            return
        if not isinstance(payload, list):
            self._lexical_index = None
            return
        index = InvertedIndex(
            k1=self.weights.k1,
            b=self.weights.b,
            scorer=self.bm25,
        )
        try:
            index.build(payload)
        except Exception:  # noqa: BLE001 - 语料畸形时同样降级
            self._lexical_index = None
            return
        self._lexical_index = index if index.usable else None
        if len(_LEXICAL_INDEX_CACHE) >= _LEXICAL_INDEX_CACHE_LIMIT:
            _LEXICAL_INDEX_CACHE.clear()
        _LEXICAL_INDEX_CACHE[stamp] = self._lexical_index

    @staticmethod
    def _where_filter(
        course_code: str | None,
        source_types: list[str] | tuple[str, ...] | None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if course_code:
            clauses.append({"course_code": course_code})
        if source_types:
            clauses.append({"source_type": {"$in": list(source_types)}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        return term_set(text)

    def _lexical_scorer(self, query: str):
        """Return a ``document -> lexical score`` callable for this query.

        BM25 needs corpus-wide document frequencies; without the statistics
        file the retriever falls back to the historical term-overlap ratio so
        that a fresh checkout still ranks sensibly.
        """
        scheme = self.weights.lexical_scheme
        if scheme == "none":
            return lambda _document: 0.0
        scorer = self.bm25 if scheme == "bm25" else None
        if scorer is not None:
            query_tokens = tokenize(query)
            return lambda document: scorer.normalized_score(
                query_tokens,
                tokenize(document),
            )
        query_terms = term_set(query)
        return lambda document: overlap_score(query_terms, term_set(document))

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        course_code: str | None = None,
        source_types: list[str] | tuple[str, ...] | None = None,
        min_score: float | None = None,
        preferred_section_types: list[str] | tuple[str, ...] | None = None,
        source_boosts: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """混合召回入口：默认 ``linear``（候选内线性加权，行为等价改造前）。

        ``config.retrieval_fusion == "rrf"`` 且倒排索引可用时，走 dense / BM25
        两路独立召回 + RRF 融合；倒排不可用（没有 ``chunks.json``）时无论哪种
        模式都退回改造前的稠密路径，保证服务可用性与既有测试不变。
        """
        if self.collection is None or self.encoder is None or self.count == 0:
            return []
        index = self.lexical_index
        if self.fusion == "bm25" and index is not None and index.usable:
            # A5 对照臂：只用内存倒排召回，不查向量库。
            return self._search_bm25(
                query,
                top_k,
                index,
                course_code=course_code,
                source_types=source_types,
                min_score=min_score,
                preferred_section_types=preferred_section_types,
                source_boosts=source_boosts,
            )
        if self.fusion == "dense":
            # A5 对照臂：去掉 BM25 独立召回，候选集只来自向量一路。
            index = None
        if self.fusion == "rrf" and index is not None and index.usable:
            return self._search_rrf(
                query,
                top_k,
                index,
                course_code=course_code,
                source_types=source_types,
                min_score=min_score,
                preferred_section_types=preferred_section_types,
                source_boosts=source_boosts,
            )
        return self._search_linear(
            query,
            top_k,
            index,
            course_code=course_code,
            source_types=source_types,
            min_score=min_score,
            preferred_section_types=preferred_section_types,
            source_boosts=source_boosts,
        )

    def _search_linear(
        self,
        query: str,
        top_k: int,
        index: InvertedIndex | None,
        *,
        course_code: str | None,
        source_types: list[str] | tuple[str, ...] | None,
        min_score: float | None,
        preferred_section_types: list[str] | tuple[str, ...] | None,
        source_boosts: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """改造前的线性加权；倒排可用时并入 BM25 独立召回的额外候选。"""
        candidate_count = min(
            max(1, top_k * config.rerank_candidate_multiplier),
            self.count,
        )
        where = self._where_filter(course_code, source_types)
        query_embedding = self.encoder.encode([query])
        query_args: dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": candidate_count,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where
        result = self.collection.query(
            **query_args,
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        lexical_score_of = self._lexical_scorer(query)
        weights = self.weights
        hits: list[dict[str, Any]] = []
        dense_ids: set[str] = set()
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            stored = dict(metadata or {})
            try:
                domain_metadata = json.loads(stored.pop("metadata_json", "{}"))
            except json.JSONDecodeError:
                domain_metadata = {}
            vector_score = 1.0 - float(distance)
            if min_score is not None and vector_score < min_score:
                continue
            dense_ids.add(str(chunk_id))
            lexical_score = lexical_score_of(document or "")
            section_type = (
                stored.get("section_type")
                or domain_metadata.get("section_type")
                or ""
            )
            section_boost = (
                weights.section_boost
                if preferred_section_types
                and section_type in preferred_section_types
                else 0.0
            )
            source_type = (
                stored.get("source_type")
                or domain_metadata.get("source_type")
                or ""
            )
            source_boost = (source_boosts or {}).get(str(source_type), 0.0)
            rerank_score = (
                vector_score * weights.vector
                + lexical_score * weights.lexical
                + section_boost
                + source_boost
            )
            hits.append(
                {
                    **stored,
                    "chunk_id": chunk_id,
                    "text": document or "",
                    "metadata": domain_metadata,
                    "score": vector_score,
                    "vector_score": vector_score,
                    "lexical_score": lexical_score,
                    "rerank_score": rerank_score,
                }
            )
        recall = self._bm25_recall(index, query, candidate_count, where)
        if recall is not None:
            hits.extend(
                self._bm25_only_hits(
                    index,
                    recall,
                    dense_ids,
                    query_embedding,
                    lexical_score_of,
                    weights,
                    min_score=min_score,
                    preferred_section_types=preferred_section_types,
                    source_boosts=source_boosts,
                )
            )
            # 附加来源标记（dense / bm25 / both）；不删改任何既有键。
            bm25_ids = {
                str(index.chunk_id(position)) for position, _ in recall
            }
            for hit in hits:
                if "retrieval" in hit:
                    continue
                hit["retrieval"] = (
                    "both" if str(hit["chunk_id"]) in bm25_ids else "dense"
                )
        hits.sort(key=lambda hit: hit["rerank_score"], reverse=True)
        return hits[:top_k]

    def _search_bm25(
        self,
        query: str,
        top_k: int,
        index: InvertedIndex,
        *,
        course_code: str | None,
        source_types: list[str] | tuple[str, ...] | None,
        min_score: float | None,
        preferred_section_types: list[str] | tuple[str, ...] | None,
        source_boosts: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """A5 对照臂：只用内存倒排召回（``RETRIEVAL_FUSION=bm25``）。

        这一路没有可比的向量相似度，因此 ``min_score`` 不生效；命中结构与稠密
        路径保持一致（额外带 ``retrieval="bm25"``），供上层直接消费。
        """
        weights = self.weights
        where = self._where_filter(course_code, source_types)
        candidate_count = min(
            max(1, top_k * config.rerank_candidate_multiplier),
            self.count,
        )
        hits: list[dict[str, Any]] = []
        for position, lexical_score in index.search(
            query, top_k=candidate_count, where=where
        ):
            attributes = index.attributes(position)
            metadata = index.metadata(position)
            section_boost = (
                weights.section_boost
                if preferred_section_types
                and str(attributes.get("section_type") or "")
                in preferred_section_types
                else 0.0
            )
            source_boost = (source_boosts or {}).get(
                str(attributes.get("source_type") or ""), 0.0
            )
            hits.append(
                {
                    **attributes,
                    "chunk_id": index.chunk_id(position),
                    "text": index.text(position),
                    "metadata": metadata,
                    "score": lexical_score,
                    "vector_score": 0.0,
                    "lexical_score": lexical_score,
                    "rerank_score": lexical_score + section_boost + source_boost,
                    "retrieval": "bm25",
                }
            )
        hits.sort(key=lambda hit: hit["rerank_score"], reverse=True)
        return hits[:top_k]

    def _bm25_recall(
        self,
        index: InvertedIndex | None,
        query: str,
        top_k: int,
        where: dict[str, Any] | None,
    ) -> list[tuple[int, float]] | None:
        """线性模式下的独立 BM25 召回；不可用时返回 ``None``（保持旧行为）。"""
        if index is None or not index.usable:
            return None
        if not callable(getattr(self.collection, "get", None)):
            # 集合不支持 ``get``（例如轻量替身）时无法补算向量分，退回旧路径。
            return None
        try:
            return index.search(query, top_k=top_k, where=where)
        except Exception:  # noqa: BLE001 - 倒排异常不允许影响主链路
            return None

    def _bm25_only_hits(
        self,
        index: InvertedIndex,
        recall: list[tuple[int, float]],
        dense_ids: set[str],
        query_embedding: Any,
        lexical_score_of,
        weights: RerankWeights,
        *,
        min_score: float | None,
        preferred_section_types: list[str] | tuple[str, ...] | None,
        source_boosts: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """把「稠密没召回、BM25 召回了」的块用 ``collection.get`` 补算向量分。"""
        wanted: list[tuple[int, str]] = []
        for position, _score in recall:
            chunk_id = str(index.chunk_id(position))
            if not chunk_id or chunk_id in dense_ids:
                continue
            wanted.append((position, chunk_id))
        if not wanted:
            return []
        embeddings = self._fetch_embeddings([chunk_id for _, chunk_id in wanted])
        query_vector = None
        if query_embedding is not None and len(query_embedding):
            query_vector = query_embedding[0]
        extra: list[dict[str, Any]] = []
        for position, chunk_id in wanted:
            vector = embeddings.get(chunk_id)
            if vector is None or query_vector is None:
                continue
            vector_score = self._cosine_similarity(query_vector, vector)
            if min_score is not None and vector_score < min_score:
                continue
            chunk = index.document_at(position)
            stored = _scalar_metadata_from_chunk(chunk)
            try:
                domain_metadata = json.loads(stored.pop("metadata_json", "{}"))
            except json.JSONDecodeError:
                domain_metadata = {}
            lexical_score = lexical_score_of(str(chunk.get("text", "") or ""))
            section_type = (
                stored.get("section_type")
                or domain_metadata.get("section_type")
                or ""
            )
            section_boost = (
                weights.section_boost
                if preferred_section_types
                and section_type in preferred_section_types
                else 0.0
            )
            source_type = (
                stored.get("source_type")
                or domain_metadata.get("source_type")
                or ""
            )
            source_boost = (source_boosts or {}).get(str(source_type), 0.0)
            extra.append(
                {
                    **stored,
                    "chunk_id": chunk_id,
                    "text": str(chunk.get("text", "") or ""),
                    "metadata": domain_metadata,
                    "score": vector_score,
                    "vector_score": vector_score,
                    "lexical_score": lexical_score,
                    "rerank_score": (
                        vector_score * weights.vector
                        + lexical_score * weights.lexical
                        + section_boost
                        + source_boost
                    ),
                    "retrieval": "bm25",
                }
            )
        return extra

    def _fetch_embeddings(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """用 Chroma ``get(include=["embeddings"])`` 补齐 BM25-only 候选的向量。"""
        if not chunk_ids:
            return {}
        try:
            result = self.collection.get(ids=chunk_ids, include=["embeddings"])
        except Exception:  # noqa: BLE001 - 补算向量失败只应降级，不应中断检索
            return {}
        if not isinstance(result, dict):
            return {}
        raw = result.get("embeddings")
        if raw is None:
            return {}
        mapping: dict[str, list[float]] = {}
        for position, chunk_id in enumerate(list(result.get("ids") or [])):
            try:
                vector = raw[position]
            except (IndexError, KeyError, TypeError):
                continue
            if vector is None:
                continue
            try:
                mapping[str(chunk_id)] = [float(value) for value in vector]
            except (TypeError, ValueError):
                continue
        return mapping

    @staticmethod
    def _cosine_similarity(left: Any, right: Any) -> float:
        """与 Chroma cosine 空间一致的余弦相似度（``vector_score = 1 - distance``）。"""
        if left is None or right is None:
            return 0.0
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
        if not left_values or len(left_values) != len(right_values):
            return 0.0
        dot = sum(x * y for x, y in zip(left_values, right_values))
        left_norm = math.sqrt(sum(x * x for x in left_values))
        right_norm = math.sqrt(sum(y * y for y in right_values))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

    def _search_rrf(
        self,
        query: str,
        top_k: int,
        index: InvertedIndex,
        *,
        course_code: str | None,
        source_types: list[str] | tuple[str, ...] | None,
        min_score: float | None,
        preferred_section_types: list[str] | tuple[str, ...] | None,
        source_boosts: dict[str, float] | None,
    ) -> list[dict[str, Any]]:
        """dense / BM25 两路独立召回后 RRF 融合，再叠加既有业务加权。"""
        candidate_count = min(
            max(1, top_k * config.rerank_candidate_multiplier),
            self.count,
        )
        where = self._where_filter(course_code, source_types)
        query_args: dict[str, Any] = {
            "query_embeddings": self.encoder.encode([query]),
            "n_results": candidate_count,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where
        result = self.collection.query(**query_args)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        entries: dict[str, dict[str, Any]] = {}
        dense_rank: dict[str, int] = {}
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            vector_score = 1.0 - float(distance)
            if min_score is not None and vector_score < min_score:
                # ``min_score`` 只作用于稠密一路（BM25-only 没有向量分可比）。
                continue
            key = str(chunk_id)
            if key in dense_rank:
                continue
            stored = dict(metadata or {})
            try:
                domain_metadata = json.loads(stored.pop("metadata_json", "{}"))
            except json.JSONDecodeError:
                domain_metadata = {}
            dense_rank[key] = len(dense_rank) + 1
            entries[key] = {
                "chunk_id": chunk_id,
                "text": document or "",
                "stored": stored,
                "metadata": domain_metadata,
                "vector_score": vector_score,
            }
        bm25_rank: dict[str, int] = {}
        for position, _score in index.search(
            query,
            top_k=candidate_count,
            where=where,
        ):
            key = str(index.chunk_id(position))
            if not key or key in bm25_rank:
                continue
            bm25_rank[key] = len(bm25_rank) + 1
            if key not in entries:
                chunk = index.document_at(position)
                stored = _scalar_metadata_from_chunk(chunk)
                try:
                    domain_metadata = json.loads(
                        stored.pop("metadata_json", "{}")
                    )
                except json.JSONDecodeError:
                    domain_metadata = {}
                entries[key] = {
                    "chunk_id": key,
                    "text": str(chunk.get("text", "") or ""),
                    "stored": stored,
                    "metadata": domain_metadata,
                    "vector_score": None,
                }
        lexical_score_of = self._lexical_scorer(query)
        weights = self.weights
        hits: list[dict[str, Any]] = []
        for key, candidate in entries.items():
            in_dense = key in dense_rank
            in_bm25 = key in bm25_rank
            rrf_score = 0.0
            if in_dense:
                rrf_score += 1.0 / (RRF_K + dense_rank[key])
            if in_bm25:
                rrf_score += 1.0 / (RRF_K + bm25_rank[key])
            stored = candidate["stored"]
            domain_metadata = candidate["metadata"]
            vector_score = float(candidate["vector_score"] or 0.0)
            lexical_score = lexical_score_of(candidate["text"])
            section_type = (
                stored.get("section_type")
                or domain_metadata.get("section_type")
                or ""
            )
            section_boost = (
                weights.section_boost
                if preferred_section_types
                and section_type in preferred_section_types
                else 0.0
            )
            source_type = (
                stored.get("source_type")
                or domain_metadata.get("source_type")
                or ""
            )
            source_boost = (source_boosts or {}).get(str(source_type), 0.0)
            hits.append(
                {
                    **stored,
                    "chunk_id": candidate["chunk_id"],
                    "text": candidate["text"],
                    "metadata": domain_metadata,
                    "score": vector_score,
                    "vector_score": vector_score,
                    "lexical_score": lexical_score,
                    "rrf_score": rrf_score,
                    "rerank_score": rrf_score + section_boost + source_boost,
                    "retrieval": (
                        "both"
                        if in_dense and in_bm25
                        else ("dense" if in_dense else "bm25")
                    ),
                }
            )
        hits.sort(key=lambda hit: hit["rerank_score"], reverse=True)
        return hits[:top_k]

    retrieve = search
