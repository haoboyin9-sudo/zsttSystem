"""Shared lexical tokenisation and BM25 scoring for retrieval reranking.

The corpus is small (a few thousand Chinese syllabus chunks), so document
frequencies are computed offline once per index build and reused at query
time.  Keeping the tokeniser in one place guarantees that the statistics
written by :mod:`src.data_processing.lexical_stats` and the scores computed by
:class:`src.online_service.chroma_retriever.ChromaRetriever` agree on what a
term is.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping

_LATIN_TOKEN = re.compile(r"[a-z][a-z0-9]+")
_NON_CJK = re.compile(r"[^\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Split text into latin words plus Chinese character bigrams.

    Bigrams are emitted with repetition so that term frequency is meaningful
    for BM25; use :func:`term_set` when only the vocabulary is needed.
    """
    lowered = (text or "").lower()
    tokens = _LATIN_TOKEN.findall(lowered)
    chinese = _NON_CJK.sub("", lowered)
    tokens.extend(
        chinese[index : index + 2] for index in range(len(chinese) - 1)
    )
    return tokens


def term_set(text: str) -> set[str]:
    """Vocabulary of ``text`` under :func:`tokenize`."""
    return set(tokenize(text))


#: chunk 顶层里参与 Chroma 标量 metadata / ``where`` 过滤的字段。
_CHUNK_SCALAR_KEYS: tuple[str, ...] = (
    "course_code",
    "course_name",
    "source_type",
    "section_type",
    "section",
    "source_file",
    "chunk_index",
    "chapter_no",
    "is_header",
    "parent_id",
    "prev_chunk_id",
    "next_chunk_id",
    "document_hash",
)


def _values_equal(left: Any, right: Any) -> bool:
    """宽松等值比较：容忍 ``"1"`` / ``1`` 这类类型差异，``None`` 只等于 ``None``。"""
    if left is right:
        return True
    if left is None or right is None:
        return False
    if left == right:
        return True
    try:
        return str(left) == str(right)
    except Exception:  # pragma: no cover - 极端自定义对象
        return False


def _match_condition(value: Any, condition: Any) -> bool:
    """判断单个字段是否满足 Chroma 风格的条件（等值或 ``$eq/$ne/$in/...``）。"""
    if isinstance(condition, Mapping):
        for operator, operand in condition.items():
            if operator in ("$eq",):
                if not _values_equal(value, operand):
                    return False
            elif operator == "$ne":
                if _values_equal(value, operand):
                    return False
            elif operator == "$in":
                if not any(_values_equal(value, item) for item in operand or ()):
                    return False
            elif operator == "$nin":
                if any(_values_equal(value, item) for item in operand or ()):
                    return False
            elif operator == "$gt":
                if value is None or not value > operand:
                    return False
            elif operator == "$gte":
                if value is None or not value >= operand:
                    return False
            elif operator == "$lt":
                if value is None or not value < operand:
                    return False
            elif operator == "$lte":
                if value is None or not value <= operand:
                    return False
            else:
                # 未知算子：保守地视为不匹配，避免把不该召回的块放进来。
                return False
        return True
    return _values_equal(value, condition)


def flat_attributes(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """把一条 chunk 压平成「用户 metadata + 顶层标量」的属性字典。

    Chroma 把顶层标量（``chunk_index``、``is_header`` 等）和业务 metadata 一起
    存进标量 metadata，过滤时看到的就是这张扁平表；:class:`InvertedIndex` 用
    同一张表做 ``where`` 过滤，两路过滤条件因此天然一致。
    """
    attributes: dict[str, Any] = dict(chunk.get("metadata") or {})
    for key in _CHUNK_SCALAR_KEYS:
        value = chunk.get(key)
        if value is None or attributes.get(key):
            continue
        attributes[key] = value
    return attributes


def overlap_score(query_terms: set[str], document_terms: set[str]) -> float:
    """Legacy lexical score: share of query terms present in the document."""
    if not query_terms:
        return 0.0
    return len(query_terms & document_terms) / len(query_terms)


def idf(document_frequency: float, document_count: float) -> float:
    """Robertson/Sparck-Jones IDF with the usual +0.5 smoothing.

    Clamped at zero so that terms occurring in almost every document (``课程``,
    ``信息``, ``管理`` in this corpus) contribute nothing instead of a negative
    score.
    """
    if document_count <= 0:
        return 0.0
    numerator = document_count - document_frequency + 0.5
    denominator = document_frequency + 0.5
    return max(0.0, math.log(1.0 + numerator / denominator))


class BM25Scorer:
    """Okapi BM25 over precomputed corpus statistics.

    ``stats`` is the payload written by
    :func:`src.data_processing.lexical_stats.build_lexical_stats`.
    """

    def __init__(
        self,
        stats: Mapping[str, object],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.document_count = float(stats.get("document_count", 0) or 0)
        self.average_length = float(stats.get("average_length", 0.0) or 0.0)
        raw_frequencies = stats.get("document_frequency") or {}
        self.document_frequency: dict[str, float] = {
            str(term): float(count)
            for term, count in dict(raw_frequencies).items()
        }
        self._idf_cache: dict[str, float] = {}

    @property
    def usable(self) -> bool:
        """Whether the statistics are complete enough to score with."""
        return (
            self.document_count > 0
            and self.average_length > 0
            and bool(self.document_frequency)
        )

    def term_idf(self, term: str) -> float:
        cached = self._idf_cache.get(term)
        if cached is None:
            cached = idf(
                self.document_frequency.get(term, 0.0),
                self.document_count,
            )
            self._idf_cache[term] = cached
        return cached

    def score(self, query_tokens: Iterable[str], document_tokens: Iterable[str]) -> float:
        """Raw BM25 score of one document against one query."""
        document = list(document_tokens)
        if not document:
            return 0.0
        frequencies = Counter(document)
        length_norm = self.k1 * (
            1.0 - self.b + self.b * len(document) / self.average_length
        )
        total = 0.0
        for term in dict.fromkeys(query_tokens):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            total += self.term_idf(term) * (
                frequency * (self.k1 + 1.0) / (frequency + length_norm)
            )
        return total

    def normalized_score(
        self,
        query_tokens: Iterable[str],
        document_tokens: Iterable[str],
    ) -> float:
        """BM25 mapped into ``[0, 1)`` so it can be linearly blended.

        The upper bound of a raw BM25 score depends on the query, so it is
        divided by the score a perfectly matching document would receive
        (every query term present with saturated term frequency).  Queries
        whose terms are all corpus-wide stopwords score 0.
        """
        query = list(dict.fromkeys(query_tokens))
        if not query:
            return 0.0
        ceiling = sum(
            self.term_idf(term) * (self.k1 + 1.0) for term in query
        )
        if ceiling <= 0.0:
            return 0.0
        return self.score(query, document_tokens) / ceiling


class InvertedIndex:
    """内存倒排索引：BM25 独立全库召回（改造方案 M5.1）。

    :class:`BM25Scorer` 只对「调用方给定的候选」打分，稠密向量没召回的块它
    无从挽救；本类对整个语料建 ``term -> [(doc_position, tf)]`` 倒排，因此含
    精确词面的块可以被独立召回，再与向量结果做 RRF 融合（M5.2）。

    分词与 IDF 口径刻意和 :class:`BM25Scorer` 保持一致（同一 ``tokenize``、
    同一 ``idf`` 公式、同一 ``k1`` / ``b``）；传入 ``scorer`` 时直接复用它的
    文档频率，保证与 ``outputs/lexical_stats.json`` 的 IDF 完全一致。

    ``documents`` 就是 ``outputs/chunks.json`` 的元素；只读取 ``text``、
    ``chunk_id``、``metadata`` 与顶层标量字段，字段缺失一律容错。
    """

    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75

    def __init__(
        self,
        *,
        k1: float | None = None,
        b: float | None = None,
        scorer: BM25Scorer | None = None,
    ) -> None:
        if scorer is not None:
            k1 = scorer.k1 if k1 is None else k1
            b = scorer.b if b is None else b
        self.k1 = self.DEFAULT_K1 if k1 is None else float(k1)
        self.b = self.DEFAULT_B if b is None else float(b)
        self._scorer = scorer
        self._documents: list[Mapping[str, Any]] = []
        self._attributes: list[dict[str, Any]] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._total_length = 0
        self._idf_cache: dict[str, float] = {}

    # -- 构建 ---------------------------------------------------------------
    def build(self, documents: Iterable[Mapping[str, Any]]) -> int:
        """清空并重建倒排，返回收录的文档数。"""
        self._documents = []
        self._attributes = []
        self._lengths = []
        self._postings = {}
        self._total_length = 0
        self._idf_cache = {}
        for chunk in documents:
            if not isinstance(chunk, Mapping):
                continue
            text = str(chunk.get("text", "") or "")
            if not text.strip():
                # 与 lexical_stats 保持一致：空文本不计入文档数与平均长度。
                continue
            tokens = tokenize(text)
            position = len(self._documents)
            self._documents.append(chunk)
            self._attributes.append(flat_attributes(chunk))
            self._lengths.append(len(tokens))
            self._total_length += len(tokens)
            for term, frequency in Counter(tokens).items():
                self._postings.setdefault(term, []).append((position, frequency))
        return len(self._documents)

    # -- 基本属性 -----------------------------------------------------------
    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def average_length(self) -> float:
        if not self._documents:
            return 0.0
        return self._total_length / len(self._documents)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    @property
    def usable(self) -> bool:
        return bool(self._postings) and self.average_length > 0.0

    # -- 文档位置 -> chunk 映射 ---------------------------------------------
    def document_at(self, position: int) -> Mapping[str, Any]:
        return self._documents[position]

    def chunk_id(self, position: int) -> str:
        return str(self._documents[position].get("chunk_id", "") or "")

    def text(self, position: int) -> str:
        return str(self._documents[position].get("text", "") or "")

    def metadata(self, position: int) -> dict[str, Any]:
        return dict(self._documents[position].get("metadata") or {})

    def attributes(self, position: int) -> dict[str, Any]:
        """扁平属性表（业务 metadata + 顶层标量），供 ``where`` 过滤与命中组装。"""
        return dict(self._attributes[position])

    # -- 打分 ---------------------------------------------------------------
    def term_idf(self, term: str) -> float:
        cached = self._idf_cache.get(term)
        if cached is None:
            if self._scorer is not None:
                cached = self._scorer.term_idf(term)
            else:
                cached = idf(
                    float(len(self._postings.get(term, ()))),
                    float(len(self._documents)),
                )
            self._idf_cache[term] = cached
        return cached

    def _length_norm(self, length: int) -> float:
        average = self.average_length
        if average <= 0.0:
            return self.k1
        return self.k1 * (1.0 - self.b + self.b * length / average)

    def score_document(self, position: int, query_terms: Iterable[str]) -> float:
        """单篇文档的原始 BM25 分（与 :meth:`BM25Scorer.score` 同口径）。"""
        if position < 0 or position >= len(self._documents):
            return 0.0
        frequencies = Counter(tokenize(self.text(position)))
        norm = self._length_norm(self._lengths[position])
        total = 0.0
        for term in dict.fromkeys(query_terms):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            total += self.term_idf(term) * (
                frequency * (self.k1 + 1.0) / (frequency + norm)
            )
        return total

    # -- 过滤 ---------------------------------------------------------------
    @staticmethod
    def _matches(
        metadata: Mapping[str, Any] | None,
        where: Mapping[str, Any] | None,
    ) -> bool:
        """判断扁平属性表是否满足 Chroma ``where`` 条件。

        覆盖 ``ChromaRetriever._where_filter`` 会产出的结构：字段等值、
        ``{"$in": [...]}``，以及 ``$and`` / ``$or`` / ``$not`` 组合；
        ``None`` 或空 dict 表示不过滤。
        """
        if not where:
            return True
        attributes = metadata or {}
        for key, condition in where.items():
            if key == "$and":
                if not all(
                    InvertedIndex._matches(attributes, clause)
                    for clause in condition or ()
                ):
                    return False
            elif key == "$or":
                if not any(
                    InvertedIndex._matches(attributes, clause)
                    for clause in condition or ()
                ):
                    return False
            elif key == "$not":
                if InvertedIndex._matches(attributes, condition):
                    return False
            elif not _match_condition(attributes.get(key), condition):
                return False
        return True

    # -- 召回 ---------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int,
        where: Mapping[str, Any] | None = None,
    ) -> list[tuple[int, float]]:
        """全库 BM25 召回，返回 ``[(doc_position, score), ...]``，按分数降序。

        ``where`` 语义与 Chroma 过滤一致（见 :meth:`_matches`）；同分按文档
        位置升序，保证结果可复现。
        """
        if top_k <= 0 or not self._postings or self.average_length <= 0.0:
            return []
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            return []
        scores: dict[int, float] = {}
        for term in query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            weight = self.term_idf(term)
            if weight <= 0.0:
                continue
            for position, frequency in postings:
                norm = self._length_norm(self._lengths[position])
                scores[position] = scores.get(position, 0.0) + weight * (
                    frequency * (self.k1 + 1.0) / (frequency + norm)
                )
        ranked = [
            (position, score)
            for position, score in scores.items()
            if score > 0.0 and self._matches(self._attributes[position], where)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]


def build_inverted_index(
    documents: Iterable[Mapping[str, Any]],
    *,
    k1: float | None = None,
    b: float | None = None,
    stats: Mapping[str, Any] | None = None,
) -> InvertedIndex:
    """从 chunk 语料构建 :class:`InvertedIndex`（可选复用离线统计的 IDF）。"""
    scorer: BM25Scorer | None = None
    if stats:
        options: dict[str, float] = {}
        if k1 is not None:
            options["k1"] = float(k1)
        if b is not None:
            options["b"] = float(b)
        candidate = BM25Scorer(stats, **options)
        scorer = candidate if candidate.usable else None
    index = InvertedIndex(k1=k1, b=b, scorer=scorer)
    index.build(documents)
    return index
