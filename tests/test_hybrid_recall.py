"""混合召回测试（改造方案 M5.1 / M5.2、P2-1）。

覆盖：BM25 独立全库召回、RRF 融合公式、两路过滤条件一致、``chunks.json``
缺失时退化为改造前行为、linear 模式下 BM25-only 候选补算向量分。
全部用 fake collection / monkeypatch，不连真实 Chroma 服务、不依赖 ``data/``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import pytest

from src.data_processing import chroma_index
from src.online_service.chroma_retriever import (
    RRF_K,
    ChromaRetriever,
    RerankWeights,
)
from src.utils.lexical import (
    BM25Scorer,
    InvertedIndex,
    build_inverted_index,
    flat_attributes,
    tokenize,
)


def _chunk(
    chunk_id: str,
    text: str,
    *,
    course_code: str = "IM399",
    source_type: str = "syllabus",
    **extra: Any,
) -> dict[str, Any]:
    """构造一条 ``chunks.json`` 形态的 chunk。"""
    metadata = {"source_type": source_type, "course_code": course_code}
    metadata.update(extra.pop("metadata", {}))
    chunk: dict[str, Any] = {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": metadata,
    }
    chunk.update(extra)
    return chunk


CORPUS = [
    _chunk("objective", "课程目标 管理运筹学 线性规划"),
    _chunk("teacher", "课程负责人：李四"),
    _chunk("plan", "培养方案课程信息 学分 3", source_type="training_plan"),
    _chunk("archive", "档案学 档案分类", course_code="IM204"),
]


class _Encoder:
    """固定输出一个查询向量的编码器替身。"""

    def __init__(self, vector: list[float]) -> None:
        self.vector = list(vector)

    def encode(self, _texts: list[str]) -> list[list[float]]:
        return [list(self.vector)]


class _FakeCollection:
    """最小 Chroma collection 替身：只实现 count / query / get。"""

    def __init__(
        self,
        dense: list[dict[str, Any]],
        *,
        count: int | None = None,
        embeddings: dict[str, list[float]] | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self._dense = list(dense)
        self._count = count
        self._embeddings = dict(embeddings or {})
        self._get_error = get_error
        self.get_calls: list[list[str]] = []

    def count(self) -> int:
        if self._count is not None:
            return self._count
        return len(self._dense)

    def query(self, **_kwargs: Any) -> dict[str, list[list[Any]]]:
        where = _kwargs.get("where")
        rows = [
            item
            for item in self._dense
            if _chroma_reference_matches(item.get("metadata", {}), where)
        ]
        return {
            "ids": [[item["chunk_id"] for item in rows]],
            "documents": [[item.get("text", "") for item in rows]],
            "metadatas": [[_chroma_style_metadata(item) for item in rows]],
            "distances": [[item.get("distance", 0.0) for item in rows]],
        }

    def get(self, ids: Any, include: Any = None) -> dict[str, Any]:
        self.get_calls.append(list(ids))
        if self._get_error is not None:
            raise self._get_error
        found = [chunk_id for chunk_id in ids if chunk_id in self._embeddings]
        return {
            "ids": found,
            "embeddings": [self._embeddings[chunk_id] for chunk_id in found],
        }


class _CollectionWithoutGet:
    """只支持 ``query`` 的替身，用来验证缺 ``get`` 时退回旧路径。"""

    def __init__(self, dense: list[dict[str, Any]], *, count: int) -> None:
        self._dense = list(dense)
        self._count = count

    def count(self) -> int:
        return self._count

    def query(self, **_kwargs: Any) -> dict[str, list[list[Any]]]:
        where = _kwargs.get("where")
        rows = [
            item
            for item in self._dense
            if _chroma_reference_matches(item.get("metadata", {}), where)
        ]
        return {
            "ids": [[item["chunk_id"] for item in rows]],
            "documents": [[item.get("text", "") for item in rows]],
            "metadatas": [[_chroma_style_metadata(item) for item in rows]],
            "distances": [[item.get("distance", 0.0) for item in rows]],
        }


def _chroma_style_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """把 fake 命中的业务 metadata 包成 Chroma 标量 metadata（含 metadata_json）。"""
    metadata = dict(item.get("metadata", {}))
    return {
        **metadata,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
    }


def _dense_hit(
    chunk_id: str,
    text: str,
    distance: float,
    *,
    course_code: str = "IM399",
    source_type: str = "syllabus",
    section_type: str | None = None,
) -> dict[str, Any]:
    """一条稠密命中；metadata 用 Chroma 标量口径（无 metadata_json 键也可）。"""
    metadata: dict[str, Any] = {
        "course_code": course_code,
        "source_type": source_type,
    }
    if section_type:
        metadata["section_type"] = section_type
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": metadata,
        "distance": distance,
    }


def _retriever(
    collection: Any,
    *,
    index: InvertedIndex | None = None,
    pinned: bool = True,
    fusion: str = "linear",
    query_vector: list[float] | None = None,
    weights: RerankWeights | None = None,
    stats_path: str | None = None,
) -> ChromaRetriever:
    """不连 Chroma 服务的检索器：手工装配属性，参考 test_lexical_scoring.py。"""
    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever.model_name = "hash"
    retriever.collection_name = "test"
    retriever.client = None
    retriever.connected = True
    retriever.collection = collection
    retriever.encoder = _Encoder(query_vector or [1.0, 0.0])
    retriever._weights = weights
    retriever._lexical_stats_path = stats_path
    retriever._bm25 = None
    retriever._bm25_loaded = False
    retriever._fusion = fusion
    if pinned:
        retriever.lexical_index = index
    else:
        retriever._lexical_index = None
        retriever._lexical_index_loaded = False
        retriever._lexical_index_pinned = False
        retriever._lexical_index_stamp = None
        retriever._lexical_index_path = None
    return retriever


# ---------------------------------------------------------------------------
# M5.1：内存倒排 + 独立 BM25 召回
# ---------------------------------------------------------------------------
def test_inverted_index_builds_postings_and_maps_chunk_ids() -> None:
    index = build_inverted_index(CORPUS)

    assert index.document_count == 4
    assert index.vocabulary_size > 0
    assert index.average_length == pytest.approx(
        sum(len(tokenize(item["text"])) for item in CORPUS) / 4
    )
    ranked = index.search("课程负责人", top_k=2)
    assert ranked, "含精确词面的块必须被召回"
    assert index.chunk_id(ranked[0][0]) == "teacher"
    assert index.metadata(ranked[0][0])["course_code"] == "IM399"
    assert index.attributes(ranked[0][0])["source_type"] == "syllabus"


def test_inverted_index_reuses_lexical_stats_idf() -> None:
    stats = {
        "document_count": 100,
        "average_length": 20,
        "document_frequency": {"课程": 99, "程负": 99, "负责": 2, "责人": 2},
    }
    index = build_inverted_index(CORPUS, stats=stats)
    scorer = BM25Scorer(stats)

    for term in ("课程", "程负", "负责", "责人", "档案"):
        assert index.term_idf(term) == pytest.approx(scorer.term_idf(term))
    # 罕见词单独就能把含它的块排到前面（口径与 BM25Scorer 一致）。
    assert index.chunk_id(index.search("课程负责人", top_k=1)[0][0]) == "teacher"


def test_inverted_index_empty_documents_are_skipped() -> None:
    index = build_inverted_index(
        [_chunk("blank", "   "), _chunk("real", "档案分类")]
    )

    assert index.document_count == 1
    assert index.chunk_id(0) == "real"
    assert index.search("   ", top_k=5) == []
    assert index.search("档案", top_k=0) == []


def test_inverted_index_where_filter_limits_recall() -> None:
    index = build_inverted_index(CORPUS)

    only_im204 = index.search(
        "档案 课程 负责人",
        top_k=10,
        where={"course_code": "IM204"},
    )
    assert [index.chunk_id(position) for position, _ in only_im204] == ["archive"]

    both = index.search(
        "档案 课程 负责人",
        top_k=10,
        where={
            "$and": [
                {"course_code": "IM399"},
                {"source_type": "syllabus"},
            ]
        },
    )
    assert {index.chunk_id(position) for position, _ in both} == {
        "objective",
        "teacher",
    }


def test_inverted_index_matches_supports_in_and_scalar_conditions() -> None:
    attributes = {"course_code": "IM204", "source_type": "syllabus"}

    assert InvertedIndex._matches(attributes, None) is True
    assert InvertedIndex._matches(attributes, {}) is True
    assert InvertedIndex._matches(attributes, {"course_code": "IM204"}) is True
    assert InvertedIndex._matches(attributes, {"course_code": "IM399"}) is False
    assert InvertedIndex._matches(
        attributes, {"source_type": {"$in": ["syllabus", "training_plan"]}}
    ) is True
    assert InvertedIndex._matches(
        attributes, {"source_type": {"$in": ["training_plan"]}}
    ) is False
    assert InvertedIndex._matches(
        attributes,
        {"$and": [{"course_code": "IM204"}, {"source_type": "syllabus"}]},
    ) is True
    assert InvertedIndex._matches(
        attributes,
        {"$and": [{"course_code": "IM204"}, {"source_type": "training_plan"}]},
    ) is False
    assert InvertedIndex._matches(attributes, {"$not": {"course_code": "IM204"}}) is False


# ---------------------------------------------------------------------------
# 两路过滤条件一致
# ---------------------------------------------------------------------------
def _chroma_reference_matches(metadata: dict[str, Any], where: Any) -> bool:
    """测试内独立实现的一份 Chroma ``where`` 语义（刻意不复用被测代码）。"""
    if not where:
        return True
    for key, condition in where.items():
        if key == "$and":
            if not all(_chroma_reference_matches(metadata, c) for c in condition):
                return False
            continue
        if key == "$or":
            if not any(_chroma_reference_matches(metadata, c) for c in condition):
                return False
            continue
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$in" in condition:
                if value not in condition["$in"]:
                    return False
                continue
            raise AssertionError(f"unexpected operator {condition!r}")
        if value != condition:
            return False
    return True


@pytest.mark.parametrize(
    "where",
    [
        None,
        {"course_code": "IM204"},
        {"source_type": "syllabus"},
        {"source_type": {"$in": ["syllabus", "training_plan"]}},
        {"$and": [{"course_code": "IM399"}, {"source_type": "syllabus"}]},
    ],
)
def test_bm25_and_chroma_filters_agree(where: Any) -> None:
    builder = getattr(chroma_index, "_chroma_metadata", None)
    if not callable(builder):
        pytest.skip("chroma_index._chroma_metadata 不存在，跳过两路口径比对")

    chroma_side = {
        chunk["chunk_id"]
        for chunk in CORPUS
        if _chroma_reference_matches(builder(chunk), where)
    }
    bm25_side = {
        chunk["chunk_id"]
        for chunk in CORPUS
        if InvertedIndex._matches(flat_attributes(chunk), where)
    }

    assert chroma_side == bm25_side


def test_where_filter_shape_matches_chroma_retriever_builder() -> None:
    assert ChromaRetriever._where_filter(None, None) is None
    assert ChromaRetriever._where_filter("IM104", None) == {"course_code": "IM104"}
    assert ChromaRetriever._where_filter(None, ["syllabus"]) == {
        "source_type": {"$in": ["syllabus"]}
    }
    combined = ChromaRetriever._where_filter("IM104", ("syllabus", "training_plan"))
    assert combined == {
        "$and": [
            {"course_code": "IM104"},
            {"source_type": {"$in": ["syllabus", "training_plan"]}},
        ]
    }
    attributes = {"course_code": "IM104", "source_type": "syllabus"}
    assert InvertedIndex._matches(attributes, combined) is True


# ---------------------------------------------------------------------------
# M5.2：linear 模式（默认）——BM25-only 候选补算向量分
# ---------------------------------------------------------------------------
def test_linear_mode_recalls_bm25_only_chunk_with_backfilled_vector_score(
    tmp_path: Path,
) -> None:
    dense = [
        _dense_hit("objective", "课程目标 管理运筹学 线性规划", 0.1),
        _dense_hit("archive", "档案学 档案分类", 0.4, course_code="IM204"),
    ]
    collection = _FakeCollection(
        dense,
        count=len(CORPUS),
        embeddings={"teacher": [1.0, 0.0]},
    )
    retriever = _retriever(
        collection,
        index=build_inverted_index(CORPUS),
        # 让 BM25 词法分固定走 overlap 回退，断言不受真实 lexical_stats 影响。
        stats_path=str(tmp_path / "absent_stats.json"),
    )

    hits = retriever.search("课程负责人", 3)

    assert [hit["chunk_id"] for hit in hits][0] == "teacher"
    teacher = hits[0]
    # 稠密没召回，BM25 独立召回后用 collection.get 补算向量分。
    assert teacher["retrieval"] == "bm25"
    assert teacher["vector_score"] == pytest.approx(1.0)
    assert teacher["lexical_score"] == pytest.approx(1.0)
    assert teacher["rerank_score"] == pytest.approx(
        1.0 * retriever.weights.vector + 1.0 * retriever.weights.lexical
    )
    assert collection.get_calls == [["teacher", "plan"]]
    assert "teacher" not in [item["chunk_id"] for item in dense]


def test_linear_mode_marks_hits_present_in_both_lists() -> None:
    dense = [_dense_hit("objective", "课程目标 管理运筹学 线性规划", 0.1)]
    collection = _FakeCollection(
        dense,
        count=len(CORPUS),
        embeddings={"teacher": [1.0, 0.0]},
    )
    retriever = _retriever(collection, index=build_inverted_index(CORPUS))

    by_id = {hit["chunk_id"]: hit for hit in retriever.search("课程负责人 课程目标", 4)}

    assert by_id["objective"]["retrieval"] == "both"
    assert by_id["teacher"]["retrieval"] == "bm25"


def test_linear_mode_backfill_survives_collection_get_error() -> None:
    dense = [_dense_hit("objective", "课程目标 管理运筹学 线性规划", 0.1)]
    collection = _FakeCollection(
        dense,
        count=len(CORPUS),
        embeddings={"teacher": [1.0, 0.0]},
        get_error=RuntimeError("boom"),
    )
    retriever = _retriever(collection, index=build_inverted_index(CORPUS))

    hits = retriever.search("课程负责人 课程目标", 4)

    assert [hit["chunk_id"] for hit in hits] == ["objective"]


def test_linear_mode_skips_backfill_without_collection_get() -> None:
    dense = [_dense_hit("objective", "课程目标 管理运筹学 线性规划", 0.1)]
    collection = _CollectionWithoutGet(dense, count=len(CORPUS))
    retriever = _retriever(collection, index=build_inverted_index(CORPUS))

    hits = retriever.search("课程负责人 课程目标", 4)

    assert [hit["chunk_id"] for hit in hits] == ["objective"]
    assert "retrieval" not in hits[0]


# ---------------------------------------------------------------------------
# M5.2：rrf 模式
# ---------------------------------------------------------------------------
def test_rrf_ranking_follows_the_manual_formula() -> None:
    corpus = [
        _chunk("A", "课程负责人 档案管理"),
        _chunk("B", "线性规划 单纯形法"),
        _chunk("C", "课程负责人 课程负责人 档案"),
        _chunk("D", "档案 分类"),
    ]
    index = build_inverted_index(corpus)
    query = "课程负责人 档案"
    bm25_order = [index.chunk_id(p) for p, _ in index.search(query, top_k=10)]
    assert bm25_order == ["C", "A", "D"]
    dense = [
        _dense_hit("A", "课程负责人 档案管理", 0.1),
        _dense_hit("B", "线性规划 单纯形法", 0.3),
        _dense_hit("C", "课程负责人 课程负责人 档案", 0.5),
    ]
    collection = _FakeCollection(dense, count=len(corpus))
    retriever = _retriever(collection, index=index, fusion="rrf")

    hits = retriever.search(query, 4)

    expected = {
        "A": 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 2),
        "C": 1.0 / (RRF_K + 3) + 1.0 / (RRF_K + 1),
        "B": 1.0 / (RRF_K + 2),
        "D": 1.0 / (RRF_K + 3),
    }
    assert [hit["chunk_id"] for hit in hits] == ["A", "C", "B", "D"]
    for hit in hits:
        assert hit["rrf_score"] == pytest.approx(expected[hit["chunk_id"]])
        # 没有 section / source 加权时，最终分就是 RRF。
        assert hit["rerank_score"] == pytest.approx(expected[hit["chunk_id"]])
    assert {hit["chunk_id"]: hit["retrieval"] for hit in hits} == {
        "A": "both",
        "C": "both",
        "B": "dense",
        "D": "bm25",
    }
    assert hits[-1]["vector_score"] == 0.0


def test_rrf_keeps_business_boosts_and_reports_them() -> None:
    corpus = [
        _chunk("plain", "线性规划 单纯形法", metadata={"section_type": "other"}),
        _chunk(
            "objective",
            "线性规划 课程目标",
            metadata={"section_type": "course_objectives"},
        ),
    ]
    index = build_inverted_index(corpus)
    dense = [
        _dense_hit("plain", "线性规划 单纯形法", 0.1, section_type="other"),
        _dense_hit(
            "objective",
            "线性规划 课程目标",
            0.3,
            section_type="course_objectives",
        ),
    ]
    collection = _FakeCollection(dense, count=len(corpus))
    retriever = _retriever(collection, index=index, fusion="rrf")

    hits = retriever.search(
        "线性规划",
        2,
        preferred_section_types=("course_objectives",),
    )

    # RRF 里 plain 名次更靠前，但 course_objectives 的 section 加权把它翻过来。
    assert [hit["chunk_id"] for hit in hits] == ["objective", "plain"]
    assert hits[0]["rerank_score"] == pytest.approx(
        hits[0]["rrf_score"] + retriever.weights.section_boost
    )


def test_rrf_falls_back_to_lazy_index_and_respects_where_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps(CORPUS, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("CHUNKS_OUTPUT_PATH", str(chunks_path))
    monkeypatch.setenv("LEXICAL_STATS_PATH", str(tmp_path / "absent_stats.json"))
    dense = [_dense_hit("archive", "档案学 档案分类", 0.2, course_code="IM204")]
    collection = _FakeCollection(dense, count=len(CORPUS))
    retriever = _retriever(
        collection,
        index=None,
        pinned=False,
        fusion="rrf",
    )

    assert retriever.lexical_index is not None
    assert retriever.lexical_index.document_count == len(CORPUS)

    hits = retriever.search("课程负责人", 5, course_code="IM399")

    assert {"objective", "teacher"} <= {hit["chunk_id"] for hit in hits}
    assert all(hit["metadata"]["course_code"] == "IM399" for hit in hits)


# ---------------------------------------------------------------------------
# 降级：没有 chunks.json 时行为与改造前一致
# ---------------------------------------------------------------------------
def test_search_degrades_to_legacy_behaviour_without_chunks_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUNKS_OUTPUT_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("LEXICAL_STATS_PATH", str(tmp_path / "absent_stats.json"))
    dense = [
        _dense_hit("objective", "课程目标 管理运筹学", 0.1),
        _dense_hit("archive", "档案学 档案分类", 0.4, course_code="IM204"),
    ]
    collection = _FakeCollection(dense, count=2)
    weights = RerankWeights()
    weights.vector = 0.75
    weights.lexical = 0.15

    for fusion in ("linear", "rrf"):
        retriever = _retriever(
            collection,
            index=None,
            pinned=False,
            fusion=fusion,
            weights=weights,
        )
        assert retriever.lexical_index is None

        hits = retriever.search("课程目标 管理运筹学", 2)

        assert [hit["chunk_id"] for hit in hits] == ["objective", "archive"]
        assert hits[0]["vector_score"] == pytest.approx(0.9)
        assert hits[0]["lexical_score"] == pytest.approx(1.0)
        assert hits[0]["rerank_score"] == pytest.approx(
            0.9 * 0.75 + 1.0 * 0.15
        )
        # 没有倒排索引时不追加来源标记，命中结构与改造前逐字段一致。
        assert set(hits[0]) == {
            "chunk_id",
            "text",
            "metadata",
            "score",
            "vector_score",
            "lexical_score",
            "rerank_score",
            "course_code",
            "source_type",
        }


def test_linear_mode_without_index_matches_legacy_scores() -> None:
    dense = [
        _dense_hit("objective", "课程目标 管理运筹学", 0.1),
        _dense_hit("archive", "档案学 档案分类", 0.4, course_code="IM204"),
    ]
    collection = _FakeCollection(dense, count=2)

    legacy = _retriever(collection, index=None)
    legacy_hits = legacy.search("课程目标 管理运筹学", 2)

    assert [hit["chunk_id"] for hit in legacy_hits] == ["objective", "archive"]
    assert "retrieval" not in legacy_hits[0]
    assert collection.get_calls == []


def test_inverted_index_usable_flag_and_empty_corpus() -> None:
    empty = build_inverted_index([])

    assert empty.document_count == 0
    assert empty.usable is False
    assert empty.search("课程", top_k=5) == []
    assert empty.average_length == 0.0


def test_fusion_property_defaults_to_linear_and_normalises_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RETRIEVAL_FUSION", raising=False)
    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever._fusion = None

    assert retriever.fusion == "linear"

    retriever._fusion = "RRF"
    assert retriever.fusion == "rrf"

    # M5.3 的 A5 对照臂：dense（只用向量一路）/ bm25（只用倒排一路）是合法取值。
    monkeypatch.setenv("RETRIEVAL_FUSION", "dense")
    retriever._fusion = None
    assert retriever.fusion == "dense"

    monkeypatch.setenv("RETRIEVAL_FUSION", "bm25")
    retriever._fusion = None
    assert retriever.fusion == "bm25"

    # 其它取值一律退回 linear。
    monkeypatch.setenv("RETRIEVAL_FUSION", "bogus")
    retriever._fusion = None
    assert retriever.fusion == "linear"


def test_constructor_accepts_injected_index_and_pins_it() -> None:
    index = build_inverted_index(CORPUS)
    retriever = ChromaRetriever(
        "hash",
        client=chromadb.EphemeralClient(),
        collection_name="hybrid-recall-missing-collection",
        lexical_index=index,
        fusion="rrf",
    )

    assert retriever.fusion == "rrf"
    assert retriever.lexical_index is index
    # 集合不存在：search 必须安静返回空表，而不是抛错。
    assert retriever.search("课程负责人", 3) == []


def test_scalar_metadata_matches_chroma_side_for_bm25_only_hits() -> None:
    builder = getattr(chroma_index, "_chroma_metadata", None)
    if not callable(builder):
        pytest.skip("chroma_index._chroma_metadata 不存在，跳过字段口径比对")
    chunk = {
        "chunk_id": "teacher",
        "text": "课程负责人：李四",
        "section": "课程基本信息",
        "source_file": "IM399.docx",
        "chunk_index": 7,
        "is_header": True,
        "metadata": {
            "source_type": "syllabus",
            "course_code": "IM399",
            "course_name": "管理运筹学",
            "section_type": "basic_info",
        },
    }
    collection = _FakeCollection(
        [],
        count=1,
        embeddings={"teacher": [1.0, 0.0]},
    )
    retriever = _retriever(collection, index=build_inverted_index([chunk]))

    hit = retriever.search("课程负责人", 2)[0]

    chroma_side = builder(chunk)
    for key, value in chroma_side.items():
        if key == "metadata_json":
            continue
        assert hit.get(key) == value, key
    assert hit["metadata"]["course_code"] == "IM399"
    assert hit["text"] == chunk["text"]
