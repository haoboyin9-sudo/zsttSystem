"""P1-6 / P1-8 结构路由测试（改造方案 §7 / §4 M4.1-M4.2 / §6 P1-8）。

覆盖三件事：

* 路由优先级：「第三章讲什么」归 structure 而不是 content，FACT 优先于 structure；
* 第 N 章不存在时走既有拒答结构（``metadata.status == "refused"``）；
* 完备性检查（M6.4）把证据没覆盖到的章节标题补齐，并回报 ``chapter_coverage``。

全部离线：课程、文档树、命中都用构造数据，不连 ChromaDB、不调 LLM。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.data_processing.doc_tree import build_doc_tree
from src.online_service.doc_tree import DocTree
from src.online_service.generator import select_generation_evidence
from src.online_service.query_router import QueryRouter, QueryType

COURSE_CODE = "IM127"
COURSE_NAME = "信息素养与创新思维"
DOC_HASH = "hashIM127"
SOURCE_FILE = "2025-1教学大纲汇总/IM127信息素养与创新思维.docx"


def _chunk(
    chunk_id: str,
    chunk_index: int,
    heading_path: list[str],
    text: str,
    *,
    section_type: str = "teaching_schedule",
    is_header: bool = False,
) -> dict[str, Any]:
    """构造一个符合新 parser 产物形态的大纲 chunk。"""
    metadata: dict[str, Any] = {
        "source_type": "syllabus",
        "section_type": section_type,
        "course_code": COURSE_CODE,
        "course_name": COURSE_NAME,
        "source_year": "2025",
        "syllabus_section": heading_path[-1] if heading_path else "",
    }
    if is_header:
        metadata["is_header"] = True
    return {
        "chunk_id": chunk_id,
        "document_hash": DOC_HASH,
        "chunk_index": chunk_index,
        "heading_path": heading_path,
        "text": text,
        "source_file": SOURCE_FILE,
        "metadata": metadata,
    }


def _chunks() -> list[dict[str, Any]]:
    """一门课：基本信息 + 第 1-3 章 + 复习 / 期末考试。"""
    schedule = ["四、教学进度"]
    return [
        _chunk(
            "h0", 0, ["一、课程基本信息"], "课程负责人：张三",
            section_type="basic_info", is_header=True,
        ),
        _chunk("h1", 1, schedule, "本表按周次安排教学进度。"),
        _chunk(
            "h2", 2, schedule + ["第一章 课程导论"],
            "章节次序及名称：第一章 课程导论\n第3周 讲授信息素养与信息意识\n学时：2",
        ),
        _chunk(
            "h3", 3, schedule + ["第二章 搜寻信息资源"],
            "章节次序及名称：第二章 搜寻信息资源\n学时：4",
        ),
        _chunk(
            "h4", 4, schedule + ["第三章 信息检索"],
            "章节次序及名称：第三章 信息检索\n学时：4",
        ),
        _chunk("h5", 5, schedule + ["复习"], "复习答疑"),
        _chunk("h6", 6, schedule + ["期末考试"], "闭卷考试"),
    ]


def _doc_tree() -> DocTree:
    return DocTree(build_doc_tree(_chunks()))


def _courses() -> list[dict[str, Any]]:
    return [{"course_code": COURSE_CODE, "course_name": COURSE_NAME, "credits": 2}]


class _StubRetriever:
    """最小可用的检索桩：只按 ``course_code`` / ``source_types`` 过滤构造命中。"""

    connected = True

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []

    @property
    def count(self) -> int:
        return len(self.hits)

    def search(
        self,
        _query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        course_code = kwargs.get("course_code")
        source_types = kwargs.get("source_types")
        matches: list[dict[str, Any]] = []
        for hit in self.hits:
            metadata = hit.get("metadata") or {}
            if course_code and metadata.get("course_code") != course_code:
                continue
            if source_types and metadata.get("source_type") not in source_types:
                continue
            matches.append(hit)
        return matches[:top_k] if top_k else matches


def _router(hits: list[dict[str, Any]] | None = None) -> QueryRouter:
    """构造一个完全离线的 router，并注入构造的文档树。"""
    router = QueryRouter(vector_retriever=_StubRetriever(hits))
    router.courses = _courses()
    router.set_doc_tree(_doc_tree())
    return router


def _inline_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 ``asyncio.to_thread`` 改成就地执行。

    本沙箱里线程池第二次 ``asyncio.to_thread`` 的 await 永不返回（纯 stdlib
    即可复现，与业务代码无关），而 content 路径要连开两次线程；就地执行既绕开
    该环境问题，又让测试保持离线、确定。
    """

    async def _inline(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _inline)


# --------------------------------------------------------------------------- #
# 1) 路由优先级
# --------------------------------------------------------------------------- #

def test_chapter_question_routes_to_structure_not_content() -> None:
    query = "《信息素养与创新思维》第三章讲什么？"

    assert QueryRouter.classify(query) is QueryType.STRUCTURE
    prediction = QueryRouter.classify_intent(query)
    assert prediction.primary is QueryType.STRUCTURE
    # 「讲什么」同时命中 content 词面，但 structure 必须排在它前面。
    assert QueryType.CONTENT in prediction.labels


def test_fact_pattern_still_wins_over_structure() -> None:
    # 「学时」是 FACT 词面，「第三章」是 STRUCTURE 词面；FACT 优先级更高。
    query = "《信息素养与创新思维》第三章一共有多少学时？"

    assert QueryRouter.classify(query) is QueryType.FACT


# --------------------------------------------------------------------------- #
# 2) 文档树确定性回答 / 缺失章节拒答
# --------------------------------------------------------------------------- #

def test_existing_chapter_answered_from_doc_tree() -> None:
    result = asyncio.run(
        _router().route("《信息素养与创新思维》第二章讲什么？", "s-chapter-2")
    )

    assert result.query_type == "structure"
    assert result.metadata["structure_source"] == "doc_tree"
    assert result.metadata["chapter_no"] == 2
    assert result.metadata["llm_used"] is False
    assert "第二章 搜寻信息资源" in result.answer
    assert "第一章 课程导论" not in result.answer
    # 确定性路径不过 NLI。
    assert "nli_status" not in result.metadata


def test_missing_chapter_uses_existing_refusal_structure() -> None:
    result = asyncio.run(
        _router().route("《信息素养与创新思维》第99章讲什么？", "s-chapter-99")
    )

    assert result.query_type == "structure"
    assert result.answer == QueryRouter._STRUCTURE_REFUSAL
    assert result.citations == []
    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "STRUCTURE_CHAPTER_NOT_FOUND"
    assert result.metadata["chapter_no"] == 99
    assert result.metadata["llm_used"] is False


def test_outline_lists_chapters_in_document_order() -> None:
    result = asyncio.run(
        _router().route("《信息素养与创新思维》一共有几章？", "s-outline")
    )

    assert result.query_type == "structure"
    assert result.metadata["structure_source"] == "doc_tree"
    assert result.metadata["chapter_coverage"] == "3/3"
    answer = result.answer
    assert (
        answer.index("第一章 课程导论")
        < answer.index("第二章 搜寻信息资源")
        < answer.index("第三章 信息检索")
    )


def test_week_question_maps_to_the_chapter_holding_that_week() -> None:
    result = asyncio.run(
        _router().route("《信息素养与创新思维》第3周学什么？", "s-week-3")
    )

    assert result.query_type == "structure"
    assert result.metadata["week_no"] == 3
    assert result.metadata["structure_source"] == "doc_tree"
    assert "第一章 课程导论" in result.answer


# --------------------------------------------------------------------------- #
# 3) 完备性检查（M6.4）：证据没覆盖的章节要补齐
# --------------------------------------------------------------------------- #

def test_completeness_check_fills_missing_chapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 只召回第一章的进度表行；文档树有 3 章 → 应补齐第二、三章并回报 1/3。
    _inline_to_thread(monkeypatch)
    hit = {
        "chunk_id": "h2",
        "text": "章节次序及名称：第一章 课程导论\n第3周 讲授信息素养与信息意识\n学时：2",
        "source_file": SOURCE_FILE,
        "section": "第一章 课程导论",
        "score": 0.9,
        "metadata": {
            "source_type": "syllabus",
            "course_code": COURSE_CODE,
            "course_name": COURSE_NAME,
            "section_type": "teaching_schedule",
        },
    }

    result = asyncio.run(
        _router([hit]).route(
            "《信息素养与创新思维》这门课主要讲什么？", "s-coverage"
        )
    )

    assert result.query_type == "content"
    assert result.metadata["chapter_coverage"] == "1/3"
    assert "第一章 课程导论" in result.answer
    assert "第二章 搜寻信息资源" in result.answer
    assert "第三章 信息检索" in result.answer


def _many_chapter_tree(chapters: int = 12) -> DocTree:
    """构造一个有 ``chapters`` 章的文档树，用来检验 10 条证据预算下的截断。"""
    numbers = [
        "一", "二", "三", "四", "五", "六",
        "七", "八", "九", "十", "十一", "十二",
    ]
    schedule = ["四、教学进度"]
    chunks = [_chunk("s0", 0, schedule, "本表按周次安排教学进度。")]
    for index, name in enumerate(numbers[:chapters], 1):
        chunks.append(
            _chunk(
                f"c{index}", index, schedule + [f"第{name}章 章节{index}"],
                "章节次序及名称：第%s章 章节%s\n讲授信息素养与信息意识\n学时：2"
                % (name, index),
            )
        )
    return DocTree(build_doc_tree(chunks))


def test_completeness_fillers_must_not_evict_retrieved_evidence() -> None:
    """章节数超过证据预算时，真实命中的正文不能被补齐的章节标题挤掉。"""
    course = _courses()[0]
    doc_tree = _many_chapter_tree(12)
    hit = {
        "chunk_id": "c1",
        "text": "章节次序及名称：第一章 章节1\n讲授信息素养与信息意识\n学时：2",
        "source_file": SOURCE_FILE,
        "section": "第一章 章节1",
        "score": 0.9,
        "metadata": {
            "source_type": "syllabus",
            "course_code": COURSE_CODE,
            "course_name": COURSE_NAME,
            "section_type": "teaching_schedule",
        },
    }
    query = "《信息素养与创新思维》这门课主要讲什么？"

    items = QueryRouter._build_evidence_items(
        query, [hit], [], course=course, doc_tree=doc_tree
    )
    items, _coverage = QueryRouter._apply_chapter_coverage(
        query, items, course, doc_tree
    )
    selected = select_generation_evidence(items)

    retrieved = [
        item for item in selected
        if item.get("chapter_no") == 1 and not item.get("coverage_filler")
    ]
    assert retrieved, "命中的第一章正文被补齐项挤出了证据预算"
    assert "信息素养与信息意识" in str(retrieved[0].get("excerpt"))
