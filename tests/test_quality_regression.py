from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from src.data_processing.doc_tree import build_doc_tree
from src.online_service.doc_tree import DocTree
from src.online_service.query_router import QueryRouter


class RegressionRetriever:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.connected = True

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
        min_score = kwargs.get("min_score")
        matches = []
        for hit in self.hits:
            metadata = hit.get("metadata") or {}
            if course_code and metadata.get("course_code") != course_code:
                continue
            if source_types and metadata.get("source_type") not in source_types:
                continue
            if min_score is not None and float(hit.get("score", 1.0)) < min_score:
                continue
            matches.append(hit)
        return matches[:top_k]


def _assert_clean_answer(answer: str) -> None:
    assert "\\n" not in answer
    assert "chunk_id" not in answer
    assert "metadata_json" not in answer
    assert not (answer.lstrip().startswith("{") or answer.lstrip().startswith("["))


def _im399_router() -> QueryRouter:
    retriever = RegressionRetriever(
        [
            {
                "chunk_id": "objective",
                "text": "课程目标\n管理运筹学学习线性规划、整数规划和网络分析。",
                "source_file": "2025/IM399管理运筹学.docx",
                "section": "课程目标",
                "score": 0.8,
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM399",
                    "course_name": "管理运筹学",
                    "section_type": "course_objectives",
                },
            },
            {
                "chunk_id": "schedule",
                "text": "教学进度 - 第一章 线性规划\n讲授数学模型与单纯形法。",
                "source_file": "2025/IM399管理运筹学.docx",
                "section": "教学进度 - 第一章 线性规划",
                "score": 0.75,
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM399",
                    "course_name": "管理运筹学",
                    "section_type": "teaching_schedule",
                },
            },
            {
                "chunk_id": "plan",
                "text": "培养方案课程信息\n课程：管理运筹学（IM399）\n学分：3\n总学时：54",
                "source_file": "training_plans/信管培养方案.xlsx",
                "section": "培养方案课程信息",
                "score": 0.9,
                "metadata": {
                    "source_type": "training_plan",
                    "course_code": "IM399",
                    "course_name": "管理运筹学",
                },
            },
            {
                "chunk_id": "unrelated",
                "text": "专业实习内容",
                "source_file": "IM441专业实习.docx",
                "score": 0.9,
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM441",
                    "course_name": "图书馆学专业实习",
                },
            },
        ]
    )
    router = QueryRouter(vector_retriever=retriever)
    router.courses = [
        {
            "course_code": "IM399",
            "course_name": "管理运筹学",
            "credits": 3,
            "hours": 54,
            "offerings": [
                {
                    "program_name": "信息管理与信息系统专业培养方案",
                    "program_type": "主修专业",
                    "course_category": "专业必修课",
                    "course_subcategory": "专业基础课",
                    "credits": 3,
                    "hours": 54,
                    "source_file": "信管培养方案.xlsx",
                }
            ],
        }
    ]
    return router


def test_quality_course_content() -> None:
    result = asyncio.run(
        _im399_router().route("管理运筹学主要学什么", "quality-content")
    )

    assert "管理运筹学" in result.answer
    assert "线性规划" in result.answer
    assert "专业实习" not in result.answer
    assert "核心内容：" in result.answer
    assert "资料来源：" in result.answer
    assert {citation["course_code"] for citation in result.citations} == {"IM399"}
    assert all(
        set(citation) <= {"source_file", "course_code", "course_name", "section"}
        for citation in result.citations
    )
    _assert_clean_answer(result.answer)


def test_quality_training_plan_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # Program names are learned from the plans themselves; "信管" is campus
    # slang that appears in no plan title, so it is deployment configuration.
    monkeypatch.setenv("PROGRAM_ALIASES", "信管=信息管理与信息系统")
    router = QueryRouter(vector_retriever=RegressionRetriever())
    router.courses = [
        {
            "course_code": "IM249",
            "course_name": "高级程序设计",
            "offerings": [
                {
                    "program_name": "信息管理与信息系统专业培养方案",
                    "program_type": "主修专业",
                    "course_category": "专业必修课",
                    "course_subcategory": "专业核心课",
                    "semester": 3,
                    "source_file": "信管培养方案.xlsx",
                }
            ],
        },
        {
            "course_code": "IM999",
            "course_name": "其他专业核心课",
            "offerings": [
                {
                    "program_name": "档案学专业培养方案",
                    "program_type": "主修专业",
                    "course_category": "专业必修课",
                    "course_subcategory": "专业核心课",
                    "semester": 3,
                    "source_file": "档案学培养方案.xlsx",
                }
            ],
        },
    ]

    result = asyncio.run(
        router.route("信管专业核心课程有哪些", "quality-catalog")
    )

    assert "高级程序设计（IM249）" in result.answer
    assert "其他专业核心课" not in result.answer
    _assert_clean_answer(result.answer)


def test_quality_exact_multi_fact() -> None:
    result = asyncio.run(
        _im399_router().route(
            "管理运筹学多少学分、多少学时",
            "quality-fact",
        )
    )

    assert "学分为 3" in result.answer
    assert "总学时为 54" in result.answer
    _assert_clean_answer(result.answer)


def test_quality_course_prerequisite() -> None:
    retriever = RegressionRetriever(
        [
            {
                "chunk_id": "prerequisite",
                "text": "课程基本信息\n先修课程：信息管理学基础",
                "source_file": "2025/IM2105信息组织基础.docx",
                "section": "课程基本信息",
                "score": 0.8,
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM2105",
                    "course_name": "信息组织基础",
                    "section_type": "basic_info",
                },
            }
        ]
    )
    router = QueryRouter(vector_retriever=retriever)
    router.courses = [
        {
            "course_code": "IM2105",
            "course_name": "信息组织基础",
            "prerequisites": [],
        }
    ]

    result = asyncio.run(
        router.route(
            "信息组织基础有哪些先修课程",
            "quality-prerequisite",
        )
    )

    assert result.query_type == "dependency"
    assert "信息管理学基础" in result.answer
    _assert_clean_answer(result.answer)


def test_quality_no_answer_and_weak_relevance() -> None:
    retriever = RegressionRetriever(
        [
            {
                "chunk_id": "weak",
                "text": "数据库课程的一般介绍",
                "source_file": "database.docx",
                "score": 0.2,
                "metadata": {
                    "source_type": "syllabus",
                    "course_code": "IM001",
                },
            }
        ]
    )
    router = QueryRouter(vector_retriever=retriever)
    router.courses = []

    for query in ("火星种植学主要学什么", "量子航天课程的培养目标是什么"):
        result = asyncio.run(router.route(query, "quality-no-answer"))
        assert "未找到足够相关" in result.answer or "无法获取足够" in result.answer
        assert result.citations == []
        _assert_clean_answer(result.answer)


def test_quality_page_never_renders_raw_payload() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "src" / "templates" / "demo.html"
    ).read_text(encoding="utf-8")

    assert 'result.textContent = data.answer' in template
    assert "JSON.stringify(data.citations" not in template
    assert 'lines.join("\\\\n")' not in template
    assert "item.chunk_id" not in template
    assert "item.score" not in template


def test_demo_page_exposes_the_feedback_loop() -> None:
    """The feedback pipeline is only demonstrable if the page can trigger it."""
    template = (
        Path(__file__).resolve().parents[1] / "src" / "templates" / "demo.html"
    ).read_text(encoding="utf-8")

    assert 'id="feedback-yes"' in template and 'id="feedback-no"' in template
    assert 'fetch("/feedback"' in template
    # The verdict must be tied to the answer it judges.
    assert "query_id: feedbackQueryId" in template
    assert "is_helpful: isHelpful" in template
    # Hidden until an answer exists, and reset before each new query.
    assert 'id="feedback" hidden' in template
    assert "showFeedback(data.query_id)" in template
    assert "hideFeedback()" in template


# --------------------------------------------------------------------------- #
# 大创改造 §7：两条固定回归（追加，不改动上面的既有用例）。
# --------------------------------------------------------------------------- #

def _syllabus_chunk(
    chunk_id: str,
    chunk_index: int,
    heading_path: list[str],
    text: str,
    *,
    section_type: str = "teaching_schedule",
    is_header: bool = False,
) -> dict[str, Any]:
    """构造一个 IM399 大纲 chunk（形态与 P1 parser 产物一致）。"""
    metadata: dict[str, Any] = {
        "source_type": "syllabus",
        "section_type": section_type,
        "course_code": "IM399",
        "course_name": "管理运筹学",
        "source_year": "2025",
        "syllabus_section": heading_path[-1] if heading_path else "",
    }
    if is_header:
        metadata["is_header"] = True
    return {
        "chunk_id": chunk_id,
        "document_hash": "hashIM399",
        "chunk_index": chunk_index,
        "heading_path": heading_path,
        "text": text,
        "source_file": "2025-1教学大纲汇总/IM399管理运筹学.docx",
        "metadata": metadata,
    }


def _im399_doc_tree() -> DocTree:
    schedule = ["四、教学进度"]
    chunks = [
        _syllabus_chunk(
            "b0", 0, ["一、课程基本信息"], "课程负责人：/",
            section_type="basic_info", is_header=True,
        ),
        _syllabus_chunk("b1", 1, schedule, "本表按周次安排教学进度。"),
        _syllabus_chunk(
            "b2", 2, schedule + ["第一章 线性规划"],
            "章节次序及名称：第一章 线性规划\n学时：8",
        ),
        _syllabus_chunk(
            "b3", 3, schedule + ["第二章 整数规划"],
            "章节次序及名称：第二章 整数规划\n学时：6",
        ),
        _syllabus_chunk(
            "b4", 4, schedule + ["第三章 网络分析"],
            "章节次序及名称：第三章 网络分析\n学时：4",
        ),
    ]
    return DocTree(build_doc_tree(chunks))


def test_quality_structure_chapter_is_deterministic() -> None:
    """「某课第三章讲什么」走 structure，答案含该章标题且结构顺序正确。"""
    router = QueryRouter(vector_retriever=RegressionRetriever())
    router.courses = [
        {"course_code": "IM399", "course_name": "管理运筹学", "credits": 3}
    ]
    router.set_doc_tree(_im399_doc_tree())

    result = asyncio.run(
        router.route("《管理运筹学》第三章讲什么？", "quality-structure")
    )

    assert result.query_type == "structure"
    assert result.metadata["structure_source"] == "doc_tree"
    assert result.metadata["chapter_no"] == 3
    # 只答本章，不串到别的章。
    assert "第三章 网络分析" in result.answer
    assert "第一章 线性规划" not in result.answer
    assert "第二章 整数规划" not in result.answer
    # 顺序：章节标题 → 正文 → 学时 → 资料来源。
    assert result.answer.index("第三章 网络分析") < result.answer.index("资料来源")
    _assert_clean_answer(result.answer)


def test_quality_instructor_placeholder_is_not_an_answer() -> None:
    """培养方案负责人为 ``/`` 且大纲无记录时，不得输出「为 /」，按「未记录」拒答。"""
    router = QueryRouter(vector_retriever=RegressionRetriever())
    router.courses = [
        {
            "course_code": "IM399",
            "course_name": "管理运筹学",
            "instructor": "/",
            "offerings": [
                {
                    "program_name": "信息管理与信息系统专业培养方案",
                    "program_type": "主修专业",
                    "instructor": "/",
                    "source_file": "信管培养方案.xlsx",
                }
            ],
        }
    ]

    result = asyncio.run(
        router.route("《管理运筹学》的任课教师是谁？", "quality-instructor")
    )

    assert result.query_type == "fact"
    assert "为 /" not in result.answer
    assert "未记录" in result.answer
    assert result.metadata["status"] == "refused"
    assert result.metadata["error_code"] == "FACT_INSTRUCTOR_NOT_RECORDED"
    _assert_clean_answer(result.answer)
