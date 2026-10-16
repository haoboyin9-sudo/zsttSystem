"""M2 文档树：构建（build_doc_tree）与在线查询（DocTree）的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_processing.doc_tree import build_doc_tree, node_id
from src.online_service.doc_tree import DocTree


def _chunk(
    document_hash: str,
    chunk_id: str,
    chunk_index: int,
    heading_path: list[str],
    text: str,
    *,
    section_type: str = "overview",
    course_code: str = "IM900",
    course_name: str = "测试课程",
    source_year: str = "2025",
    is_header: bool = False,
    source_type: str = "syllabus",
    source_file: str = "2025-1教学大纲汇总/IM900测试课程.docx",
) -> dict[str, Any]:
    """构造一个符合新 parser 产物形态的 chunk。"""
    metadata: dict[str, Any] = {
        "source_type": source_type,
        "section_type": section_type,
        "course_code": course_code,
        "course_name": course_name,
        "source_year": source_year,
        "syllabus_section": heading_path[-1] if heading_path else "",
    }
    if is_header:
        metadata["is_header"] = True
    return {
        "chunk_id": chunk_id,
        "document_hash": document_hash,
        "chunk_index": chunk_index,
        "heading_path": heading_path,
        "text": text,
        "source_file": source_file,
        "metadata": metadata,
    }


def _sample_chunks(
    document_hash: str = "hashA",
    course_code: str = "IM900",
    source_year: str = "2025",
) -> list[dict[str, Any]]:
    """一门课：基本信息 + 两个章（第一章两行）+ 复习 / 期末考试。"""
    schedule = ["四、教学进度"]
    return [
        _chunk(
            document_hash, "h0", 0, ["一、课程基本信息"],
            "课程负责人：张三\n主讲教师：李四",
            section_type="basic_info", is_header=True,
            course_code=course_code, source_year=source_year,
        ),
        _chunk(
            document_hash, "h1", 1, schedule,
            "本表按周次安排教学进度。",
            section_type="teaching_schedule",
            course_code=course_code, source_year=source_year,
        ),
        _chunk(
            document_hash, "h2", 2, schedule + ["第一章 绪论"],
            "章节次序及名称：第一章 绪论\n学时：2",
            section_type="teaching_schedule",
            course_code=course_code, source_year=source_year,
        ),
        _chunk(
            document_hash, "h3", 3, schedule + ["第一章 绪论"],
            "章节次序及名称：第一章 绪论（续）\n所需学时(必填)：4",
            section_type="teaching_schedule",
            course_code=course_code, source_year=source_year,
        ),
        _chunk(
            document_hash, "h4", 4, schedule + ["第二章 数据模型"],
            "章节次序及名称：第二章 数据模型\n学时：6",
            section_type="teaching_schedule",
            course_code=course_code, source_year=source_year,
        ),
        _chunk(
            document_hash, "h5", 5, schedule + ["复习"],
            "复习答疑",
            section_type="teaching_schedule",
            course_code=course_code, source_year=source_year,
        ),
        _chunk(
            document_hash, "h6", 6, schedule + ["期末考试"],
            "闭卷考试",
            section_type="teaching_schedule",
            course_code=course_code, source_year=source_year,
        ),
    ]


def _doc(tree: dict[str, Any], key: str = "hashA") -> dict[str, Any]:
    return tree["documents"][key]


def test_same_chapter_rows_merge_into_one_node() -> None:
    tree = build_doc_tree(_sample_chunks())
    nodes = _doc(tree)["nodes"]
    chapter_ids = [
        nid
        for nid, node in nodes.items()
        if node["kind"] == "chapter" and node["chapter_no"] == 1
    ]
    assert len(chapter_ids) == 1
    chapter = nodes[chapter_ids[0]]
    assert chapter["chunk_ids"] == ["h2", "h3"]
    assert chapter["hours"] == 6
    assert "第一章" in chapter["text"] and "续" in chapter["text"]


def test_outline_ordered_by_chunk_index() -> None:
    # 打乱输入顺序，输出仍应按 chunk_index 有序（不依赖 dict 迭代顺序）。
    tree = build_doc_tree(list(reversed(_sample_chunks())))
    document = DocTree(tree)
    outline = document.outline("IM900")
    assert [node.chapter_no for node in outline] == [1, 2]
    assert [node.order_index for node in outline] == [2, 4]
    assert outline[0].chunk_ids == ["h2", "h3"]


def test_chapter_count_excludes_non_chapter_rows() -> None:
    document = _doc(build_doc_tree(_sample_chunks()))
    assert document["chapter_count"] == 2
    assert document["non_chapter_rows"] == ["复习", "期末考试"]


def test_node_id_stable_within_document_and_scoped_by_hash() -> None:
    path = ["四、教学进度", "第一章 绪论"]
    assert node_id("hashA", path) == node_id("hashA", list(path))
    assert node_id("hashA", path) != node_id("hashB", path)
    assert node_id("hashA", path) != node_id("hashA", path[:1])


def test_rebuild_syncs_after_document_removed_or_replaced() -> None:
    assert "hashA" in build_doc_tree(_sample_chunks())["documents"]

    removed = build_doc_tree(
        [chunk for chunk in _sample_chunks() if chunk["document_hash"] != "hashA"]
    )
    assert "hashA" not in removed["documents"]

    replaced_chunks = _sample_chunks()
    replaced_chunks[4]["text"] = "章节次序及名称：第二章 数据模型\n学时：9"
    replaced = build_doc_tree(replaced_chunks)
    chapter_two = [
        node
        for node in _doc(replaced)["nodes"].values()
        if node["kind"] == "chapter" and node["chapter_no"] == 2
    ][0]
    assert chapter_two["hours"] == 9


def test_missing_fields_do_not_raise() -> None:
    legacy: list[dict[str, Any]] = [
        {
            "chunk_id": "o1",
            "document_hash": "hashOld",
            "text": "文档概览",
            "section": "文档概览",
            "metadata": {
                "source_type": "syllabus",
                "section_type": "overview",
                "course_code": "IM901",
            },
        },
        {"chunk_id": "o2", "text": "没有 document_hash"},
        {"chunk_id": "o3", "metadata": {"source_type": "training_plan"}},
        {
            "chunk_id": "o4",
            "document_hash": "hashOld",
            "text": "教学进度 - 第一章 绪论",
            "section": "教学进度 - 第一章 绪论",
            "metadata": {
                "source_type": "syllabus",
                "section_type": "teaching_schedule",
                "course_code": "IM901",
            },
        },
    ]
    tree = build_doc_tree(legacy)
    assert set(tree["documents"]) == {"hashOld"}
    document = _doc(tree, "hashOld")
    assert document["chunk_ids"] == ["o1", "o4"]
    assert any(node["kind"] == "chapter" for node in document["nodes"].values())


def test_online_queries_against_built_tree() -> None:
    document = DocTree(build_doc_tree(_sample_chunks()))
    latest = document.latest_document("IM900")
    assert latest is not None
    assert latest.source_file.endswith("IM900测试课程.docx")
    assert document.chapter_numbers("IM900") == [1, 2]
    assert document.version == 1

    chapter = document.chapter("IM900", 1)
    assert chapter is not None and chapter.hours == 6

    block = document.parent_block("h3")
    assert block is not None and block.kind == "chapter" and block.chapter_no == 1
    assert block.chunk_ids == ["h2", "h3"] and "学时：2" in block.text

    schedule = document.parent_block("h5", kind="schedule")
    assert schedule is not None and schedule.kind == "schedule"
    assert "h2" in schedule.chunk_ids and "h6" in schedule.chunk_ids

    assert document.order_key("h4") == ("IM900", 4)
    assert document.neighbors("h3", window=1) == ["h2", "h4"]
    assert document.header_chunks("IM900") == ["h0"]


def test_latest_document_prefers_newer_year() -> None:
    chunks = _sample_chunks("hash2024", source_year="2024") + _sample_chunks(
        "hash2025", source_year="2025"
    )
    document = DocTree(build_doc_tree(chunks))
    latest = document.latest_document("IM900")
    assert latest is not None
    assert latest.document_hash == "hash2025"
    assert latest.source_year == "2025"


def test_doc_tree_missing_file_is_graceful() -> None:
    document = DocTree("/tmp/definitely-missing-doc-tree.json")
    assert document.latest_document("IM900") is None
    assert document.outline("IM900") == []
    assert document.chapter("IM900", 1) is None
    assert document.chapter_numbers("IM900") == []
    assert document.parent_block("h2") is None
    assert document.neighbors("h2") == []
    assert document.order_key("h2") == ("", -1)
    assert document.header_chunks("IM900") == []
    assert document.version is None
    # None / 结构不符同样不能抛异常
    assert DocTree(None).outline("IM900") == []
    assert DocTree({"unexpected": 1}).outline("IM900") == []


def test_load_from_file_and_reload(tmp_path: Path) -> None:
    tree_path = tmp_path / "doc_tree.json"
    tree_path.write_text(
        json.dumps(build_doc_tree(_sample_chunks()), ensure_ascii=False),
        encoding="utf-8",
    )
    document = DocTree(tree_path)
    assert document.latest_document("IM900") is not None
    assert document.version == 1

    # 加载失败时保留旧状态（原子热替换）
    document.reload(tmp_path / "missing.json")
    assert document.latest_document("IM900") is not None

    document.reload({"version": 1, "documents": {}})
    assert document.latest_document("IM900") is None
