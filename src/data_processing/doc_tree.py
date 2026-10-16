"""文档结构树：从大纲 chunk 派生章节层级，供 small-to-big 与结构问答使用。

本模块只依赖 ``chunks.json`` 的产物形态，不导入 ``parser_chunker``，避免循环依赖。
``node_id`` 是跨模块冻结接口：``parser_chunker`` 用它给 chunk 计算 ``parent_id``，
``build_doc_tree`` 用它给节点命名，两处实现必须一致。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

__all__ = ["build_doc_tree", "node_id"]


def node_id(document_hash: str, heading_path: Sequence[str]) -> str:
    """Return the deterministic node id for one heading path inside a document."""
    joined = "|".join(str(part).strip() for part in heading_path)
    digest = hashlib.sha1(f"{document_hash}|{joined}".encode("utf-8")).hexdigest()
    return f"n{digest[:15]}"


# --------------------------------------------------------------------------- #
# M2.1 / M2.2 文档树构建
# --------------------------------------------------------------------------- #

#: 进度表类标题的关键词，用于定位「整张进度表」节点。
_SCHEDULE_KEYWORDS = re.compile(
    r"教学进度|进度安排|教学安排|授课计划|章节安排|课程进度|教学计划|教学日历"
)

#: 章号解析：支持中文数字与阿拉伯数字。
_CHAPTER_RE = re.compile(r"第\s*([0-9一二三四五六七八九十百零〇两]+)\s*(章|单元|模块|部分|讲|篇)")

#: 学时解析：兼容「学时：2」「所需学时(必填)：2」等写法。
_HOURS_RE = re.compile(r"学\s*时\s*(?:[（(][^）)]*[）)]\s*)?[:：]\s*(\d+)")

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}

#: 头部块（课程基本信息 / 文档概览）的 section_type。
_HEADER_SECTION_TYPES = {"basic_info", "overview"}
_HEADER_TITLES = {"文档概览", "课程基本信息", "基本信息"}


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """安全地取出 chunk 的 metadata，非字典时返回空字典。"""
    meta = chunk.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _chunk_field(chunk: dict[str, Any], name: str) -> Any:
    """先读 chunk 顶层字段，缺失时回退到 metadata 同名键。"""
    if name in chunk and chunk[name] not in (None, "", []):
        return chunk[name]
    return _metadata(chunk).get(name)


def _cn_to_int(text: str) -> int | None:
    """把中文数字（含「十二」「二十三」）转为整数，失败返回 None。"""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    section = 0
    number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            section += (number or 1) * _CN_UNITS[char]
            number = 0
        else:
            return None
    return section + number


def _parse_chapter_no(title: str) -> int | None:
    """从章节标题（如「第3章 存储与索引」）解析章号，无法解析返回 None。"""
    match = _CHAPTER_RE.search(str(title))
    if match is None:
        return None
    return _cn_to_int(match.group(1))


def _normalize_heading_path(value: Any) -> list[str]:
    """把 heading_path 归一成非空字符串列表。"""
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in re.split(r"[|/]", value) if part.strip()]
    return []


def _heading_path_of(chunk: dict[str, Any]) -> list[str]:
    """取 chunk 的标题路径；旧产物缺字段时退化到单节点路径。"""
    path = _normalize_heading_path(_chunk_field(chunk, "heading_path"))
    if path:
        return path

    meta = _metadata(chunk)
    section_type = str(meta.get("section_type") or "")
    section = (
        str(chunk.get("section") or "")
        or str(meta.get("syllabus_section") or "")
        or str(meta.get("parent_section") or "")
        or str(meta.get("section_title") or "")
    ).strip()
    if not section:
        return ["未分类"]
    # 旧进度表块形如「教学进度 - 第一章 …」，尽量拆出「进度表 + 行」两级。
    if section_type == "teaching_schedule" and " - " in section:
        head, row = section.split(" - ", 1)
        parts = [head.strip(), row.strip()]
        return [part for part in parts if part] or ["未分类"]
    return [section]


def _schedule_prefix(path: list[str]) -> list[str]:
    """定位一张进度表所在章节的路径前缀（即「整张进度表」节点）。"""
    for index in range(len(path) - 1, -1, -1):
        if _SCHEDULE_KEYWORDS.search(path[index].replace(" ", "")):
            return path[: index + 1]
    if len(path) >= 2:
        return path[:-1]
    return path[:]


def _is_header(meta: dict[str, Any], section_type: str) -> bool:
    """判断该 chunk 是否为头部块（课程基本信息 / 文档概览）。"""
    if meta.get("is_header") is True:
        return True
    if section_type in _HEADER_SECTION_TYPES:
        return True
    title = re.sub(
        r"\s+",
        "",
        str(meta.get("syllabus_section") or meta.get("parent_section") or ""),
    )
    return title in _HEADER_TITLES


def _group_syllabus_chunks(
    chunks: list[dict[str, Any]],
) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """只保留大纲 chunk，并按 document_hash 分组（保留原始顺序）。"""
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for order, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        if _metadata(chunk).get("source_type") != "syllabus":
            continue
        document_hash = _chunk_field(chunk, "document_hash")
        if not document_hash:
            continue
        groups.setdefault(str(document_hash), []).append((order, chunk))
    return groups


def _build_document(
    document_hash: str, items: list[tuple[int, dict[str, Any]]]
) -> dict[str, Any]:
    """把同一份大纲的全部 chunk 组装成文档树节点集合。"""
    indexed: list[tuple[int, int, dict[str, Any]]] = []
    for order, chunk in items:
        try:
            index = int(_chunk_field(chunk, "chunk_index"))
        except (TypeError, ValueError):
            index = order
        indexed.append((index, order, chunk))
    indexed.sort(key=lambda item: (item[0], item[1]))

    nodes: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    chunk_meta: dict[str, tuple[int, str]] = {}
    non_chapter_rows: list[str] = []
    header_chunk_ids: list[str] = []
    all_chunk_ids: list[str] = []

    for index, _order, chunk in indexed:
        chunk_id = str(_chunk_field(chunk, "chunk_id") or "")
        if not chunk_id:
            continue
        path = _heading_path_of(chunk)
        text = str(chunk.get("text") or "")
        meta = _metadata(chunk)
        section_type = str(meta.get("section_type") or _chunk_field(chunk, "section_type") or "")

        previous: str | None = None
        for depth, title in enumerate(path):
            current = node_id(document_hash, path[: depth + 1])
            node = nodes.get(current)
            if node is None:
                node = {
                    "level": depth + 1,
                    "title": title,
                    "kind": "section",
                    "chapter_no": None,
                    "children": [],
                    "chunk_ids": [],
                    "hours": None,
                    "text": "",
                    "section_type": None,
                }
                nodes[current] = node
                node_order.append(current)
            if previous is not None and current not in nodes[previous]["children"]:
                nodes[previous]["children"].append(current)
            if section_type and node["section_type"] is None:
                node["section_type"] = section_type
            node["chunk_ids"].append(chunk_id)
            previous = current

        if section_type == "teaching_schedule":
            schedule_prefix = _schedule_prefix(path)
            schedule_node = nodes.get(node_id(document_hash, schedule_prefix))
            if schedule_node is not None:
                schedule_node["kind"] = "schedule"
            if len(path) > len(schedule_prefix):
                row_title = path[-1]
                chapter_no = _parse_chapter_no(row_title)
                row_node = nodes.get(node_id(document_hash, path))
                if chapter_no is not None and row_node is not None:
                    row_node["kind"] = "chapter"
                    row_node["chapter_no"] = chapter_no
                elif row_title and row_title not in non_chapter_rows:
                    non_chapter_rows.append(row_title)

        chunk_meta[chunk_id] = (index, text)
        all_chunk_ids.append(chunk_id)
        if _is_header(meta, section_type) and chunk_id not in header_chunk_ids:
            header_chunk_ids.append(chunk_id)

    for node in nodes.values():
        ordered_ids = sorted(
            dict.fromkeys(node["chunk_ids"]),
            key=lambda cid: chunk_meta.get(cid, (1 << 30, ""))[0],
        )
        node["chunk_ids"] = ordered_ids
        node["text"] = "\n\n".join(
            chunk_meta[cid][1] for cid in ordered_ids if chunk_meta[cid][1]
        )
        if node["kind"] == "chapter":
            total_hours = 0
            found = False
            for cid in ordered_ids:
                for match in _HOURS_RE.finditer(chunk_meta[cid][1]):
                    total_hours += int(match.group(1))
                    found = True
            node["hours"] = total_hours if found else None

    first_meta = _metadata(indexed[0][2]) if indexed else {}
    first_chunk = indexed[0][2] if indexed else {}
    course_code = str(first_meta.get("course_code") or "")
    course_name = str(first_meta.get("course_name") or "")
    source_file = str(
        first_chunk.get("source_file")
        or first_meta.get("parent_document")
        or first_meta.get("source_file")
        or ""
    )
    source_year = str(first_meta.get("source_year") or "")

    def position(chunk_id: str) -> int:
        return chunk_meta.get(chunk_id, (1 << 30, ""))[0]

    ordered_all = sorted(dict.fromkeys(all_chunk_ids), key=position)
    roots = [nid for nid in node_order if nodes[nid]["level"] == 1]
    chapter_count = sum(1 for node in nodes.values() if node["kind"] == "chapter")

    return {
        "course_code": course_code,
        "course_name": course_name,
        "source_file": source_file,
        "source_year": source_year,
        "chunk_ids": ordered_all,
        "nodes": nodes,
        "roots": roots,
        "chapter_count": chapter_count,
        "non_chapter_rows": non_chapter_rows,
        "header_chunk_ids": sorted(dict.fromkeys(header_chunk_ids), key=position),
    }


def build_doc_tree(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """从全量 chunk 列表构建文档树 JSON（``version: 1``）。

    只消费 ``metadata.source_type == "syllabus"`` 的 chunk，按 ``document_hash``
    分组；缺字段的旧 chunk 不会抛异常，会退化到单节点或直接跳过。
    """
    if not isinstance(chunks, list):
        return {"version": 1, "documents": {}}
    documents: dict[str, Any] = {}
    for document_hash, items in _group_syllabus_chunks(chunks).items():
        documents[document_hash] = _build_document(document_hash, items)
    return {"version": 1, "documents": documents}
