"""在线文档树查询：读取 ``outputs/doc_tree.json``，提供章节与父块查询。

本模块是 M2.3 的在线接口，只消费 ``build_doc_tree`` 的产物形态，
不导入 ``parser_chunker``，也不依赖 chromadb；缺失文件或结构不符时
构造过程不会抛异常，所有查询退化为空结果，便于上层回退到 content 路径。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Block", "ChapterNode", "DocTree", "Document"]

_INFINITE_POSITION = 1 << 30


def _as_int(value: Any) -> int | None:
    """把 JSON 里的数字（可能是字符串）安全地转成 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _year_key(text: str) -> tuple[int, ...]:
    """把 ``source_year``（如 ``2025-1``）转成可比较的整数元组。"""
    numbers = re.findall(r"\d+", str(text or ""))
    return tuple(int(item) for item in numbers) or (0,)


@dataclass
class Block:
    """文档树里的一个节点（section / chapter / schedule）。"""

    node_id: str
    title: str
    kind: str = "section"
    level: int = 0
    chapter_no: int | None = None
    hours: int | None = None
    chunk_ids: list[str] = field(default_factory=list)
    text: str = ""

    @property
    def char_count(self) -> int:
        """父块全文长度，供 small-to-big 判定是否需要截断。"""
        return len(self.text)


@dataclass
class ChapterNode:
    """章节目录项，``order_index`` 为文档内最小 chunk 序号。"""

    node_id: str
    title: str
    chapter_no: int | None = None
    hours: int | None = None
    chunk_ids: list[str] = field(default_factory=list)
    order_index: int = _INFINITE_POSITION


@dataclass
class Document:
    """一份大纲（一个 ``document_hash``）的元数据。"""

    document_hash: str
    course_code: str = ""
    course_name: str = ""
    source_file: str = ""
    source_year: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    chapter_count: int = 0
    header_chunk_ids: list[str] = field(default_factory=list)
    non_chapter_rows: list[str] = field(default_factory=list)


class DocTree:
    """文档树查询入口，构造时建立 chunk_id → 文档/章节 索引。"""

    def __init__(self, tree: dict[str, Any] | str | Path | None = None) -> None:
        self._reset()
        self.reload(tree)

    # -- 加载 -------------------------------------------------------------- #

    def load(self, path: dict[str, Any] | str | Path | None) -> DocTree:
        """加载（或替换）文档树内容，失败时保持原状态并返回自身。"""
        return self.reload(path)

    def reload(self, path: dict[str, Any] | str | Path | None) -> DocTree:
        """重新加载文档树，供 ``/admin/data/reindex`` 热替换使用。"""
        data = self._read_source(path)
        if data is None:
            return self
        state = self._build_state(data)
        if state is None:
            return self
        self._install(state, data)
        return self

    def _reset(self) -> None:
        self._documents: dict[str, Document] = {}
        self._blocks: dict[str, Block] = {}
        self._chapters: dict[str, list[ChapterNode]] = {}
        self._chunk_doc: dict[str, str] = {}
        self._chunk_pos: dict[str, int] = {}
        self._chunk_nodes: dict[str, list[str]] = {}
        self._course_docs: dict[str, list[str]] = {}
        self._version: int | str | None = None

    def _install(self, state: dict[str, Any], data: dict[str, Any]) -> None:
        self._documents = state["documents"]
        self._blocks = state["blocks"]
        self._chapters = state["chapters"]
        self._chunk_doc = state["chunk_doc"]
        self._chunk_pos = state["chunk_pos"]
        self._chunk_nodes = state["chunk_nodes"]
        self._course_docs = state["course_docs"]
        self._version = self._compute_version(data)

    @staticmethod
    def _read_source(source: dict[str, Any] | str | Path | None) -> dict[str, Any] | None:
        """把 dict / 路径 / None 统一读成字典；任何异常都返回 None。"""
        if source is None:
            return None
        if isinstance(source, dict):
            return source
        if isinstance(source, (str, Path)):
            try:
                path = Path(source)
                if not path.is_file():
                    return None
                with path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except (OSError, ValueError):
                return None
            return loaded if isinstance(loaded, dict) else None
        return None

    @staticmethod
    def _compute_version(data: dict[str, Any]) -> int | str | None:
        """优先返回文件自带的 version，否则退化为内容哈希。"""
        if isinstance(data, dict) and "version" in data:
            return data.get("version")
        try:
            payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            payload = str(data)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def version(self) -> int | str | None:
        """文件内容的 version 或哈希，供 ``/health`` 使用。"""
        return self._version

    # -- 索引构建 ---------------------------------------------------------- #

    def _build_state(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """从树 JSON 构建全部查询索引；结构不符时返回 None。"""
        if not isinstance(data, dict):
            return None
        documents_raw = data.get("documents")
        if not isinstance(documents_raw, dict):
            return None

        documents: dict[str, Document] = {}
        blocks: dict[str, Block] = {}
        chapters: dict[str, list[ChapterNode]] = {}
        chunk_doc: dict[str, str] = {}
        chunk_pos: dict[str, int] = {}
        chunk_nodes: dict[str, list[str]] = {}
        course_docs: dict[str, list[str]] = {}

        for raw_hash, raw_doc in documents_raw.items():
            if not isinstance(raw_doc, dict):
                continue
            doc_hash = str(raw_hash)
            chunk_ids = list(
                dict.fromkeys(str(c) for c in raw_doc.get("chunk_ids") or [] if c)
            )
            for position, chunk_id in enumerate(chunk_ids):
                chunk_doc[chunk_id] = doc_hash
                chunk_pos[chunk_id] = position

            doc_nodes = raw_doc.get("nodes")
            doc_chapters: list[ChapterNode] = []
            if isinstance(doc_nodes, dict):
                for raw_node_id, raw_node in doc_nodes.items():
                    if not isinstance(raw_node, dict):
                        continue
                    node_id = str(raw_node_id)
                    block = Block(
                        node_id=node_id,
                        title=str(raw_node.get("title") or ""),
                        kind=str(raw_node.get("kind") or "section"),
                        level=_as_int(raw_node.get("level")) or 0,
                        chapter_no=_as_int(raw_node.get("chapter_no")),
                        hours=_as_int(raw_node.get("hours")),
                        chunk_ids=[
                            str(c) for c in raw_node.get("chunk_ids") or [] if c
                        ],
                        text=str(raw_node.get("text") or ""),
                    )
                    blocks[node_id] = block
                    for chunk_id in block.chunk_ids:
                        chunk_nodes.setdefault(chunk_id, []).append(node_id)
                    if block.kind == "chapter":
                        order_index = min(
                            (chunk_pos.get(c, _INFINITE_POSITION) for c in block.chunk_ids),
                            default=_INFINITE_POSITION,
                        )
                        doc_chapters.append(
                            ChapterNode(
                                node_id=block.node_id,
                                title=block.title,
                                chapter_no=block.chapter_no,
                                hours=block.hours,
                                chunk_ids=list(block.chunk_ids),
                                order_index=order_index,
                            )
                        )

            doc_chapters.sort(
                key=lambda node: (node.order_index, node.chapter_no or 0, node.title)
            )
            chapters[doc_hash] = doc_chapters

            header_ids = list(
                dict.fromkeys(
                    str(c) for c in raw_doc.get("header_chunk_ids") or [] if c
                )
            )
            document = Document(
                document_hash=doc_hash,
                course_code=str(raw_doc.get("course_code") or ""),
                course_name=str(raw_doc.get("course_name") or ""),
                source_file=str(raw_doc.get("source_file") or ""),
                source_year=str(raw_doc.get("source_year") or ""),
                chunk_ids=chunk_ids,
                chapter_count=_as_int(raw_doc.get("chapter_count")) or len(doc_chapters),
                header_chunk_ids=header_ids,
                non_chapter_rows=[
                    str(row) for row in raw_doc.get("non_chapter_rows") or [] if row
                ],
            )
            documents[doc_hash] = document
            course_docs.setdefault(document.course_code, []).append(doc_hash)

        for hashes in course_docs.values():
            hashes.sort(
                key=lambda item: _year_key(documents[item].source_year), reverse=True
            )

        return {
            "documents": documents,
            "blocks": blocks,
            "chapters": chapters,
            "chunk_doc": chunk_doc,
            "chunk_pos": chunk_pos,
            "chunk_nodes": chunk_nodes,
            "course_docs": course_docs,
        }

    # -- 查询接口 ---------------------------------------------------------- #

    def latest_document(self, course_code: str) -> Document | None:
        """同一课程可能有多份大纲，按 ``source_year`` 取最新。"""
        hashes = self._course_docs.get(str(course_code))
        if not hashes:
            return None
        return self._documents.get(hashes[0])

    def outline(self, course_code: str) -> list[ChapterNode]:
        """返回有序章节目录（按文档内 chunk 序号）。"""
        document = self.latest_document(course_code)
        if document is None:
            return []
        return list(self._chapters.get(document.document_hash, []))

    def chapter(self, course_code: str, chapter_no: int) -> ChapterNode | None:
        """按章号取章节；不存在返回 None。"""
        for node in self.outline(course_code):
            if node.chapter_no == chapter_no:
                return node
        return None

    def chapter_numbers(self, course_code: str) -> list[int]:
        """便利方法：返回该课程全部可解析章号。"""
        return [
            node.chapter_no
            for node in self.outline(course_code)
            if node.chapter_no is not None
        ]

    def parent_block(self, chunk_id: str, kind: str = "chapter") -> Block | None:
        """small-to-big：返回包含该 chunk 的最近指定类型节点（含全文）。"""
        candidates = self._chunk_nodes.get(str(chunk_id))
        if not candidates:
            return None
        best: Block | None = None
        for node_id in candidates:
            block = self._blocks.get(node_id)
            if block is None or block.kind != kind:
                continue
            if best is None or block.level > best.level:
                best = block
        return best

    def neighbors(self, chunk_id: str, window: int = 1) -> list[str]:
        """返回同文档内前后各 ``window`` 个 chunk_id（不含自身）。"""
        chunk_id = str(chunk_id)
        if window < 0:
            window = 0
        doc_hash = self._chunk_doc.get(chunk_id)
        if doc_hash is None:
            return []
        document = self._documents.get(doc_hash)
        position = self._chunk_pos.get(chunk_id)
        if document is None or position is None:
            return []
        ids = document.chunk_ids
        low = max(0, position - window)
        high = min(len(ids), position + window + 1)
        return ids[low:position] + ids[position + 1 : high]

    def order_key(self, chunk_id: str) -> tuple[str, int]:
        """证据排序键：``(course_code, chunk_index)``。"""
        chunk_id = str(chunk_id)
        doc_hash = self._chunk_doc.get(chunk_id)
        if doc_hash is None:
            return ("", -1)
        document = self._documents.get(doc_hash)
        course_code = document.course_code if document is not None else ""
        return (course_code, self._chunk_pos.get(chunk_id, -1))

    def header_chunks(self, course_code: str) -> list[str]:
        """返回该课程最新大纲的头部块 chunk_id（基本信息 / 文档概览）。"""
        document = self.latest_document(course_code)
        if document is None:
            return []
        return list(document.header_chunk_ids)
