"""embedding_text 面包屑与 Chroma metadata 白名单的单元测试（P1-5）。"""

from __future__ import annotations

import json

import pytest

from src.data_processing.chroma_index import (
    _breadcrumb_enabled,
    _chroma_metadata,
    build_embedding_text,
)

BREADCRUMB = "【数据结构 | 四、教学进度 | 第3章 存储与索引】"
MINILM_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BGE_MODEL = "BAAI/bge-large-zh-v1.5"


def test_build_embedding_text_prepends_breadcrumb_when_enabled() -> None:
    chunk = {"text": "正文内容", "breadcrumb": BREADCRUMB}

    assert (
        build_embedding_text(chunk, breadcrumb_enabled=True)
        == f"{BREADCRUMB}\n正文内容"
    )


def test_build_embedding_text_returns_original_text_when_disabled() -> None:
    chunk = {"text": "正文内容", "breadcrumb": BREADCRUMB}

    assert build_embedding_text(chunk, breadcrumb_enabled=False) == "正文内容"


def test_build_embedding_text_ignores_missing_or_blank_breadcrumb() -> None:
    assert build_embedding_text({"text": "正文内容"}, breadcrumb_enabled=True) == "正文内容"
    assert (
        build_embedding_text({"text": "正文内容", "breadcrumb": ""}, breadcrumb_enabled=True)
        == "正文内容"
    )
    assert (
        build_embedding_text({"text": "正文内容", "breadcrumb": "   "}, breadcrumb_enabled=True)
        == "正文内容"
    )


def test_build_embedding_text_does_not_double_inject() -> None:
    chunk = {"text": f"{BREADCRUMB}\n正文内容", "breadcrumb": BREADCRUMB}

    result = build_embedding_text(chunk, breadcrumb_enabled=True)

    assert result == f"{BREADCRUMB}\n正文内容"
    assert result.count(BREADCRUMB) == 1


def test_breadcrumb_enabled_on_and_off_override_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_BREADCRUMB", "on")
    monkeypatch.setenv("EMBEDDING_MODEL", MINILM_MODEL)
    assert _breadcrumb_enabled() is True

    monkeypatch.setenv("EMBEDDING_BREADCRUMB", "off")
    monkeypatch.setenv("EMBEDDING_MODEL", BGE_MODEL)
    assert _breadcrumb_enabled() is False


def test_breadcrumb_enabled_auto_skips_minilm_and_enables_bge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_BREADCRUMB", "auto")
    monkeypatch.setenv("EMBEDDING_MODEL", MINILM_MODEL)
    assert _breadcrumb_enabled() is False

    monkeypatch.setenv("EMBEDDING_MODEL", BGE_MODEL)
    assert _breadcrumb_enabled() is True


def test_chroma_metadata_dispatches_scalars_and_json_payload() -> None:
    chunk = {
        "chunk_id": "c1",
        "text": "正文",
        "section": "第3章 存储与索引",
        "source_file": "a.docx",
        "chunk_index": 7,
        "chapter_no": 3,
        "is_header": False,
        "prev_chunk_id": "c0",
        "next_chunk_id": None,
        "parent_id": None,
        "heading_path": ["四、教学进度", "第3章 存储与索引"],
        "breadcrumb": BREADCRUMB,
        "metadata": {"course_code": "IM101", "source_type": "syllabus"},
    }

    metadata = _chroma_metadata(chunk)

    # 标量字段进 Chroma metadata 白名单
    assert metadata["chunk_id"] == "c1"
    assert metadata["chunk_index"] == 7
    assert metadata["chapter_no"] == 3
    assert metadata["is_header"] is False
    assert metadata["prev_chunk_id"] == "c0"
    # None 值不写入 Chroma（Chroma 不接受 None）
    assert "next_chunk_id" not in metadata
    assert "parent_id" not in metadata
    # 列表 / 长文本不进标量白名单
    assert "heading_path" not in metadata
    assert "breadcrumb" not in metadata

    # 列表 / 长文本并入 metadata_json
    payload = json.loads(metadata["metadata_json"])
    assert payload["heading_path"] == ["四、教学进度", "第3章 存储与索引"]
    assert payload["breadcrumb"] == BREADCRUMB
    assert payload["course_code"] == "IM101"


def test_chroma_metadata_omits_empty_breadcrumb_and_heading_path() -> None:
    chunk = {
        "chunk_id": "c2",
        "text": "正文",
        "heading_path": [],
        "breadcrumb": "",
        "metadata": {"source_type": "syllabus"},
    }

    metadata = _chroma_metadata(chunk)
    payload = json.loads(metadata["metadata_json"])

    assert "heading_path" not in payload
    assert "breadcrumb" not in payload
