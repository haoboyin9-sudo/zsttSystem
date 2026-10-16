"""M1.6 旧→新 chunk_id 映射的单元测试。

新切分会改变 chunk 的 section 文本，使 chunk_id 全量变化；反馈样本里引用的
旧 id 必须能按 ``(source_file, 文本哈希)`` 找回新 id，找不回的样本要标成
``stale`` 而不是被静默丢弃。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maintenance.retraining_updater import (
    ChunkIdMapper,
    chunk_text_hash,
    normalize_chunk_text,
    remap_samples_file,
)

SOURCE_FILE = "2025-1教学大纲汇总/IM127信息素养与创新思维.docx"
TEXT_A = "四、教学进度\n\n第1章 课程导论（理论教学）所需学时：4"
TEXT_B = "五、考核方式\n\n平时成绩包括4次作业和考勤，占课程总评成绩的40%"


def _chunk(
    chunk_id: str,
    text: str,
    source_file: str = SOURCE_FILE,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source_file": source_file,
        "metadata": {"source_type": "syllabus"},
    }


def test_normalize_ignores_whitespace_and_case() -> None:
    assert normalize_chunk_text(" 第1章\n\t课程导论 ") == "第1章课程导论"
    assert chunk_text_hash("第1章 课程导论") == chunk_text_hash("  第1章\n课程导论  ")
    assert chunk_text_hash("a") != chunk_text_hash("b")


def test_maps_an_old_chunk_id_by_source_file_and_text() -> None:
    mapper = ChunkIdMapper([_chunk("new-a", TEXT_A), _chunk("new-b", TEXT_B)])

    record = {"chunk_id": "old-a", "text": TEXT_A, "source_file": SOURCE_FILE}

    assert mapper.map_record(record) is True
    assert record["chunk_id"] == "new-a"
    assert record["previous_chunk_id"] == "old-a"
    assert "stale" not in record
    assert "stale_reason" not in record


def test_keeps_a_still_valid_chunk_id_untouched() -> None:
    mapper = ChunkIdMapper([_chunk("new-a", TEXT_A)])

    record = {"chunk_id": "new-a", "text": TEXT_A}

    assert mapper.map_record(record) is True
    assert record["chunk_id"] == "new-a"
    assert "previous_chunk_id" not in record


def test_marks_an_unmappable_sample_as_stale() -> None:
    mapper = ChunkIdMapper([_chunk("new-a", TEXT_A)])

    record = {
        "chunk_id": "old-gone",
        "text": "这段正文在新语料里已经完全不存在了",
        "source_file": SOURCE_FILE,
    }

    assert mapper.map_record(record) is False
    assert record["stale"] is True
    assert record["stale_reason"]
    assert record["previous_chunk_id"] == "old-gone"
    # 标记过期而不是丢弃：原 id 仍然保留在记录里。
    assert record["chunk_id"] == "old-gone"


def test_falls_back_to_text_only_when_the_source_file_is_unknown() -> None:
    mapper = ChunkIdMapper([_chunk("new-b", TEXT_B)])

    record = {"chunk_id": "old-b", "text": TEXT_B}

    assert mapper.map_record(record) is True
    assert record["chunk_id"] == "new-b"


def test_does_not_guess_from_a_short_text() -> None:
    mapper = ChunkIdMapper([_chunk("new-c", "教材")])

    record = {"chunk_id": "old-c", "text": "教材"}

    assert mapper.map_record(record) is False
    assert record["stale"] is True


def test_does_not_guess_from_an_ambiguous_text() -> None:
    mapper = ChunkIdMapper(
        [
            _chunk("new-1", TEXT_A, "a.docx"),
            _chunk("new-2", TEXT_A, "b.docx"),
        ]
    )

    record = {"chunk_id": "old", "text": TEXT_A}

    assert mapper.map_record(record) is False
    assert record["stale"] is True


def test_maps_nested_citation_records() -> None:
    mapper = ChunkIdMapper([_chunk("new-a", TEXT_A)])
    records = [
        {
            "query_id": "q1",
            "citations": [
                {
                    "chunk_id": "old-a",
                    "text": TEXT_A,
                    "source_file": SOURCE_FILE,
                }
            ],
        }
    ]

    stats = mapper.map_records(records)

    assert stats == {"mapped": 1, "unchanged": 0, "stale": 0}
    assert records[0]["citations"][0]["chunk_id"] == "new-a"


def test_remap_samples_file_rewrites_json_and_marks_stale(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        json.dumps([_chunk("new-a", TEXT_A)], ensure_ascii=False),
        encoding="utf-8",
    )
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "citations": [
                        {
                            "chunk_id": "old-a",
                            "text": TEXT_A,
                            "source_file": SOURCE_FILE,
                        }
                    ],
                },
                {
                    "query_id": "q2",
                    "chunk_id": "old-gone",
                    "text": "完全对不上的旧正文内容",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = remap_samples_file(samples_path, chunks_path=chunks_path)

    assert summary["mapped"] == 1
    assert summary["stale"] == 1
    saved = json.loads(samples_path.read_text(encoding="utf-8"))
    assert saved[0]["citations"][0]["chunk_id"] == "new-a"
    assert saved[1]["stale"] is True
    assert saved[1]["stale_reason"]


def test_apply_updates_skips_stale_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """映射不上的修正样本既不落盘也不影响其余流程。"""
    from maintenance import retraining_updater as module

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(
        json.dumps([_chunk("new-a", TEXT_A)], ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CHUNKS_PATHS", (chunks_path,))

    updater = module.RetrainingUpdater()
    try:
        summary = updater.apply_updates(
            [
                {
                    "issue_type": "chunking",
                    "chunk_id": "old-gone",
                    "corrected_text": "正文对不上",
                },
                {"issue_type": "unknown-issue"},
            ]
        )
    finally:
        updater.close()

    assert summary["counts"]["stale"] == 1
    assert summary["counts"]["chunking"] == 0
    assert summary["counts"]["skipped"] == 1
    assert summary["rebuilt"] is None
