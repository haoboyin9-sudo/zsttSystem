"""Apply reviewed corrections to the local vector index and knowledge graph.

Corrections are never applied to the collection the service is reading.  A
round builds ``<alias>_v<n+1>``, optionally scores it against the gold set,
and only then repoints the alias -- so a bad correction is undone by
``--rollback`` (one pointer write) instead of by reparsing the corpus, and the
before/after numbers needed to claim the human-in-the-loop cycle *works* fall
out of the same run.

    python maintenance/retraining_updater.py --evaluate
    python maintenance/retraining_updater.py --rollback
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from chromadb.errors import ChromaError
from neo4j import GraphDatabase

from src.config import config
from src.data_processing.chroma_index import build_index, create_chroma_client
from src.data_processing.collection_registry import (
    next_version_name,
    read_alias_record,
    resolve_active_collection,
    write_alias_record,
)
from src.data_processing.lexical_stats import write_lexical_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTED_SAMPLES_PATH = PROJECT_ROOT / "outputs" / "corrected_samples.json"
CHUNKS_PATHS = (
    PROJECT_ROOT / "outputs" / "chunks.json",
)
VECTOR_OUTPUT_DIR = PROJECT_ROOT / "outputs"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
EMBEDDING_FINETUNE_DATASET_PATH = PROJECT_ROOT / "outputs" / "embedding_finetune_samples.jsonl"
FEEDBACK_LOG_PATH = PROJECT_ROOT / "outputs" / "feedback_log.jsonl"
REVIEW_SAMPLES_PATH = PROJECT_ROOT / "outputs" / "samples_for_review.json"

# 纯文本兜底匹配所需的最短归一化长度：过短的文本（“考核方式”之类）在语料里
# 并不唯一，只按文本匹配会误伤。
MIN_TEXT_FALLBACK_LENGTH = 8


def normalize_chunk_text(text: str) -> str:
    """归一化 chunk 文本：折叠所有空白并转小写，作为跨版本的文本指纹输入。"""
    return re.sub(r"\s+", "", str(text or "")).lower()


def chunk_text_hash(text: str) -> str:
    """归一化文本的 SHA256，用于跨版本识别「同一个物理块」（M1.6）。"""
    return hashlib.sha256(normalize_chunk_text(text).encode("utf-8")).hexdigest()


def _record_source_file(record: dict[str, Any], source_key: str = "source_file") -> str:
    """从反馈 / 样本记录里取来源文件，兼容顶层、``metadata`` 与 ``citations``。"""
    for container in (record, record.get("metadata")):
        if not isinstance(container, dict):
            continue
        value = str(container.get(source_key) or "").strip()
        if value:
            return value
    citations = record.get("citations")
    if isinstance(citations, list):
        chunk_id = str(record.get("chunk_id") or "").strip()
        matched = {
            str(item.get(source_key) or "").strip()
            for item in citations
            if isinstance(item, dict)
            and str(item.get("chunk_id") or "").strip() == chunk_id
        }
        matched.discard("")
        if len(matched) == 1:
            return next(iter(matched))
    return ""


class ChunkIdMapper:
    """用 ``(source_file, 文本哈希)`` 把旧 chunk_id 映射到重切分后的新 chunk_id（M1.6）。

    ``chunk_id`` 的哈希输入包含 section 文本，新切分会让大纲的 chunk_id 全量变化，
    于是 ``feedback_log.jsonl``、主动学习样本里引用的旧 id 全部失效。映射依据是
    **原文**而不是 id：同一 ``source_file`` 下正文相同的块视为同一个块；来源文件
    缺失时退化为「全语料唯一文本」匹配（归一化长度不足
    ``MIN_TEXT_FALLBACK_LENGTH`` 不参与）。映射不到的样本标记 ``stale`` /
    ``stale_reason``，绝不静默丢弃（§8 风险表）。
    """

    STALE_NO_MATCH = "chunk_id 已失效且按 (source_file, 文本哈希) 找不到对应块"

    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self._by_fingerprint: dict[tuple[str, str], str] = {}
        self._by_text: dict[str, str | None] = {}
        self._known_ids: set[str] = set()
        if chunks:
            self.rebuild(chunks)

    def rebuild(self, chunks: list[dict[str, Any]]) -> None:
        """用最新的 ``chunks.json`` 重建映射索引。"""
        by_fingerprint: dict[tuple[str, str], str] = {}
        by_text: dict[str, str | None] = {}
        known: set[str] = set()
        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            known.add(chunk_id)
            fingerprint = chunk_text_hash(chunk.get("text") or "")
            source_file = str(chunk.get("source_file") or "").strip()
            by_fingerprint.setdefault((source_file, fingerprint), chunk_id)
            # 同一文本出现两次即视为有歧义：纯文本兜底不再使用它。
            by_text[fingerprint] = chunk_id if fingerprint not in by_text else None
        self._by_fingerprint = by_fingerprint
        self._by_text = by_text
        self._known_ids = known

    @property
    def known_chunk_ids(self) -> set[str]:
        """当前语料里仍然有效的 chunk_id。"""
        return set(self._known_ids)

    def resolve(self, chunk_id: str, text: str, source_file: str = "") -> str | None:
        """返回旧 id 对应的新 id；无法安全映射时返回 ``None``。"""
        if chunk_id and chunk_id in self._known_ids:
            return chunk_id
        normalized = normalize_chunk_text(text)
        if not normalized:
            return None
        fingerprint = chunk_text_hash(text)
        if source_file:
            mapped = self._by_fingerprint.get((source_file, fingerprint))
            if mapped is not None:
                return mapped
        if len(normalized) < MIN_TEXT_FALLBACK_LENGTH:
            return None
        return self._by_text.get(fingerprint)

    def map_record(
        self,
        record: dict[str, Any],
        *,
        id_key: str = "chunk_id",
        text_key: str = "text",
        source_key: str = "source_file",
    ) -> bool:
        """就地重写一条记录的 ``chunk_id``。

        返回 ``True`` 表示记录仍然可用（id 未变或已成功映射）；``False`` 表示已
        标成 ``stale``，调用方应跳过这条样本。
        """
        current = str(record.get(id_key) or "").strip()
        text = str(record.get(text_key) or record.get("corrected_text") or "")
        if not current and not normalize_chunk_text(text):
            return True  # 没有引用任何 chunk，无需映射
        mapped = self.resolve(current, text, _record_source_file(record, source_key))
        if mapped is None:
            record["stale"] = True
            record["stale_reason"] = self.STALE_NO_MATCH
            if current:
                record["previous_chunk_id"] = current
            return False
        if mapped != current:
            record["previous_chunk_id"] = current or record.get("previous_chunk_id")
            record[id_key] = mapped
        record.pop("stale", None)
        record.pop("stale_reason", None)
        return True

    def map_records(self, records: list[dict[str, Any]]) -> dict[str, int]:
        """批量映射记录本身与其中的 ``citations``，返回 mapped / unchanged / stale。"""
        stats = {"mapped": 0, "unchanged": 0, "stale": 0}
        for record in records or []:
            if not isinstance(record, dict):
                continue
            self._map_one(record, stats)
            citations = record.get("citations")
            if isinstance(citations, list):
                for citation in citations:
                    if isinstance(citation, dict):
                        self._map_one(citation, stats)
        return stats

    def _map_one(self, record: dict[str, Any], stats: dict[str, int]) -> None:
        current = str(record.get("chunk_id") or "").strip()
        text = str(record.get("text") or record.get("corrected_text") or "")
        if not current and not normalize_chunk_text(text):
            return  # 这条记录没有引用任何 chunk，不计入统计
        before = str(record.get("chunk_id") or "")
        if not self.map_record(record):
            stats["stale"] += 1
        elif str(record.get("chunk_id") or "") != before:
            stats["mapped"] += 1
        else:
            stats["unchanged"] += 1


def remap_samples_file(
    path: str | Path = REVIEW_SAMPLES_PATH,
    *,
    chunks_path: str | Path | None = None,
    output_path: str | Path | None = None,
    mapper: ChunkIdMapper | None = None,
) -> dict[str, Any]:
    """把样本文件里的旧 chunk_id 就地映射到新 id（M1.6）。

    支持 ``.json`` 列表与 ``.jsonl`` 逐行两种格式，默认原地写回（临时文件 +
    ``replace``）；映射不到的样本被标成 ``stale`` 而不是删除。
    """
    source = Path(path)
    if not source.exists():
        return {"error": f"样本文件不存在：{source}"}
    if mapper is None:
        chunk_file = (
            Path(chunks_path)
            if chunks_path is not None
            else next((item for item in CHUNKS_PATHS if item.exists()), None)
        )
        if chunk_file is None or not chunk_file.exists():
            return {"error": "chunks.json 不存在，无法建立 chunk_id 映射"}
        try:
            mapper = ChunkIdMapper(json.loads(chunk_file.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            return {"error": f"无法读取 {chunk_file}：{exc}"}

    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
        if not isinstance(records, list):
            return {"error": f"{source.name} 不是 JSON 列表"}

    stats = mapper.map_records(records)
    # 序列化必须在映射之后：否则写回的是映射前的快照。
    if source.suffix == ".jsonl":
        serialized = (
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
            + "\n"
        )
    else:
        serialized = json.dumps(records, ensure_ascii=False, indent=2)
    target = Path(output_path) if output_path is not None else source
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(target)
    return {"path": str(target), "total": len(records), **stats}


class RetrainingUpdater:
    """Update retrieval and graph assets from manually corrected samples."""

    def __init__(self) -> None:
        self.neo4j_driver = None

    def close(self) -> None:
        """Close external clients."""
        if self.neo4j_driver is not None:
            self.neo4j_driver.close()

    def _get_neo4j_driver(self):
        if self.neo4j_driver is None:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
        return self.neo4j_driver

    def load_corrected_samples(self, path: str | Path = CORRECTED_SAMPLES_PATH) -> list[dict[str, Any]]:
        """Load corrected samples exported by human reviewers."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Corrected samples file not found: {file_path}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("corrected_samples.json must contain a list.")
        return data

    def handle_chunking_issue(self, sample: dict[str, Any]) -> bool:
        """Update persisted chunk text/metadata after manual corrections."""
        chunk_id = str(sample.get("chunk_id", "")).strip()
        corrected_text = str(sample.get("corrected_text", "")).strip()
        corrected_metadata = sample.get("corrected_metadata", {})
        if not chunk_id or not corrected_text:
            return False
        if not isinstance(corrected_metadata, dict):
            corrected_metadata = {}

        updated = False
        for path in CHUNKS_PATHS:
            if not path.exists():
                continue
            chunks = json.loads(path.read_text(encoding="utf-8"))
            path_updated = False
            for chunk in chunks:
                if str(chunk.get("chunk_id", "")) != chunk_id:
                    continue
                chunk["text"] = corrected_text
                chunk["metadata"] = {
                    **(chunk.get("metadata") or {}),
                    **corrected_metadata,
                }
                path_updated = True
            if path_updated:
                path.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                updated = True
        return updated

    def handle_kg_issue(self, sample: dict[str, Any]) -> None:
        """Apply Cypher delete/create fixes for corrected KG relations."""
        delete_relations = sample.get("delete_relations", [])
        create_relations = sample.get("create_relations", [])
        with self._get_neo4j_driver().session() as session:
            for relation in delete_relations:
                if not isinstance(relation, dict):
                    continue
                session.run(
                    """
                    MATCH (a {name: $source})-[r]->(b {name: $target})
                    WHERE type(r) = $relation_type
                    DELETE r
                    """,
                    source=relation.get("source", ""),
                    target=relation.get("target", ""),
                    relation_type=relation.get("type", ""),
                )
            for relation in create_relations:
                if not isinstance(relation, dict):
                    continue
                relation_type = str(relation.get("type", "")).strip()
                if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", relation_type):
                    continue
                session.run(
                    f"""
                    MATCH (a {{name: $source}}), (b {{name: $target}})
                    MERGE (a)-[:{relation_type}]->(b)
                    """,
                    source=relation.get("source", ""),
                    target=relation.get("target", ""),
                )

    def handle_embedding_issue(self, sample: dict[str, Any]) -> None:
        """Append problematic query-response pairs to a future embedding fine-tune set."""
        record = {
            "query": sample.get("query", ""),
            "positive_text": sample.get("corrected_text", ""),
            "reason": sample.get("issue_type", "embedding"),
        }
        EMBEDDING_FINETUNE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EMBEDDING_FINETUNE_DATASET_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load_chunk_id_mapper(self) -> ChunkIdMapper | None:
        """用当前 ``chunks.json`` 构建旧→新 chunk_id 映射器；产物缺失时返回 None。"""
        source_path = next((path for path in CHUNKS_PATHS if path.exists()), None)
        if source_path is None:
            return None
        try:
            chunks = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[retrain] 无法读取 {source_path}（{exc}），跳过 chunk_id 映射")
            return None
        return ChunkIdMapper(chunks)

    def remap_sample_chunk_ids(self, samples: list[dict[str, Any]]) -> dict[str, int]:
        """对反馈 / 主动学习样本做一次旧→新 chunk_id 映射（M1.6）。"""
        mapper = self.load_chunk_id_mapper()
        if mapper is None:
            print("[retrain] 缺少 chunks.json，chunk_id 映射已跳过")
            return {"mapped": 0, "unchanged": 0, "stale": 0}
        return mapper.map_records(samples)

    def rebuild_index(self) -> dict[str, Any] | None:
        """Build the corrected chunks into a new collection version.

        The alias is left alone: the caller decides whether the new version is
        good enough to serve.
        """
        source_path = next((path for path in CHUNKS_PATHS if path.exists()), None)
        if source_path is None:
            return None
        chunks = json.loads(source_path.read_text(encoding="utf-8"))
        model_name = config.retrieval_model_name
        client = None
        existing: list[str] = []
        try:
            client = create_chroma_client()
            existing = [collection.name for collection in client.list_collections()]
        except (ChromaError, ValueError, OSError, RuntimeError) as exc:
            # Without a reachable store the version number still advances from
            # the alias record, and build_index reports the real failure.
            print(f"[retrain] could not list collections ({exc})")
        target = next_version_name(config.chroma_collection, existing=existing)
        summary = build_index(
            chunks,
            VECTOR_OUTPUT_DIR,
            model_name=model_name,
            dimensions=config.simple_embedding_dimensions,
            client=client,
            collection_name=target,
        )
        # Document frequencies shift with the corrected text, so BM25 stats
        # are rebuilt alongside the vectors rather than left stale.
        write_lexical_stats(
            chunks,
            Path(VECTOR_OUTPUT_DIR) / config.lexical_stats_path.name,
        )
        return summary

    def apply_updates(self, corrected_samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Dispatch corrected samples and rebuild the index when needed.

        先按 M1.6 把旧 chunk_id 映射到重切分后的新 id：映射不上的样本标成
        ``stale`` 并跳过 —— 既不会静默丢弃，也不会把修正写到错误的块上。
        """
        mapper = self.load_chunk_id_mapper()
        chunks_changed = False
        counts = {"chunking": 0, "kg": 0, "embedding": 0, "skipped": 0, "stale": 0}
        for sample in corrected_samples:
            if not isinstance(sample, dict):
                counts["skipped"] += 1
                continue
            if mapper is not None and not mapper.map_record(sample):
                counts["stale"] += 1
                continue
            issue_type = str(sample.get("issue_type", "")).strip().lower()
            if issue_type == "chunking":
                chunks_changed = self.handle_chunking_issue(sample) or chunks_changed
                counts["chunking"] += 1
            elif issue_type == "kg":
                self.handle_kg_issue(sample)
                counts["kg"] += 1
            elif issue_type == "embedding":
                self.handle_embedding_issue(sample)
                counts["embedding"] += 1
            else:
                counts["skipped"] += 1

        summary: dict[str, Any] = {"counts": counts, "rebuilt": None}
        if chunks_changed:
            summary["rebuilt"] = self.rebuild_index()
        return summary


def _score_collection(collection_name: str | None, dataset: str | None) -> dict[str, Any]:
    """Retrieval metrics for one collection, or an explanation of why not."""
    try:
        from eval.run_eval import evaluate, load_dataset, resolve_dataset_path
    except ImportError as exc:  # pragma: no cover - depends on checkout layout
        return {"error": f"eval package unavailable: {exc}"}
    try:
        items = load_dataset(resolve_dataset_path(dataset))
    except SystemExit as exc:
        return {"error": str(exc)}
    report = evaluate(
        items,
        stages=["retrieval"],
        tag="retraining",
        collection_name=collection_name,
        keep_details=False,
    )
    return report.get("retrieval", {"error": "retrieval stage did not run"})


def _print_comparison(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Print a before/after table; return whether the new version regressed."""
    keys = ("recall@1", "recall@5", "recall@10", "mrr")
    print("\n指标对比（修正前 -> 修正后）")
    print("| 指标 | before | after | delta |")
    print("| --- | --- | --- | --- |")
    regressed = False
    for key in keys:
        old = before.get(key)
        new = after.get(key)
        if old is None or new is None:
            print(f"| {key} | {old} | {new} | - |")
            continue
        delta = round(new - old, 4)
        regressed = regressed or delta < 0
        print(f"| {key} | {old} | {new} | {delta:+} |")
    return regressed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="score the old and new collections on the gold set before promoting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="promote the new version even if the metrics regress",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="point the alias back at the previous collection and exit",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--remap-chunk-ids",
        nargs="*",
        default=None,
        metavar="PATH",
        help=(
            "只做 M1.6 的旧→新 chunk_id 映射后退出；默认处理 "
            "outputs/samples_for_review.json，可显式给出多个样本文件"
        ),
    )
    arguments = parser.parse_args()

    if arguments.remap_chunk_ids is not None:
        updater = RetrainingUpdater()
        try:
            mapper = updater.load_chunk_id_mapper()
        finally:
            updater.close()
        for target in arguments.remap_chunk_ids or [REVIEW_SAMPLES_PATH]:
            summary = remap_samples_file(target, mapper=mapper)
            print(json.dumps(summary, ensure_ascii=False))
        return

    if arguments.rollback:
        record = read_alias_record()
        history = record.get("history") or []
        previous = next(
            (
                entry.get("previous")
                for entry in reversed(history)
                if entry.get("previous")
            ),
            None,
        )
        if not previous:
            raise SystemExit("no previous collection recorded; nothing to roll back to")
        write_alias_record(previous, note="rollback")
        print(f"alias now points at {previous}")
        return

    updater = RetrainingUpdater()
    try:
        corrected_samples = updater.load_corrected_samples()
        summary = updater.apply_updates(corrected_samples)
    finally:
        updater.close()

    print(f"Applied updates for {len(corrected_samples)} corrected samples.")
    print(json.dumps(summary["counts"], ensure_ascii=False))

    rebuilt = summary.get("rebuilt")
    if not rebuilt:
        print("No chunk corrections applied; the served collection is unchanged.")
        return

    candidate = rebuilt["collection"]
    active = resolve_active_collection()
    print(f"Built {candidate} ({rebuilt['count']} chunks); active alias is {active}.")

    metrics_payload: dict[str, Any] = {}
    if arguments.evaluate:
        before = _score_collection(active, arguments.dataset)
        after = _score_collection(candidate, arguments.dataset)
        metrics_payload = {"before": before, "after": after}
        if "error" in before or "error" in after:
            print(f"[retrain] evaluation skipped: {before.get('error') or after.get('error')}")
        elif _print_comparison(before, after) and not arguments.force:
            print(
                f"\n新版本在金标集上出现指标下降，未切换别名。"
                f"确认无误后可执行：\n"
                f"  python -c \"from src.data_processing.collection_registry import "
                f"write_alias_record; write_alias_record('{candidate}', note='manual')\""
            )
            return

    write_alias_record(
        candidate,
        note="retraining round",
        metrics=metrics_payload,
    )
    print(f"alias now points at {candidate}")


if __name__ == "__main__":
    main()
