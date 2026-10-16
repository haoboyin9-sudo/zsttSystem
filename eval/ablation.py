"""Ablation runner: same gold set, one component removed at a time.

    python eval/ablation.py                     # retrieval arms only (fast)
    python eval/ablation.py --with-answers      # also grade answers per arm
    python eval/ablation.py --arms full,no-rerank

Arms differ only in the retrieval configuration, so any metric gap is
attributable to the ablated component rather than to sampling.  The embedding
arm is the exception: swapping ``hash`` for the sentence-transformer model
requires rebuilding the collection, so it is run as a separate pass with
``EMBEDDING_PROVIDER=hash python run_pipeline.py embed`` and compared by tag.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import (  # noqa: E402
    evaluate,
    load_dataset,
    resolve_dataset_path,
)
from src.config import config  # noqa: E402
from src.online_service.chroma_retriever import RerankWeights  # noqa: E402

# name -> (description, RerankWeights kwargs)
ARMS: dict[str, tuple[str, dict[str, Any]]] = {
    "full": ("完整重排（向量 + BM25 词面 + 章节/来源加权）", {}),
    "no-rerank": (
        "仅向量相似度，去掉词面分与章节加权",
        {"lexical": 0.0, "section_boost": 0.0, "lexical_scheme": "none"},
    ),
    "no-lexical": (
        "保留章节加权，去掉词面分",
        {"lexical": 0.0, "lexical_scheme": "none"},
    ),
    "overlap-lexical": (
        "词面分用改造前的 bigram 重合率（无 IDF）",
        {"lexical_scheme": "overlap"},
    ),
    "no-section-boost": ("去掉章节偏好加权", {"section_boost": 0.0}),
    "lexical-heavy": ("词面分权重提到 0.35", {"vector": 0.55, "lexical": 0.35}),
}

#: A1–A7 对照实验（M5.3）：每个臂通过子进程改环境变量后跑一次 run_eval.py。
#: 环境变量名对应并行实现（P1–P3）的开关；尚未实现的臂等价于基线，仅作占位。
ENV_ARMS: dict[str, dict[str, Any]] = {
    "A1-legacy-chunker": {
        "group": "A1",
        "description": "A1 旧切分（对照）",
        "env": {"CHUNKER_HIERARCHY": "off"},
    },
    "A1-hierarchy-chunker": {
        "group": "A1",
        "description": "A1 层级切分",
        "env": {"CHUNKER_HIERARCHY": "on"},
    },
    "A2-breadcrumb-off": {
        "group": "A2",
        "description": "A2 无面包屑",
        "env": {"EMBEDDING_BREADCRUMB": "off"},
    },
    "A2-breadcrumb-on": {
        "group": "A2",
        "description": "A2 有面包屑",
        "env": {"EMBEDDING_BREADCRUMB": "on"},
    },
    "A3-row-evidence": {
        "group": "A3",
        "description": "A3 行级证据",
        "env": {"EVIDENCE_SMALL_TO_BIG": "off"},
    },
    "A3-small-to-big": {
        "group": "A3",
        "description": "A3 small-to-big 父块",
        "env": {"EVIDENCE_SMALL_TO_BIG": "on"},
    },
    "A4-score-order": {
        "group": "A4",
        "description": "A4 证据按相关度分数排序",
        "env": {"EVIDENCE_ORDER": "score"},
    },
    "A4-source-order": {
        "group": "A4",
        "description": "A4 证据按原文顺序排序",
        "env": {"EVIDENCE_ORDER": "source"},
    },
    "A5-dense": {
        "group": "A5",
        "description": "A5 仅 dense 召回",
        "env": {"RETRIEVAL_FUSION": "dense"},
    },
    "A5-bm25": {
        "group": "A5",
        "description": "A5 仅 BM25 召回",
        "env": {"RETRIEVAL_FUSION": "bm25"},
    },
    "A5-linear": {
        "group": "A5",
        "description": "A5 候选内线性加权（现状）",
        "env": {"RETRIEVAL_FUSION": "linear"},
    },
    "A5-rrf": {
        "group": "A5",
        "description": "A5 dense/BM25 独立召回 + RRF 融合",
        "env": {"RETRIEVAL_FUSION": "rrf"},
    },
    "A6-llm-structure": {
        "group": "A6",
        "description": "A6 structure 由 LLM 生成",
        "env": {"STRUCTURE_MODE": "llm"},
    },
    "A6-deterministic-structure": {
        "group": "A6",
        "description": "A6 structure 确定性回答",
        "env": {"STRUCTURE_MODE": "deterministic"},
    },
    "A7-cross-encoder-off": {
        "group": "A7",
        "description": "A7 无 Cross-Encoder（P3 占位）",
        "env": {"CROSS_ENCODER": "off"},
        "placeholder": True,
    },
    "A7-cross-encoder-on": {
        "group": "A7",
        "description": "A7 有 Cross-Encoder（P3 占位）",
        "env": {"CROSS_ENCODER": "on"},
        "placeholder": True,
    },
}


def run_arms(
    arm_names: list[str],
    *,
    dataset_path: Path,
    with_answers: bool,
    top_k: int,
    limit: int = 0,
) -> dict[str, Any]:
    items = load_dataset(dataset_path)
    if limit:
        items = items[:limit]
    stages = ["retrieval", "answer"] if with_answers else ["retrieval"]

    arms: dict[str, Any] = {}
    for name in arm_names:
        description, overrides = ARMS[name]
        print(f"[ablation] running arm {name}: {description}")
        report = evaluate(
            items,
            stages=stages,
            tag=f"ablation-{name}",
            dataset_path=str(dataset_path),
            top_k=top_k,
            weights=RerankWeights(**overrides),
            keep_details=False,
        )
        arms[name] = {
            "description": description,
            "weights": report["environment"]["rerank_weights"],
            "retrieval": report.get("retrieval", {}),
            "generation": report.get("generation", {}),
            "refusal": report.get("refusal", {}),
        }
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "item_count": len(items),
        "arms": arms,
    }


#: 报告里跨臂汇总的字段（weight 臂与 env 臂共用）。
_ARM_REPORT_KEYS = ("retrieval", "generation", "refusal", "structure", "instructor")


def run_env_arm(
    name: str,
    *,
    dataset_path: Path,
    stages: list[str],
    top_k: int,
    timeout: int = 3600,
) -> dict[str, Any]:
    """在子进程里设置环境变量后跑一次 ``run_eval.py``，回收其报告。"""
    spec = ENV_ARMS[name]
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(spec.get("env") or {})
    env.setdefault("PYTHONPATH", str(project_root))
    with tempfile.TemporaryDirectory(prefix=f"ablation-{name}-") as tmp:
        command = [
            sys.executable,
            str(project_root / "eval" / "run_eval.py"),
            "--dataset",
            str(dataset_path),
            "--stages",
            ",".join(stages),
            "--top-k",
            str(top_k),
            "--tag",
            f"ablation-{name}",
            "--output-dir",
            tmp,
        ]
        print(f"[ablation] env arm {name}: {spec['description']}")
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"env arm {name} failed (exit {completed.returncode}):\n"
                f"{(completed.stdout or '')[-2000:]}\n{(completed.stderr or '')[-2000:]}"
            )
        reports = sorted(Path(tmp).glob("*.json"))
        if not reports:
            raise SystemExit(f"env arm {name} produced no report json")
        report = json.loads(reports[-1].read_text(encoding="utf-8"))

    return {
        "description": spec["description"],
        "group": spec.get("group", ""),
        "env": dict(spec.get("env") or {}),
        "placeholder": bool(spec.get("placeholder")),
        **{key: report.get(key, {}) for key in _ARM_REPORT_KEYS},
    }


def run_env_arms(
    arm_names: list[str],
    *,
    dataset_path: Path,
    stages: list[str],
    top_k: int,
    timeout: int = 3600,
) -> dict[str, Any]:
    """依次跑完选定的 A1–A7 环境变量臂，汇总为一份报告。"""
    arms: dict[str, Any] = {}
    for name in arm_names:
        arms[name] = run_env_arm(
            name,
            dataset_path=dataset_path,
            stages=stages,
            top_k=top_k,
            timeout=timeout,
        )
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "stages": list(stages),
        "env_arms": arms,
    }


def render_markdown(report: dict[str, Any]) -> str:
    weight_arms = report.get("arms", {})
    env_arms = report.get("env_arms", {})
    lines = [
        "# 检索消融实验",
        "",
        f"- 数据集：`{report['dataset']}`（{report.get('item_count', '-')} 题）",
        f"- 生成时间：{report['created_at']}",
        "",
        "| 实验组 | 说明 | Recall@1 | Recall@5 | Recall@10 | MRR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, arm in weight_arms.items():
        retrieval = arm.get("retrieval") or {}
        lines.append(
            f"| `{name}` | {arm['description']} | "
            f"{retrieval.get('recall@1', '-')} | {retrieval.get('recall@5', '-')} | "
            f"{retrieval.get('recall@10', '-')} | {retrieval.get('mrr', '-')} |"
        )
    answer_arms = {
        name: arm
        for name, arm in weight_arms.items()
        if arm.get("generation")
    }
    if answer_arms:
        lines.extend(
            [
                "",
                "| 实验组 | 答案要点命中率 | 引用正确率 | 正确拒答率 | 误拒率 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for name, arm in answer_arms.items():
            generation = arm["generation"]
            refusal = arm.get("refusal") or {}
            lines.append(
                f"| `{name}` | {generation.get('answer_key_coverage', '-')} | "
                f"{generation.get('citation_precision', '-')} | "
                f"{refusal.get('correct_refusal_rate', '-')} | "
                f"{refusal.get('false_refusal_rate', '-')} |"
            )
    if env_arms:
        lines.extend(
            [
                "",
                "### A1–A7 环境变量消融",
                "",
                "| 实验组 | 分组 | 说明 | 环境变量 | Recall@5 | 答案要点命中率 | "
                "章节覆盖率 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, arm in env_arms.items():
            retrieval = arm.get("retrieval") or {}
            generation = arm.get("generation") or {}
            structure = arm.get("structure") or {}
            env_text = ",".join(
                f"{key}={value}" for key, value in (arm.get("env") or {}).items()
            )
            lines.append(
                f"| `{name}` | {arm.get('group', '-')} | {arm.get('description', '-')} | "
                f"`{env_text}` | {retrieval.get('recall@5', '-')} | "
                f"{generation.get('answer_key_coverage', '-')} | "
                f"{structure.get('chapter_coverage', '-')} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument(
        "--env-arms",
        default="",
        help="A1–A7 环境变量消融臂，逗号分隔；用 --list 查看",
    )
    parser.add_argument("--list", action="store_true", help="列出全部可用实验臂后退出")
    parser.add_argument("--with-answers", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--timeout", type=int, default=3600, help="每个 env 臂子进程超时（秒）"
    )
    arguments = parser.parse_args()

    if arguments.list:
        for name, (description, _overrides) in ARMS.items():
            print(f"weight  {name}: {description}")
        for name, spec in ENV_ARMS.items():
            print(f"env     {name} [{spec.get('group')}]: {spec['description']}")
        return

    dataset_path = resolve_dataset_path(arguments.dataset)
    env_arm_names = [
        name.strip() for name in arguments.env_arms.split(",") if name.strip()
    ]
    if env_arm_names:
        unknown_env = [name for name in env_arm_names if name not in ENV_ARMS]
        if unknown_env:
            raise SystemExit(
                f"unknown env arms: {unknown_env}; available: {sorted(ENV_ARMS)}"
            )
        report = run_env_arms(
            env_arm_names,
            dataset_path=dataset_path,
            stages=["retrieval", "answer"] if arguments.with_answers else ["retrieval"],
            top_k=arguments.top_k,
            timeout=arguments.timeout,
        )
    else:
        arm_names = [name.strip() for name in arguments.arms.split(",") if name.strip()]
        unknown = [name for name in arm_names if name not in ARMS]
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}; available: {sorted(ARMS)}")
        report = run_arms(
            arm_names,
            dataset_path=dataset_path,
            with_answers=arguments.with_answers,
            top_k=arguments.top_k,
            limit=arguments.limit,
        )
    directory = config.eval_report_dir
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{report['run_id']}-ablation"
    (directory / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = render_markdown(report)
    (directory / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\n[ablation] wrote {directory / f'{stem}.md'}")


if __name__ == "__main__":
    main()
