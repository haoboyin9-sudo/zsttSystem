from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.online_service.program_graph import (
    aggregate_concept_links,
    build_program_graph,
    corpus_overview,
    parse_semester,
    program_catalog,
)

MAJOR = "信息管理学院2025级档案学专业培养方案"
MINOR = "信息管理学院2025级档案学专业辅修专业培养方案"
BROAD = "信息管理学院学院2025级图书情报与档案管理类培养方案"
OTHER = "信息管理学院2025级图书馆学专业培养方案"


def _offering(program: str, semester: Any, sub: str = "专业基础课", **extra: Any) -> dict[str, Any]:
    program_type = (
        "主修专业类" if "大类" in program or "管理类" in program
        else "辅修专业" if "辅修专业" in program
        else "主修专业"
    )
    return {
        "program_name": program,
        "program_type": program_type,
        "course_category": "专业必修课",
        "course_subcategory": sub,
        "credits": 3,
        "hours": 54,
        "semester": semester,
        "instructor": extra.get("instructor", "/"),
    }


def _course(code: str, name: str, offerings: list[dict[str, Any]], prerequisites=()) -> dict[str, Any]:
    return {
        "course_code": code,
        "course_name": name,
        "prerequisites": list(prerequisites),
        "prerequisite_evidence": [
            {"course_code": p, "source_file": f"2025-1/{code}.docx", "source_type": "syllabus"}
            for p in prerequisites
        ],
        "offerings": offerings,
    }


COURSES = [
    _course("IM104", "档案学概论", [_offering(MAJOR, 2), _offering(MINOR, 1), _offering(BROAD, 2)]),
    _course("IM221", "档案管理学", [_offering(MAJOR, 3), _offering(MINOR, 2)], ["IM104"]),
    _course("IM3109", "数字档案馆", [_offering(MAJOR, 6, "专业核心课")], ["IM221"]),
    _course("IM289", "图书馆学基础", [_offering(OTHER, 3)]),
    _course("IM400", "档案前沿", [_offering(MAJOR, "1-6", "本研贯通课程")], ["IM289"]),
    _course("IM500", "档案信息化", [_offering(MAJOR, 5, instructor="张三")]),
]


def test_parse_semester_keeps_ranges_out_of_fixed_columns() -> None:
    assert parse_semester(3) == (3, "第 3 学期")
    assert parse_semester("4") == (4, "第 4 学期")
    assert parse_semester("1-6") == (None, "第 1–6 学期（弹性）")
    assert parse_semester("/") == (None, "未标注")


def test_catalog_groups_plans_by_college_and_major() -> None:
    catalog = program_catalog(COURSES)

    assert catalog["program_count"] == 4
    college = catalog["colleges"][0]
    assert college["name"] == "信息管理学院"
    majors = {major["name"]: major for major in college["majors"]}
    # The broad-category plan comes first; the 档案学 major carries both plans.
    assert college["majors"][0]["name"] == "图书情报与档案管理类"
    assert [p["program_type"] for p in majors["档案学"]["plans"]] == ["主修专业", "辅修专业"]
    assert majors["档案学"]["plans"][0]["course_count"] == 5


def test_graph_places_courses_by_semester_and_keeps_prerequisite_evidence() -> None:
    graph = build_program_graph(COURSES, MAJOR)

    nodes = {node["course_code"]: node for node in graph["nodes"]}
    assert nodes["IM104"]["semester"] == 2
    assert nodes["IM400"]["semester"] is None
    assert nodes["IM3109"]["group"] == "core"
    edges = {(e["source"], e["target"]): e for e in graph["edges"]}
    assert ("IM104", "IM221") in edges
    assert edges[("IM104", "IM221")]["source_file"] == "IM221.docx"
    assert edges[("IM104", "IM221")]["evidence_type"] == "syllabus"
    assert graph["stats"]["longest_chain"] == ["IM104", "IM221", "IM3109"]


def test_prerequisite_outside_the_plan_is_marked_external() -> None:
    graph = build_program_graph(COURSES, MAJOR)

    nodes = {node["course_code"]: node for node in graph["nodes"]}
    assert nodes["IM289"]["external"] is True
    assert nodes["IM289"]["home_program"] == OTHER
    assert graph["stats"]["course_count"] == 5
    assert graph["stats"]["external_count"] == 1


def test_placeholder_instructor_is_treated_as_missing() -> None:
    graph = build_program_graph(COURSES, MAJOR)

    nodes = {node["course_code"]: node for node in graph["nodes"]}
    assert nodes["IM104"]["instructor"] == ""
    assert nodes["IM500"]["instructor"] == "张三"


def test_semester_is_taken_from_the_selected_plan() -> None:
    graph = build_program_graph(COURSES, MINOR)

    nodes = {node["course_code"]: node for node in graph["nodes"]}
    assert nodes["IM104"]["semester"] == 1
    assert nodes["IM221"]["semester"] == 2
    assert "IM3109" not in nodes


def test_unknown_plan_returns_none() -> None:
    assert build_program_graph(COURSES, "不存在的方案") is None


def _registry() -> list[dict[str, Any]]:
    return [
        {"id": "c1", "canonical_name": "档案价值", "source_course_codes": ["IM104"]},
        {"id": "c2", "canonical_name": "档案鉴定", "source_course_codes": ["IM500"]},
        {"id": "c3", "canonical_name": "电子文件", "source_course_codes": ["IM500"]},
        {"id": "c4", "canonical_name": "元数据", "source_course_codes": ["IM104"]},
        {"id": "c5", "canonical_name": "数字档案", "source_course_codes": ["IM3109"]},
    ]


def _edge(source: str, target: str, **extra: Any) -> dict[str, Any]:
    return {
        "source_id": source,
        "target_id": target,
        "requires": True,
        "verification_source": "llm",
        "confidence": 0.9,
        **extra,
    }


def test_concept_links_apply_the_graph_stage_gate() -> None:
    links = aggregate_concept_links(
        _registry(),
        [
            _edge("c1", "c2"),
            _edge("c1", "c3", requires="true"),  # 字符串不算
            _edge("c1", "c3", verification_source="rule"),
            _edge("c4", "c3", confidence=0.3),
        ],
        min_confidence=0.6,
    )

    assert links["pairs"] == {("IM104", "IM500"): ["档案价值 → 档案鉴定"]}
    assert links["verified_relation_count"] == 1


def test_knowledge_edges_must_point_forward_and_skip_existing_prerequisites() -> None:
    links = aggregate_concept_links(
        _registry(),
        [
            _edge("c1", "c2"),  # IM104(2) → IM500(5)：保留
            _edge("c3", "c4"),  # IM500(5) → IM104(2)：逆时间，丢弃
            _edge("c5", "c3"),  # IM3109(6) → IM500(5)：逆时间，丢弃
        ],
        min_confidence=0.6,
    )
    links["pairs"][("IM104", "IM221")] = ["已是先修边"]

    graph = build_program_graph(COURSES, MAJOR, concept_links=links)

    pairs = {(e["source"], e["target"]) for e in graph["knowledge_edges"]}
    assert pairs == {("IM104", "IM500")}
    assert graph["knowledge_edges"][0]["weight"] == 1
    assert graph["knowledge_available"] is True


def test_knowledge_edges_in_both_directions_are_dropped_as_overlap() -> None:
    courses = [
        _course("A1", "甲", [_offering(MAJOR, 3)]),
        _course("B1", "乙", [_offering(MAJOR, 3)]),
    ]
    links = {"pairs": {("A1", "B1"): ["x → y"], ("B1", "A1"): ["y → x"]}}

    graph = build_program_graph(courses, MAJOR, concept_links=links)

    assert graph["knowledge_edges"] == []


def test_overview_counts_resolved_prerequisites_only() -> None:
    courses = [*COURSES, _course("IM900", "孤课", [_offering(MAJOR, 1)], ["不存在的课"])]

    overview = corpus_overview(courses, concept_links=None)

    assert overview["course_count"] == 7
    assert overview["prerequisite_edge_count"] == 3
    assert overview["major_count"] == 3
    assert overview["concept_count"] is None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    courses_path = tmp_path / "courses.json"
    courses_path.write_text(json.dumps(COURSES, ensure_ascii=False), encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "20260101T000000Z-old.json").write_text(
        json.dumps({"run_id": "old", "routing": {"accuracy": 0.5}}), encoding="utf-8"
    )
    (reports / "20260914T134003Z-new.json").write_text(
        json.dumps(
            {
                "run_id": "new",
                "routing": {"accuracy": 0.95},
                "answer_details": [{"id": "x"}],
                "routing_details": [{"id": "y"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("COURSES_OUTPUT_PATH", str(courses_path))
    monkeypatch.setenv("CONCEPT_REGISTRY_PATH", str(tmp_path / "missing_registry.json"))
    monkeypatch.setenv("CONCEPT_VERIFIED_EDGE_PATH", str(tmp_path / "missing_edges.json"))
    monkeypatch.setenv("EVAL_REPORT_DIR", str(reports))
    monkeypatch.setattr(main_module, "QUERY_LOG_PATH", tmp_path / "query.jsonl")
    monkeypatch.setattr(main_module, "FEEDBACK_LOG_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(main_module, "create_deepseek_client", lambda: None)
    monkeypatch.setattr(main_module, "ChromaRetriever", lambda *_a, **_k: None)
    monkeypatch.setattr(
        main_module.GraphDatabase,
        "driver",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("no neo4j")),
    )
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_programs_endpoint_lists_plans(client: TestClient) -> None:
    response = client.get("/programs")

    assert response.status_code == 200
    assert response.json()["program_count"] == 4


def test_program_graph_endpoint_works_without_neo4j_or_concepts(client: TestClient) -> None:
    response = client.get("/programs/graph", params={"program_name": MAJOR})

    assert response.status_code == 200
    data = response.json()
    assert data["stats"]["prerequisite_edge_count"] == 3
    assert data["knowledge_available"] is False
    assert data["knowledge_edges"] == []


def test_program_graph_endpoint_404s_for_unknown_plan(client: TestClient) -> None:
    response = client.get("/programs/graph", params={"program_name": "火星学院培养方案"})

    assert response.status_code == 404


def test_overview_endpoint_reports_backend_state(client: TestClient) -> None:
    data = client.get("/overview").json()

    assert data["course_count"] == 6
    assert data["neo4j"] == "unavailable"
    assert data["llm"] == "unavailable"


def test_personas_endpoint_exposes_configured_profiles(client: TestClient) -> None:
    from src.online_service.persona import PERSONA_PROFILES

    data = client.get("/personas").json()

    assert set(data["profiles"]) == {"student", "teacher", "visitor"}
    assert data["profiles"]["student"]["syllabus_top_k"] == PERSONA_PROFILES["student"]["syllabus_top_k"]


def test_latest_eval_returns_newest_report_without_item_rows(client: TestClient) -> None:
    data = client.get("/eval/latest").json()

    assert data["run_id"] == "new"
    assert "answer_details" not in data and "routing_details" not in data


def test_demo_page_has_the_four_views_and_program_graph() -> None:
    template = (
        Path(__file__).resolve().parents[1] / "src" / "templates" / "demo.html"
    ).read_text(encoding="utf-8")

    for view in ("view-overview", "view-qa", "view-graph", "view-method"):
        assert f'id="{view}"' in template
    assert 'fetch("/query"' in template
    assert "/programs/graph?" in template
    assert 'id="compare-mode"' in template
    # 每个角色都有示例问题。
    for persona in ("student", "teacher", "visitor"):
        assert f"{persona}: {{" in template
