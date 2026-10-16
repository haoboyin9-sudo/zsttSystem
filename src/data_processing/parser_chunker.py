"""Parse training plans and course syllabi into structure-aware chunks."""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import docx
import pandas as pd
from docx.document import Document as DocxDocumentType
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.config import config


@dataclass(frozen=True)
class CourseOffering:
    """One course occurrence in one training plan."""

    program_name: str
    program_type: str
    course_category: str
    course_subcategory: str
    credits: int | float | None
    theory_credits: int | float | None
    practice_credits: int | float | None
    hours: int | float | str | None
    theory_hours: int | float | str | None
    practice_hours: int | float | str | None
    semester: int | float | str | None
    instructor: str
    is_honors: str
    graduation_requirements: str
    source_file: str
    sheet_name: str


@dataclass
class CourseCatalogEntry:
    """Canonical course plus all of its training-plan memberships."""

    course_code: str
    course_name: str
    english_name: str = ""
    prerequisites: list[str] = field(default_factory=list)
    offerings: list[CourseOffering] = field(default_factory=list)

    def add_offering(self, offering: CourseOffering) -> None:
        if offering not in self.offerings:
            self.offerings.append(offering)

    def to_dict(self) -> dict[str, Any]:
        primary = self.offerings[0] if self.offerings else None
        return {
            "course_code": self.course_code,
            "course_name": self.course_name,
            "english_name": self.english_name,
            "credits": primary.credits if primary else None,
            "hours": primary.hours if primary else None,
            "semester": primary.semester if primary else None,
            "instructor": primary.instructor if primary else "",
            "course_category": primary.course_category if primary else "",
            "course_subcategory": primary.course_subcategory if primary else "",
            "prerequisites": self.prerequisites,
            "program_names": list(
                dict.fromkeys(offering.program_name for offering in self.offerings)
            ),
            "offerings": [asdict(offering) for offering in self.offerings],
        }


class SyllabusChunker:
    """Parse XLSX training plans and DOCX syllabi into teaching-oriented chunks."""

    #: M1.1 标题分级阈值：集中为类常量，便于按其它学校模板调整。
    HEADING_KEYWORD_MAX_CHARS = 24
    HEADING_LEVEL3_MAX_CHARS = 20
    #: 含冒号的列表项标题必须两侧都是短标签才算标题（"：+长文本" 回到正文）。
    HEADING_LEVEL3_COLON_MAX_CHARS = 12
    HEADING_LEVEL3_COLON_TAIL_MAX_CHARS = 8
    #: 以这些标点结尾的列表项一律视为正文（多为完整句子）。
    HEADING_LEVEL3_FORBIDDEN_TAILS = "。；;，"
    #: 一级「一、」标题阈值：分类标题通常是短标签，而「一、总成绩构成：…」这类
    #: 正文列表项往往以句读或冒号收尾（实测 432 个候选中 53% 是这样的正文行）。
    HEADING_LEVEL1_MAX_CHARS = 24
    HEADING_LEVEL1_FORBIDDEN_TAILS = "。；;，：:"
    #: 栈为空时，关键词标题的兜底层级。
    KEYWORD_HEADING_DEFAULT_LEVEL = 1
    #: 「第一周 …」「第一讲 …」的语义层级：进度表内的一行。
    WEEK_TITLE_LEVEL = 2

    OVERVIEW_SECTION = "文档概览"

    #: chunk 文本长度上限（含标题前缀），与原 ``tests/test_training_plan_parser.py`` 的断言一致。
    MAX_TEXT_CHARS = 750

    #: M1.4 进度表表头短名：原文标签关键词 → 短名。
    TABLE_HEADER_SHORT_NAMES = (
        (("章节", "次序"), "章节"),
        (("教学内容", "内容"), "内容"),
        (("学时",), "学时"),
        (("方式",), "教学方式"),
    )

    #: M1.5 实习类大纲模板的字段映射（先于通用规则匹配）。
    INTERNSHIP_SECTION_TYPES = (
        ("实习目的", "course_objectives"),
        ("实习内容", "teaching_content"),
        ("实习具体安排", "teaching_schedule"),
        ("考核体系", "assessment"),
        ("实习的性质", "basic_info"),
    )

    def __init__(self) -> None:
        self.section_keywords = (
            "课程目标",
            "教学内容",
            "课程基本内容",
            "教学要求",
            "教学安排",
            "教学进度",
            "课程简介",
            "课程性质",
            "课程任务",
            "实践环节",
            "实验内容",
            "课程考核",
            "考核方式",
            "成绩评定",
            "教材",
            "参考书",
            "先修课程",
            "毕业要求",
        )
        self.course_code_pattern = re.compile(r"\b[A-Za-z]{2,}\d{2,}|\b\d{5,}\b")
        self.week_title_pattern = re.compile(
            r"^\s*(第[一二三四五六七八九十百0-9]+周.*|第[一二三四五六七八九十百0-9]+讲.*)$"
        )
        self.level1_chinese_number_pattern = re.compile(r"^\s*[一二三四五六七八九十]+、")
        self.level1_chapter_pattern = re.compile(
            r"^\s*第[一二三四五六七八九十百0-9]+(?:章|部分|单元|模块)"
        )
        self.level2_heading_pattern = re.compile(
            r"^\s*(?:[（(][一二三四五六七八九十]+[）)]"
            r"|第[一二三四五六七八九十百0-9]+节"
            r"|\d+\.\d+\s*\S)"
        )
        self.level3_number_pattern = re.compile(r"^\s*\d+[.、．]\s*\S")
        self.style_heading_pattern = re.compile(
            r"(?:heading|标题)\s*(\d+)",
            re.IGNORECASE,
        )
        self.max_chunk_chars = 700
        self.chunk_overlap_chars = 80
        self.max_text_chars = self.MAX_TEXT_CHARS
        #: M1.2 解析过程中的标题栈 ``(level, title)``，由 ``parse_syllabus`` 维护。
        self._heading_stack: list[tuple[int, str]] = []
        self.template_noise_patterns = (
            re.compile(r"^注[：:].*(A4|教学大纲|打印|纸张)", re.I),
            re.compile(r"^[（(].*(要求有一定的字数|对各种教学环节的安排|包括课堂讲授|推荐若干参考书).*[）)]$"),
        )

    def _normalize_text(self, value: Any) -> str:
        """Normalize cell or paragraph text for matching and output."""
        if value is None:
            return ""
        text = str(value).replace("\r", "\n")
        text = re.sub(r"[ \t\u3000]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_key(self, value: str) -> str:
        """Normalize header names for flexible column matching."""
        return re.sub(r"[\s_（）()\-:/：]+", "", value).lower()

    def _find_column(self, columns: list[str], candidates: tuple[str, ...]) -> str | None:
        """Resolve the best matching column for a logical field."""
        normalized_map = {column: self._normalize_key(column) for column in columns}
        normalized_candidates = [self._normalize_key(candidate) for candidate in candidates]

        for column, normalized in normalized_map.items():
            if normalized in normalized_candidates:
                return column

        for column, normalized in normalized_map.items():
            if any(candidate in normalized for candidate in normalized_candidates):
                return column

        return None

    def _find_column_index(
        self,
        columns: list[str],
        candidates: tuple[str, ...],
    ) -> int | None:
        """Resolve a logical field to its column position."""
        matched = self._find_column(columns, candidates)
        return columns.index(matched) if matched is not None else None

    def _looks_like_course_table_header(self, row_values: list[str]) -> bool:
        """Detect the first header row of a course table inside a worksheet."""
        normalized = [self._normalize_key(value) for value in row_values if self._normalize_text(value)]
        if not normalized:
            return False

        has_code = any(
            token in item
            for item in normalized
            for token in ("课程编码", "课程代码", "课程编号")
        )
        has_name = any(
            token in item
            for item in normalized
            for token in ("课程中文名称", "课程名称", "课程")
        )
        return has_code and has_name

    def _build_header_row(
        self,
        dataframe: pd.DataFrame,
        header_row_index: int,
    ) -> tuple[list[str], int]:
        """Flatten one-row or two-row worksheet headers into a single header list."""
        primary = [
            self._normalize_text(value)
            for value in dataframe.iloc[header_row_index].tolist()
        ]

        data_start_index = header_row_index + 1
        if header_row_index + 1 >= len(dataframe):
            return primary, data_start_index

        secondary = [
            self._normalize_text(value)
            for value in dataframe.iloc[header_row_index + 1].tolist()
        ]
        secondary_normalized = [self._normalize_key(value) for value in secondary if value]

        if any(
            token in item
            for item in secondary_normalized
            for token in ("总学分", "理论学分", "实验实践学分")
        ):
            merged = []
            for first, second in zip(primary, secondary):
                if first and second:
                    merged.append(f"{first}-{second}")
                else:
                    merged.append(first or second)
            return merged, header_row_index + 2

        return primary, data_start_index

    def _extract_program_name(self, dataframe: pd.DataFrame, xlsx_path: Path) -> str:
        """Infer a program name from the first few rows of a worksheet."""
        for _, row in dataframe.head(12).iterrows():
            for value in row.tolist():
                text = self._normalize_text(value)
                if text and "培养方案" in text:
                    return text
        return xlsx_path.stem

    def _extract_sheet_course_catalog(
        self,
        dataframe: pd.DataFrame,
        xlsx_path: Path,
        sheet_name: str = "",
    ) -> dict[str, dict[str, Any]]:
        """Extract course metadata from a single worksheet."""
        dataframe = dataframe.fillna("")

        header_row_index: int | None = None
        for idx in range(len(dataframe)):
            row_values = [self._normalize_text(value) for value in dataframe.iloc[idx].tolist()]
            if self._looks_like_course_table_header(row_values):
                header_row_index = idx
                break

        if header_row_index is None:
            return {}

        columns, data_start_index = self._build_header_row(dataframe, header_row_index)
        if not any(self._normalize_text(column) for column in columns):
            return {}

        code_index = self._find_column_index(
            columns,
            ("课程代码", "课程编码", "课程编号"),
        )
        name_index = self._find_column_index(
            columns,
            ("课程名称", "课程中文名称", "课程中文名称/英文名称"),
        )
        credits_index = self._find_column_index(
            columns,
            ("总学分", "学分情况-总学分", "学分", "课程学分"),
        )
        theory_credits_index = self._find_column_index(
            columns,
            ("理论学分", "学分情况-理论学分"),
        )
        practice_credits_index = self._find_column_index(
            columns,
            ("实验实践学分", "学分情况-实验实践学分"),
        )
        hours_index = self._find_column_index(
            columns,
            ("总学时", "学时情况-总学时", "学时", "课程学时"),
        )
        theory_hours_index = self._find_column_index(
            columns,
            ("理论学时", "学时情况-理论学时"),
        )
        practice_hours_index = self._find_column_index(
            columns,
            ("实验实践学时", "学时情况-实验实践学时"),
        )
        semester_index = self._find_column_index(columns, ("开课学期", "学期"))
        instructor_index = self._find_column_index(
            columns,
            ("课程负责人", "任课教师", "教师"),
        )
        honors_index = self._find_column_index(columns, ("是否荣誉课程",))
        graduation_index = self._find_column_index(
            columns,
            ("对应毕业要求", "毕业要求"),
        )
        prereq_index = self._find_column_index(
            columns,
            ("先修课程代码", "先修课程", "前置课程", "先修"),
        )
        category_index = self._find_column_index(
            columns,
            ("课程细类", "课程类别", "课程类型"),
        )
        subcategory_index = None
        if (
            category_index is not None
            and code_index is not None
            and category_index + 1 < code_index
            and not self._normalize_text(columns[category_index + 1])
        ):
            subcategory_index = category_index + 1

        if code_index is None or name_index is None:
            return {}

        program_name = self._extract_program_name(dataframe, xlsx_path)
        program_type = self._extract_program_type(program_name, xlsx_path)
        entries: dict[str, CourseCatalogEntry] = {}
        current_category = ""
        current_subcategory = ""

        for row_index in range(data_start_index, len(dataframe)):
            values = [
                self._normalize_text(value)
                for value in dataframe.iloc[row_index].tolist()
            ]
            values += [""] * (len(columns) - len(values))

            category = self._value_at(values, category_index)
            if category:
                current_category = category
                current_subcategory = ""
            subcategory = self._value_at(values, subcategory_index)
            if subcategory:
                current_subcategory = subcategory

            raw_code = self._value_at(values, code_index)
            raw_name = self._value_at(values, name_index)
            course_codes = self._extract_course_codes_from_text(raw_code)
            if not course_codes or not raw_name:
                continue

            course_name = raw_name.splitlines()[0].strip()
            english_name = (
                raw_name.splitlines()[1].strip()
                if len(raw_name.splitlines()) > 1
                else ""
            )
            if not course_name:
                continue

            offering = CourseOffering(
                program_name=program_name,
                program_type=program_type,
                course_category=current_category,
                course_subcategory=current_subcategory,
                credits=self._number_or_text(self._value_at(values, credits_index)),
                theory_credits=self._number_or_text(
                    self._value_at(values, theory_credits_index)
                ),
                practice_credits=self._number_or_text(
                    self._value_at(values, practice_credits_index)
                ),
                hours=self._number_or_text(self._value_at(values, hours_index)),
                theory_hours=self._number_or_text(
                    self._value_at(values, theory_hours_index)
                ),
                practice_hours=self._number_or_text(
                    self._value_at(values, practice_hours_index)
                ),
                semester=self._number_or_text(
                    self._value_at(values, semester_index)
                ),
                instructor=self._value_at(values, instructor_index),
                is_honors=self._value_at(values, honors_index),
                graduation_requirements=self._value_at(
                    values,
                    graduation_index,
                ),
                source_file=xlsx_path.name,
                sheet_name=sheet_name,
            )
            prerequisites = self._split_prerequisites(
                self._value_at(values, prereq_index)
            )
            for course_code in course_codes:
                entry = entries.setdefault(
                    course_code,
                    CourseCatalogEntry(
                        course_code=course_code,
                        course_name=course_name,
                        english_name=english_name,
                        prerequisites=prerequisites,
                    ),
                )
                entry.add_offering(offering)

        return {code: entry.to_dict() for code, entry in entries.items()}

    @staticmethod
    def _value_at(values: list[str], index: int | None) -> str:
        if index is None or index >= len(values):
            return ""
        return values[index]

    def _number_or_text(self, value: Any) -> int | float | str | None:
        text = self._normalize_text(value)
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if number.is_integer() else number

    def _extract_program_type(self, program_name: str, xlsx_path: Path) -> str:
        text = f"{program_name} {xlsx_path.stem}"
        for value in ("辅修微专业", "辅修专业", "主修专业类", "主修专业"):
            if value in text:
                return value
        return "培养方案"

    @staticmethod
    def _merge_catalog(
        target: dict[str, dict[str, Any]],
        incoming: dict[str, dict[str, Any]],
    ) -> None:
        for code, course in incoming.items():
            if code not in target:
                target[code] = course
                continue
            existing = target[code]
            known_offerings = {
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                for item in existing.get("offerings", [])
            }
            for offering in course.get("offerings", []):
                key = json.dumps(offering, ensure_ascii=False, sort_keys=True)
                if key not in known_offerings:
                    existing.setdefault("offerings", []).append(offering)
                    known_offerings.add(key)
            existing["program_names"] = list(
                dict.fromkeys(
                    [
                        *existing.get("program_names", []),
                        *course.get("program_names", []),
                    ]
                )
            )

    def _split_prerequisites(self, value: Any) -> list[str]:
        """Split prerequisite course codes or names into a list."""
        text = self._normalize_text(value)
        if not text:
            return []
        parts = re.split(r"[、,，;/；\n]+", text)
        empty_values = {
            "无",
            "无要求",
            "无先修课程",
            "无先修课",
            "暂无",
            "不要求",
            "没有",
        }
        return [
            part.strip()
            for part in parts
            if part.strip() and part.strip("。 ") not in empty_values
        ]

    @staticmethod
    def _normalize_course_alias(value: Any) -> str:
        text = str(value or "").strip().casefold()
        text = text.strip("《》〈〉“”\"'()（）[]【】。 ")
        return re.sub(r"[\s\u3000·・_\-—:：/]+", "", text)

    def _course_alias_index(
        self,
        course_catalog: dict[str, dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build an exact alias index; fuzzy matches never create hard edges."""
        aliases: dict[str, list[dict[str, Any]]] = {}
        for code, course in course_catalog.items():
            values = [
                code,
                course.get("course_code"),
                course.get("course_name"),
                course.get("english_name"),
                *(course.get("aliases") or []),
            ]
            for value in values:
                normalized = self._normalize_course_alias(value)
                if not normalized:
                    continue
                bucket = aliases.setdefault(normalized, [])
                if all(item.get("course_code") != code for item in bucket):
                    bucket.append(course)
        return aliases

    def resolve_prerequisites(
        self,
        raw_names: list[str],
        course_catalog: dict[str, dict[str, Any]],
        *,
        dependent_course_code: str,
        source_file: str,
        section: str,
        source_year: str,
        source_type: str = "syllabus",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve explicit names by exact aliases and retain unresolved evidence."""
        alias_index = self._course_alias_index(course_catalog)
        resolved: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        seen_resolved: set[str] = set()
        seen_unresolved: set[tuple[str, str]] = set()

        for raw_name in raw_names:
            raw_name = self._normalize_text(raw_name).strip("。 ")
            if not raw_name:
                continue
            matches = alias_index.get(self._normalize_course_alias(raw_name), [])
            matches = [
                match
                for match in matches
                if str(match.get("course_code", "")).casefold()
                != dependent_course_code.casefold()
            ]
            if len(matches) == 1:
                match = matches[0]
                code = str(match.get("course_code", ""))
                if code in seen_resolved:
                    continue
                resolved.append(
                    {
                        "raw_name": raw_name,
                        "course_code": code,
                        "course_name": self._normalize_text(
                            match.get("course_name", "")
                        ),
                        "relation": "PREREQUISITE_OF",
                        "confidence": 1.0,
                        "source_type": source_type,
                        "source_file": source_file,
                        "section": section,
                        "source_year": source_year,
                        "verified": True,
                    }
                )
                seen_resolved.add(code)
                continue

            reason = "ambiguous" if len(matches) > 1 else "not_found"
            key = (raw_name, reason)
            if key in seen_unresolved:
                continue
            unresolved.append(
                {
                    "raw_name": raw_name,
                    "dependent_course_code": dependent_course_code,
                    "source_file": source_file,
                    "section": section,
                    "source_year": source_year,
                    "reason": reason,
                }
            )
            seen_unresolved.add(key)
        return resolved, unresolved

    def _explicit_prerequisite_names(
        self,
        parsed_chunks: list[dict[str, str]],
    ) -> list[str]:
        values: list[str] = []
        pattern = re.compile(
            r"(?:先修课程|前置课程|预修课程)\s*[：:]\s*([^\n。；;]+)"
        )
        for chunk in parsed_chunks:
            if chunk.get("section_type") != "basic_info":
                continue
            for match in pattern.finditer(str(chunk.get("content", ""))):
                values.extend(self._split_prerequisites(match.group(1)))
        return list(dict.fromkeys(values))

    def _extract_course_code_from_text(self, text: str) -> str:
        """Extract a likely course code from a syllabus title or filename."""
        match = self.course_code_pattern.search(text)
        return match.group(0) if match else ""

    def _extract_course_codes_from_text(self, text: str) -> list[str]:
        """Extract and deduplicate valid course codes from a cell."""
        return list(dict.fromkeys(self.course_code_pattern.findall(text)))

    def _extract_course_name_from_path(self, path: Path) -> str:
        """Guess course name from the DOCX filename."""
        stem = path.stem
        stem = re.sub(r"[_\-]+", " ", stem)
        stem = re.sub(self.course_code_pattern, "", stem)
        return self._normalize_text(stem)

    def _is_heading(self, paragraph: Paragraph) -> bool:
        """薄封装：段落是否为一个标题（保留给既有调用点）。"""
        return self._heading_level(paragraph) is not None

    def _heading_level(self, paragraph: Paragraph) -> int | None:
        """返回段落的标题层级，``None`` 表示正文（层级数字越小越高）。

        M1.1 规则（按优先级）：

        1. Word 样式名含 ``Heading N`` / ``标题 N`` → ``N``；
        2. ``第一周 …`` / ``第一讲 …`` → ``WEEK_TITLE_LEVEL``（沿用旧行为，
           视为进度表内的一行，不改变它「是标题」的语义）；
        3. 一级：``一、`` / ``第3章`` / ``第三部分`` / ``第一单元`` / ``第一模块``。
           ``第…章`` 形式按规范无条件成立；``一、`` 形式另加短标签约束
           （``HEADING_LEVEL1_*``）：去掉括注后 ≤ ``HEADING_LEVEL1_MAX_CHARS``
           且不以句读/冒号收尾，用来把 ``一、总成绩构成：平时考核占比40%…``、
           ``二、平时考核说明：`` 这类编号正文挡在标题之外，且不再回落到关键词规则；
        4. 二级：``（三）`` / ``第3节`` / ``3.2 索引结构``；
        5. 三级：``1. 教材`` 这类编号列表项，须同时满足长度、结尾标点与
           冒号后缀三条阈值。编号列表项**不再回落到关键词规则**，这样
           ``3. 期末考核方式为闭卷考试。``、``2. 平时成绩包括4次作业和考勤：……``
           这类正文行会被挡在标题之外（修 D1）；
        6. 关键词：命中 ``section_keywords`` 且长度 ≤ ``HEADING_KEYWORD_MAX_CHARS``
           → 继承当前栈顶层级（同级替换）；栈为空时取
           ``KEYWORD_HEADING_DEFAULT_LEVEL``。
        """
        text = self._normalize_text(paragraph.text)
        if not text:
            return None

        if not config.chunker_hierarchy:
            # A1 对照臂：改造前的单层标题判定（只有是/否）。
            return self._legacy_heading_level(paragraph, text)

        style_level = self._style_heading_level(paragraph)
        if style_level is not None:
            return style_level

        if self.week_title_pattern.match(text):
            return self.WEEK_TITLE_LEVEL

        if self.level1_chapter_pattern.match(text):
            return 1

        if self.level1_chinese_number_pattern.match(text):
            return 1 if self._is_chinese_number_heading(text) else None

        if self.level2_heading_pattern.match(text):
            return 2

        if self.level3_number_pattern.match(text):
            return 3 if self._is_short_list_heading(text) else None

        return self._keyword_heading_level(text)

    def _legacy_heading_level(self, paragraph: Paragraph, text: str) -> int | None:
        """A1 对照臂：复刻改造前 ``_is_heading()`` 的是/否判定（统一当作一级）。

        规则逐条照搬旧实现，只是把 ``bool`` 换成 ``1`` / ``None``，好让标题栈
        走同一条代码路径。``CHUNKER_HIERARCHY=off`` 时启用，用于跑 A1 消融。
        """
        if self._style_heading_level(paragraph) is not None:
            return 1
        if self.week_title_pattern.match(text):
            return 1
        heading_text = re.sub(r"[（(].*?[）)]", "", text).strip(" ：:")
        if any(keyword in heading_text for keyword in self.section_keywords):
            return 1 if len(heading_text) <= 24 else None
        if re.match(
            r"^\s*(第[一二三四五六七八九十百0-9]+[章节部分单元].*|\d+(\.\d+){0,2}\s*.+)$",
            text,
        ):
            return 1
        return None

    def _style_heading_level(self, paragraph: Paragraph) -> int | None:
        """从 Word 样式名解析标题层级，如 ``Heading 2`` / ``标题 2``。"""
        style = getattr(paragraph, "style", None)
        style_name = getattr(style, "name", "") or ""
        if not style_name:
            return None
        match = self.style_heading_pattern.search(style_name)
        if match is not None:
            return int(match.group(1))
        if "heading" in style_name.lower() or "标题" in style_name:
            return 1
        return None

    def _is_short_list_heading(self, text: str) -> bool:
        """判断编号列表项是否短到可以当作第三级标题（修 D1 的关键阈值）。"""
        if len(text) > self.HEADING_LEVEL3_MAX_CHARS:
            return False
        if text.endswith(tuple(self.HEADING_LEVEL3_FORBIDDEN_TAILS)):
            return False
        for separator in ("：", ":"):
            if separator in text:
                if len(text) > self.HEADING_LEVEL3_COLON_MAX_CHARS:
                    return False
                _, _, tail = text.partition(separator)
                if len(tail.strip()) > self.HEADING_LEVEL3_COLON_TAIL_MAX_CHARS:
                    return False
                break
        return True

    def _is_chinese_number_heading(self, text: str) -> bool:
        """判断「一、」形式的一级标题：短标签而非「键：值」式正文列表项。"""
        if text.rstrip().endswith(tuple(self.HEADING_LEVEL1_FORBIDDEN_TAILS)):
            return False
        title = re.sub(r"[（(].*?[）)]", "", text).strip(" ：:")
        return 0 < len(title) <= self.HEADING_LEVEL1_MAX_CHARS

    def _keyword_heading_level(self, text: str) -> int | None:
        """关键词标题：命中 ``section_keywords`` 时继承栈顶层级（同级）。"""
        heading_text = re.sub(r"[（(].*?[）)]", "", text).strip(" ：:")
        if not heading_text or len(heading_text) > self.HEADING_KEYWORD_MAX_CHARS:
            return None
        if not any(keyword in heading_text for keyword in self.section_keywords):
            return None
        if self._heading_stack:
            return self._heading_stack[-1][0]
        return self.KEYWORD_HEADING_DEFAULT_LEVEL

    def _push_heading(self, level: int, title: str) -> list[str]:
        """M1.2 入栈语义：弹出 ``level >= 新level`` 的项后压入，返回新路径。"""
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        self._heading_stack.append((level, title))
        return [stacked_title for _, stacked_title in self._heading_stack]

    @staticmethod
    def _strip_heading_number(title: str) -> str:
        """去掉标题前的中文/阿拉伯编号，便于判断表格标题与章节标题是否重复。"""
        stripped = re.sub(
            r"^\s*(?:[一二三四五六七八九十]+、"
            r"|[（(][一二三四五六七八九十]+[）)]"
            r"|第[一二三四五六七八九十百0-9]+[章节部分单元模块讲]"
            r"|\d+(?:\.\d+)*[.、．]?)\s*",
            "",
            title,
        )
        return stripped.strip(" ：:")

    @staticmethod
    def _heading_title(text: str) -> str:
        """标题展示名：保留前导编号（如「（二）」），去掉括注说明。"""
        match = re.match(r"^\s*([（(][一二三四五六七八九十]+[）)])", text)
        number_prefix = match.group(1) if match is not None else ""
        rest = text[len(number_prefix) :] if number_prefix else text
        rest = re.sub(r"[（(].*?[）)]", "", rest)
        title = f"{number_prefix}{rest}".strip(" ：:")
        return title or text.strip()

    def _paragraph_is_effectively_empty(self, paragraph: Paragraph) -> bool:
        """Skip blank paragraphs."""
        return not self._normalize_text(paragraph.text)

    def _is_template_noise(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        return any(pattern.search(normalized) for pattern in self.template_noise_patterns)

    def _unique_row_values(self, row: Any) -> list[str]:
        """Read a row without repeating values from horizontally merged cells."""
        values: list[str] = []
        seen_cells: set[int] = set()
        for cell in row.cells:
            cell_id = id(cell._tc)
            if cell_id in seen_cells:
                continue
            seen_cells.add(cell_id)
            values.append(self._normalize_text(cell.text))
        return values

    def _table_to_markdown(self, table: Table) -> str:
        """Convert a DOCX table into Markdown text."""
        rows: list[list[str]] = []
        for row in table.rows:
            row_values = self._unique_row_values(row)
            if any(row_values):
                rows.append(row_values)

        if not rows:
            return ""

        column_count = max(len(row) for row in rows)
        normalized_rows = [row + [""] * (column_count - len(row)) for row in rows]

        header = normalized_rows[0]
        separator = ["---"] * column_count
        markdown_lines = [
            f"| {' | '.join(header)} |",
            f"| {' | '.join(separator)} |",
        ]
        for row in normalized_rows[1:]:
            markdown_lines.append(f"| {' | '.join(row)} |")

        return "\n".join(markdown_lines)

    @staticmethod
    def _section_type(title: str) -> str:
        normalized = re.sub(r"\s+", "", title)
        for keyword, section_type in SyllabusChunker.INTERNSHIP_SECTION_TYPES:
            if keyword in normalized:
                return section_type
        if "课程目标" in normalized:
            return "course_objectives"
        if re.search(r"教学进度|章节|第.+章", normalized):
            return "teaching_schedule"
        if "教学内容" in normalized or "课程基本内容" in normalized:
            return "teaching_content"
        if re.search(r"考核|成绩评定", normalized):
            return "assessment"
        if "参考书" in normalized:
            return "references"
        if "教材" in normalized:
            return "textbook"
        if "先修" in normalized:
            return "prerequisites"
        if re.search(r"基本信息|课程简介|课程性质|课程任务", normalized):
            return "basic_info"
        return "overview"

    def _split_content(self, content: str, limit: int | None = None) -> list[str]:
        """Bound chunk size while retaining a small, deterministic overlap.

        ``limit`` 允许调用方按标题前缀长度收紧单块预算（``build_syllabus_chunks``
        保证最终 ``text`` 不超过 ``max_text_chars``）。
        """
        content = self._normalize_text(content)
        max_chars = limit or self.max_chunk_chars
        if len(content) <= max_chars:
            return [content] if content else []

        units = [
            item.strip()
            for item in re.split(r"(?<=[。！？；])|\n+", content)
            if item.strip()
        ]
        parts: list[str] = []
        current = ""
        for unit in units:
            remaining = unit
            while len(remaining) > max_chars:
                if current:
                    parts.append(current)
                    current = current[-self.chunk_overlap_chars :]
                take = max_chars - len(current)
                current += remaining[:take]
                remaining = remaining[take:]
            candidate = f"{current}\n{remaining}".strip() if current else remaining
            if len(candidate) <= max_chars:
                current = candidate
                continue
            parts.append(current)
            overlap = current[-self.chunk_overlap_chars :]
            current = f"{overlap}\n{remaining}".strip()
        if current:
            parts.append(current)
        return parts

    def _short_table_header(self, label: str) -> str:
        """M1.4 去掉表头括注（``(必填)`` / ``(细化至节，必填)``）并映射为短名。"""
        cleaned = re.sub(r"[（(][^）)]*[）)]", "", self._normalize_text(label))
        cleaned = cleaned.strip(" ：:")
        if not cleaned:
            return ""
        for keywords, short_name in self.TABLE_HEADER_SHORT_NAMES:
            if any(keyword in cleaned for keyword in keywords):
                return short_name
        return cleaned

    @classmethod
    def _chapter_number(cls, title: str) -> int | None:
        """从进度表行标题解析章号（中文数字→int，阿拉伯数字直接用），失败为 None。"""
        match = re.match(
            r"^\s*第\s*([0-9一二三四五六七八九十百零〇两]+)\s*(?:章|单元|模块|部分|讲|篇|节)",
            title,
        )
        if match is None:
            return None
        return cls._chinese_number_to_int(match.group(1))

    @staticmethod
    def _chinese_number_to_int(text: str) -> int | None:
        """把「三」「十二」「二十三」这类中文数字转为整数。"""
        text = text.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        digits = {
            "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
        }
        units = {"十": 10, "百": 100}
        total = 0
        number = 0
        for char in text:
            if char in digits:
                number = digits[char]
            elif char in units:
                total += (number or 1) * units[char]
                number = 0
            else:
                return None
        return total + number

    def _table_sections(self, table: Table) -> list[dict[str, Any]]:
        """Turn common syllabus tables into semantic, row-level sections."""
        rows = [self._unique_row_values(row) for row in table.rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            return []

        combined = " ".join(" ".join(row) for row in rows)
        if "章节次序" in combined and "主要教学内容" in combined:
            header = [self._short_table_header(label) for label in rows[0]]
            sections: list[dict[str, Any]] = []
            for row in rows[1:]:
                if not any(row):
                    continue
                values = row + [""] * max(0, len(header) - len(row))
                chapter = (values[0] or "教学进度").splitlines()[0]
                facts = [
                    f"{label}：{value}"
                    for label, value in zip(header, values)
                    if label and value
                ]
                sections.append(
                    {
                        "section_title": f"教学进度 - {chapter}",
                        "path_title": chapter,
                        "content": "\n".join(facts),
                        "section_type": "teaching_schedule",
                        "chapter_no": self._chapter_number(chapter),
                    }
                )
            return sections

        if (
            ("课程代码" in combined or "课程编码" in combined)
            and ("学分" in combined or "课程类别" in combined)
        ):
            basic_facts: list[str] = []
            objective_parts: list[str] = []
            for row in rows:
                for index in range(0, len(row) - 1, 2):
                    key, value = row[index].strip(" ：:"), row[index + 1]
                    if not key or not value or key == value:
                        continue
                    if "课程目标" in key:
                        objective_parts.append(value)
                    else:
                        basic_facts.append(f"{key}：{value}")
                if row and "课程目标" in row[0] and len(row) > 1:
                    objective_parts.extend(row[1:])
            sections = []
            if basic_facts:
                sections.append(
                    {
                        "section_title": "课程基本信息",
                        "path_title": "课程基本信息",
                        "content": "\n".join(dict.fromkeys(basic_facts)),
                        "section_type": "basic_info",
                        "chapter_no": None,
                    }
                )
            objectives = "\n".join(dict.fromkeys(filter(None, objective_parts)))
            if objectives:
                sections.append(
                    {
                        "section_title": "课程目标",
                        "path_title": "课程目标",
                        "content": objectives,
                        "section_type": "course_objectives",
                        "chapter_no": None,
                    }
                )
            return sections

        markdown = self._table_to_markdown(table)
        return (
            [
                {
                    "section_title": "教学内容表",
                    "path_title": "教学内容表",
                    "content": markdown,
                    "section_type": "teaching_content",
                    "chapter_no": None,
                }
            ]
            if markdown
            else []
        )

    def _iter_block_items(self, document: DocxDocumentType) -> Iterator[Paragraph | Table]:
        """Yield paragraphs and tables in their original document order."""
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                yield Paragraph(child, document)
            elif child.tag.endswith("}tbl"):
                yield Table(child, document)

    def _match_course_metadata(
        self,
        syllabus_path: Path,
        parsed_chunks: list[dict[str, str]],
        course_catalog: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Match a syllabus file against the loaded training plan metadata."""
        filename_code = self._extract_course_code_from_text(syllabus_path.stem)
        if filename_code and filename_code in course_catalog:
            return dict(course_catalog[filename_code])

        filename_name = self._extract_course_name_from_path(syllabus_path)
        if filename_code:
            return {
                "course_code": filename_code,
                "course_name": filename_name,
                "credits": None,
                "prerequisites": [],
            }

        for metadata in course_catalog.values():
            course_name = self._normalize_text(metadata.get("course_name", ""))
            if filename_name and course_name and filename_name == course_name:
                return dict(metadata)

        combined_text = "\n".join(
            [chunk.get("section_title", "") + "\n" + chunk.get("content", "") for chunk in parsed_chunks]
        )
        content_code = self._extract_course_code_from_text(combined_text)
        if content_code and content_code in course_catalog:
            return dict(course_catalog[content_code])

        for metadata in course_catalog.values():
            course_name = self._normalize_text(metadata.get("course_name", ""))
            if course_name and course_name in combined_text:
                return dict(metadata)

        return {
            "program_name": "",
            "course_code": filename_code,
            "course_name": filename_name,
            "credits": None,
            "prerequisites": [],
        }

    def parse_training_plan(self, xlsx_path: str | Path) -> dict[str, dict[str, Any]]:
        """Load structured course metadata from an Excel training plan."""
        xlsx_path = Path(xlsx_path)
        workbook = pd.ExcelFile(xlsx_path)
        course_catalog: dict[str, dict[str, Any]] = {}

        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
            self._merge_catalog(
                course_catalog,
                self._extract_sheet_course_catalog(
                    dataframe,
                    xlsx_path,
                    sheet_name,
                ),
            )

        if course_catalog:
            return course_catalog

        raise ValueError(
            f"Could not find a course table header in training plan: {xlsx_path.name}"
        )

    def parse_training_plans(
        self,
        plan_dir: str | Path,
    ) -> dict[str, dict[str, Any]]:
        """Load and merge every training plan without overwriting memberships."""
        plan_dir = Path(plan_dir)
        catalog: dict[str, dict[str, Any]] = {}
        for xlsx_file in sorted(plan_dir.glob("*.xlsx")):
            self._merge_catalog(catalog, self.parse_training_plan(xlsx_file))
        if not catalog:
            raise FileNotFoundError("未找到可解析的培养方案 Excel 文件（.xlsx）。")
        return catalog

    def build_training_plan_chunks(
        self,
        course_catalog: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create independently retrievable chunks for every plan offering."""
        chunks: list[dict[str, Any]] = []
        for code, course in sorted(course_catalog.items()):
            for offering in course.get("offerings", []):
                category = " / ".join(
                    part
                    for part in (
                        offering.get("course_category", ""),
                        offering.get("course_subcategory", ""),
                    )
                    if part
                )
                facts = [
                    f"培养方案：{offering.get('program_name', '')}",
                    f"方案类型：{offering.get('program_type', '')}",
                    f"课程类别：{category}",
                    f"课程：{course.get('course_name', '')}（{code}）",
                    f"英文名称：{course.get('english_name', '')}",
                    f"学分：{offering.get('credits')}",
                    f"理论学分：{offering.get('theory_credits')}",
                    f"实验实践学分：{offering.get('practice_credits')}",
                    f"总学时：{offering.get('hours')}",
                    f"理论学时：{offering.get('theory_hours')}",
                    f"实验实践学时：{offering.get('practice_hours')}",
                    f"开课学期：{offering.get('semester')}",
                    f"课程负责人：{offering.get('instructor', '')}",
                ]
                text = "\n".join(
                    fact
                    for fact in facts
                    if not fact.endswith(("：", "：None"))
                )
                source_file = (
                    f"training_plans/{offering.get('source_file', '')}"
                    f"#{offering.get('sheet_name', '')}"
                )
                chunk_id = hashlib.sha256(
                    f"training-plan|{source_file}|{code}".encode("utf-8")
                ).hexdigest()
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": text,
                        "source_file": source_file,
                        "section": "培养方案课程信息",
                        "metadata": {
                            "source_type": "training_plan",
                            "course_code": code,
                            "course_name": course.get("course_name", ""),
                            "program_name": offering.get("program_name", ""),
                            "program_type": offering.get("program_type", ""),
                            "course_category": offering.get(
                                "course_category",
                                "",
                            ),
                            "course_subcategory": offering.get(
                                "course_subcategory",
                                "",
                            ),
                            "credits": offering.get("credits"),
                            "hours": offering.get("hours"),
                            "semester": offering.get("semester"),
                            "instructor": offering.get("instructor", ""),
                            "prerequisites": course.get("prerequisites", []),
                        },
                    }
                )
        return chunks

    def parse_syllabus(self, docx_path: str | Path) -> list[dict[str, Any]]:
        """Parse a DOCX syllabus into bounded, semantically typed chunks.

        每个条目除 ``section_title`` / ``content`` / ``section_type`` 外，还带
        ``heading_path``（M1.2 标题栈得到的完整路径）与 ``chapter_no``（进度表行
        章号，非章号行为 ``None``）。表格块不入栈，继承当前栈作为父路径；进度表
        行的 ``chapter`` 作为额外一级追加。
        """
        docx_path = Path(docx_path)
        document = docx.Document(docx_path)

        self._heading_stack = []
        section_chunks: list[dict[str, Any]] = []
        current_title = self.OVERVIEW_SECTION
        current_path: list[str] = [current_title]
        current_lines: list[str] = []

        def flush_current_section() -> None:
            content = self._normalize_text("\n\n".join(current_lines))
            if not content:
                return
            section_type = self._section_type(current_title)
            for part in self._split_content(content):
                section_chunks.append(
                    {
                        "section_title": current_title,
                        "heading_path": list(current_path),
                        "content": part,
                        "section_type": section_type,
                        "chapter_no": None,
                    }
                )

        for block in self._iter_block_items(document):
            if isinstance(block, Paragraph):
                if self._paragraph_is_effectively_empty(block):
                    continue

                paragraph_text = self._normalize_text(block.text)
                if self._is_template_noise(paragraph_text):
                    continue
                level = self._heading_level(block)
                if level is not None:
                    flush_current_section()
                    section_title = re.sub(
                        r"[（(].*?[）)]",
                        "",
                        paragraph_text,
                    ).strip(" ：:")
                    # section_title 决定 section_type / chunk_id 稳定性；
                    # heading_path 用保留编号的展示名（M1.2）。
                    current_title = section_title or paragraph_text
                    current_path = self._push_heading(
                        level,
                        self._heading_title(paragraph_text),
                    )
                    current_lines = []
                else:
                    current_lines.append(paragraph_text)
            else:
                flush_current_section()
                current_lines = []
                parent_path = [title for _, title in self._heading_stack]
                for table_section in self._table_sections(block):
                    path_title = self._normalize_text(
                        table_section.get("path_title")
                        or table_section["section_title"]
                    )
                    heading_path = list(parent_path)
                    if path_title and not self._table_heading_is_redundant(
                        parent_path,
                        path_title,
                    ):
                        heading_path.append(path_title)
                    for part in self._split_content(table_section["content"]):
                        section_chunks.append(
                            {
                                "section_title": table_section["section_title"],
                                "heading_path": heading_path,
                                "content": part,
                                "section_type": table_section["section_type"],
                                "chapter_no": table_section.get("chapter_no"),
                            }
                        )

        flush_current_section()
        return section_chunks

    @classmethod
    def _table_heading_is_redundant(
        cls,
        parent_path: list[str],
        path_title: str,
    ) -> bool:
        """表格自带标题与栈顶章节同名（如「一、课程基本信息」+「课程基本信息」）时跳过。"""
        if not parent_path:
            return False
        return cls._strip_heading_number(parent_path[-1]) == cls._strip_heading_number(
            path_title
        )

    @staticmethod
    def _source_year(source_file: str) -> str:
        match = re.search(r"(20\d{2})", source_file)
        return match.group(1) if match else ""

    def build_syllabus_chunks(
        self,
        docx_path: str | Path,
        syllabus_dir: str | Path,
        course_catalog: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build final index records for one syllabus document."""
        docx_path = Path(docx_path)
        syllabus_dir = Path(syllabus_dir)
        parsed_chunks = self.parse_syllabus(docx_path)
        if not parsed_chunks:
            return []

        course_metadata = self._match_course_metadata(
            docx_path,
            parsed_chunks,
            course_catalog,
        )
        source_file = str(docx_path.relative_to(syllabus_dir).as_posix())
        source_year = self._source_year(source_file)
        catalog_prerequisites = course_metadata.get("prerequisites") or []
        if not isinstance(catalog_prerequisites, list):
            catalog_prerequisites = [catalog_prerequisites]
        raw_prerequisites = [
            *(
                name
                for value in catalog_prerequisites
                for name in self._split_prerequisites(value)
            ),
            *self._explicit_prerequisite_names(parsed_chunks),
        ]
        raw_prerequisites = list(dict.fromkeys(raw_prerequisites))
        prerequisite_evidence, unresolved_prerequisites = (
            self.resolve_prerequisites(
                raw_prerequisites,
                course_catalog,
                dependent_course_code=self._normalize_text(
                    course_metadata.get("course_code", "")
                ),
                source_file=source_file,
                section="课程基本信息",
                source_year=source_year,
            )
        )
        prerequisite_codes = [
            item["course_code"] for item in prerequisite_evidence
        ]
        document_hash = hashlib.sha256(docx_path.read_bytes()).hexdigest()
        # 延迟导入：doc_tree 会消费本模块的产物，模块级导入会形成循环依赖。
        from src.data_processing.doc_tree import node_id

        course_code = self._normalize_text(course_metadata.get("course_code", ""))
        course_name = self._normalize_text(course_metadata.get("course_name", ""))
        final_chunks: list[dict[str, Any]] = []
        title_counts: dict[str, int] = {}
        for chunk in parsed_chunks:
            section = self._normalize_text(chunk.get("section_title", ""))
            content = self._normalize_text(chunk.get("content", ""))
            if not content:
                continue
            heading_path = [
                normalized
                for normalized in (
                    self._normalize_text(part)
                    for part in chunk.get("heading_path") or []
                )
                if normalized
            ]
            if not heading_path:
                heading_path = [section or self.OVERVIEW_SECTION]
            section_type = chunk.get("section_type") or self._section_type(section)
            chapter_no = chunk.get("chapter_no")
            if not isinstance(chapter_no, int):
                chapter_no = None
            breadcrumb = self._build_breadcrumb(course_name, heading_path)
            parent_id: str | None = None
            if len(heading_path) >= 2:
                parent_id = node_id(document_hash, heading_path[:-1])
            prefix = " / ".join(heading_path[-2:])
            # 标题前缀也占长度预算：必要时按更小的上限重新切分正文。
            budget = max(
                self.chunk_overlap_chars + 1,
                self.max_text_chars - len(prefix) - 2,
            )
            if len(prefix) + len(content) + 2 > self.max_text_chars:
                parts = self._split_content(content, budget)
            else:
                parts = [content]
            for part in parts:
                title_counts[section] = title_counts.get(section, 0) + 1
                part_number = title_counts[section]
                text = self._normalize_text(f"{prefix}\n\n{part}")
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                chunk_index = len(final_chunks)
                final_chunks.append(
                    {
                        "chunk_id": hashlib.sha256(
                            f"{source_file}|{section}|{part_number}|{text_hash}".encode(
                                "utf-8"
                            )
                        ).hexdigest(),
                        "text": text,
                        "source_file": source_file,
                        "document_hash": document_hash,
                        "section": section,
                        "chunk_index": chunk_index,
                        "heading_path": list(heading_path),
                        "breadcrumb": breadcrumb,
                        "parent_id": parent_id,
                        "chapter_no": chapter_no,
                        "is_header": section_type == "basic_info"
                        or section == self.OVERVIEW_SECTION,
                        "prev_chunk_id": None,
                        "next_chunk_id": None,
                        "metadata": {
                            "source_type": "syllabus",
                            "course_code": course_code,
                            "course_name": course_name,
                            "syllabus_section": section,
                            "section_type": section_type,
                            "parent_document": source_file,
                            "parent_section": section,
                            "chunk_part": part_number,
                            "chunk_index": chunk_index,
                            "heading_path": list(heading_path),
                            "breadcrumb": breadcrumb,
                            "parent_id": parent_id,
                            "chapter_no": chapter_no,
                            "is_header": section_type == "basic_info"
                            or section == self.OVERVIEW_SECTION,
                            "prev_chunk_id": None,
                            "next_chunk_id": None,
                            "source_year": source_year,
                            "prerequisites": prerequisite_codes,
                            "prerequisite_evidence": prerequisite_evidence,
                            "unresolved_prerequisites": unresolved_prerequisites,
                            "credits": course_metadata.get("credits"),
                        },
                    }
                )
        for index, final_chunk in enumerate(final_chunks):
            previous_id = final_chunks[index - 1]["chunk_id"] if index > 0 else None
            next_id = (
                final_chunks[index + 1]["chunk_id"]
                if index + 1 < len(final_chunks)
                else None
            )
            final_chunk["prev_chunk_id"] = previous_id
            final_chunk["next_chunk_id"] = next_id
            final_chunk["metadata"]["prev_chunk_id"] = previous_id
            final_chunk["metadata"]["next_chunk_id"] = next_id
        return final_chunks

    def _build_breadcrumb(self, course_name: str, heading_path: list[str]) -> str:
        """M1.3 面包屑：``【课程名 | 倒数第二级 | 最后一级】``（课程名为空则省略）。"""
        parts = list(heading_path)
        if len(parts) > 2:
            parts = parts[-2:]
        if course_name:
            parts = [course_name, *parts]
        return "【" + " | ".join(parts) + "】"

    def run(
        self,
        plan_dir: str | Path,
        syllabus_dir: str | Path,
        output_path: str | Path,
    ) -> list[dict[str, Any]]:
        """Load all plans, process all syllabi recursively, and save merged chunks."""
        plan_dir = Path(plan_dir)
        syllabus_dir = Path(syllabus_dir)
        output_path = Path(output_path)

        if not plan_dir.exists():
            raise FileNotFoundError(f"培养方案目录不存在: {plan_dir}")
        if not syllabus_dir.exists():
            raise FileNotFoundError(f"课程大纲目录不存在: {syllabus_dir}")

        all_course_metadata = self.parse_training_plans(plan_dir)
        final_chunks = self.build_training_plan_chunks(all_course_metadata)
        docx_files = sorted(
            path for path in syllabus_dir.rglob("*.docx") if not path.name.startswith("~$")
        )

        for docx_file in docx_files:
            final_chunks.extend(
                self.build_syllabus_chunks(
                    docx_file,
                    syllabus_dir,
                    all_course_metadata,
                )
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(final_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return final_chunks


# --------------------------------------------------------------------------- #
# M3.1–M3.2 大纲结构化字段（纯规则抽取，不调用 LLM、不读环境变量、不联网）
# --------------------------------------------------------------------------- #

#: ``parse_basic_info`` 抽取的规范字段（顺序即返回字典的键顺序）。
_BASIC_INFO_FIELDS: tuple[str, ...] = (
    "course_owner",
    "lecturers",
    "credits",
    "hours",
    "theory_hours",
    "practice_hours",
    "grade",
    "audience",
    "department",
    "intended_semester",
    "prerequisites_raw",
)

#: M3.1 键名归一表：规范字段 → 原文键名别名（首个为方案文档给出的主键名）。
_BASIC_INFO_KEY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("course_owner", ("课程负责人",)),
    ("lecturers", ("主讲教师", "任课教师", "授课教师")),
    ("credits", ("学分",)),
    ("hours", ("学时", "总学时")),
    ("theory_hours", ("理论学时",)),
    ("practice_hours", ("实践（含实验）学时", "实验实践学时", "实践学时")),
    ("grade", ("授课年级",)),
    ("audience", ("面向专业/大类", "面向专业", "适用专业")),
    ("department", ("开课单位",)),
    ("intended_semester", ("意向开课学期",)),
    ("prerequisites_raw", ("先修课程", "前置课程", "预修课程")),
)

#: 缺失值占位符：统一归一为 ``None``。
_MISSING_BASIC_INFO_VALUES = frozenset(
    {"", "/", "／", "\\", "无", "-", "—", "－", "暂无"}
)

_LECTURER_SEPARATOR_PATTERN = re.compile(r"[,，、;；/／|]+")

#: assessment 块里的两个纯文本字段。
_ASSESSMENT_MODE_PATTERN = re.compile(r"考\s*核\s*方\s*式\s*[:：]\s*([^\n]+)")
_GRADING_MODE_PATTERN = re.compile(r"成\s*绩\s*录\s*入\s*方\s*式\s*[:：]\s*([^\n]+)")


def _normalize_basic_info_key(value: Any) -> str:
    """归一化键名：去掉空白、括注符号与分隔符后转小写。"""
    return re.sub(r"[\s_（）()\-—－:/：、,，.。\[\]【】'\"]+", "", str(value)).lower()


def _split_basic_info_line(line: str) -> tuple[str, str]:
    """按第一个中英文冒号拆出 ``键：值``，不是键值行时返回空键。"""
    match = re.match(r"^([^:：]{1,40})[:：]\s*(.*)$", line)
    if match is None:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def _match_basic_info_field(
    key: str,
    alias_table: list[tuple[str, str]],
) -> str | None:
    """先精确后包含地匹配规范字段名，包含匹配优先取更长的别名。"""
    normalized_key = _normalize_basic_info_key(key)
    if not normalized_key:
        return None
    for field_name, alias in alias_table:
        if normalized_key == alias:
            return field_name
    for field_name, alias in sorted(
        alias_table,
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        if alias and alias in normalized_key:
            return field_name
    return None


def _clean_basic_info_value(field_name: str, value: str) -> Any:
    """清洗字段值：缺失占位符归一为 ``None``，教师字段拆成列表。"""
    cleaned = str(value).strip().strip("　")
    if cleaned in _MISSING_BASIC_INFO_VALUES:
        return None
    if field_name == "lecturers":
        names = [
            name.strip()
            for name in _LECTURER_SEPARATOR_PATTERN.split(cleaned)
            if name.strip() and name.strip() not in _MISSING_BASIC_INFO_VALUES
        ]
        return names or None
    return cleaned


def parse_basic_info(text: str) -> dict[str, Any]:
    """从大纲「课程基本信息」文本中按规则抽取结构化字段（M3.1）。

    输入是 ``键：值`` 行（也容忍标题前缀行），输出的键固定为
    ``_BASIC_INFO_FIELDS``；同一个字段出现多次时以第一条非缺失值为准。
    缺失值（``None`` / 空串 / ``/`` / ``无`` / ``-`` / ``—`` / ``暂无``）
    统一为 ``None``；``lecturers`` 为 ``list[str]``。纯函数，不联网。
    """
    profile: dict[str, Any] = {field_name: None for field_name in _BASIC_INFO_FIELDS}
    alias_table = [
        (field_name, _normalize_basic_info_key(alias))
        for field_name, aliases in _BASIC_INFO_KEY_ALIASES
        for alias in aliases
    ]
    for raw_line in re.split(r"[\r\n]+", str(text or "")):
        line = raw_line.strip()
        if not line:
            continue
        key, value = _split_basic_info_line(line)
        if not key:
            continue
        field_name = _match_basic_info_field(key, alias_table)
        if field_name is None or profile[field_name] is not None:
            continue
        parsed_value = _clean_basic_info_value(field_name, value)
        if parsed_value is not None:
            profile[field_name] = parsed_value
    return profile


def collect_syllabus_profile(
    parsed_chunks: list[dict[str, Any]],
    *,
    source_file: str,
    course_code: str,
    course_name: str,
    source_year: str,
) -> dict[str, Any]:
    """汇总一份大纲的结构化字段（M3.2）：基本信息 + 考核/成绩录入方式。

    只读取 ``metadata.section_type`` 为 ``basic_info`` / ``assessment`` 的块，
    通过 ``parse_basic_info`` 做规则抽取；``field_chunk_ids`` 记录每个字段落在
    哪个 chunk 上，便于回答时出引用。纯函数，不读环境变量、不联网。
    """
    profile: dict[str, Any] = {
        **{field_name: None for field_name in _BASIC_INFO_FIELDS},
        "assessment_mode": None,
        "grading_mode": None,
    }
    field_chunk_ids: dict[str, str] = {}

    for chunk in parsed_chunks:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata") or {}
        if metadata.get("section_type") != "basic_info":
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or chunk.get("content") or "")
        for field_name, value in parse_basic_info(chunk_text).items():
            if value is None or profile[field_name] is not None:
                continue
            profile[field_name] = value
            if chunk_id:
                field_chunk_ids[field_name] = chunk_id

    assessment_patterns = (
        ("assessment_mode", _ASSESSMENT_MODE_PATTERN),
        ("grading_mode", _GRADING_MODE_PATTERN),
    )
    for chunk in parsed_chunks:
        if not isinstance(chunk, dict):
            continue
        metadata = chunk.get("metadata") or {}
        if metadata.get("section_type") != "assessment":
            continue
        chunk_id = str(chunk.get("chunk_id") or "")
        chunk_text = str(chunk.get("text") or chunk.get("content") or "")
        for field_name, pattern in assessment_patterns:
            if profile[field_name] is not None:
                continue
            match = pattern.search(chunk_text)
            if match is None:
                continue
            value = _clean_basic_info_value(field_name, match.group(1))
            if value is None:
                continue
            profile[field_name] = value
            if chunk_id:
                field_chunk_ids[field_name] = chunk_id

    return {
        **profile,
        "source_file": source_file,
        "course_code": course_code,
        "course_name": course_name,
        "source_year": source_year,
        "field_chunk_ids": field_chunk_ids,
    }


def main() -> None:
    """Provide a small CLI for parsing syllabi and training plans."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse Excel training plans and DOCX syllabi into chunks."
    )
    parser.add_argument(
        "--plan-dir",
        default="data/training_plans",
        help="Directory containing one or more training-plan .xlsx files.",
    )
    parser.add_argument(
        "--syllabus-dir",
        default="data/syllabi",
        help="Directory containing syllabus .docx files, scanned recursively.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/chunks.json",
        help="Destination JSON file for the processed chunks.",
    )
    args = parser.parse_args()

    chunker = SyllabusChunker()
    results = chunker.run(
        plan_dir=args.plan_dir,
        syllabus_dir=args.syllabus_dir,
        output_path=args.output_path,
    )
    print(f"Generated {len(results)} chunks.")


if __name__ == "__main__":
    main()
