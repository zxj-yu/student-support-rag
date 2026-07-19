"""Unit tests for the evaluation harness (no backend or API key required)."""
import json
from pathlib import Path

import pytest

from evaluation.evaluate import parse_judge_response, retrieval_hit

CASES_PATH = Path(__file__).parent.parent / "evaluation" / "cases.json"


def test_retrieval_hit_true_when_expected_title_present():
    sources = [{"title": "GPA Calculation", "score": 0.8}]
    assert retrieval_hit(sources, "GPA Calculation")


def test_retrieval_hit_false_when_absent():
    sources = [{"title": "Tuition Refund Schedule", "score": 0.5}]
    assert not retrieval_hit(sources, "GPA Calculation")


def test_parse_judge_response_plain_json():
    score, reasoning = parse_judge_response(
        '{"score": 4, "reasoning": "mostly consistent"}'
    )
    assert score == 4
    assert reasoning == "mostly consistent"


def test_parse_judge_response_strips_code_fences():
    text = '```json\n{"score": 5, "reasoning": "exact match"}\n```'
    score, reasoning = parse_judge_response(text)
    assert score == 5


def test_parse_judge_response_rejects_garbage():
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parse_judge_response("I think it deserves a 4 out of 5")


def test_cases_file_matches_knowledge_base():
    """Every expected_source must be a real document title in the KB, so the
    eval can never silently pass/fail against nonexistent documents."""
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
    kb_titles = {d["title"] for d in json.loads(kb_path.read_text(encoding="utf-8"))}

    assert cases, "cases.json must not be empty"
    for case in cases:
        assert case["expected_source"] in kb_titles, (
            f"expected_source {case['expected_source']!r} not found in KB"
        )
        assert case["question"].strip()
        assert case["reference_answer"].strip()
