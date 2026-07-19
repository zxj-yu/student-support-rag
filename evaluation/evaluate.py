"""Evaluation harness for the RAG pipeline.

Runs the cases in evaluation/cases.json against a live backend over HTTP and
reports two layers of quality metrics:

1. Retrieval hit rate (always available, no API key needed): did the expected
   source document appear in the top-k results returned by /chat? This checks
   the semantic search layer in isolation.
2. Answer quality via LLM-as-judge (requires ANTHROPIC_API_KEY): a judge model
   scores each generated answer against a reference answer on a 1-5 scale.

Usage:
    # start the stack first: docker-compose up  +  POST /ingest
    python -m evaluation.evaluate                       # local backend
    python -m evaluation.evaluate --base-url https://...run.app
"""
import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

CASES_PATH = Path(__file__).parent / "cases.json"

JUDGE_PROMPT = """You are an evaluation assistant. Compare the two answers \
below for semantic consistency and give a score from 1-5:
- 5: fully consistent, covers all key information
- 3: mostly consistent, but misses some details
- 1: clearly inconsistent or off-topic

Question: {question}
Reference answer: {reference}
Generated answer: {generated}

Respond with JSON only: {{"score": <int>, "reasoning": "<brief explanation>"}}"""


@dataclass
class CaseResult:
    question: str
    expected_source: str
    retrieved_sources: list = field(default_factory=list)
    hit: bool = False
    answer: str = ""
    score: int | None = None
    reasoning: str = ""


def retrieval_hit(sources: list[dict], expected_title: str) -> bool:
    """True if the expected document title appears in the returned sources."""
    return any(s.get("title") == expected_title for s in sources)


def parse_judge_response(text: str) -> tuple[int, str]:
    """Parse the judge's JSON, tolerating markdown code fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    parsed = json.loads(cleaned.strip())
    return int(parsed["score"]), str(parsed.get("reasoning", ""))


def judge_answer(question: str, generated: str, reference: str, api_key: str):
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6"),
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question, reference=reference, generated=generated
                ),
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return parse_judge_response(text)


def run(base_url: str, api_key: str = "") -> list[CaseResult]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results: list[CaseResult] = []

    with httpx.Client(base_url=base_url, timeout=60) as client:
        for case in cases:
            res = client.post("/chat", json={"question": case["question"]})
            res.raise_for_status()
            body = res.json()
            sources = body.get("sources", [])
            result = CaseResult(
                question=case["question"],
                expected_source=case["expected_source"],
                retrieved_sources=[s.get("title") for s in sources],
                hit=retrieval_hit(sources, case["expected_source"]),
                answer=body.get("answer", ""),
            )
            if api_key:
                result.score, result.reasoning = judge_answer(
                    case["question"], result.answer, case["reference_answer"], api_key
                )
            results.append(result)

    return results


def print_report(results: list[CaseResult]) -> None:
    hits = sum(r.hit for r in results)
    print(f"\n{'=' * 60}")
    print(f"Retrieval hit rate: {hits}/{len(results)} "
          f"({hits / len(results):.0%}) — expected source in top-k")
    scored = [r for r in results if r.score is not None]
    if scored:
        avg = sum(r.score for r in scored) / len(scored)
        print(f"Answer quality (LLM judge): {avg:.2f} / 5.0 across {len(scored)} cases")
    else:
        print("Answer quality: skipped (set ANTHROPIC_API_KEY to enable the judge)")
    print(f"{'=' * 60}\n")
    for r in results:
        mark = "PASS" if r.hit else "MISS"
        print(f"[{mark}] {r.question}")
        print(f"       expected: {r.expected_source} | got: {r.retrieved_sources}")
        if r.score is not None:
            print(f"       judge: {r.score}/5 — {r.reasoning}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    results = run(args.base_url, api_key)
    print_report(results)


if __name__ == "__main__":
    main()
