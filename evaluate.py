"""
evaluate.py

A minimal but legitimate evaluation framework:
1. Prepare a set of "question + reference answer" test cases
2. Have the system generate an answer for each question
3. Use another LLM as a judge to compare the generated answer against
   the reference answer and assign a score
4. Aggregate into an accuracy summary — write it into your README or
   CI report

Most entry-level projects skip this step entirely, so doing it well
is a clear differentiator. It proves you're not just thinking about
"does it run" but also "how do I know if it's actually any good."
"""

import json
from dataclasses import dataclass
from anthropic import Anthropic

client = Anthropic()

# ---- Test set: replace with questions your project actually covers ----
TEST_CASES = [
    {
        "question": "What chapters does the COMP 472 final exam cover?",
        "reference_answer": "Search algorithms, machine learning (decision trees, Naive Bayes, evaluation metrics, perceptrons, word embeddings, K-means), NLP, and uncertainty/utility theory",
    },
    {
        "question": "What is the late submission policy for this course?",
        "reference_answer": "Per the syllabus, a 10% deduction per day late, with submissions not accepted after 3 days",
    },
    # Add more of your own test cases here...
]


@dataclass
class EvalResult:
    question: str
    generated_answer: str
    reference_answer: str
    score: int  # 1-5
    reasoning: str


JUDGE_PROMPT = """You are an evaluation assistant. Compare the two answers below for semantic consistency and give a score from 1-5:
- 5: Fully consistent, covers all key information
- 3: Mostly consistent, but misses some details
- 1: Clearly inconsistent or off-topic

Question: {question}
Reference answer: {reference}
Generated answer: {generated}

Return your response as JSON: {{"score": <int>, "reasoning": "<brief explanation>"}}
Return only the JSON, no other text."""


def judge_answer(question: str, generated: str, reference: str) -> tuple[int, str]:
    response = client.messages.create(
        model="claude-sonnet-4-6",
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
    text = "".join(block.text for block in response.content if block.type == "text")
    parsed = json.loads(text)
    return parsed["score"], parsed["reasoning"]


def run_evaluation(rag_system) -> list[EvalResult]:
    """
    rag_system: your existing RAG/Agent system — needs an
    answer(question) -> str method
    """
    results = []
    for case in TEST_CASES:
        generated = rag_system.answer(case["question"])
        score, reasoning = judge_answer(
            case["question"], generated, case["reference_answer"]
        )
        results.append(
            EvalResult(
                question=case["question"],
                generated_answer=generated,
                reference_answer=case["reference_answer"],
                score=score,
                reasoning=reasoning,
            )
        )
    return results


def print_report(results: list[EvalResult]) -> None:
    avg_score = sum(r.score for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"Evaluation Report — Average Score: {avg_score:.2f} / 5.0")
    print(f"{'='*50}\n")
    for r in results:
        print(f"Question: {r.question}")
        print(f"Score: {r.score}/5 — {r.reasoning}")
        print("-" * 50)


if __name__ == "__main__":
    # Example usage:
    # from main import rag_system
    # results = run_evaluation(rag_system)
    # print_report(results)
    print("Wire this script up to your RAG system and run it — it will print an evaluation report")
    print("It's worth pasting the results into your README as a quantitative proof of project quality")
