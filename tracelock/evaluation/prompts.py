from __future__ import annotations

import json
from typing import Any


ACC_SYSTEM_PROMPT = (
    "You are a strict grading assistant. "
    "Return exactly one JSON object and nothing else."
)

RANK_SYSTEM_PROMPT = (
    "You are a strict ranking assistant. "
    "Return exactly one JSON object and nothing else."
)

SCORE_SYSTEM_PROMPT = (
    "You are a strict scoring assistant. "
    "Return exactly one JSON object and nothing else."
)


def _json_schema_text(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2)


def build_acc_messages(
    *,
    question: str,
    reference_answer: str,
    reference_final_answer: str | None,
    candidate_answer: str,
) -> list[dict[str, str]]:
    schema = {
        "verdict": "correct or incorrect",
        "confidence": "high or medium or low",
        "reason": "short explanation",
    }
    user_prompt = (
        "Judge whether the candidate answer should be marked correct for the given question.\n\n"
        "Rules:\n"
        "1. Use the reference answer as the grading standard.\n"
        "2. Focus on whether the candidate's final answer matches the reference.\n"
        "3. Do not reward style, verbosity, or partial progress if the final answer is wrong.\n"
        "4. If the candidate answer is ambiguous, malformed, or missing a final answer, mark incorrect.\n"
        "5. Output valid JSON only, matching this schema:\n"
        f"{_json_schema_text(schema)}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Reference final answer:\n{reference_final_answer or ''}\n\n"
        f"Candidate answer:\n{candidate_answer}\n"
    )
    return [
        {"role": "system", "content": ACC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_qa_lenient_critic_messages(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> list[dict[str, str]]:
    schema = {
        "verdict": "correct or incorrect or skip",
        "confidence": "high or medium or low",
        "reason": "short explanation",
    }
    user_prompt = (
        "Judge an open-ended QA or instruction-following answer against a reference answer.\n\n"
        "This is a lenient critic. Many prompts are open-ended, so do not require exact wording, exact coverage, "
        "or the same level of detail as the reference answer.\n\n"
        "Rules:\n"
        "1. Mark `correct` if the candidate is broadly consistent with the reference and gives a reasonable answer to the question.\n"
        "2. Mark `incorrect` only when the candidate is clearly and obviously wrong, contradicts the reference on a central fact, "
        "fails to answer the question, is nonsensical, or gives harmful/irrelevant content.\n"
        "3. Mark `skip` when correctness is ambiguous, subjective, underspecified, or when both answers are plausible but different.\n"
        "4. For partial but not clearly wrong answers, prefer `skip` over `incorrect`.\n"
        "5. For stylistic differences, shorter answers, or missing minor details, prefer `correct` or `skip`; do not mark incorrect.\n"
        "6. Output valid JSON only, matching this schema:\n"
        f"{_json_schema_text(schema)}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{candidate_answer}\n"
    )
    return [
        {"role": "system", "content": ACC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_acc_extract_final_answer_messages(
    *,
    question: str,
    candidate_answer: str,
) -> list[dict[str, str]]:
    schema = {
        "status": "found or not_found",
        "final_answer": "the candidate's final numeric result when found, else empty string",
        "reason": "short explanation",
    }
    user_prompt = (
        "Extract only the candidate's final numeric answer for the given question.\n\n"
        "Rules:\n"
        "1. Ignore the candidate's reasoning steps. Extract only the final result.\n"
        "2. If the candidate clearly gives a final numeric answer, copy that final numeric answer into `final_answer`.\n"
        "3. If the candidate does not provide a clear final numeric answer, set `status` to `not_found` and set `final_answer` to an empty string.\n"
        "4. Do not solve the problem yourself.\n"
        "5. Output valid JSON only, matching this schema:\n"
        f"{_json_schema_text(schema)}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{candidate_answer}\n"
    )
    return [
        {"role": "system", "content": ACC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]


def build_acc_final_answer_compare_messages(
    *,
    question: str,
    reference_answer: str,
    candidate_final_answer: str,
) -> list[dict[str, str]]:
    schema = {
        "verdict": "correct or incorrect",
        "confidence": "high or medium or low",
        "reason": "short explanation",
    }
    user_prompt = (
        "Judge whether the candidate final answer should be marked correct for the given question.\n\n"
        "Rules:\n"
        "1. Use the reference answer as the grading standard.\n"
        "2. Focus only on whether the candidate final answer matches the reference answer's final result.\n"
        "3. Ignore any missing or incorrect intermediate reasoning from the candidate. Only the final result matters here.\n"
        "4. If the candidate final answer is ambiguous, malformed, or not a clear numeric result, mark incorrect.\n"
        "5. Output valid JSON only, matching this schema:\n"
        f"{_json_schema_text(schema)}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate final answer:\n{candidate_final_answer}\n"
    )
    return [
        {"role": "system", "content": ACC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_acc_reasoning_review_messages(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> list[dict[str, str]]:
    schema = {
        "verdict": "correct or incorrect",
        "confidence": "high or medium or low",
        "reason": "short explanation",
    }
    user_prompt = (
        "Review whether the candidate answer should be considered mathematically acceptable for the given question.\n\n"
        "Rules:\n"
        "1. Use the reference answer as the grading standard.\n"
        "2. This stage is about the candidate's overall reasoning quality, but be lenient about wording and style.\n"
        "3. Mark incorrect only when the candidate contains clearly wrong reasoning, a clear arithmetic or logic mistake, or a final result that does not match the reference.\n"
        "4. Do not require the candidate's reasoning to match the reference step by step.\n"
        "5. Minor omissions, different valid solution paths, or concise reasoning are acceptable if the mathematics is still sound.\n"
        "6. Output valid JSON only, matching this schema:\n"
        f"{_json_schema_text(schema)}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{candidate_answer}\n"
    )
    return [
        {"role": "system", "content": ACC_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_rank_messages(
    *,
    question: str,
    candidate_map: dict[str, str],
) -> list[dict[str, str]]:
    schema = {
        "ranking": ["candidate_1", "candidate_2"],
        "ties": [["candidate_1", "candidate_2"]],
        "best_candidate": "candidate_1",
        "reason": "short explanation",
    }
    candidate_names = list(candidate_map.keys())
    candidates_text = "\n\n".join(f"{name}:\n{text}" for name, text in candidate_map.items())
    candidate_name_text = ", ".join(candidate_names)
    user_prompt = (
        "Rank the candidate answers for the same question from best to worst.\n\n"
        "Rules:\n"
        "1. Judge correctness first, then completeness, relevance, and clarity.\n"
        "2. Do not use the candidate ids as a signal.\n"
        "3. Prefer the answer that best solves the user's request.\n"
        "4. You must output a COMPLETE ranking of ALL candidates, not just the top few.\n"
        "5. `ranking` must list every candidate exactly once, with no missing candidates, no duplicates, and no extra names.\n"
        "6. `ties` is optional extra information and does NOT replace `ranking`. Even if some candidates are tied, `ranking` must still contain all candidates.\n"
        "7. `best_candidate` must be one of the candidates in the top tie group.\n"
        "8. Before producing the final JSON, verify that the set of names in `ranking` is exactly equal to the full candidate list given below.\n"
        "9. If you are uncertain about lower-ranked candidates, you must still place them in `ranking`.\n"
        "10. Output valid JSON only, matching this schema:\n"
        f"{_json_schema_text(schema)}\n\n"
        f"Total candidates: {len(candidate_names)}\n"
        f"Candidate ids that MUST all appear exactly once in `ranking`:\n{candidate_name_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidates:\n{candidates_text}\n"
    )
    return [
        {"role": "system", "content": RANK_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_score_messages(
    *,
    question: str,
    candidate_map: dict[str, str],
) -> list[dict[str, str]]:
    schema = {
        "scores": {
            "candidate_1": 8.4,
            "candidate_2": 6.1,
        },
        "reason": "short explanation",
    }
    candidate_names = list(candidate_map.keys())
    candidates_text = "\n\n".join(f"{name}:\n{text}" for name, text in candidate_map.items())
    candidate_name_text = ", ".join(candidate_names)
    user_prompt = (
        "Score the candidate answers for the same question.\n\n"
        "Rules:\n"
        "1. Score candidates only relative to each other within this batch, not against a global scale across questions.\n"
        "2. Use a 0 to 10 scale, where higher is better.\n"
        "3. Judge correctness first, then completeness, relevance, and clarity.\n"
        "4. Do not reward candidate ids, style, verbosity, or politeness.\n"
        "5. `scores` must include every candidate exactly once, with no missing candidates and no extra names.\n"
        "6. Equal scores are allowed when candidates are truly tied.\n"
        "7. Output valid JSON only, matching this schema:\n"
        f"{_json_schema_text(schema)}\n\n"
        f"Total candidates: {len(candidate_names)}\n"
        f"Candidate ids that MUST all appear exactly once in `scores`:\n{candidate_name_text}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidates:\n{candidates_text}\n"
    )
    return [
        {"role": "system", "content": SCORE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
