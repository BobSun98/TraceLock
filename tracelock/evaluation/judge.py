from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


TASK_TYPE_ACC = "acc"
TASK_TYPE_RANK = "rank"
TASK_TYPE_SCORE = "score"
SUPPORTED_TASK_TYPES = (TASK_TYPE_ACC, TASK_TYPE_RANK, TASK_TYPE_SCORE)

ACC_VERDICTS = {"correct", "incorrect"}
QA_CRITIC_VERDICTS = {"correct", "incorrect", "skip"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
ACC_EXTRACT_STATUSES = {"found", "not_found"}


@dataclass(frozen=True)
class JudgeTask:
    task_type: str
    question_index: int
    sample_id: str
    prompt_text: str
    set_name: str | None = None
    reference_answer: str | None = None
    reference_final_answer: str | None = None
    candidate_answer: str | None = None
    candidate_final_answer: str | None = None
    acc_stage: str | None = None
    candidates: dict[str, str] | None = None
    candidate_to_set: dict[str, str] | None = None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "JudgeTask":
        return cls(**payload)


def extract_gsm8k_final_answer(text: str) -> str | None:
    normalized = str(text).strip()
    if not normalized:
        return None
    match = re.search(r"####\s*([^\n]+)", normalized)
    if match:
        return _normalize_final_answer(match.group(1))
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return None
    return _normalize_final_answer(lines[-1])


def _normalize_final_answer(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    return text


def extract_json_object(text: str) -> dict[str, Any]:
    payload = str(text).strip()
    if not payload:
        raise ValueError("Empty model output.")
    try:
        loaded = json.loads(payload)
        if not isinstance(loaded, dict):
            raise ValueError("JSON output must be an object.")
        return loaded
    except json.JSONDecodeError:
        pass

    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Could not find JSON object in model output.")
    loaded = json.loads(payload[start : end + 1])
    if not isinstance(loaded, dict):
        raise ValueError("JSON output must be an object.")
    return loaded


def validate_acc_output(output: dict[str, Any]) -> dict[str, Any]:
    verdict = str(output.get("verdict", "")).strip().lower()
    if verdict not in ACC_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    confidence = str(output.get("confidence", "medium")).strip().lower()
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"Invalid confidence: {confidence!r}")
    reason = str(output.get("reason", "")).strip()
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
    }


def validate_qa_critic_output(output: dict[str, Any]) -> dict[str, Any]:
    verdict = str(output.get("verdict", "")).strip().lower()
    if verdict not in QA_CRITIC_VERDICTS:
        raise ValueError(f"Invalid QA critic verdict: {verdict!r}")
    confidence = str(output.get("confidence", "medium")).strip().lower()
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"Invalid confidence: {confidence!r}")
    reason = str(output.get("reason", "")).strip()
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reason": reason,
    }


def validate_acc_extract_output(output: dict[str, Any]) -> dict[str, Any]:
    status = str(output.get("status", "")).strip().lower()
    if status not in ACC_EXTRACT_STATUSES:
        raise ValueError(f"Invalid extraction status: {status!r}")
    final_answer = str(output.get("final_answer", "")).strip()
    if status == "found":
        final_answer = _normalize_final_answer(final_answer)
        if not final_answer:
            raise ValueError("`final_answer` must be non-empty when status=found.")
    else:
        final_answer = ""
    reason = str(output.get("reason", "")).strip()
    return {
        "status": status,
        "final_answer": final_answer,
        "reason": reason,
    }


def validate_rank_output(output: dict[str, Any], allowed_candidates: set[str]) -> dict[str, Any]:
    ranking = output.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError("`ranking` must be a list.")
    normalized_ranking = [str(item).strip() for item in ranking]
    if len(normalized_ranking) != len(allowed_candidates):
        deduped_ranking: list[str] = []
        seen: set[str] = set()
        for item in normalized_ranking:
            if item in allowed_candidates and item not in seen:
                deduped_ranking.append(item)
                seen.add(item)
        normalized_ranking = deduped_ranking
    else:
        if len(set(normalized_ranking)) != len(normalized_ranking):
            raise ValueError("`ranking` contains duplicate candidates.")

    unknown_candidates = [item for item in normalized_ranking if item not in allowed_candidates]
    if unknown_candidates:
        normalized_ranking = [item for item in normalized_ranking if item in allowed_candidates]

    deduped_ranking = []
    seen_candidates: set[str] = set()
    for item in normalized_ranking:
        if item in seen_candidates:
            continue
        deduped_ranking.append(item)
        seen_candidates.add(item)
    normalized_ranking = deduped_ranking

    missing_candidates = [item for item in sorted(allowed_candidates) if item not in seen_candidates]
    auto_appended_tie_group: list[str] = []
    if missing_candidates:
        normalized_ranking.extend(missing_candidates)
        auto_appended_tie_group = missing_candidates

    ties_raw = output.get("ties", [])
    if ties_raw is None:
        ties_raw = []
    if not isinstance(ties_raw, list):
        raise ValueError("`ties` must be a list.")

    normalized_ties: list[list[str]] = []
    seen_tie_members: set[str] = set()
    for group in ties_raw:
        if not isinstance(group, list):
            raise ValueError("Each tie group must be a list.")
        normalized_group = [str(item).strip() for item in group]
        if len(normalized_group) < 2:
            continue
        if not set(normalized_group).issubset(allowed_candidates):
            raise ValueError("Tie group contains unknown candidate.")
        if len(set(normalized_group)) != len(normalized_group):
            raise ValueError("Tie group contains duplicates.")
        overlap = seen_tie_members.intersection(normalized_group)
        if overlap:
            raise ValueError(f"Candidate appears in multiple tie groups: {sorted(overlap)}")
        seen_tie_members.update(normalized_group)
        normalized_ties.append(normalized_group)

    if auto_appended_tie_group:
        auto_appended_tie_group = [item for item in auto_appended_tie_group if item not in seen_tie_members]
    if auto_appended_tie_group:
        normalized_ties.append(auto_appended_tie_group)

    best_candidate = str(output.get("best_candidate", "")).strip()
    if best_candidate not in allowed_candidates:
        raise ValueError("`best_candidate` must be one of the candidates.")

    top_group = {normalized_ranking[0]}
    for group in normalized_ties:
        if normalized_ranking[0] in group:
            top_group = set(group)
            break
    if best_candidate not in top_group:
        raise ValueError("`best_candidate` must be in the top tie group.")

    reason = str(output.get("reason", "")).strip()
    return {
        "ranking": normalized_ranking,
        "ties": normalized_ties,
        "best_candidate": best_candidate,
        "reason": reason,
    }


def validate_score_output(output: dict[str, Any], allowed_candidates: set[str]) -> dict[str, Any]:
    scores = output.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("`scores` must be an object.")

    normalized_scores: dict[str, float] = {}
    for key, value in scores.items():
        candidate_name = str(key).strip()
        if candidate_name not in allowed_candidates:
            raise ValueError(f"Unknown candidate in `scores`: {candidate_name!r}")
        try:
            numeric_score = float(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Score for {candidate_name!r} must be numeric.") from exc
        if not (0.0 <= numeric_score <= 10.0):
            raise ValueError(f"Score for {candidate_name!r} must be within [0, 10].")
        normalized_scores[candidate_name] = numeric_score

    if set(normalized_scores) != allowed_candidates:
        missing = sorted(allowed_candidates.difference(normalized_scores))
        extra = sorted(set(normalized_scores).difference(allowed_candidates))
        raise ValueError(f"`scores` must contain every candidate exactly once. missing={missing}, extra={extra}")

    reason = str(output.get("reason", "")).strip()
    return {
        "scores": normalized_scores,
        "reason": reason,
    }


def compute_rank_map(ranking: list[str], ties: list[list[str]]) -> dict[str, float]:
    group_map: dict[str, set[str]] = {}
    for candidate in ranking:
        group_map[candidate] = {candidate}
    for group in ties:
        group_set = set(group)
        for candidate in group:
            group_map[candidate] = group_set

    rank_map: dict[str, float] = {}
    visited: set[str] = set()
    position = 1
    for candidate in ranking:
        if candidate in visited:
            continue
        group = group_map[candidate]
        sorted_group = [item for item in ranking if item in group]
        visited.update(sorted_group)
        start = position
        end = position + len(sorted_group) - 1
        group_rank = (start + end) / 2.0
        for item in sorted_group:
            rank_map[item] = group_rank
        position = end + 1
    return rank_map


def derive_score_order(scores: dict[str, float]) -> tuple[list[str], list[list[str]], dict[str, float]]:
    grouped: dict[float, list[str]] = {}
    for candidate_name, score in scores.items():
        grouped.setdefault(float(score), []).append(candidate_name)

    ranking: list[str] = []
    ties: list[list[str]] = []
    for score in sorted(grouped.keys(), reverse=True):
        group = sorted(grouped[score])
        ranking.extend(group)
        if len(group) >= 2:
            ties.append(group)
    rank_map = compute_rank_map(ranking, ties)
    return ranking, ties, rank_map


def summarize_acc_results(results: dict[tuple[str, int], dict[str, Any]], all_set_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len({question_index for _set_name, question_index in results}),
        "task_type": TASK_TYPE_ACC,
        "sets": {},
    }
    for set_name in all_set_names:
        records = [record for (name, _question_index), record in results.items() if name == set_name]
        records.sort(key=lambda item: item["question_index"])
        elapsed = [float(record["elapsed_sec"]) for record in records if record["elapsed_sec"] is not None]
        steps = [int(record["executed_steps"]) for record in records if record["executed_steps"] is not None]
        success = [record for record in records if record["status"] == "ok"]
        skipped = [record for record in records if record["status"] == "skipped"]
        failed = [record for record in records if record["status"] == "error"]
        extract_found = [record for record in records if record.get("judge_extract_status") == "found"]
        extract_not_found = [record for record in records if record.get("judge_extract_status") == "not_found"]
        extract_error = [record for record in records if record.get("judge_extract_status") == "error"]
        final_judged = [
            record for record in records if record.get("judge_final_status") == "ok" and record.get("judge_final_verdict") is not None
        ]
        final_correct = [record for record in final_judged if record["judge_final_verdict"] == "correct"]
        final_incorrect = [record for record in final_judged if record["judge_final_verdict"] == "incorrect"]
        reasoning_judged = [
            record
            for record in records
            if record.get("judge_reasoning_status") == "ok" and record.get("judge_reasoning_verdict") is not None
        ]
        reasoning_correct = [record for record in reasoning_judged if record["judge_reasoning_verdict"] == "correct"]
        reasoning_incorrect = [record for record in reasoning_judged if record["judge_reasoning_verdict"] == "incorrect"]
        reward_judged = [record for record in records if record.get("reward_status") == "ok" and record.get("reward_rank") is not None]
        reward_best = [record for record in reward_judged if record.get("reward_is_best")]
        reward_ranks = [float(record["reward_rank"]) for record in reward_judged]
        summary["sets"][set_name] = {
            "count": len(records),
            "success_count": len(success),
            "skipped_count": len(skipped),
            "failure_count": len(failed),
            "extract_found_count": len(extract_found),
            "extract_not_found_count": len(extract_not_found),
            "extract_error_count": len(extract_error),
            "final_judge_success_count": len(final_judged),
            "final_judge_failure_count": len([record for record in records if record.get("judge_final_status") == "error"]),
            "final_answer_correct_count": len(final_correct),
            "final_answer_incorrect_count": len(final_incorrect),
            "final_answer_acc": (len(final_correct) / len(final_judged)) if final_judged else None,
            "reasoning_judge_success_count": len(reasoning_judged),
            "reasoning_judge_failure_count": len([record for record in records if record.get("judge_reasoning_status") == "error"]),
            "reasoning_correct_count": len(reasoning_correct),
            "reasoning_incorrect_count": len(reasoning_incorrect),
            "reasoning_acc": (len(reasoning_correct) / len(reasoning_judged)) if reasoning_judged else None,
            "judge_success_count": len(reasoning_judged),
            "judge_failure_count": len([record for record in records if record.get("judge_status") == "error"]),
            "correct_count": len(reasoning_correct),
            "incorrect_count": len(reasoning_incorrect),
            "acc": (len(reasoning_correct) / len(reasoning_judged)) if reasoning_judged else None,
            "reward_success_count": len(reward_judged),
            "reward_failure_count": len([record for record in records if record.get("reward_status") == "error"]),
            "reward_best_count": len(reward_best),
            "reward_average_rank": mean(reward_ranks) if reward_ranks else None,
            "average_elapsed_sec": mean(elapsed) if elapsed else None,
            "average_executed_steps": mean(steps) if steps else None,
        }
    return summary


def summarize_rank_results(results: dict[tuple[str, int], dict[str, Any]], all_set_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len({question_index for _set_name, question_index in results}),
        "task_type": TASK_TYPE_RANK,
        "sets": {},
    }
    for set_name in all_set_names:
        records = [record for (name, _question_index), record in results.items() if name == set_name]
        records.sort(key=lambda item: item["question_index"])
        elapsed = [float(record["elapsed_sec"]) for record in records if record["elapsed_sec"] is not None]
        steps = [int(record["executed_steps"]) for record in records if record["executed_steps"] is not None]
        success = [record for record in records if record["status"] == "ok"]
        skipped = [record for record in records if record["status"] == "skipped"]
        failed = [record for record in records if record["status"] == "error"]
        judged = [record for record in records if record.get("judge_status") == "ok" and record.get("judge_rank") is not None]
        best = [record for record in judged if record.get("judge_is_best")]
        top2 = [record for record in judged if float(record["judge_rank"]) <= 2.0]
        ranks = [float(record["judge_rank"]) for record in judged]
        reward_judged = [record for record in records if record.get("reward_status") == "ok" and record.get("reward_rank") is not None]
        reward_best = [record for record in reward_judged if record.get("reward_is_best")]
        reward_ranks = [float(record["reward_rank"]) for record in reward_judged]
        summary["sets"][set_name] = {
            "count": len(records),
            "success_count": len(success),
            "skipped_count": len(skipped),
            "failure_count": len(failed),
            "judge_success_count": len(judged),
            "judge_failure_count": len([record for record in records if record.get("judge_status") == "error"]),
            "best_count": len(best),
            "top2_count": len(top2),
            "average_rank": mean(ranks) if ranks else None,
            "reward_success_count": len(reward_judged),
            "reward_failure_count": len([record for record in records if record.get("reward_status") == "error"]),
            "reward_best_count": len(reward_best),
            "reward_average_rank": mean(reward_ranks) if reward_ranks else None,
            "average_elapsed_sec": mean(elapsed) if elapsed else None,
            "average_executed_steps": mean(steps) if steps else None,
        }
    return summary


def summarize_score_results(results: dict[tuple[str, int], dict[str, Any]], all_set_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "question_count": len({question_index for _set_name, question_index in results}),
        "task_type": TASK_TYPE_SCORE,
        "sets": {},
    }
    for set_name in all_set_names:
        records = [record for (name, _question_index), record in results.items() if name == set_name]
        records.sort(key=lambda item: item["question_index"])
        elapsed = [float(record["elapsed_sec"]) for record in records if record["elapsed_sec"] is not None]
        steps = [int(record["executed_steps"]) for record in records if record["executed_steps"] is not None]
        success = [record for record in records if record["status"] == "ok"]
        skipped = [record for record in records if record["status"] == "skipped"]
        failed = [record for record in records if record["status"] == "error"]
        judged = [record for record in records if record.get("judge_status") == "ok" and record.get("judge_score") is not None]
        scores = [float(record["judge_score"]) for record in judged]
        best = [record for record in judged if record.get("judge_is_best_from_score")]
        top2 = [record for record in judged if record.get("judge_rank_from_score") is not None and float(record["judge_rank_from_score"]) <= 2.0]
        derived_ranks = [float(record["judge_rank_from_score"]) for record in judged if record.get("judge_rank_from_score") is not None]
        reward_judged = [record for record in records if record.get("reward_status") == "ok" and record.get("reward_rank") is not None]
        reward_best = [record for record in reward_judged if record.get("reward_is_best")]
        reward_ranks = [float(record["reward_rank"]) for record in reward_judged]
        average_score = mean(scores) if scores else None
        score_std = None
        if scores:
            variance = sum((score - average_score) ** 2 for score in scores) / len(scores)
            score_std = variance ** 0.5
        summary["sets"][set_name] = {
            "count": len(records),
            "success_count": len(success),
            "skipped_count": len(skipped),
            "failure_count": len(failed),
            "judge_success_count": len(judged),
            "judge_failure_count": len([record for record in records if record.get("judge_status") == "error"]),
            "average_score": average_score,
            "score_std": score_std,
            "best_score_count": len(best),
            "best_count": len(best),
            "top2_count": len(top2),
            "average_rank": mean(derived_ranks) if derived_ranks else None,
            "reward_success_count": len(reward_judged),
            "reward_failure_count": len([record for record in records if record.get("reward_status") == "error"]),
            "reward_best_count": len(reward_best),
            "reward_average_rank": mean(reward_ranks) if reward_ranks else None,
            "average_elapsed_sec": mean(elapsed) if elapsed else None,
            "average_executed_steps": mean(steps) if steps else None,
        }
    return summary
