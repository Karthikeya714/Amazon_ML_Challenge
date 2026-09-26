"""Scoring utilities: the leaderboard's macro F0.5, plus blocking diagnostics
(recall ceiling, reduction ratio) used to judge candidate_pairs.tsv quality.
"""
from __future__ import annotations

from typing import Dict, List


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f_beta(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    beta: float = 0.5,
) -> Dict[str, float]:
    """Per-entity F_beta, macro-averaged over every Source-1 id in
    ``ground_truth``. A singleton (empty truth) scores 1.0 for an empty
    prediction and 0.0 for any non-empty prediction, matching the rule
    stated in the challenge materials -- this falls out of the F_beta
    formula automatically as long as we special-case the 0/0 (empty vs
    empty) precision/recall as perfect.
    """
    scores = []
    for s1_id, truth in ground_truth.items():
        truth_set = set(truth)
        pred_set = set(predictions.get(s1_id, []))

        if not truth_set and not pred_set:
            scores.append(1.0)
            continue
        if not pred_set:  # truth non-empty, predicted nothing -> recall 0
            scores.append(0.0)
            continue

        tp = len(truth_set & pred_set)
        precision = tp / len(pred_set)
        recall = tp / len(truth_set) if truth_set else 0.0
        scores.append(f_beta(precision, recall, beta))

    macro = sum(scores) / len(scores) if scores else 0.0
    return {"macro_f_beta": macro, "n_entities": len(scores)}


def blocking_diagnostics(
    candidates: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, float]:
    """Recall ceiling (fraction of true match edges present in the
    candidate set) and reduction ratio (avg candidates per Source-1 entity)
    -- the two numbers the final ranking judges candidate_pairs.tsv on.
    """
    total_true_edges = 0
    recovered_edges = 0
    total_candidates = 0
    n_entities = 0

    for s1_id, truth in ground_truth.items():
        truth_set = set(truth)
        cand_set = set(candidates.get(s1_id, []))
        total_true_edges += len(truth_set)
        recovered_edges += len(truth_set & cand_set)
        total_candidates += len(cand_set)
        n_entities += 1

    recall_ceiling = recovered_edges / total_true_edges if total_true_edges else 1.0
    avg_candidates = total_candidates / n_entities if n_entities else 0.0
    return {
        "recall_ceiling": recall_ceiling,
        "avg_candidates_per_s1": avg_candidates,
        "total_true_edges": total_true_edges,
        "recovered_edges": recovered_edges,
    }
