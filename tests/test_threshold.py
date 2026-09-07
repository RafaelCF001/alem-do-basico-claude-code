"""Testes de unidade para a regra de decisão por custo (src/threshold.py)."""

from __future__ import annotations

import numpy as np

from src.threshold import evaluate_cost, expected_value_decision


def test_expected_value_decision_is_not_a_global_threshold() -> None:
    """O mesmo p̂ (0.5) decide diferente dependendo do valor da transação."""
    y_prob = np.array([0.5, 0.5])
    amount = np.array([10.0, 100.0])  # 0.5*10=5 <= 15; 0.5*100=50 > 15
    decision = expected_value_decision(y_prob, amount, fp_cost=15.0)
    assert decision.tolist() == [False, True]


def test_evaluate_cost_counts_false_negative_as_transaction_amount() -> None:
    """Uma fraude não detectada (p̂ baixo) custa o valor da transação, não fp_cost."""
    y_true = np.array([1])
    y_prob = np.array([0.01])  # 0.01 * 1000 = 10 <= 15 -> não decide revisar
    amount = np.array([1000.0])

    result = evaluate_cost(y_true, y_prob, amount, fp_cost=15.0)

    assert result["n_false_negatives"] == 1
    assert result["n_false_positives"] == 0
    assert result["fn_cost_total"] == 1000.0
    assert result["total_cost"] == 1000.0


def test_evaluate_cost_counts_false_positive_as_fp_cost() -> None:
    """Uma transação legítima mandada para revisão custa fp_cost, não o valor da transação."""
    y_true = np.array([0])
    y_prob = np.array([0.9])  # 0.9 * 1000 = 900 > 15 -> decide revisar
    amount = np.array([1000.0])

    result = evaluate_cost(y_true, y_prob, amount, fp_cost=15.0)

    assert result["n_false_positives"] == 1
    assert result["n_false_negatives"] == 0
    assert result["fp_cost_total"] == 15.0
    assert result["total_cost"] == 15.0
