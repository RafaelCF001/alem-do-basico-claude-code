"""Regra de decisão por custo (SPEC: FN = valor da transação, FP = R$15).

Pergunta em aberto #1 da `reports/research.md`, confirmada com o humano: o custo do falso
negativo varia por transação (é o próprio `amount`), então a regra correta é decisão por valor
esperado (`p̂ × amount > custo_fp`) — não um único ponto de corte de probabilidade `τ` global,
que seria ótimo para uma transação e subótimo para outra de valor diferente.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

# MLflow imprime emojis; no console cp1252 do Windows isso derruba chamadas internas com
# UnicodeEncodeError (mesmo bug de src/train.py).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import mlflow
import numpy as np

from src.data import load_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FP_COST = 15.0  # SPEC: custo de revisão manual


def expected_value_decision(
    y_prob: np.ndarray, amount: np.ndarray, fp_cost: float = FP_COST
) -> np.ndarray:
    """Bloqueia/revisa quando o custo esperado de deixar passar (`p̂ × amount`) supera o custo
    de revisão manual. O mesmo `p̂` decide diferente dependendo do valor da transação — por
    design, não é um único ponto de corte de probabilidade.
    """
    return (np.asarray(y_prob) * np.asarray(amount)) > fp_cost


def evaluate_cost(
    y_true: np.ndarray, y_prob: np.ndarray, amount: np.ndarray, fp_cost: float = FP_COST
) -> dict[str, float]:
    """Custo realizado no teste salvo, aplicando `expected_value_decision`.

    FN: transação fraudulenta que a regra deixou passar (custo = `amount`).
    FP: transação legítima que a regra mandou para revisão manual (custo = `fp_cost`).
    """
    decision = expected_value_decision(y_prob, amount, fp_cost)
    y_true = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype=float)

    false_negatives = (~decision) & y_true
    false_positives = decision & (~y_true)

    fn_cost_total = float(amount[false_negatives].sum())
    fp_cost_total = float(false_positives.sum() * fp_cost)
    total_cost = fn_cost_total + fp_cost_total
    n = len(y_true)

    return {
        "n_transactions": int(n),
        "n_reviewed": int(decision.sum()),
        "n_false_negatives": int(false_negatives.sum()),
        "n_false_positives": int(false_positives.sum()),
        "fn_cost_total": fn_cost_total,
        "fp_cost_total": fp_cost_total,
        "total_cost": total_cost,
        "avg_cost_per_transaction": total_cost / n if n else 0.0,
        "fp_cost_unit": fp_cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="run_id do modelo aprovado no gate")
    ap.add_argument("--fp-cost", type=float, default=FP_COST)
    ap.add_argument("--gate-file", default="reports/eval_gate.json")
    args = ap.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "databricks"
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks":
        mlflow.set_registry_uri("databricks-uc")

    _, test = load_split()
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")
    y_prob = model.predict_proba(test)[:, 1]

    result = evaluate_cost(
        test["isFraud"].to_numpy(), y_prob, test["amount"].to_numpy(), args.fp_cost
    )
    logger.info("custo no teste: %s", result)
    print(json.dumps(result, indent=2))

    gate_path = pathlib.Path(args.gate_file)
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    gate["decision_rule"] = "expected_value: p_hat * amount > fp_cost (nao ha tau global)"
    gate["threshold_run_id"] = args.run_id
    gate["fp_cost"] = args.fp_cost
    gate["expected_cost"] = result
    gate_path.write_text(json.dumps(gate, indent=2))
    logger.info("gravado em %s", gate_path)


if __name__ == "__main__":
    main()
