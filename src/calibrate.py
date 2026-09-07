"""Calibração pós-hoc do candidato `hgb` (`reports/research.md` decisão #10):
`CalibratedClassifierCV` sobre os hiperparâmetros vencedores do `/tune`, para reduzir o brier
sem reabrir a busca de hiperparâmetros (`pr_auc`/`recall_at_p90` já folgados no gate).

CV de calibração é `TimeSeriesSplit` — nunca `KFold` aleatório, mesma regra não-negociável de
qualquer CV neste projeto (research.md decisão #8; SPEC/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# MLflow imprime emojis no fim de cada run; no console cp1252 do Windows isso derruba end_run()
# com UnicodeEncodeError e deixa o run travado em RUNNING (mesmo bug de src/train.py).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import mlflow
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.data import load_split
from src.train import EXPERIMENT_NAME, TARGET, build_pipeline, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TUNED_RUN_ID = "07e9a97d402b44bfb8b28660891e770e"  # melhor run do /tune (orçamento máximo)


def _load_tuned_params(run_id: str) -> dict[str, object]:
    """Lê os hiperparâmetros vencedores logados por `src/tune.py` (params `best_clf__...`)."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    raw = {
        k.removeprefix("best_"): v for k, v in run.data.params.items() if k.startswith("best_clf__")
    }
    if not raw:
        raise ValueError(f"run {run_id} não tem params 'best_clf__*' — não é um run de /tune?")

    # RandomizedSearchCV loga todo param como string; reconverte para o tipo original.
    typed: dict[str, object] = {}
    literals = {"None": None, "True": True, "False": False}
    for k, v in raw.items():
        if v in literals:
            typed[k] = literals[v]
            continue
        try:
            typed[k] = int(v)
        except ValueError:
            try:
                typed[k] = float(v)
            except ValueError:
                typed[k] = v
    return typed


def build_calibrated_pipeline(
    tuned_run_id: str, method: str = "isotonic"
) -> CalibratedClassifierCV:
    """`hgb` com os hiperparâmetros do `/tune`, envelopado em calibração (Platt=sigmoid ou isotônica)."""
    base = build_pipeline("hgb")
    base.set_params(**_load_tuned_params(tuned_run_id))
    return CalibratedClassifierCV(estimator=base, method=method, cv=TimeSeriesSplit(n_splits=3))


def train_and_log_calibrated(
    tuned_run_id: str, method: str, train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, object]:
    model = build_calibrated_pipeline(tuned_run_id, method=method)
    y_train = train[TARGET]
    model.fit(train, y_train)

    y_test = test[TARGET]
    y_prob = model.predict_proba(test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)

    with mlflow.start_run(run_name=f"classification-hgb-calibrated-{method}") as run:
        mlflow.set_tags({"stage": "calibrate", "task": "classification", "model": "hgb_calibrated"})
        mlflow.log_params(
            {
                "base_model": "hgb",
                "tuned_run_id": tuned_run_id,
                "calibration_method": method,
                "calibration_cv": "TimeSeriesSplit(n_splits=3)",
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        mlflow.log_metrics(metrics)
        input_example = train.iloc[[0]]
        signature = mlflow.models.infer_signature(input_example, model.predict_proba(input_example))
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id

    logger.info("run_id=%s model=hgb_calibrated(%s) metrics=%s", run_id, method, metrics)
    return {
        "run_id": run_id,
        "model": "hgb_calibrated",
        "method": method,
        "task": "classification",
        **metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuned-run-id", default=DEFAULT_TUNED_RUN_ID)
    ap.add_argument("--method", choices=["isotonic", "sigmoid", "both"], default="both")
    args = ap.parse_args()
    methods = ["isotonic", "sigmoid"] if args.method == "both" else [args.method]

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "databricks"
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks":
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, test = load_split()
    results = [
        train_and_log_calibrated(args.tuned_run_id, method, train, test) for method in methods
    ]

    header = (
        f"{'model':<16} {'method':<10} {'run_id':<34} {'pr_auc':>8} {'recall@p90':>10} {'brier':>8}"
    )
    print(header)
    for r in results:
        print(
            f"{r['model']:<16} {r['method']:<10} {r['run_id']:<34} "
            f"{r['pr_auc']:>8.4f} {r['recall_at_p90']:>10.4f} {r['brier']:>8.5f}"
        )


if __name__ == "__main__":
    main()
