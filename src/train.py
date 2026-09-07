"""Treino de classificação de fraude por transação: baseline + candidato de gradient boosting.

Decisões desta etapa vêm de `reports/research.md` ("Decisões para este projeto") e
`reports/eda.md` ("Implicações para /train"). Métricas sempre em `data/test.parquet` intacto.
"""

from __future__ import annotations

import logging
import os
import sys

# MLflow imprime emojis (ex.: "View run") no fim de cada run; no console cp1252 do Windows
# isso derruba end_run() com UnicodeEncodeError e deixa o run travado em RUNNING. Reconfigura
# aqui (não via env var) porque env vars não sobrevivem quando o processo é movido pra background.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.data import load_split
from src.features import FORBIDDEN_FEATURES
from src.features.build import (
    CATEGORICAL_FEATURES,
    DERIVED_FEATURES,
    NUMERIC_FEATURES,
    add_derived_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "/Shared/fraud-serving"
TARGET = "isFraud"
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DERIVED_FEATURES

assert not (set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES), "feature proibida em FEATURE_COLUMNS"


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS]


def build_pipeline(name: str) -> Pipeline:
    """Pipeline sklearn completo: derivação de features -> encoding -> classificador.

    `name` in {"logreg", "hgb"}. Aceita a linha bruta (schema de data/train.parquet) —
    seleciona só FEATURE_COLUMNS internamente, então colunas proibidas ou o alvo em X são ignorados.
    """
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    if name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    elif name == "hgb":
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"modelo desconhecido: {name!r} (use 'logreg' ou 'hgb')")

    return Pipeline(
        [
            ("derive", FunctionTransformer(add_derived_features)),
            ("select", FunctionTransformer(_select_features)),
            ("prep", preprocessor),
            ("clf", clf),
        ]
    )


def compute_metrics(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    """pr_auc (primária), roc_auc (informativa), recall_at_p90, brier. Threshold fica para /evaluate.

    `neg_brier` = -brier: o gate determinístico (.claude/skills/evaluate/scripts/gate.py, nunca
    editado por mim) só sabe checar `métrica >= mínimo`; como o gate da SPEC é `brier ≤ 0.002`
    (menor é melhor), logo essa métrica auxiliar para o script conseguir aplicar o mesmo critério
    (`neg_brier >= -0.002`) sem precisar de lógica de "menor é melhor".
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = precision >= 0.90
    recall_at_p90 = float(recall[mask].max()) if mask.any() else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall_at_p90": recall_at_p90,
        "brier": brier,
        "neg_brier": -brier,
    }


def train_and_log(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    pipeline = build_pipeline(name)
    y_train = train[TARGET]
    pipeline.fit(train, y_train)

    y_test = test[TARGET]
    y_prob = pipeline.predict_proba(test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)

    with mlflow.start_run(run_name=f"classification-{name}") as run:
        mlflow.set_tags({"stage": "train", "task": "classification", "model": name})
        mlflow.log_params(
            {
                "model": name,
                "class_weight": "balanced",
                "n_train": len(train),
                "n_train_positives": int(y_train.sum()),
                "n_test": len(test),
                "n_test_positives": int(y_test.sum()),
                "feature_columns": ",".join(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metrics(metrics)
        input_example = train.iloc[[0]]
        signature = mlflow.models.infer_signature(
            input_example, pipeline.predict_proba(input_example)
        )
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=input_example,
            signature=signature,
            # pipeline usa FunctionTransformer com funções de src.features.build: cloudpickle
            # serializa closures/funções custom sem a checagem de tipos "confiáveis" do skops.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id

    logger.info("run_id=%s model=%s metrics=%s", run_id, name, metrics)
    return {"run_id": run_id, "model": name, "task": "classification", **metrics}


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "databricks"
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks":
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, test = load_split()

    results = [train_and_log(name, train, test) for name in ("logreg", "hgb")]

    header = f"{'model':<10} {'task':<14} {'run_id':<34} {'pr_auc':>8} {'roc_auc':>8} {'recall@p90':>10} {'brier':>8}"
    print(header)
    for r in results:
        print(
            f"{r['model']:<10} {r['task']:<14} {r['run_id']:<34} "
            f"{r['pr_auc']:>8.4f} {r['roc_auc']:>8.4f} {r['recall_at_p90']:>10.4f} {r['brier']:>8.5f}"
        )


if __name__ == "__main__":
    main()
