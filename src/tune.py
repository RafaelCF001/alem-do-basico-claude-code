"""Busca de hiperparâmetros com orçamento fixo da SPEC para o candidato reprovado no gate.

Parte do `hgb` de `reports/eval_gate.json` (reaproveita `build_pipeline("hgb")` de src/train.py,
não reimplementa o pipeline). CV temporal (`TimeSeriesSplit`) — nunca `KFold` aleatório, por
causa do drift (research.md decisão #8). Avaliação final só em `data/test.parquet` salvo, nunca
no CV. Orçamento da SPEC: `n_iter` <= 25 (nunca acima).

O gate reprovou por `brier` (0.00618 > 0.002), não por `pr_auc`/`recall_at_p90` (já folgados).
`scoring="average_precision"` é o que a skill /tune manda usar — não troco a métrica de busca,
mas o espaço de hiperparâmetros abaixo é escolhido para favorecer modelos menos overconfident
(menos folhas, mais regularização), que tende a ajudar a calibração como efeito colateral.
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
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.data import load_split
from src.train import EXPERIMENT_NAME, TARGET, build_pipeline, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_N_ITER = 25  # orçamento da SPEC — nunca acima

# Espaço pequeno; cada parâmetro com 1 linha de justificativa (skill /tune).
PARAM_DISTRIBUTIONS = {
    # menos folhas / mais amostras por folha => árvores menos complexas, tendem a superajustar
    # menos a fraude rara (~2% na amostra de treino) e produzir probabilidades menos
    # overconfident — é exatamente o que o brier penaliza.
    "clf__max_leaf_nodes": randint(7, 63),
    "clf__min_samples_leaf": randint(20, 200),
    # taxa de aprendizado menor + mais iterações é o par clássico para reduzir overfitting de
    # boosting sem perder capacidade de discriminação (pr_auc).
    "clf__learning_rate": uniform(0.01, 0.29),  # amostra em [0.01, 0.30)
    "clf__max_iter": randint(100, 400),
    # regularização L2 explícita — reduz a confiança excessiva nas folhas raras.
    "clf__l2_regularization": uniform(0.0, 2.0),
    # a skill pede para incluir class_weight no espaço de busca, não fixo no baseline.
    "clf__class_weight": [None, "balanced"],
}


def tune(n_iter: int) -> dict[str, object]:
    train, test = load_split()
    pipeline = build_pipeline("hgb")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="average_precision",
        cv=TimeSeriesSplit(n_splits=3),
        random_state=42,
        n_jobs=-1,
    )
    search.fit(train, train[TARGET])

    best_pipeline = search.best_estimator_
    y_test = test[TARGET]
    y_prob = best_pipeline.predict_proba(test)[:, 1]
    metrics = compute_metrics(y_test, y_prob)

    with mlflow.start_run(run_name="classification-hgb-tuned") as run:
        mlflow.set_tags({"stage": "tune", "task": "classification", "model": "hgb"})
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_params(
            {
                "n_iter": n_iter,
                "cv": "TimeSeriesSplit(n_splits=3)",
                "scoring": "average_precision",
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_best_average_precision", float(search.best_score_))
        input_example = train.iloc[[0]]
        signature = mlflow.models.infer_signature(
            input_example, best_pipeline.predict_proba(input_example)
        )
        mlflow.sklearn.log_model(
            best_pipeline,
            name="model",
            input_example=input_example,
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        run_id = run.info.run_id

    logger.info(
        "run_id=%s model=hgb-tuned metrics=%s best_params=%s",
        run_id,
        metrics,
        search.best_params_,
    )
    return {
        "run_id": run_id,
        "model": "hgb",
        "task": "classification",
        "metrics": metrics,
        "best_params": search.best_params_,
        "cv_results": search.cv_results_,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=MAX_N_ITER)
    args = ap.parse_args()
    n_iter = min(args.n_iter, MAX_N_ITER)
    if args.n_iter > MAX_N_ITER:
        logger.warning(
            "n_iter pedido (%d) > orçamento da SPEC (%d); usando %d",
            args.n_iter,
            MAX_N_ITER,
            n_iter,
        )

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "databricks"
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks":
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)

    result = tune(n_iter)
    print(f"run_id={result['run_id']} model=hgb-tuned metrics={result['metrics']}")
    print(f"best_params={result['best_params']}")

    cv = pd.DataFrame(result["cv_results"])
    top2 = cv.sort_values("rank_test_score").head(2)[
        ["rank_test_score", "mean_test_score", "params"]
    ]
    print("Top 2 tentativas do CV (por average_precision):")
    print(top2.to_string(index=False))


if __name__ == "__main__":
    main()
