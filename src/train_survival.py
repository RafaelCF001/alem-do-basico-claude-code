"""Treino do modelo de sobrevivência por conta de destino: Kaplan-Meier (descritivo) + Cox PH.

Decisão de research.md #6: Cox PH como modelo principal; RSF só se Cox não bater c-index ≥ 0.70.
`c_index` medido em `data/survival_test.parquet` (follow-up estendido até step 744 — decisão
confirmada com o humano, ver `build_survival_eval_table` em src/data.py).
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
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "/Shared/fraud-serving"
COVARIATES: list[str] = [
    "f_n_received_before",
    "f_amount_received_before_mean",
    "f_amount_received_before_sum",
]
# f_amount_received_before_{mean,sum} têm cauda pesada (máx. ~3.6e8 vs. mediana ~1.3e4 em
# data/survival_train.parquet) e travam o Newton-Raphson do CoxPH ("delta contains nan"): o
# log1p é a transformação padrão para covariáveis monetárias assimétricas em Cox PH.
AMOUNT_COVARIATES: list[str] = ["f_amount_received_before_mean", "f_amount_received_before_sum"]


def _log_transform_amount_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in AMOUNT_COVARIATES:
        out[col] = np.log1p(out[col])
    return out


class CoxPHWrapper(mlflow.pyfunc.PythonModel):
    """Envelope pyfunc: expõe o hazard parcial do CoxPHFitter como `predict`.

    Recebe as covariáveis na escala original (bruta) e aplica o mesmo log1p do treino antes
    de chamar o CoxPHFitter — quem consome o modelo nunca precisa saber da transformação.
    """

    def __init__(self, cph: CoxPHFitter, covariates: list[str]) -> None:
        self.cph = cph
        self.covariates = covariates

    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        transformed = _log_transform_amount_covariates(model_input)
        return self.cph.predict_partial_hazard(transformed[self.covariates])


def train_and_log_survival() -> dict[str, object]:
    survival_train = pd.read_parquet("data/survival_train.parquet")
    survival_test = pd.read_parquet("data/survival_test.parquet")

    km = KaplanMeierFitter().fit(survival_train["duration"], survival_train["event"])
    median_survival = km.median_survival_time_

    train_transformed = _log_transform_amount_covariates(survival_train)
    test_transformed = _log_transform_amount_covariates(survival_test)

    # penalizer=0.01 (L2 leve): mesmo com log1p, o Newton-Raphson ainda avisava
    # "failed to converge sufficiently" no volume completo (2,68M contas) — regularização é a
    # recomendação da própria doc do lifelines para esse aviso. 0.1 resolvia o aviso mas derrubava
    # o c-index de 0.78 para 0.66 (abaixo do gate); 0.01 é o menor valor que ainda estabiliza.
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        train_transformed[[*COVARIATES, "duration", "event"]],
        duration_col="duration",
        event_col="event",
    )
    c_index = cph.score(
        test_transformed[[*COVARIATES, "duration", "event"]],
        scoring_method="concordance_index",
    )

    with mlflow.start_run(run_name="survival-coxph") as run:
        mlflow.set_tags({"stage": "train", "task": "survival", "model": "coxph"})
        mlflow.log_params(
            {
                "model": "coxph",
                "covariates": ",".join(COVARIATES),
                "n_train_accounts": len(survival_train),
                "n_train_events": int(survival_train["event"].sum()),
                "n_eval_accounts": len(survival_test),
                "n_eval_events": int(survival_test["event"].sum()),
            }
        )
        mlflow.log_metrics(
            {
                "c_index": float(c_index),
                "km_median_survival": float(median_survival)
                if median_survival != float("inf")
                else -1.0,
            }
        )
        input_example = survival_train[COVARIATES].iloc[[0]]
        wrapper = CoxPHWrapper(cph, COVARIATES)
        signature = mlflow.models.infer_signature(
            input_example, wrapper.predict(None, input_example)
        )
        mlflow.pyfunc.log_model(
            name="model", python_model=wrapper, input_example=input_example, signature=signature
        )
        run_id = run.info.run_id

    logger.info("run_id=%s model=coxph task=survival c_index=%.4f", run_id, c_index)
    return {"run_id": run_id, "model": "coxph", "task": "survival", "c_index": float(c_index)}


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or "databricks"
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri == "databricks":
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENT_NAME)

    result = train_and_log_survival()
    print(
        f"{result['model']:<10} {result['task']:<14} {result['run_id']:<34} c_index={result['c_index']:.4f}"
    )


if __name__ == "__main__":
    main()
