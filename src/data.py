"""Carga, split temporal e tabela de sobrevivência do dataset PaySim.

Fonte de verdade: SPEC.md (dataset, split, amostra) e CLAUDE.md (regras não-negociáveis).

Split temporal feito UMA vez aqui: treino = step <= TRAIN_MAX_STEP, teste = step > TRAIN_MAX_STEP.
Nunca re-splitar a jusante; nunca train_test_split aleatório.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
TRAIN_MAX_STEP = 600  # não-negociável: SPEC.md / CLAUDE.md
SURVIVAL_CENSOR_STEP = 600  # censura administrativa no fim do treino
NEGATIVE_SAMPLE_SIZE = 300_000  # SPEC.md: amostra para a demo, mantendo TODOS os positivos
RANDOM_STATE = 42

PAYSIM_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê `data/train.parquet` e `data/test.parquet` já splitados por `src/data.py`.

    Nunca re-splita: só lê o que já foi decidido temporalmente por `temporal_split()`.
    """
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    test = pd.read_parquet(DATA_DIR / "test.parquet")
    return train, test


def _find_local_paysim_csv() -> Path | None:
    """Procura um CSV do PaySim já baixado em data/ (evita rede na hora da demo)."""
    if not DATA_DIR.exists():
        return None
    candidates = [p for p in DATA_DIR.glob("*.csv") if "paysim" in p.name.lower()]
    return candidates[0] if candidates else None


def _download_paysim_via_kagglehub() -> Path:
    """Baixa o PaySim via kagglehub (credenciais só via KAGGLE_* no ambiente)."""
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download("ealaxi/paysim1"))
    csvs = list(dataset_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"kagglehub baixou {dataset_path}, mas nenhum CSV foi encontrado")
    return csvs[0]


def load_paysim() -> pd.DataFrame:
    """Carrega o PaySim bruto (local, se existir; senão kagglehub)."""
    csv_path = _find_local_paysim_csv()
    if csv_path is not None:
        logger.info("Usando PaySim local: %s", csv_path)
    else:
        logger.info("PaySim local não encontrado; baixando via kagglehub")
        csv_path = _download_paysim_via_kagglehub()

    df = pd.read_csv(csv_path, dtype=PAYSIM_DTYPES)
    logger.info("PaySim carregado: %d linhas, %d colunas", len(df), df.shape[1])
    return df


def load_openml_creditcard_fallback() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback sem Kaggle: OpenML `creditcard` (só classificação, sem `step`/sobrevivência).

    Usa `Time` (segundos, ~48h) como proxy temporal: split no mesmo quantil que
    TRAIN_MAX_STEP/744 representaria no PaySim (~0.806), nunca aleatório.
    """
    from sklearn.datasets import fetch_openml

    logger.warning(
        "Usando fallback OpenML creditcard: sobrevivência será pulada (sem conta de destino)"
    )
    bunch = fetch_openml(name="creditcard", version=1, as_frame=True, parser="auto")
    df = bunch.frame.rename(columns={"Class": "isFraud"})
    df["isFraud"] = df["isFraud"].astype("int8")

    cutoff = df["Time"].quantile(TRAIN_MAX_STEP / 744)
    train = df[df["Time"] <= cutoff].reset_index(drop=True)
    test = df[df["Time"] > cutoff].reset_index(drop=True)
    return train, test


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal fixo por `step`. Não re-splitar a jusante."""
    train = df[df["step"] <= TRAIN_MAX_STEP].reset_index(drop=True)
    test = df[df["step"] > TRAIN_MAX_STEP].reset_index(drop=True)
    logger.info(
        "Split temporal: treino step<=%d (%d linhas), teste step>%d (%d linhas)",
        TRAIN_MAX_STEP,
        len(train),
        TRAIN_MAX_STEP,
        len(test),
    )
    return train, test


def downsample_negatives(
    train: pd.DataFrame, target_col: str = "isFraud", n_negatives: int = NEGATIVE_SAMPLE_SIZE
) -> pd.DataFrame:
    """Downsample de negativos no treino mantendo TODOS os positivos (SPEC.md).

    Nunca aplicar em teste. Nunca aplicar antes do split temporal.
    """
    positives = train[train[target_col] == 1]
    negatives = train[train[target_col] == 0]
    if len(negatives) > n_negatives:
        negatives = negatives.sample(n=n_negatives, random_state=RANDOM_STATE)
    sampled = pd.concat([positives, negatives]).sort_values("step").reset_index(drop=True)
    logger.info(
        "Downsample de negativos: %d positivos + %d negativos (de %d) = %d linhas",
        len(positives),
        len(negatives),
        len(train) - len(positives),
        len(sampled),
    )
    return sampled


def build_survival_table(
    train_full: pd.DataFrame, censor_step: int = SURVIVAL_CENSOR_STEP
) -> pd.DataFrame:
    """Tabela de sobrevivência por conta de destino, a partir do treino COMPLETO (não downsampled).

    unidade = nameDest; duration = steps desde a 1a transação recebida até a 1a fraude recebida;
    event=1 se houve fraude no treino, senão censurado em `censor_step`.
    Covariáveis (`f_*`) usam só transações estritamente anteriores ao evento/censura.
    """
    received = train_full[train_full["step"] <= censor_step].copy()

    first_seen = received.groupby("nameDest", observed=True)["step"].min().rename("start_step")

    fraud_received = received[received["isFraud"] == 1]
    first_fraud = (
        fraud_received.groupby("nameDest", observed=True)["step"].min().rename("fraud_step")
    )

    accounts = first_seen.to_frame().join(first_fraud, how="left")
    accounts["event"] = accounts["fraud_step"].notna().astype("int8")
    accounts["duration"] = (
        accounts["fraud_step"].fillna(censor_step) - accounts["start_step"]
    ).clip(lower=0)

    # Covariáveis causais: só transações recebidas ANTES do step do evento/censura de cada conta.
    accounts = accounts.reset_index()
    event_step = accounts["fraud_step"].fillna(censor_step)
    received_indexed = received.merge(
        accounts[["nameDest"]].assign(_cutoff=event_step.values), on="nameDest", how="inner"
    )
    prior = received_indexed[received_indexed["step"] < received_indexed["_cutoff"]]

    agg = prior.groupby("nameDest", observed=True).agg(
        f_n_received_before=("amount", "count"),
        f_amount_received_before_mean=("amount", "mean"),
        f_amount_received_before_sum=("amount", "sum"),
    )
    accounts = accounts.merge(agg, on="nameDest", how="left").fillna(
        {
            "f_n_received_before": 0,
            "f_amount_received_before_mean": 0.0,
            "f_amount_received_before_sum": 0.0,
        }
    )
    accounts = accounts.drop(columns=["fraud_step"])
    logger.info(
        "Tabela de sobrevivência: %d contas, %d eventos (%.4f%% taxa de evento)",
        len(accounts),
        accounts["event"].sum(),
        100 * accounts["event"].mean(),
    )
    return accounts


def build_survival_eval_table(
    survival_train: pd.DataFrame, test: pd.DataFrame, max_step: int = 744
) -> pd.DataFrame:
    """Estende o desfecho de sobrevivência ao período de teste (step>600) para medir c-index "no teste".

    Decisão confirmada com o humano (pergunta em aberto da research não cobria isso):
    a verdade de `duration`/`event` passa a considerar fraudes recebidas depois do step 600,
    mas as covariáveis (`f_*`) continuam as mesmas de `survival_train` — calculadas só com
    dados até o step 600. Isso evita leakage de features, só estende o rótulo/desfecho.
    """
    already_event = survival_train[survival_train["event"] == 1].copy()
    censored = survival_train[survival_train["event"] == 0].copy()

    fraud_test = test[test["isFraud"] == 1]
    first_fraud_test = (
        fraud_test.groupby("nameDest", observed=True)["step"].min().rename("fraud_step_test")
    )
    censored = censored.merge(first_fraud_test, on="nameDest", how="left")
    has_future_fraud = censored["fraud_step_test"].notna()
    censored.loc[has_future_fraud, "event"] = 1
    censored.loc[has_future_fraud, "duration"] = (
        censored.loc[has_future_fraud, "fraud_step_test"]
        - censored.loc[has_future_fraud, "start_step"]
    )
    censored.loc[~has_future_fraud, "duration"] = (
        max_step - censored.loc[~has_future_fraud, "start_step"]
    )
    censored = censored.drop(columns=["fraud_step_test"])

    eval_table = pd.concat([already_event, censored], ignore_index=True)
    logger.info(
        "Tabela de sobrevivência (avaliação, follow-up até step %d): %d contas, %d eventos",
        max_step,
        len(eval_table),
        int(eval_table["event"].sum()),
    )
    return eval_table


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    try:
        df = load_paysim()
        train_full, test = temporal_split(df)
        survival = build_survival_table(train_full)
        train = downsample_negatives(train_full)
        survival.to_parquet(DATA_DIR / "survival_train.parquet", index=False)
        survival_eval = build_survival_eval_table(survival, test)
        survival_eval.to_parquet(DATA_DIR / "survival_test.parquet", index=False)
    except Exception as exc:  # noqa: BLE001 - kagglehub falha de formas variadas; fallback deliberado
        logger.warning("PaySim indisponível (%s); usando fallback OpenML creditcard", exc)
        train, test = load_openml_creditcard_fallback()

    train.to_parquet(DATA_DIR / "train.parquet", index=False)
    test.to_parquet(DATA_DIR / "test.parquet", index=False)
    logger.info(
        "Salvos: data/train.parquet (%d linhas), data/test.parquet (%d linhas)",
        len(train),
        len(test),
    )


if __name__ == "__main__":
    main()
