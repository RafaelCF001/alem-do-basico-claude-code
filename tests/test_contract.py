"""Contrato mínimo exigido antes de treinar (`.claude/skills/train/SKILL.md`).

Não editar estes testes para fazê-los passar — se algo aqui falhar, é `src/` que precisa mudar.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data import build_survival_table, load_split, temporal_split
from src.features import FORBIDDEN_FEATURES
from src.train import FEATURE_COLUMNS, build_pipeline
from src.train_survival import COVARIATES


def test_load_split_is_temporal() -> None:
    """`load_split()` devolve treino/teste temporais: nenhum step de treino é >= algum step de teste."""
    train, test = load_split()
    assert train["step"].max() < test["step"].min()


def test_temporal_split_has_no_row_overlap(synthetic_paysim: pd.DataFrame) -> None:
    """`temporal_split()` particiona por `step`: nenhuma linha (por id) cai nos dois lados,
    e nenhuma linha é perdida ou duplicada. (Índice pandas não serve de checagem aqui: os dois
    lados fazem `reset_index(drop=True)` de propósito, então ambos recomeçam em 0.)
    """
    df = synthetic_paysim.assign(row_id=range(len(synthetic_paysim)))
    train, test = temporal_split(df)
    assert set(train["row_id"]).isdisjoint(set(test["row_id"]))
    assert len(train) + len(test) == len(df)


def test_forbidden_features_never_in_feature_columns() -> None:
    """Nenhuma `FORBIDDEN_FEATURES` (isFlaggedFraud, nameOrig, nameDest crus, isFraud) é usada em X."""
    assert not (set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES)


def test_pipeline_select_step_excludes_forbidden_columns(synthetic_paysim: pd.DataFrame) -> None:
    """Mesmo alimentando o pipeline com a linha bruta (todas as colunas), a etapa de seleção
    de features nunca deixa passar uma coluna proibida adiante para o classificador."""
    pipeline = build_pipeline("logreg")
    derived = pipeline.named_steps["derive"].transform(synthetic_paysim)
    selected = pipeline.named_steps["select"].transform(derived)
    assert not (set(selected.columns) & FORBIDDEN_FEATURES)


@pytest.mark.parametrize("name", ["logreg", "hgb"])
def test_build_pipeline_returns_sklearn_pipeline_with_encoder_inside(name: str) -> None:
    """`build_pipeline(name)` devolve `sklearn.Pipeline`; encoder/scaler está dentro dela."""
    pipeline = build_pipeline(name)
    assert isinstance(pipeline, Pipeline)
    assert "prep" in pipeline.named_steps


def test_pipeline_predicts_proba_for_one_row() -> None:
    """`pipeline.predict_proba(input_example)` funciona para 1 linha, no schema de data/train.parquet."""
    train, _ = load_split()
    sample = train.sample(n=min(5000, len(train)), random_state=0)
    pipeline = build_pipeline("logreg")
    pipeline.fit(sample, sample["isFraud"])

    input_example = train.iloc[[0]]
    proba = pipeline.predict_proba(input_example)

    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_survival_table_has_duration_and_event_columns(synthetic_paysim: pd.DataFrame) -> None:
    """`build_survival_table()` tem `duration` e `event`."""
    survival = build_survival_table(synthetic_paysim, censor_step=605)
    assert {"duration", "event", "nameDest"}.issubset(survival.columns)


def test_survival_covariates_exclude_post_event_transactions(
    causal_leakage_frame: pd.DataFrame,
) -> None:
    """Covariáveis (`f_*`) só usam transações estritamente anteriores ao evento/censura da conta —
    não ao censor_step global. Ver `causal_leakage_frame` em conftest.py."""
    survival = build_survival_table(causal_leakage_frame, censor_step=100)
    row = survival[survival["nameDest"] == "DX"].iloc[0]

    assert row["event"] == 1
    assert row["duration"] == 5 - 1
    assert row["f_n_received_before"] == 1  # só o step=1; nunca o step=8 (pós-evento)
    assert row["f_amount_received_before_sum"] == 10.0


def test_survival_covariates_are_not_forbidden() -> None:
    """As covariáveis usadas no CoxPH não incluem nenhuma FORBIDDEN_FEATURES."""
    assert not (set(COVARIATES) & FORBIDDEN_FEATURES)
