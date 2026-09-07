"""Testes de unidade para src/data.py: split temporal, downsample e tabela de sobrevivência."""

from __future__ import annotations

import pandas as pd

from src.data import (
    build_survival_eval_table,
    build_survival_table,
    downsample_negatives,
    temporal_split,
)


def test_temporal_split_boundary_is_inclusive_on_train(synthetic_paysim: pd.DataFrame) -> None:
    """step == TRAIN_MAX_STEP vai para o treino; step > TRAIN_MAX_STEP vai para o teste."""
    df = synthetic_paysim.copy()
    df.loc[0, "step"] = 600
    df.loc[1, "step"] = 601
    train, test = temporal_split(df)
    assert 600 in set(train["step"])
    assert 601 in set(test["step"])
    assert 601 not in set(train["step"])


def test_downsample_negatives_keeps_all_positives() -> None:
    positives = pd.DataFrame({"step": range(5), "isFraud": [1] * 5})
    negatives = pd.DataFrame({"step": range(100), "isFraud": [0] * 100})
    train = pd.concat([positives, negatives], ignore_index=True)

    sampled = downsample_negatives(train, n_negatives=10)

    assert (sampled["isFraud"] == 1).sum() == 5
    assert (sampled["isFraud"] == 0).sum() == 10
    assert len(sampled) == 15


def test_downsample_negatives_is_noop_when_fewer_negatives_than_target() -> None:
    positives = pd.DataFrame({"step": range(3), "isFraud": [1] * 3})
    negatives = pd.DataFrame({"step": range(4), "isFraud": [0] * 4})
    train = pd.concat([positives, negatives], ignore_index=True)

    sampled = downsample_negatives(train, n_negatives=1000)

    assert len(sampled) == len(train)


def test_build_survival_table_censors_account_without_fraud(synthetic_paysim: pd.DataFrame) -> None:
    """Conta sem fraude recebida: event=0, duration = censor_step - start_step."""
    survival = build_survival_table(synthetic_paysim, censor_step=605)
    d2 = survival[survival["nameDest"] == "D2"].iloc[0]  # D2 só recebe PAYMENT não-fraude
    assert d2["event"] == 0
    assert d2["duration"] == 605 - 3  # D2 aparece pela 1a vez no step 3


def test_build_survival_eval_table_extends_censored_event_from_test(
    synthetic_paysim: pd.DataFrame,
) -> None:
    """Conta censurada no treino que recebe fraude no período de teste vira evento na avaliação,
    mas as covariáveis (f_*) continuam as do treino — sem recalcular com dados do teste."""
    train_full = synthetic_paysim[synthetic_paysim["step"] <= 600]
    test = synthetic_paysim[synthetic_paysim["step"] > 600].copy()

    survival_train = build_survival_table(train_full, censor_step=600)
    d1_before = survival_train[survival_train["nameDest"] == "D1"].iloc[0]
    assert d1_before["event"] == 0  # D1 não tem fraude no treino

    # D1 recebe fraude no teste (step 602)
    test.loc[test["nameDest"] == "D1", "isFraud"] = 1
    eval_table = build_survival_eval_table(survival_train, test, max_step=744)

    d1_after = eval_table[eval_table["nameDest"] == "D1"].iloc[0]
    assert d1_after["event"] == 1
    assert d1_after["duration"] == 602 - d1_before["start_step"]
    assert (
        d1_after["f_n_received_before"] == d1_before["f_n_received_before"]
    )  # covariável intocada
