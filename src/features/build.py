"""Engenharia de features pura para classificação de fraude por transação. Ver CLAUDE.md desta pasta.

Hipóteses de features vêm de `reports/eda.md`. Nenhuma função aqui tem I/O ou estado global;
qualquer transformação que aprende parâmetros fica no `Pipeline` do sklearn (src/train.py), não aqui.
"""

from __future__ import annotations

import pandas as pd

NUMERIC_FEATURES: list[str] = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
CATEGORICAL_FEATURES: list[str] = ["type"]
DERIVED_FEATURES: list[str] = ["f_orig_balance_drained"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """f_orig_balance_drained: saldo de origem chega positivo e sai zerado (reports/eda.md).

    Ressalva registrada na research (pergunta em aberto #3): pode ser artefato do
    simulador PaySim, não necessariamente causal em dados reais.
    """
    out = df.copy()
    out["f_orig_balance_drained"] = (
        (out["newbalanceOrig"] == 0) & (out["oldbalanceOrg"] > 0)
    ).astype("int8")
    return out
