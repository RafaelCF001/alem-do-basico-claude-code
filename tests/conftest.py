"""Fixtures compartilhadas: DataFrames sintéticos pequenos no schema do PaySim.

Contrato dos testes (ver CLAUDE.md e .claude/skills/train/SKILL.md): rápidos, sem depender dos
parquets grandes em data/ quando o comportamento pode ser isolado com poucas linhas.
"""

from __future__ import annotations

import pandas as pd
import pytest

PAYSIM_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


@pytest.fixture
def synthetic_paysim() -> pd.DataFrame:
    """~10 linhas cobrindo TRANSFER/CASH_OUT/PAYMENT, fraude e saldo de origem dreno/não-dreno."""
    return pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5, 601, 602, 603, 604, 605],
            "type": [
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "TRANSFER",
                "CASH_OUT",
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
            ],
            "amount": [100.0, 200.0, 50.0, 300.0, 400.0, 150.0, 250.0, 60.0, 350.0, 450.0],
            "nameOrig": [f"C{i}" for i in range(10)],
            "oldbalanceOrg": [100.0, 200.0, 50.0, 300.0, 400.0, 150.0, 250.0, 60.0, 350.0, 450.0],
            "newbalanceOrig": [0.0, 0.0, 0.0, 300.0, 400.0, 0.0, 0.0, 0.0, 350.0, 450.0],
            "nameDest": ["D0", "D1", "D2", "D0", "D1", "D0", "D1", "D2", "D3", "D4"],
            "oldbalanceDest": [0.0] * 10,
            "newbalanceDest": [100.0, 200.0, 50.0, 300.0, 400.0, 150.0, 250.0, 60.0, 350.0, 450.0],
            "isFraud": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "isFlaggedFraud": [0] * 10,
        }
    )[PAYSIM_COLUMNS]


@pytest.fixture
def causal_leakage_frame() -> pd.DataFrame:
    """Uma conta (DX) recebe uma fraude no step 5; uma transação legítima chega depois, no step 8.

    Um agregador causal correto de covariáveis (`f_*`) NUNCA deve contar a transação do step 8,
    porque ela é posterior ao evento (fraude) daquela conta — mesmo estando antes do censor_step
    global. Serve para pegar exatamente o tipo de leakage que a research.md aponta (armadilha #3).
    """
    return pd.DataFrame(
        {
            "step": [1, 5, 8],
            "type": ["CASH_OUT", "TRANSFER", "CASH_OUT"],
            "amount": [10.0, 999.0, 500.0],
            "nameOrig": ["C1", "C2", "C3"],
            "oldbalanceOrg": [10.0, 999.0, 500.0],
            "newbalanceOrig": [0.0, 0.0, 0.0],
            "nameDest": ["DX", "DX", "DX"],
            "oldbalanceDest": [0.0, 10.0, 1009.0],
            "newbalanceDest": [10.0, 1009.0, 1509.0],
            "isFraud": [0, 1, 0],
            "isFlaggedFraud": [0, 0, 0],
        }
    )[PAYSIM_COLUMNS]
