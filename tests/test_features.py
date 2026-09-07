"""Testes de unidade para src/features/build.py: pureza e regra de negócio de add_derived_features."""

from __future__ import annotations

import builtins

import pandas as pd
import pytest

from src.features.build import add_derived_features


def test_add_derived_features_does_not_mutate_input(synthetic_paysim: pd.DataFrame) -> None:
    before = synthetic_paysim.copy(deep=True)
    add_derived_features(synthetic_paysim)
    pd.testing.assert_frame_equal(synthetic_paysim, before)


def test_add_derived_features_has_no_io(
    synthetic_paysim: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regra local (src/features/CLAUDE.md): funções de feature são puras, sem I/O."""

    def _blow_up(*args: object, **kwargs: object) -> None:
        raise AssertionError("add_derived_features não deveria fazer I/O (chamou open())")

    monkeypatch.setattr(builtins, "open", _blow_up)
    add_derived_features(synthetic_paysim)


def test_add_derived_features_flags_drained_balance() -> None:
    df = pd.DataFrame(
        {
            "oldbalanceOrg": [100.0, 0.0, 50.0],
            "newbalanceOrig": [0.0, 0.0, 50.0],
        }
    )
    out = add_derived_features(df)
    assert out["f_orig_balance_drained"].tolist() == [1, 0, 0]
