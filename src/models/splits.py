"""
Out-of-time (OOT) splitting — the only valid train/validation/test partition for
PD evaluation.

PD models are graded on their ability to rank *future* defaults, so the folds
must be carved by ``observation_date`` against fixed cutoffs (config
``validation.train_cutoff`` / ``validation_cutoff`` / ``test_cutoff``), never by
random cross-validation. Records observed after the test cutoff are excluded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OOTSplit:
    """Boolean masks (aligned to the input order) for the three OOT folds."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    @property
    def counts(self) -> dict[str, int]:
        return {
            "train": int(self.train.sum()),
            "validation": int(self.validation.sum()),
            "test": int(self.test.sum()),
        }


def oot_split(
    observation_dates: Sequence | pd.Series,
    train_cutoff: str | pd.Timestamp,
    validation_cutoff: str | pd.Timestamp,
    test_cutoff: str | pd.Timestamp,
) -> OOTSplit:
    """
    Partition rows into train / validation / test by observation date.

    Folds are half-open on the left, closed on the right::

        train       : date <= train_cutoff
        validation  : train_cutoff < date <= validation_cutoff
        test        : validation_cutoff < date <= test_cutoff

    Parameters
    ----------
    observation_dates : sequence | pd.Series
        Per-row observation dates (anything ``pd.to_datetime`` accepts).
    train_cutoff, validation_cutoff, test_cutoff : str | pd.Timestamp
        Strictly increasing fold boundaries.

    Returns
    -------
    OOTSplit
        Boolean masks aligned to the input order.
    """
    tc, vc, sc = (pd.Timestamp(c) for c in (train_cutoff, validation_cutoff, test_cutoff))
    if not (tc < vc < sc):
        raise ValueError(
            "Cutoffs must be strictly increasing "
            f"(train < validation < test); got {tc.date()}, {vc.date()}, {sc.date()}"
        )

    dates = pd.to_datetime(pd.Series(list(observation_dates)).reset_index(drop=True))
    if dates.isna().any():
        raise ValueError("observation_dates contains values that could not be parsed as dates.")

    train = (dates <= tc).to_numpy()
    validation = ((dates > tc) & (dates <= vc)).to_numpy()
    test = ((dates > vc) & (dates <= sc)).to_numpy()

    if not train.any():
        raise ValueError(f"Train fold is empty (no observations on/before {tc.date()}).")
    if not test.any():
        raise ValueError(f"Test fold is empty (no observations in ({vc.date()}, {sc.date()}]).")
    return OOTSplit(train=train, validation=validation, test=test)


__all__ = ["OOTSplit", "oot_split"]
