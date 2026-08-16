"""Unit tests for the e-BH procedure (Wang & Ramdas 2022)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evalue.ebh import ebh_threshold, ebh_rejection_set


def test_single_strong_evalue_rejects():
    # K=2, alpha=0.05: threshold K/(alpha*k). k=1 needs e_(1) >= 40.
    s = pd.Series([100.0, 0.1], index=["a", "b"])
    assert ebh_rejection_set(s, alpha=0.05) == {"a"}
    thr, n = ebh_threshold(s.to_numpy(), alpha=0.05)
    assert thr == 40.0 and n == 1


def test_null_case_rejects_nothing():
    # All e-values at or below 1 can never reach K/(alpha*k) >= 20/0.05*k.
    s = pd.Series(np.linspace(0.1, 1.0, 100))
    thr, n = ebh_threshold(s.to_numpy(), alpha=0.05)
    assert thr == float("inf") and n == 0


def test_descending_property_invariant_to_order():
    s = pd.Series([30.0, 5.0, 0.5, 0.2], index=["a", "b", "c", "d"])
    set1 = ebh_rejection_set(s, alpha=0.05)
    set2 = ebh_rejection_set(s.iloc[::-1], alpha=0.05)
    assert set1 == set2


def test_nan_handled():
    s = pd.Series([50.0, np.nan, 0.1], index=["a", "b", "c"])
    assert ebh_rejection_set(s, alpha=0.05) == {"a"}
