"""Reproducibility is a documented invariant (project-brief.md, README): the
same SEED must reproduce byte-identical output. This is the regression test
for that promise, not a test of the generator's business logic."""

import numpy as np
import pandas as pd

from data_gen.config import SEED
from data_gen.generate import generate_clean_reports, generate_facilities, inject_issues


def _run_once():
    rng = np.random.default_rng(SEED)
    facilities = generate_facilities(rng)
    clean_reports = generate_clean_reports(facilities, rng)
    raw_reports, ground_truth = inject_issues(clean_reports, rng)
    return facilities, raw_reports, ground_truth


def test_same_seed_produces_identical_output():
    facilities_a, raw_a, ground_truth_a = _run_once()
    facilities_b, raw_b, ground_truth_b = _run_once()

    pd.testing.assert_frame_equal(facilities_a, facilities_b)
    pd.testing.assert_frame_equal(raw_a, raw_b)
    pd.testing.assert_frame_equal(ground_truth_a, ground_truth_b)


def test_ground_truth_ledger_only_references_generated_facilities():
    facilities, _, ground_truth = _run_once()
    known_ids = set(facilities["facility_id"])
    assert set(ground_truth["facility_id"]).issubset(known_ids)
