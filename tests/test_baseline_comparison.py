"""Tests for the prespecified GRU-versus-persistence decision rule."""

from scripts.compare_baselines import classify_result


def test_category_a_requires_at_least_point_two_improvement_across_cases() -> None:
    assert classify_result(-0.2, improved_case_count=8)[0] == "A"
    assert classify_result(-0.3, improved_case_count=2)[0] == "C"


def test_categories_b_and_c_follow_operational_mae_threshold() -> None:
    assert classify_result(0.2, improved_case_count=7)[0] == "B"
    assert classify_result(0.2001, improved_case_count=7)[0] == "C"

