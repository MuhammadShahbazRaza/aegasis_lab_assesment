import pytest

from app.domain import QueryInsight, VisibilityStatus
from app.scoring import (
    difficulty_from,
    normalize_volume,
    opportunity_score,
    priority_for,
    visibility_gap,
)


def insight(**overrides) -> QueryInsight:
    base = {
        "query_uuid": "q1",
        "query_text": "best crm",
        "estimated_search_volume": 5000,
        "competitive_difficulty": 50,
        "visibility_status": VisibilityStatus.NOT_VISIBLE,
        "domain_visible": False,
        "visibility_position": None,
        "ai_surface_present": False,
    }
    return QueryInsight(**{**base, **overrides})


def test_volume_normalization_is_monotonic_and_bounded():
    values = [normalize_volume(v, 50_000) for v in (0, 10, 100, 5_000, 50_000, 500_000)]
    assert values[0] == 0.0
    assert values == sorted(values)
    assert values[-1] == 1.0


def test_log_compression_keeps_small_volumes_meaningful():
    ceiling = 50_000
    # Linear normalization would score a 100-search query at 0.002 and bury it under
    # every head term; the point of the log is that it still lands above 0.4.
    assert 100 / ceiling < 0.01
    assert normalize_volume(100, ceiling) > 0.4

    # A doubling is worth roughly the same wherever it happens, instead of being
    # dominated by whatever the largest query in the set happens to be.
    small = normalize_volume(200, ceiling) - normalize_volume(100, ceiling)
    large = normalize_volume(40_000, ceiling) - normalize_volume(20_000, ceiling)
    assert small == pytest.approx(large, rel=0.02)


@pytest.mark.parametrize(
    ("status", "position", "expected"),
    [
        (VisibilityStatus.NOT_VISIBLE, None, 1.0),
        (VisibilityStatus.UNKNOWN, None, 0.6),
        (VisibilityStatus.VISIBLE, 1, 0.1),
        (VisibilityStatus.VISIBLE, 7, 0.35),
        (VisibilityStatus.VISIBLE, 40, 0.7),
    ],
)
def test_visibility_gap_shrinks_as_the_position_improves(status, position, expected):
    assert visibility_gap(status, position) == expected


def test_absent_beats_ranked_when_everything_else_matches(settings):
    absent = opportunity_score(insight(), settings)
    ranked = opportunity_score(
        insight(visibility_status=VisibilityStatus.VISIBLE, domain_visible=True,
                visibility_position=2),
        settings,
    )
    assert absent > ranked


def test_easier_queries_score_higher(settings):
    easy = opportunity_score(insight(competitive_difficulty=10), settings)
    hard = opportunity_score(insight(competitive_difficulty=90), settings)
    assert easy > hard


def test_higher_volume_scores_higher(settings):
    assert opportunity_score(insight(estimated_search_volume=40_000), settings) > opportunity_score(
        insight(estimated_search_volume=200), settings
    )


def test_ai_surface_bonus_applies_only_where_the_brand_is_missing(settings):
    plain = opportunity_score(insight(), settings)
    with_ai = opportunity_score(insight(ai_surface_present=True), settings)
    assert with_ai == pytest.approx(
        min(plain + settings.score_ai_surface_bonus, 1.0), abs=1e-4
    )

    visible = {
        "visibility_status": VisibilityStatus.VISIBLE,
        "domain_visible": True,
        "visibility_position": 2,
    }
    assert opportunity_score(insight(**visible, ai_surface_present=True), settings) == (
        opportunity_score(insight(**visible), settings)
    )


def test_score_stays_inside_the_unit_interval(settings):
    extreme = insight(
        estimated_search_volume=10_000_000, competitive_difficulty=0, ai_surface_present=True
    )
    assert 0.0 <= opportunity_score(extreme, settings) <= 1.0
    floor = insight(
        estimated_search_volume=0,
        competitive_difficulty=100,
        visibility_status=VisibilityStatus.VISIBLE,
        domain_visible=True,
        visibility_position=1,
    )
    assert opportunity_score(floor, settings) >= 0.0


def test_weights_are_configurable(settings):
    settings.score_weight_visibility_gap = 0.0
    settings.score_weight_volume = 1.0
    settings.score_weight_difficulty = 0.0
    scored = opportunity_score(insight(estimated_search_volume=50_000), settings)
    assert scored == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    ("score", "expected"), [(0.9, "high"), (0.66, "high"), (0.5, "medium"), (0.1, "low")]
)
def test_priority_bands(score, expected):
    assert priority_for(score).value == expected


def test_difficulty_defaults_to_the_midpoint_without_a_signal():
    assert difficulty_from([]) == 50
