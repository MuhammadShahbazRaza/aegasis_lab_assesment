import math

from app.config import Settings
from app.domain import NormalizedRecord, Priority, QueryInsight, VisibilityStatus


def normalize_volume(volume: int | None, ceiling: int) -> float:
    """Search volume is heavy-tailed: 200 vs 2,000 monthly searches is a far bigger
    jump in practical value than 100,000 vs 102,000. Log compression keeps the head
    of the distribution from flattening everything else to zero."""
    if not volume or volume <= 0:
        return 0.0
    return min(math.log10(1 + volume) / math.log10(1 + max(ceiling, 10)), 1.0)


def visibility_gap(status: VisibilityStatus, position: int | None) -> float:
    """How much is left to win. Already ranking first means there is almost no
    opportunity left on that query, however attractive its volume looks."""
    if status is VisibilityStatus.NOT_VISIBLE:
        return 1.0
    if status is VisibilityStatus.UNKNOWN:
        return 0.6
    if position is None:
        return 0.5
    if position <= 3:
        return 0.1
    if position <= 10:
        return 0.35
    return 0.7


def opportunity_score(insight: QueryInsight, settings: Settings) -> float:
    volume = normalize_volume(insight.estimated_search_volume, settings.score_volume_ceiling)
    difficulty = 1.0 - (insight.competitive_difficulty / 100.0)
    gap = visibility_gap(insight.visibility_status, insight.visibility_position)

    weighted = (
        settings.score_weight_volume * volume
        + settings.score_weight_difficulty * difficulty
        + settings.score_weight_visibility_gap * gap
    )
    total_weight = (
        settings.score_weight_volume
        + settings.score_weight_difficulty
        + settings.score_weight_visibility_gap
    )
    score = weighted / total_weight if total_weight else 0.0

    # Queries that trigger an AI Overview or get answered by an assistant are the
    # ones this product exists to influence, so they outrank an equivalent blue-link
    # opportunity rather than merely tying with it.
    if insight.ai_surface_present and insight.visibility_status is not VisibilityStatus.VISIBLE:
        score += settings.score_ai_surface_bonus

    return round(max(0.0, min(score, 1.0)), 4)


def priority_for(score: float) -> Priority:
    if score >= 0.66:
        return Priority.HIGH
    if score >= 0.4:
        return Priority.MEDIUM
    return Priority.LOW


def difficulty_from(records: list[NormalizedRecord]) -> int:
    indices = [r.competition_index for r in records if r.competition_index is not None]
    if indices:
        return int(round(sum(indices) / len(indices)))
    # No paid-competition signal available; assume the middle rather than a value
    # that would silently flatter or punish the score.
    return 50
