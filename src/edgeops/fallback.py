"""Fallback helpers (Chapter 19).

When confidence is low or the model errors out, the system must do *something* —
the wrong default is silently passing the bad prediction through. These helpers
implement the standard fallbacks the course recommends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FallbackAction:
    """What the system does when the model is not trusted."""
    decision: str                 # 'use_prediction' | 'rule_based' | 'human_review' | 'safe_stop'
    reason: str                   # short human-readable
    suggested_action: Any | None  # the action to take instead of the model's


def confidence_fallback(
    prediction: Any,
    confidence: float,
    *,
    min_confidence: float = 0.5,
    rule_based_default: Any | None = None,
    require_human_below: float | None = 0.3,
) -> FallbackAction:
    """Decide what to do based on the model's confidence.

    - confidence >= min_confidence: use the model prediction.
    - require_human_below set and confidence < require_human_below: send to human review.
    - else: use the rule_based_default (e.g. "do nothing", "go to safe state").
    """
    if confidence >= min_confidence:
        return FallbackAction(
            decision="use_prediction",
            reason=f"confidence({confidence:.2f}) >= min({min_confidence:.2f})",
            suggested_action=prediction,
        )
    if require_human_below is not None and confidence < require_human_below:
        return FallbackAction(
            decision="human_review",
            reason=f"confidence({confidence:.2f}) < require_human({require_human_below:.2f})",
            suggested_action=None,
        )
    return FallbackAction(
        decision="rule_based",
        reason=f"confidence({confidence:.2f}) below threshold; using rule-based default",
        suggested_action=rule_based_default,
    )
