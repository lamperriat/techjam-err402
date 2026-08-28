"""Immutable boundary for the post-selection Architecture Lab winner gates."""

from __future__ import annotations


FROZEN_WINNER_ID = "R08.coverage_cascade"
SELECTION_COMMIT = "e5d0d4966d01da9932d835cb3a754475b6fa13e2"
SELECTION_CORPUS_SHA256 = (
    "38c6a9fedd4a3e02d8f581e2d04d8467203d7275c3ff0eb691a57f5025c010ae"
)
SELECTION_RESULT_SHA256 = (
    "bedf4c8048186a9ca9d64a64fb9a8ee7184c5810ff13e5e69e138f15faa5e177"
)


def validate_frozen_winner_configuration(
    architecture_variant: str | None,
    *,
    question_policy: str,
    rerank_mode: str,
) -> None:
    """Reject post-selection combinations that were not frozen by the matrix."""

    if architecture_variant not in {None, FROZEN_WINNER_ID}:
        raise ValueError(
            f"post-selection gates only allow the frozen winner {FROZEN_WINNER_ID}"
        )
    if architecture_variant and question_policy != "fast":
        raise ValueError("the frozen winner requires question_policy=fast")
    if architecture_variant and rerank_mode != "off":
        raise ValueError("the frozen winner requires rerank_mode=off")
