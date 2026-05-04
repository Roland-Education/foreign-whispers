"""Clip-level alignment quality metrics.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M8-align).
Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


# ──────────────────────────────────────────────────────────────────────
# Multi-dimensional dubbing quality scorecard
# ──────────────────────────────────────────────────────────────────────


# Default normalisation thresholds. A clip with mean_abs_duration_error
# of MAX_TIMING_ERROR_S or worse scores 0 on timing_accuracy.
_MAX_TIMING_ERROR_S = 1.0
_MAX_DRIFT_S = 5.0
_MAX_RATE_VARIANCE = 8.0  # syllables/second variance — anything beyond this is incoherent

# Weights for the overall score (must sum to 1 for the dimensions actually used)
_DIMENSION_WEIGHTS = {
    "timing_accuracy":   0.30,
    "coverage":          0.25,
    "drift_control":     0.15,
    "naturalness":       0.15,
    "intelligibility":   0.075,  # optional; redistributed if absent
    "semantic_fidelity": 0.075,  # optional; redistributed if absent
}


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
    intelligibility_wer: float | None = None,
    semantic_similarity: float | None = None,
) -> dict:
    """Multi-dimensional quality scorecard for a dubbed clip.

    Each dimension is normalised to [0, 1] where 1 is best:

    - **timing_accuracy** — mean per-segment overflow (predicted TTS minus
      source window) as a fraction of an acceptable threshold. Underflow
      (TTS shorter than window) does not penalise; the dub can pad silence.
    - **coverage** — fraction of segments resolved without a severe action.
      FAIL and REQUEST_SHORTER count as failures; segments stretched
      beyond 1.3x count as half-failures.
    - **drift_control** — penalises end-of-clip cumulative drift.
    - **naturalness** — measures variance in segment-level speaking rate.
    - **intelligibility** *(optional)* — supplied externally as the word
      error rate of a TTS-then-STT round-trip.
    - **semantic_fidelity** *(optional)* — supplied externally as the
      cosine similarity between source and back-translated embeddings.

    Returns:
        Dict with keys per dimension plus ``overall`` (weighted average).
        Each value is a float in [0, 1].
    """
    if not metrics or not aligned:
        return _zero_scorecard()

    report = align_report if align_report is not None else clip_evaluation_report(metrics, aligned)

    scores = {
        "timing_accuracy":   _score_timing_accuracy(metrics),
        "coverage":          _score_coverage(aligned),
        "drift_control":     _score_drift_control(report),
        "naturalness":       _score_naturalness(metrics, aligned),
    }

    if intelligibility_wer is not None:
        scores["intelligibility"] = _clip_unit(1.0 - float(intelligibility_wer))
    if semantic_similarity is not None:
        scores["semantic_fidelity"] = _clip_unit((float(semantic_similarity) + 1.0) / 2.0)

    scores["overall"] = round(_weighted_overall(scores), 3)
    return {k: round(v, 3) for k, v in scores.items()}


def _zero_scorecard() -> dict:
    return {
        "timing_accuracy":   0.0,
        "coverage":          0.0,
        "drift_control":     0.0,
        "naturalness":       0.0,
        "overall":           0.0,
    }


def _score_timing_accuracy(metrics: list[SegmentMetrics]) -> float:
    """Mean per-segment overflow as a [0,1] score (1 = no overflow).

    Underflow (predicted TTS shorter than window) doesn't penalise — the
    dub just pads silence. Only overflow degrades timing quality.
    """
    if not metrics:
        return 1.0
    overflows = [m.overflow_s for m in metrics]
    mean_overflow = sum(overflows) / len(overflows)
    return _clip_unit(1.0 - mean_overflow / _MAX_TIMING_ERROR_S)


def _score_coverage(aligned: list[AlignedSegment]) -> float:
    """Fraction of segments resolved without a severe action.

    Looks at ``AlignedSegment.action`` rather than the upstream report,
    which under-counts failures because greedy alignment leaves
    REQUEST_SHORTER and FAIL segments at stretch_factor=1.0.
    """
    if not aligned:
        return 1.0
    n_total = len(aligned)
    n_bad = sum(
        1 for a in aligned
        if a.action in (AlignAction.REQUEST_SHORTER, AlignAction.FAIL)
    )
    n_severe = sum(1 for a in aligned if a.stretch_factor > 1.3)
    bad_total = n_bad + 0.5 * n_severe
    return _clip_unit(1.0 - bad_total / n_total)


def _score_drift_control(report: dict) -> float:
    drift = abs(float(report.get("total_cumulative_drift_s", 0.0)))
    return _clip_unit(1.0 - drift / _MAX_DRIFT_S)


def _score_naturalness(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> float:
    """Score speaking-rate consistency across segments.

    For each scheduled segment we compute the effective speaking rate
    (syllables / scheduled_duration). A natural clip has low variance
    across segments. We score 1 when variance is zero and 0 when
    variance reaches ``_MAX_RATE_VARIANCE``.
    """
    if len(aligned) < 2:
        return 1.0
    rates = []
    for m, a in zip(metrics, aligned):
        scheduled_dur = a.scheduled_end - a.scheduled_start
        if scheduled_dur <= 0:
            continue
        # Reuse predicted_tts_s as a proxy for syllable content
        syl_proxy = m.predicted_tts_s * 4.5  # syllables = predicted_tts_s * rate
        rate = syl_proxy / scheduled_dur
        rates.append(rate)
    if len(rates) < 2:
        return 1.0
    variance = _stats.pvariance(rates)
    return _clip_unit(1.0 - variance / _MAX_RATE_VARIANCE)


def _weighted_overall(scores: dict) -> float:
    """Compute weighted overall score, redistributing weight from missing dimensions."""
    used = {k: scores[k] for k in _DIMENSION_WEIGHTS if k in scores}
    if not used:
        return 0.0
    weight_sum = sum(_DIMENSION_WEIGHTS[k] for k in used)
    weighted = sum(scores[k] * _DIMENSION_WEIGHTS[k] for k in used)
    return weighted / weight_sum if weight_sum > 0 else 0.0


def _clip_unit(value: float) -> float:
    """Clamp a value to the [0, 1] range."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
