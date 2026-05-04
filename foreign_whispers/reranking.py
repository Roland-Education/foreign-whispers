"""Deterministic failure analysis and translation re-ranking.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function (``get_shorter_translations``)
implements rule-based Spanish shortening: verbose-phrase substitution,
adverbial filler removal, and discourse-marker pruning.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


# ──────────────────────────────────────────────────────────────────────
# Spanish shortening rules used by get_shorter_translations.
#
# Three groups, applied in increasing aggressiveness:
#
#   _PHRASE_SUBS    — verbose constructions with concise equivalents.
#                     Safe; preserves meaning.
#   _FILLER_ADVERBS — common discourse adverbs that carry little
#                     semantic weight.  Removable in most contexts.
#   _DISCOURSE_MARKERS — connectors that can be dropped at sentence
#                        starts without losing meaning ("Bueno,", "Pues,").
#
# All patterns use word boundaries to avoid mid-word matches.
# ──────────────────────────────────────────────────────────────────────

_PHRASE_SUBS: list[tuple[str, str]] = [
    (r"\ben este momento\b", "ahora"),
    (r"\ben estos momentos\b", "ahora"),
    (r"\ba pesar de que\b", "aunque"),
    (r"\bcon el fin de\b", "para"),
    (r"\bcon el objetivo de\b", "para"),
    (r"\bcon la finalidad de\b", "para"),
    (r"\bcon el propósito de\b", "para"),
    (r"\ben relación con\b", "sobre"),
    (r"\ben relación a\b", "sobre"),
    (r"\ben lo que respecta a\b", "sobre"),
    (r"\bpor lo tanto\b", "así"),
    (r"\bpor consiguiente\b", "así"),
    (r"\bdebido a que\b", "porque"),
    (r"\bdado que\b", "porque"),
    (r"\bpuesto que\b", "porque"),
    (r"\ben caso de que\b", "si"),
    (r"\ben tanto que\b", "mientras"),
    (r"\bes posible que\b", "puede que"),
    (r"\bes necesario\b", "hay que"),
    (r"\bes importante\b", "hay que"),
    (r"\btener la oportunidad de\b", "poder"),
    (r"\bllevar a cabo\b", "hacer"),
    (r"\bdar lugar a\b", "causar"),
    (r"\btomar la decisión de\b", "decidir"),
    (r"\bhacer una pregunta\b", "preguntar"),
    (r"\bhacer referencia a\b", "referirse a"),
    (r"\bponer en marcha\b", "iniciar"),
    (r"\btener en cuenta\b", "considerar"),
    (r"\bsin embargo\b", "pero"),
    (r"\bno obstante\b", "pero"),
    (r"\bde manera que\b", "para que"),
    (r"\bde modo que\b", "para que"),
    (r"\ba través de\b", "por"),
    (r"\bpor medio de\b", "por"),
]

_FILLER_ADVERBS: list[str] = [
    r"\brealmente\b",
    r"\bverdaderamente\b",
    r"\befectivamente\b",
    r"\bbásicamente\b",
    r"\besencialmente\b",
    r"\bobviamente\b",
    r"\bevidentemente\b",
    r"\bclaramente\b",
    r"\bgeneralmente\b",
    r"\bnormalmente\b",
    r"\babsolutamente\b",
    r"\bcompletamente\b",
    r"\btotalmente\b",
    r"\bperfectamente\b",
    r"\bsimplemente\b",
]

_DISCOURSE_MARKERS: list[str] = [
    r"^bueno,\s*",
    r"^pues,\s*",
    r"^entonces,\s*",
    r"^así que\s*",
    r"^o sea,\s*",
    r"^es decir,\s*",
    r"^en otras palabras,\s*",
]


def _apply_phrase_subs(text: str) -> str:
    """Apply each verbose-to-concise phrase substitution once."""
    out = text
    for pattern, replacement in _PHRASE_SUBS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return _collapse_spaces(out)


def _remove_filler_adverbs(text: str) -> str:
    """Strip common Spanish filler adverbs."""
    out = text
    for pattern in _FILLER_ADVERBS:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE)
    return _collapse_spaces(out)


def _strip_discourse_markers(text: str) -> str:
    """Remove leading discourse markers ('Bueno,', 'Pues,', etc.)."""
    out = text
    for pattern in _DISCOURSE_MARKERS:
        new = re.sub(pattern, "", out, flags=re.IGNORECASE)
        if new != out:
            # Re-capitalise the first letter that survived
            out = new[:1].upper() + new[1:] if new else new
    return _collapse_spaces(out)


def _collapse_spaces(text: str) -> str:
    """Collapse runs of whitespace and trim, while preserving punctuation."""
    # Remove space before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Collapse multiple spaces and strip
    return re.sub(r"\s+", " ", text).strip()


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    Uses rule-based Spanish shortening in three progressively aggressive
    passes:

    1. **Phrase substitution** — replace verbose constructions with concise
       equivalents (e.g. "en este momento" → "ahora", "con el fin de" → "para").
       Meaning-preserving.
    2. **Filler-adverb removal** — drop common discourse adverbs that carry
       little semantic weight (e.g. "realmente", "básicamente", "obviamente").
       Mild meaning loss in some contexts.
    3. **Discourse-marker stripping** — remove leading markers like
       "Bueno,", "Pues,", "Es decir,". Most aggressive.

    Each successful pass produces one ``TranslationCandidate``.  Candidates
    are returned sorted shortest first so the caller can pick the most
    aggressive option that still fits the budget.

    Returns an empty list when:

    - ``baseline_es`` is empty
    - ``baseline_es`` already fits within ``target_duration_s * 15`` chars
    - no rule produces anything shorter than the baseline

    Args:
        source_text: Original source-language text (currently unused; kept
            for interface stability and future paraphrase-from-source
            backends).
        baseline_es: Baseline target-language translation from argostranslate.
        target_duration_s: Time budget in seconds for this segment.
        context_prev: Preceding segment text (currently unused).
        context_next: Following segment text (currently unused).

    Returns:
        List of ``TranslationCandidate`` objects, sorted shortest first.
    """
    if not baseline_es or not baseline_es.strip():
        return []

    budget_chars = max(1, int(target_duration_s * 15))
    baseline = baseline_es.strip()

    # Already fits — caller will use the baseline.
    if len(baseline) <= budget_chars:
        return []

    candidates: list[TranslationCandidate] = []
    seen: set[str] = set()

    def _emit(text: str, rationale: str) -> None:
        cleaned = _collapse_spaces(text)
        if not cleaned or cleaned == baseline or cleaned in seen:
            return
        if len(cleaned) >= len(baseline):
            # Only keep candidates strictly shorter than the baseline.
            return
        seen.add(cleaned)
        candidates.append(TranslationCandidate(
            text=cleaned,
            char_count=len(cleaned),
            brevity_rationale=rationale,
        ))

    # Pass 1: phrase substitutions only.
    pass1 = _apply_phrase_subs(baseline)
    _emit(pass1, "verbose phrases replaced with concise equivalents")

    # Pass 2: phrase subs + filler-adverb removal.
    pass2 = _remove_filler_adverbs(pass1)
    _emit(pass2, "phrase substitution + filler adverbs removed")

    # Pass 3: pass 2 + leading discourse markers stripped.
    pass3 = _strip_discourse_markers(pass2)
    _emit(pass3, "phrase subs + filler removal + discourse markers dropped")

    candidates.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: baseline=%d chars, budget=%d chars (%.1fs * 15), "
        "produced %d candidate(s)",
        len(baseline), budget_chars, target_duration_s, len(candidates),
    )
    return candidates
