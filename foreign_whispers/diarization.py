"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []


# ──────────────────────────────────────────────────────────────────────
# Segment-speaker merge
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_SPEAKER = "SPEAKER_00"


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Algorithm: for each segment with start ``s`` and end ``e``, and each
    diarization interval with start ``ds`` and end ``de``, the overlap
    is ``max(0, min(e, de) - max(s, ds))``. The interval with the
    largest overlap wins; ties go to the first (earlier) interval.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated; each segment is shallow-copied.
    """
    labeled: list[dict] = []
    for seg in segments:
        copy = dict(seg)
        copy["speaker"] = _best_speaker(seg["start"], seg["end"], diarization)
        labeled.append(copy)
    return labeled


def _best_speaker(seg_start: float, seg_end: float, diarization: list[dict]) -> str:
    """Return the speaker label whose diarization interval overlaps most with [seg_start, seg_end]."""
    if not diarization:
        return _DEFAULT_SPEAKER
    best_overlap = 0.0
    best_speaker = _DEFAULT_SPEAKER
    for interval in diarization:
        ds = interval["start_s"]
        de = interval["end_s"]
        overlap = max(0.0, min(seg_end, de) - max(seg_start, ds))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = interval["speaker"]
    return best_speaker
