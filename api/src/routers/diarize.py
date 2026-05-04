"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Resolve title from video_id (404 if missing)
    2. Return cached diarization if already computed
    3. Extract 16 kHz mono PCM audio from the video via ffmpeg
    4. Run pyannote speaker diarization (offloaded to a thread)
    5. Cache speakers + segments JSON
    6. Merge speaker labels into the transcript JSON so downstream
       stages (translate, TTS) know which speaker said what
    7. Return the response
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    # Step 3: Extract audio from video
    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video file {video_path.name} not found — run download stage first",
        )
    audio_path = diar_dir / f"{title}.wav"
    proc = subprocess.run(
        [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            "-y", str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg audio extraction failed: {proc.stderr[-500:]}",
        )

    # Step 4: Run diarization off the event loop
    diar_segments = await asyncio.to_thread(
        _alignment_service.diarize, str(audio_path)
    )

    # Step 5: Extract unique speaker labels
    speakers = sorted({s["speaker"] for s in diar_segments})

    # Step 6: Cache the result
    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result))

    # Step 7: Merge speaker labels into the transcript JSON
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if transcript_path.exists() and diar_segments:
        try:
            transcript = json.loads(transcript_path.read_text())
            labeled = assign_speakers(transcript.get("segments", []), diar_segments)
            transcript["segments"] = labeled
            transcript_path.write_text(json.dumps(transcript))
        except Exception as exc:  # noqa: BLE001 — non-fatal merge failure
            # Diarization itself succeeded; merge is best-effort
            print(f"transcript merge failed for {title}: {exc}")

    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
    )
