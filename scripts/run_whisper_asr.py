"""Offline transcription of one WAV into the transcript JSON used by text models.

Stage: data preparation (before launch). Output goes to text_root/<filename>.json,
read by KintsugiTextDataset (DAM 2, DAM 3 phase 1). Not called by the launcher.
"""

import json
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import whisper_timestamped as whisper

from kirad.constants import EXPECTED_SAMPLE_RATE


def run_whisper_asr(
    audio_filename: str | Path,
    output_filename: str | Path,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    """Transcribe a 16 kHz WAV (optional start/end in s) with Whisper large-v3 to JSON.

    Uses whisper_timestamped (word timestamps, VAD, disfluencies), English only.
    Note: raises if the file is not 16 kHz; does not resample.
    Hardcoded: model "large-v3", language="en", beam 5, temperature fallback.
    """
    audio_filename = Path(audio_filename).expanduser().resolve()
    output_dir = Path(output_filename).expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # HARDCODED: ASR model "large-v3"; reloaded on every call.
    model = whisper.load_model("large-v3")

    if start_time is None:
        frame_offset = 0
    else:
        frame_offset = int(start_time * EXPECTED_SAMPLE_RATE)
    if end_time is None:
        num_frames = -1
    else:
        end_frame = int(end_time * EXPECTED_SAMPLE_RATE)
        num_frames = end_frame - frame_offset
    audio, fs = torchaudio.load(
        audio_filename, frame_offset=frame_offset, num_frames=num_frames
    )
    if fs != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"Sampling rate of audio ({fs}) doesn't match `EXPECTED_SAMPLE_RATE` "
            f"({EXPECTED_SAMPLE_RATE})."
        )
    if audio.dim() == 2:
        audio = audio.squeeze()

    result = whisper.transcribe(
        model,
        audio,
        # HARDCODED: English only; change for Congo/Colombia/Jordan/Ukraine data.
        language="en",
        task="transcribe",
        vad=True,
        detect_disfluencies=True,
        # HARDCODED: decoding params (beam 5, best_of 5, temperature fallback).
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    with open(output_filename, "w") as jsonfile:
        json.dump(result, jsonfile, indent=2, ensure_ascii=False)
