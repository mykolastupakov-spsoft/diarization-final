#!/usr/bin/env python3
"""
SpeechBrain-based speaker separation for overlap diarization (Mode 3).

This script mirrors the JSON contract of pyannote_separation.py so that
the Node.js backend can swap separation engines without changing the
rest of the pipeline.

Usage:
    python speechbrain_separation.py <audio_path> <output_dir>
"""

import sys
import os
import json
import tempfile
from pathlib import Path

import torch
import pyannote_patch  # noqa: F401  # ensures torchaudio compatibility on Python 3.14+
import torchaudio
import soundfile as sf
from speechbrain.inference.separation import (
    SepformerSeparation as Separator,
)


def log_error(message):
    print(message, file=sys.stderr)


def ensure_output_dir(path: str) -> str:
    if not path:
        return tempfile.mkdtemp(prefix="speechbrain_separation_")
    os.makedirs(path, exist_ok=True)
    return path


def get_device():
    env_device = os.getenv("SPEECHBRAIN_DEVICE")
    if env_device:
        return env_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_waveform(audio_path: str):
    """
    Loads audio as torch tensor (shape [1, samples]) and sample rate.
    Prefers soundfile to avoid torchcodec dependency, falls back to torchaudio.
    """
    try:
        data, sample_rate = sf.read(audio_path, dtype='float32', always_2d=True)
        waveform = torch.from_numpy(data.T)
        log_error(f"[SpeechBrain] Loaded via soundfile: shape={waveform.shape}, sr={sample_rate}")
        return waveform, sample_rate
    except Exception as sf_error:
        log_error(f"[SpeechBrain] soundfile load failed ({sf_error}), falling back to torchaudio")
        waveform, sample_rate = torchaudio.load(audio_path)
        return waveform, sample_rate


def separate(audio_path: str, output_dir: str):
    device = get_device()
    cache_dir = os.path.expanduser(
        os.getenv("SPEECHBRAIN_CACHE_DIR", "~/.cache/speechbrain/sepformer-wsj02mix")
    )

    log_error(f"[SpeechBrain] Using device: {device}")
    log_error(f"[SpeechBrain] Cache dir: {cache_dir}")

    model = Separator.from_hparams(
        source="speechbrain/sepformer-wsj02mix",
        savedir=cache_dir,
        run_opts={"device": device},
    )

    waveform, sample_rate = load_waveform(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 8000:
        log_error(f"[SpeechBrain] Resampling from {sample_rate}Hz to 8000Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, 8000)
        waveform = resampler(waveform)
        sample_rate = 8000

    total_samples = waveform.shape[1]
    max_chunk_seconds = float(os.getenv("SPEECHBRAIN_CHUNK_SECONDS", "30"))
    max_chunk_samples = int(max_chunk_seconds * sample_rate)
    max_chunk_samples = max(max_chunk_samples, sample_rate * 5)  # ensure at least 5 seconds

    log_error(f"[SpeechBrain] Waveform shape before separation: {waveform.shape}, dtype={waveform.dtype}, chunk={max_chunk_samples} samples")

    def separate_chunk(chunk_tensor: torch.Tensor):
        chunk_tensor = chunk_tensor.to(device)
        with torch.no_grad():
            result = model.separate_batch(chunk_tensor)
        return result.cpu()

    if total_samples > max_chunk_samples:
        log_error(f"[SpeechBrain] Processing in chunks (total samples {total_samples})")
        chunk_outputs = []
        for start in range(0, total_samples, max_chunk_samples):
            end = min(start + max_chunk_samples, total_samples)
            log_error(f"[SpeechBrain] Separating chunk {start}:{end}")
            chunk = waveform[:, start:end]
            chunk_outputs.append(separate_chunk(chunk))
        est_sources = torch.cat(chunk_outputs, dim=1)
    else:
        est_sources = separate_chunk(waveform)

    if est_sources.dim() == 3:
        est_sources = est_sources[0]  # [time, num_speakers]

    if est_sources.dim() == 2:
        if est_sources.shape[0] == model.hparams.num_spks:
            # shape [num_speakers, time]
            sources_tensor = est_sources
        elif est_sources.shape[1] == model.hparams.num_spks:
            sources_tensor = est_sources.transpose(0, 1)
        else:
            raise ValueError(f"Unexpected est_sources shape: {est_sources.shape}")
    else:
        raise ValueError(f"Unsupported est_sources dimension: {est_sources.dim()}")

    if est_sources is None or len(est_sources) == 0:
        return {
            "success": False,
            "error": "No speakers detected in audio",
        }

    speakers = []
    timeline = []

    for idx, source in enumerate(sources_tensor):
        speaker_name = f"SPEAKER_{idx:02d}"
        output_path = os.path.join(output_dir, f"{speaker_name}.wav")

        source_np = source.squeeze().numpy()
        sf.write(output_path, source_np, sample_rate)

        speakers.append(
            {
                "name": speaker_name,
                "format": "wav",
                "local_path": output_path,
                "isBackground": False,
            }
        )

        duration = source.shape[-1] / sample_rate
        timeline.append(
            {
                "speaker": speaker_name,
                "start": 0.0,
                "end": round(duration, 2),
                "duration": round(duration, 2),
            }
        )

    return {
        "success": True,
        "speakers": speakers,
        "timeline": timeline,
        "output_dir": output_dir,
        "num_speakers": len(speakers),
    }


def main():
    if len(sys.argv) < 3:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": "Usage: python speechbrain_separation.py <audio_path> <output_dir>",
                }
            )
        )
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = ensure_output_dir(sys.argv[2])

    if not os.path.isfile(audio_path):
        print(
            json.dumps(
                {"success": False, "error": f"Audio file not found: {audio_path}"}
            )
        )
        sys.exit(1)

    try:
        result = separate(audio_path, output_dir)
        print(json.dumps(result))
        sys.exit(0 if result.get("success") else 1)
    except Exception as exc:
        log_error(f"[SpeechBrain] Error: {exc}")
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

