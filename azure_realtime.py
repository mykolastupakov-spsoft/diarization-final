import json
import logging
import os
import threading
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover
    speechsdk = None


def _ticks_to_seconds(value: int) -> float:
    if not value:
        return 0.0
    return value / 10_000_000.0  # 100-nanosecond units


class AzureRealtimeDiarization:
    """Real-time transcription with speaker diarization using Azure Speech SDK."""

    def __init__(self, subscription_key: str, region: str):
        if speechsdk is None:
            raise ImportError(
                "azure-cognitiveservices-speech is required for Azure real-time STT. "
                "Install it via pip or remove the real-time engine option."
            )
        if not subscription_key or not region:
            raise ValueError("Azure Speech subscription key and region are required")

        self.subscription_key = subscription_key
        self.region = region
        self.segments: List[Dict[str, object]] = []
        self.raw_messages: List[Dict[str, object]] = []
        self.speaker_map: Dict[str, str] = {}

    def transcribe_from_file(self, audio_file_path: str, language: str = "en-US") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        if not audio_file_path or not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Reset run-specific buffers
        self.segments = []
        self.raw_messages = []
        self.speaker_map = {}

        speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )
        speech_config.speech_recognition_language = language
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
        transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config
        )

        done = threading.Event()
        error_holder: Dict[str, str] = {}

        def map_speaker(speaker_id: str) -> str:
            if speaker_id not in self.speaker_map:
                # Map Azure speaker IDs to consistent SPEAKER_XX labels
                # Azure uses "Guest-1", "Guest-2", etc. - extract number and map to SPEAKER_01, SPEAKER_02
                if speaker_id.startswith("Guest-"):
                    try:
                        guest_num = int(speaker_id.split("-")[1])
                        # Map Guest-1 -> SPEAKER_01, Guest-2 -> SPEAKER_02, etc.
                        self.speaker_map[speaker_id] = f"SPEAKER_{guest_num:02d}"
                    except (ValueError, IndexError):
                        # Fallback: use order of appearance
                        self.speaker_map[speaker_id] = f"SPEAKER_{len(self.speaker_map) + 1:02d}"
                else:
                    # For other speaker IDs, use order of appearance
                    self.speaker_map[speaker_id] = f"SPEAKER_{len(self.speaker_map) + 1:02d}"
            return self.speaker_map[speaker_id]

        def handle_transcribed(evt: Any):
            result = evt.result
            try:
                if result.reason == speechsdk.ResultReason.RecognizedSpeech and result.text:
                    logger.info("[Azure Realtime] Final: %s", result.text[:80])
                    speaker_id = result.speaker_id or getattr(result, "speakerId", None) or "Unknown"
                    speaker_label = map_speaker(str(speaker_id))
                    start = _ticks_to_seconds(result.offset)
                    end = start + _ticks_to_seconds(result.duration)

                    segment = {
                        "speaker": speaker_label,
                        "text": result.text,
                        "start": start,
                        "end": end,
                        "words": [],
                        "confidence": 0.9,
                        "pauses": []
                    }
                    self.segments.append(segment)
                    self.raw_messages.append({
                        "type": "final",
                        "speakerId": speaker_id,
                        "text": result.text,
                        "offset": start,
                        "duration": end - start
                    })
                    logger.debug("[Azure Realtime] %s: %s", speaker_label, result.text)
                else:
                    logger.debug(
                        "[Azure Realtime] Transcribed event ignored (reason=%s, text_len=%s)",
                        getattr(result, "reason", "unknown"),
                        len(result.text or ""),
                    )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Error in handle_transcribed: %s", exc)

        def handle_transcribing(evt: Any):
            result = evt.result
            if result.reason in (
                speechsdk.ResultReason.RecognizingSpeech,
                speechsdk.ResultReason.TranscribingSpeech,
            ) and result.text:
                logger.debug("[Azure Realtime] Interim: %s", result.text[:80])
                self.raw_messages.append({
                    "type": "intermediate",
                    "speakerId": result.speaker_id or getattr(result, "speakerId", None) or "Unknown",
                    "text": result.text
                })
            else:
                if result.reason != speechsdk.ResultReason.NoMatch:
                    logger.debug(
                        "[Azure Realtime] Transcribing event skipped (reason=%s, text=%s)",
                        getattr(result, "reason", "unknown"),
                        result.text,
                    )

        def handle_canceled(evt: Any):
            details = getattr(evt, "cancellation_details", None)
            if details:
                logger.warning(
                    "[Azure Realtime] Cancellation details: reason=%s, code=%s, error=%s",
                    getattr(details, "reason", None),
                    getattr(details, "error_code", None),
                    getattr(details, "error_details", None),
                )
            logger.warning(
                "[Azure Realtime] Canceled event: reason=%s, error=%s, attrs=%s",
                getattr(evt, "reason", None),
                getattr(evt, "error_details", None),
                dir(evt),
            )
            if evt.reason == speechsdk.CancellationReason.Error:
                error_holder["message"] = evt.error_details or "Azure real-time transcription canceled"
                logger.error("[Azure Realtime] ❌ %s", error_holder["message"])
            if not done.is_set():
                done.set()

        def handle_session_stopped(evt: Any):
            logger.info("[Azure Realtime] Session stopped event received")
            if not done.is_set():
                done.set()

        # Store handlers on the instance to avoid garbage-collection.
        self._handle_transcribed = handle_transcribed
        self._handle_transcribing = handle_transcribing
        self._handle_canceled = handle_canceled
        self._handle_session_stopped = handle_session_stopped

        transcriber.transcribed.connect(self._handle_transcribed)
        transcriber.transcribing.connect(self._handle_transcribing)
        transcriber.canceled.connect(self._handle_canceled)
        transcriber.session_stopped.connect(self._handle_session_stopped)

        logger.info("[Azure Realtime] Starting transcription for %s", os.path.basename(audio_file_path))
        transcriber.start_transcribing_async()
        done.wait()
        try:
            stop_future = transcriber.stop_transcribing_async()
            if stop_future:
                stop_future.get()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

        if error_holder.get("message"):
            raise RuntimeError(error_holder["message"])

        raw_payload = {
            "service": "azure_realtime",
            "language": language,
            "segments": self.raw_messages,
            "speakerMap": self.speaker_map
        }
        return self.segments, raw_payload

