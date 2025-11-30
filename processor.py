import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from audioshake_client import AudioShakeClient, AudioShakeError


logger = logging.getLogger(__name__)


@dataclass
class TrackResult:
    track_id: str
    role: str
    confidence: float
    summary: str
    transcript: str
    audio_path: Path

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["audio_path"] = str(self.audio_path)
        payload["file_name"] = self.audio_path.name
        return payload


class SpeechToTextService:
    """Placeholder for Speech-to-Text. Transcription is handled by Speechmatics in Node.js."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mock",
        language: Optional[str] = None,
    ) -> None:
        logger.warning("SpeechToTextService is a mock. Transcription is expected from Speechmatics.")
        self.model = model
        self.language = language

    def transcribe(self, audio_path: Path) -> str:
        """Returns a placeholder transcript. Real transcription is done by Speechmatics in Node.js."""
        return f"[Speechmatics transcription for {audio_path.name}]"


class SpeakerRoleClassifier:
    """Uses OpenRouter LLM to classify whether a speaker is an Agent or Client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        app_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        # Використовуємо fast модель за замовчуванням
        self.model = model or os.getenv("FAST_MODEL_ID") or "gpt-oss-120b"
        self.app_url = app_url or os.getenv("APP_URL") or "http://localhost:3000"
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            logger.warning(
                "OPENROUTER_API_KEY missing. Speaker role classification will be heuristic."
            )

    def classify(self, track_id: str, transcript: str) -> Dict[str, Any]:
        if not transcript.strip():
            return {
                "role": "Unknown",
                "confidence": 0.0,
                "summary": "No speech detected.",
            }

        if not self.api_key:
            # Fallback heuristic
            role = "Agent" if "help" in transcript.lower() else "Client"
            return {
                "role": role,
                "confidence": 0.5,
                "summary": transcript[:120] + ("..." if len(transcript) > 120 else ""),
            }

        prompt = (
            "You are analyzing a single-speaker transcript extracted from a call.\n"
            "Determine if the speaker is an Agent (support professional, host, or moderator) "
            "or a Client (customer, attendee, guest). Consider cues such as greeting style, "
            "problem descriptions, or offers of assistance.\n"
            "Respond with strict JSON using this schema:\n"
            '{\n'
            '  "track_id": "<original track id>",\n'
            '  "role": "Agent" | "Client",\n'
            '  "confidence": number between 0 and 1,\n'
            '  "summary": "one-sentence synopsis of the speaker\'s intent"\n'
            "}\n"
            "Never include extra text before or after the JSON."
        )

        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": f"Track ID: {track_id}\nTranscript:\n{transcript}",
                        },
                    ],
                    "temperature": 0,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.app_url,
                    "X-Title": "Speaker Role Classification",
                },
                timeout=30,
            )
            response.raise_for_status()
            
            output_text = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not output_text:
                raise ValueError("Empty response from OpenRouter API")
            
            # Спробувати витягти JSON з відповіді (може бути обгорнутий в markdown)
            output_text = output_text.strip()
            if output_text.startswith("```json"):
                output_text = output_text[7:]
            if output_text.startswith("```"):
                output_text = output_text[3:]
            if output_text.endswith("```"):
                output_text = output_text[:-3]
            output_text = output_text.strip()
            
            parsed = json.loads(output_text)
        except requests.exceptions.RequestException as error:
            logger.error(f"OpenRouter API error: {error}")
            # Fallback heuristic
            role = "Agent" if "help" in transcript.lower() else "Client"
            return {
                "role": role,
                "confidence": 0.5,
                "summary": transcript[:120] + ("..." if len(transcript) > 120 else ""),
            }
        except json.JSONDecodeError as error:
            logger.error(f"Failed to parse OpenRouter response: {output_text}")
            raise ValueError(f"LLM returned non-JSON output: {output_text}") from error

        for field in ("track_id", "role", "confidence", "summary"):
            if field not in parsed:
                raise ValueError(f"LLM JSON missing '{field}': {parsed}")

        return parsed


class ProcessingPipeline:
    """Coordinates separation, transcription, and classification."""

    def __init__(
        self,
        separation_client: Optional[AudioShakeClient] = None,
        stt_service: Optional[SpeechToTextService] = None,
        classifier: Optional[SpeakerRoleClassifier] = None,
        stems_dir: Optional[str] = None,
    ) -> None:
        self.separation_client = separation_client or AudioShakeClient()
        self.stt_service = stt_service or SpeechToTextService()
        self.classifier = classifier or SpeakerRoleClassifier()
        self.stems_dir = Path(stems_dir or tempfile.mkdtemp(prefix="audioshake_stems_"))

    def run(self, audio_path: str, audio_url: Optional[str] = None) -> List[TrackResult]:
        """
        Run the processing pipeline.
        
        Args:
            audio_path: Local path to audio file (for reference)
            audio_url: Public HTTPS URL to audio file (required for Tasks API)
        """
        logger.info("Starting processing pipeline for %s", audio_path)
        
        if not audio_url:
            raise ValueError(
                "AudioShake Tasks API requires a public HTTPS URL. "
                "Please provide audio_url parameter with a publicly accessible URL."
            )
        
        job_id = self.separation_client.submit_multi_speaker_job(
            audio_path=audio_path,
            audio_url=audio_url
        )
        job_payload = self.separation_client.poll_job(job_id)
        stems = self.separation_client.download_stems(job_payload, str(self.stems_dir))

        results: List[TrackResult] = []
        for stem in stems:
            track_id = stem["track_id"]
            stem_path = Path(stem["path"])
            logger.info("Transcribing %s", stem_path)
            transcript = self.stt_service.transcribe(stem_path)

            logger.info("Classifying role for %s", track_id)
            classification = self.classifier.classify(track_id, transcript)

            results.append(
                TrackResult(
                    track_id=track_id,
                    role=classification.get("role", "Unknown"),
                    confidence=float(classification.get("confidence", 0.0)),
                    summary=classification.get("summary", ""),
                    transcript=transcript,
                    audio_path=stem_path,
                )
            )

        return results


__all__ = [
    "TrackResult",
    "SpeechToTextService",
    "SpeakerRoleClassifier",
    "ProcessingPipeline",
    "AudioShakeError",
]

