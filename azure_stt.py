import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from azure.storage.blob import (
        BlobSasPermissions,
        BlobServiceClient,
        generate_blob_sas,
    )
except ImportError:  # pragma: no cover - optional dependency until Azure is enabled
    BlobSasPermissions = None
    BlobServiceClient = None
    generate_blob_sas = None

logger = logging.getLogger(__name__)


class AzureSpeechClient:
    """Azure Speech-to-Text batch transcription client with diarization support."""

    def __init__(
        self,
        subscription_key: str,
        region: str,
        *,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        if not subscription_key:
            raise ValueError("Azure Speech subscription key is required")
        if not region:
            raise ValueError("Azure Speech region is required")

        self.subscription_key = subscription_key
        self.region = region
        self.endpoint = endpoint or f"https://{region}.api.cognitive.microsoft.com"
        self.api_version = api_version or os.getenv("AZURE_SPEECH_API_VERSION", "2024-11-15")
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.subscription_key,
            "Content-Type": "application/json",
        }

    def submit_batch_transcription(
        self,
        *,
        content_urls: List[str],
        locale: str,
        display_name: str,
        diarization_enabled: bool,
        min_speakers: int,
        max_speakers: int,
        word_level_timestamps: bool = True,
        channel_list: Optional[List[int]] = None,
        time_to_live_hours: int = 48,
    ) -> Dict[str, Any]:
        """Create a batch transcription job."""
        payload: Dict[str, Any] = {
            "contentUrls": content_urls,
            "locale": locale,
            "displayName": display_name,
            "properties": {
                "wordLevelTimestampsEnabled": word_level_timestamps,
                "timeToLiveHours": time_to_live_hours,
            },
        }

        if diarization_enabled:
            payload["properties"]["diarization"] = {
                "enabled": True,
                "minSpeakers": min_speakers,
                "maxSpeakers": max_speakers,
            }

        if channel_list:
            payload["properties"]["channels"] = channel_list

        url = f"{self.endpoint}/speechtotext/transcriptions:submit?api-version={self.api_version}"
        response = requests.post(url, headers=self.headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def poll_transcription(
        self,
        transcription_id: str,
        *,
        max_retries: int = 120,
        interval_seconds: int = 5,
    ) -> Dict[str, Any]:
        """Poll Azure until transcription succeeded or failed."""
        url = f"{self.endpoint}/speechtotext/transcriptions/{transcription_id}?api-version={self.api_version}"
        for attempt in range(max_retries):
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            logger.info(
                "[Azure STT] Poll attempt %s/%s status=%s",
                attempt + 1,
                max_retries,
                status,
            )

            if status == "Succeeded":
                return data
            if status == "Failed":
                raise RuntimeError(f"Azure transcription failed: {json.dumps(data)}")

            if attempt < max_retries - 1:
                time.sleep(interval_seconds)

        raise TimeoutError(
            f"Azure transcription polling timed out after {max_retries * interval_seconds}s",
        )

    def list_transcription_files(self, transcription_id: str) -> List[Dict[str, Any]]:
        """List output files for a transcription job."""
        url = (
            f"{self.endpoint}/speechtotext/transcriptions/{transcription_id}/files"
            f"?api-version={self.api_version}"
        )
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload.get("values", [])

    @staticmethod
    def download_transcription_result(content_url: str) -> Dict[str, Any]:
        response = requests.get(content_url, timeout=60)
        response.raise_for_status()
        return response.json()

    def parse_response_to_segments(self, azure_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize Azure response into Speechmatics-compatible segments."""
        phrases = azure_response.get("recognizedPhrases", [])
        phrases.sort(key=lambda item: self._parse_iso_duration(item.get("offset", "PT0S")))

        speaker_map: Dict[int, str] = {}
        segments: List[Dict[str, Any]] = []

        for phrase in phrases:
            if phrase.get("recognitionStatus") != "Success":
                continue

            n_best = phrase.get("nBest") or []
            if not n_best:
                continue

            best = n_best[0]
            azure_speaker = phrase.get("speaker")
            speaker_label = self._map_speaker_id(azure_speaker, speaker_map)

            start = self._parse_iso_duration(phrase.get("offset", "PT0S"))
            end = start + self._parse_iso_duration(phrase.get("duration", "PT0S"))

            words = []
            last_word_end: Optional[float] = None
            pauses: List[Dict[str, float]] = []

            for word_info in best.get("words", []):
                word_start = self._parse_iso_duration(word_info.get("offset", "PT0S"))
                duration = self._parse_iso_duration(word_info.get("duration", "PT0S"))
                word_end = word_start + duration

                if last_word_end is not None:
                    gap = word_start - last_word_end
                    if gap > 0.3:
                        pauses.append(
                            {
                                "start": last_word_end,
                                "end": word_start,
                                "duration": gap,
                            }
                        )

                words.append(
                    {
                        "word": word_info.get("word", ""),
                        "start": word_start,
                        "end": word_end,
                        "speaker": speaker_label,
                        "confidence": best.get("confidence", 0),
                    }
                )
                last_word_end = word_end

            segment = {
                "speaker": speaker_label,
                "text": best.get("display") or best.get("lexical") or "",
                "start": start,
                "end": end,
                "words": words,
                "confidence": best.get("confidence", 0),
            }

            if pauses:
                segment["pauses"] = pauses

            segments.append(segment)

        return segments

    def full_workflow(
        self,
        *,
        audio_urls: List[str],
        locale: str,
        diarization_enabled: bool,
        min_speakers: int,
        max_speakers: int,
        channel_list: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Submit job, poll, fetch result, return normalized segments and raw JSON."""
        display_name = f"Diarization {datetime.utcnow().isoformat()}"
        submit_response = self.submit_batch_transcription(
            content_urls=audio_urls,
            locale=locale,
            display_name=display_name,
            diarization_enabled=diarization_enabled,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            channel_list=channel_list,
        )

        transcription_id = self._extract_transcription_id(submit_response.get("self", ""))
        if not transcription_id:
            raise RuntimeError("Azure response missing transcription ID")

        logger.info("[Azure STT] Job submitted id=%s", transcription_id)
        poll_result = self.poll_transcription(transcription_id)
        logger.info(
            "[Azure STT] Job succeeded id=%s status=%s",
            transcription_id,
            poll_result.get("status"),
        )
        files = self.list_transcription_files(transcription_id)
        logger.info("[Azure STT] Transcription files found=%s", len(files))

        transcription_files = [
            file_info for file_info in files if file_info.get("kind") == "Transcription"
        ]
        if not transcription_files:
            raise RuntimeError("Azure transcription did not return any result files")
        logger.info("[Azure STT] Result files ready=%s", len(transcription_files))

        content_url = transcription_files[0].get("links", {}).get("contentUrl")
        if not content_url:
            raise RuntimeError("Azure transcription file missing content URL")

        azure_payload = self.download_transcription_result(content_url)
        recognized_phrases = len(azure_payload.get("recognizedPhrases", []))
        logger.info("[Azure STT] Payload recognized phrases=%s", recognized_phrases)
        segments = self.parse_response_to_segments(azure_payload)
        return segments, azure_payload

    @staticmethod
    def _extract_transcription_id(self_link: str) -> Optional[str]:
        match = re.search(r"/transcriptions/([0-9a-fA-F-]+)", self_link)
        return match.group(1) if match else None

    @staticmethod
    def _parse_iso_duration(duration_str: str) -> float:
        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?"
        match = re.match(pattern, duration_str or "")
        if not match:
            return 0.0
        hours, minutes, seconds = match.groups(default="0")
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    @staticmethod
    def _map_speaker_id(speaker_id: Optional[int], speaker_map: Dict[int, str]) -> str:
        if speaker_id is None:
            return "SPEAKER_00"
        if speaker_id not in speaker_map:
            speaker_map[speaker_id] = f"SPEAKER_{len(speaker_map):02d}"
        return speaker_map[speaker_id]


def upload_file_to_azure_blob(file_path: str) -> str:
    """Upload file to Azure Blob Storage and return a SAS URL."""
    if not BlobServiceClient or not generate_blob_sas:
        raise ImportError(
            "azure-storage-blob is required for Azure STT file uploads. "
            "Install it or set AZURE_BLOB_* env variables.",
        )

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_STORAGE_CONTAINER")

    if not connection_string or not container_name:
        raise ValueError(
            "AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER must be set "
            "to upload local files for Azure STT",
        )

    logger.info("[Azure STT] Uploading local file '%s' to blob storage", os.path.basename(file_path))
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()

    blob_name = f"uploads/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{os.path.basename(file_path)}"
    with open(file_path, "rb") as file_handle:
        container_client.upload_blob(name=blob_name, data=file_handle, overwrite=True)
    logger.info("[Azure STT] Blob upload completed: %s", blob_name)

    account_name = blob_service.account_name
    account_key = _extract_account_key_from_connection_string(connection_string) or os.getenv(
        "AZURE_STORAGE_ACCOUNT_KEY",
    )
    if not account_key:
        raise ValueError(
            "Account key is required to generate SAS tokens. "
            "Include AccountKey in the connection string or set AZURE_STORAGE_ACCOUNT_KEY.",
        )

    expiry = datetime.utcnow() + timedelta(hours=2)
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"


def _extract_account_key_from_connection_string(connection_string: str) -> Optional[str]:
    parts = dict(
        part.split("=", 1) for part in connection_string.split(";") if "=" in part  # type: ignore[arg-type]
    )
    return parts.get("AccountKey")

