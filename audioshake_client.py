import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


class AudioShakeError(RuntimeError):
    """Raised when the AudioShake API returns an error response."""


class AudioShakeClient:
    """Typed helper around the AudioShake multi-speaker separation API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.audioshake.ai",
        session: Optional[requests.Session] = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("AUDIOSHAKE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "AudioShake API key missing. Set AUDIOSHAKE_API_KEY or pass api_key."
            )
        
        # Перевірка чи це не placeholder
        if self.api_key == "your_audioshake_api_key_here" or not self.api_key.strip():
            raise ValueError(
                "AudioShake API key is not set. Please set AUDIOSHAKE_API_KEY in your .env file with a real API key."
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        # Tasks API використовує x-api-key header, не Authorization: Bearer
        self.session.headers.update({"x-api-key": self.api_key})
        self.session.headers.update({"Content-Type": "application/json"})

    # --------------------------------------------------------------------- API
    def submit_multi_speaker_job(
        self,
        audio_path: str,
        preset: str = "multi_speaker",
        metadata: Optional[Dict[str, Any]] = None,
        audio_url: Optional[str] = None,
    ) -> str:
        """
        Create an AudioShake task for multi-speaker separation.
        
        Note: Tasks API requires a public HTTPS URL, not file upload.
        If audio_path is provided, it must be accessible via a public URL.
        Use audio_url parameter to provide the public URL directly.

        Returns:
            task_id returned by AudioShake.
        """
        # Tasks API використовує публічний URL, не завантаження файлу
        if not audio_url:
            raise ValueError(
                "AudioShake Tasks API requires a public HTTPS URL. "
                "Please provide audio_url parameter with a publicly accessible URL."
            )
        
        # Валідація URL
        if not audio_url.startswith("https://"):
            raise ValueError(f"AudioShake requires HTTPS URL, got: {audio_url}")

        url = f"{self.base_url}/tasks"
        
        # Формуємо payload для Tasks API (як в JavaScript клієнті)
        payload = {
            "url": audio_url,
            "targets": [
                {
                    "model": "multi_voice",
                    "formats": ["wav"],
                    "variant": preset if preset != "multi_speaker" else "n_speaker",
                    "language": metadata.get("language", "en") if metadata else "en",
                }
            ],
        }

        # Логування перед відправкою
        logger.info("Creating AudioShake task:")
        logger.info("  URL: %s", url)
        logger.info("  Base URL: %s", self.base_url)
        logger.info("  Audio URL: %s", audio_url)
        logger.info("  Payload: %s", payload)
        logger.info("  Headers: %s", {k: v[:20] + "..." if len(v) > 20 else v for k, v in self.session.headers.items() if k != "x-api-key"})

        try:
            response = self.session.post(
                url, json=payload, timeout=self.timeout
            )
        except requests.exceptions.RequestException as e:
            logger.error("Request exception during AudioShake task creation:")
            logger.error("  Exception type: %s", type(e).__name__)
            logger.error("  Exception message: %s", str(e))
            raise AudioShakeError(f"Failed to create AudioShake task: {str(e)}") from e

        task_data = self._raise_for_api_error(response)
        task_id = task_data.get("id")
        if not task_id:
            logger.error("AudioShake response payload: %s", task_data)
            raise AudioShakeError("AudioShake response did not include a task id.")

        logger.info("Created AudioShake task %s", task_id)
        return task_id

    def poll_job(
        self,
        job_id: str,
        poll_interval: int = 5,
        timeout_seconds: int = 600,
    ) -> Dict[str, Any]:
        """
        Poll AudioShake for the status of a task until completion or timeout.

        Returns:
            Final task payload containing download URLs for each stem.
        """
        url = f"{self.base_url}/tasks/{job_id}"
        deadline = time.time() + timeout_seconds
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            try:
                response = self.session.get(url, timeout=self.timeout)
                payload = self._raise_for_api_error(response)
            except requests.exceptions.RequestException as e:
                logger.warning("Request error during polling (attempt %d): %s", attempt, str(e))
                time.sleep(poll_interval)
                continue

            # Tasks API структура: targets[0].status
            target = payload.get("targets", [{}])[0] if payload.get("targets") else {}
            status = target.get("status") or payload.get("status")
            
            logger.debug("AudioShake task %s status: %s (attempt %d)", job_id, status, attempt)

            if status == "completed":
                logger.info("AudioShake task %s completed", job_id)
                return payload
            if status == "failed":
                error_info = target.get("error") or {}
                error_message = error_info.get("message") or error_info.get("detail") or "unknown error"
                logger.error("AudioShake task %s failed: %s", job_id, error_message)
                raise AudioShakeError(f"AudioShake task {job_id} failed: {error_message}")

            time.sleep(poll_interval)

        raise TimeoutError(f"AudioShake task {job_id} polling timed out after {timeout_seconds}s.")

    def download_stems(
        self, job_payload: Dict[str, Any], output_dir: str
    ) -> List[Dict[str, Path]]:
        """
        Download stem files produced by a completed AudioShake task.

        Returns:
            List of dicts with `track_id` and `path` keys.
        """
        # Tasks API структура: targets[0].output[]
        target = job_payload.get("targets", [{}])[0] if job_payload.get("targets") else {}
        outputs = target.get("output", [])
        
        # Fallback до старої структури для сумісності
        if not outputs:
            outputs = (
                job_payload.get("stems")
                or job_payload.get("results", {}).get("stems")
                or []
            )
        
        if not outputs:
            logger.error("No stems found in AudioShake task payload. Payload keys: %s", list(job_payload.keys()))
            raise AudioShakeError("No stems found in AudioShake task payload.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        downloaded: List[Dict[str, Path]] = []
        for output in outputs:
            # Tasks API структура: output.name, output.link, output.format
            track_id = output.get("name") or output.get("id") or output.get("stem_id")
            download_url = output.get("link") or output.get("url") or output.get("audio_url") or output.get("download_url")
            extension = output.get("format") or "wav"
            
            # Пропускаємо background stems
            if track_id == "_background" or output.get("isBackground"):
                logger.info("Skipping background stem: %s", track_id)
                continue
            
            if not download_url:
                logger.warning("Skipping stem without download URL: %s", track_id)
                continue
            
            if not track_id:
                logger.warning("Skipping stem without name/id: %s", output)
                continue

            file_name = f"{track_id}.{extension}".replace(" ", "_").replace("/", "_")
            destination = output_path / file_name
            self._stream_download(download_url, destination)

            downloaded.append({"track_id": track_id, "path": destination})
            logger.info("Downloaded stem %s -> %s", track_id, destination)

        if not downloaded:
            raise AudioShakeError("No stems could be downloaded from AudioShake task.")

        return downloaded

    # ----------------------------------------------------------------- Helpers
    def _stream_download(self, url: str, destination: Path) -> None:
        with self.session.get(url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            with destination.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=8192):
                    output_file.write(chunk)

    @staticmethod
    def _raise_for_api_error(response: requests.Response) -> Dict[str, Any]:
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            # Детальне логування помилки
            status_code = response.status_code
            url = response.url
            headers = dict(response.headers)
            
            # Спробувати отримати текст відповіді
            try:
                response_text = response.text
            except Exception as e:
                response_text = f"<failed to read response text: {e}>"
            
            # Спробувати отримати JSON відповідь
            error_message = ""
            error_details = {}
            try:
                json_response = response.json()
                error_message = (
                    json_response.get("error") 
                    or json_response.get("message") 
                    or json_response.get("detail")
                    or str(json_response)
                )
                error_details = json_response
            except (ValueError, AttributeError):
                error_message = response_text if response_text else "No error message provided"
            
            # Логувати всі деталі
            logger.error("AudioShake API HTTP error:")
            logger.error("  Status Code: %s", status_code)
            logger.error("  URL: %s", url)
            logger.error("  Response Headers: %s", headers)
            logger.error("  Response Body: %s", response_text[:2000])  # Перші 2000 символів
            if error_details:
                logger.error("  Error Details: %s", error_details)
            
            # Формувати повідомлення про помилку
            if not error_message or error_message.strip() == "":
                error_message = f"HTTP {status_code}"
            
            full_error_message = f"AudioShake API error ({status_code}): {error_message}"
            if response_text and len(response_text) > 0 and response_text != error_message:
                full_error_message += f" | Response: {response_text[:500]}"
            
            raise AudioShakeError(full_error_message) from error

        try:
            return response.json()
        except ValueError as error:
            response_text = ""
            try:
                response_text = response.text[:1000]
            except Exception:
                response_text = "<unable to read response>"
            
            logger.error("AudioShake JSON decode error:")
            logger.error("  Status Code: %s", response.status_code)
            logger.error("  URL: %s", response.url)
            logger.error("  Response Text (first 1000 chars): %s", response_text)
            raise AudioShakeError(f"AudioShake API returned invalid JSON. Status: {response.status_code}, Response: {response_text[:200]}") from error

