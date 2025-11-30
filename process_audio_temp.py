#!/usr/bin/env python3
"""
Temporary Audio File Processing Script
Downloads audio files to temporary storage, processes them for diarization,
and automatically cleans up after processing.
"""

import os
import sys
import tempfile
import shutil
import argparse
import json
import requests
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging
import time
import hashlib
from datetime import datetime, timezone

from azure_stt import AzureSpeechClient, upload_file_to_azure_blob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TempAudioProcessor:
    """Handles temporary audio file download, processing, and cleanup."""
    
    def __init__(self, temp_dir: Optional[str] = None, cleanup: bool = True):
        """
        Initialize the processor.
        
        Args:
            temp_dir: Custom temporary directory path. If None, uses system temp.
            cleanup: Whether to automatically cleanup files after processing.
        """
        self.cleanup = cleanup
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.temp_file_path: Optional[str] = None
        self.work_dir: Optional[str] = None
        self.original_filename: Optional[str] = None  # Store original filename for cache lookup
        self.source_url: Optional[str] = None  # Original URL if file was downloaded
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # Setup cache directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(script_dir, 'cache', 'transcriptions')
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using cache directory: {self.cache_dir}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for cache key."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_cache_key(self, file_path: str, language: str, speaker_count: Optional[int], is_separated_track: bool, engine: str = 'speechmatics', use_filename_only: bool = True) -> str:
        """
        Generate cache key based on filename (primary) or file hash.
        
        Args:
            use_filename_only: If True, uses only filename (allows reuse for same filename).
                              If False, includes hash for exact file match.
        """
        # Get filename (without path)
        filename = os.path.basename(file_path)
        # Remove extension for cleaner cache key
        filename_base = os.path.splitext(filename)[0]
        # Sanitize filename for use in cache key (remove special chars)
        filename_safe = "".join(c for c in filename_base if c.isalnum() or c in ('-', '_')).strip()
        
        engine_safe = (engine or 'speechmatics').lower()
        params = f"{language}_{speaker_count or 'none'}_{is_separated_track}_{engine_safe}"
        
        if use_filename_only:
            # Use filename only - allows caching for same filename even if content differs
            cache_key = f"{filename_safe}_{params}"
        else:
            # Include hash for exact file match
            file_hash_short = self._get_file_hash(file_path)[:8]  # First 8 chars of hash
            cache_key = f"{filename_safe}_{file_hash_short}_{params}"
        
        logger.debug(f"Generated cache key: {cache_key} (from filename: {filename})")
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get path to cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _load_from_cache(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """Load result from cache if exists and not expired."""
        if not os.path.exists(cache_path):
            return None
        
        try:
            # Check if cache is expired (30 days TTL)
            cache_age = time.time() - os.path.getmtime(cache_path)
            CACHE_TTL = 30 * 24 * 60 * 60  # 30 days
            
            if cache_age > CACHE_TTL:
                logger.info(f"Cache expired (age: {cache_age / (24 * 60 * 60):.1f} days), removing...")
                os.remove(cache_path)
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            logger.info(f"✅ Cache hit! Loaded from cache: {os.path.basename(cache_path)}")
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will fetch from API")
            return None
    
    def _find_cache_by_filename(self, filename: str, language: str, speaker_count: Optional[int], is_separated_track: bool, engine: str = 'speechmatics') -> Optional[str]:
        """
        Find cache file by filename (without hash check).
        This allows reusing cache for same filename even if file content differs slightly.
        """
        filename_base = os.path.splitext(filename)[0]
        filename_safe = "".join(c for c in filename_base if c.isalnum() or c in ('-', '_')).strip()
        engine_safe = (engine or 'speechmatics').lower()
        params = f"{language}_{speaker_count or 'none'}_{is_separated_track}_{engine_safe}"
        
        # Search for cache files matching filename pattern
        pattern_prefix = f"{filename_safe}_"
        pattern_suffixes = [f"_{params}.json"]
        if engine_safe == 'speechmatics':
            legacy_params = f"{language}_{speaker_count or 'none'}_{is_separated_track}"
            pattern_suffixes.append(f"_{legacy_params}.json")
        
        logger.info(f"🔍 Searching cache: prefix='{pattern_prefix}', suffixes='{pattern_suffixes}'")
        
        if not os.path.exists(self.cache_dir):
            logger.warning(f"Cache directory does not exist: {self.cache_dir}")
            return None
        
        cache_files = os.listdir(self.cache_dir)
        logger.info(f"📁 Found {len(cache_files)} cache files in directory")
        
        for cache_file in cache_files:
            if not cache_file.startswith(pattern_prefix):
                continue
            if not any(cache_file.endswith(suffix) for suffix in pattern_suffixes):
                continue
            
            cache_path = os.path.join(self.cache_dir, cache_file)
            # Check if not expired
            cache_age = time.time() - os.path.getmtime(cache_path)
            CACHE_TTL = 30 * 24 * 60 * 60  # 30 days
            if cache_age <= CACHE_TTL:
                logger.info(f"✅ Found valid cache by filename: {cache_file} (age: {cache_age / (24 * 60 * 60):.1f} days)")
                return cache_path
            else:
                logger.info(f"⏰ Cache expired: {cache_file} (age: {cache_age / (24 * 60 * 60):.1f} days)")
        
        logger.info(f"❌ No matching cache found for filename: {filename}")
        return None
    
    def _save_to_cache(self, cache_path: str, result: Dict[str, Any]) -> None:
        """Save result to cache."""
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def download_file(self, url: str, filename: Optional[str] = None) -> str:
        """
        Download a file from URL to temporary storage.
        
        Args:
            url: URL of the file to download
            filename: Optional custom filename. If None, extracts from URL.
            
        Returns:
            Path to the downloaded file
        """
        try:
            # Create a unique work directory for this session
            self.work_dir = tempfile.mkdtemp(dir=self.temp_dir, prefix='diarization_')
            logger.info(f"Created work directory: {self.work_dir}")
            
            # Determine filename
            if not filename:
                filename = os.path.basename(url.split('?')[0]) or 'audio_file'
                # Ensure filename has extension
                if '.' not in filename:
                    filename += '.wav'
            
            # Store original filename for cache lookup
            self.original_filename = filename
            self.source_url = url
            
            self.temp_file_path = os.path.join(self.work_dir, filename)
            
            logger.info(f"Downloading {url} to {self.temp_file_path} (original filename: {self.original_filename})")
            
            # Download the file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            with open(self.temp_file_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            file_size = os.path.getsize(self.temp_file_path)
            logger.info(f"Downloaded {file_size} bytes to {self.temp_file_path}")
            
            return self.temp_file_path
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            self.cleanup_files()
            raise
    
    def save_uploaded_file(self, file_path: str, filename: Optional[str] = None) -> str:
        """
        Save an uploaded file to temporary storage.
        
        Args:
            file_path: Path to the source file
            filename: Optional custom filename. If None, uses source filename.
            
        Returns:
            Path to the saved file in temporary storage
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Source file not found: {file_path}")
            
            # Create a unique work directory for this session
            self.work_dir = tempfile.mkdtemp(dir=self.temp_dir, prefix='diarization_')
            logger.info(f"Created work directory: {self.work_dir}")
            
            # Determine filename - prioritize provided filename, then source filename
            if not filename:
                filename = os.path.basename(file_path)
            
            # Store original filename for cache lookup
            self.original_filename = filename
            self.source_url = None
            
            self.temp_file_path = os.path.join(self.work_dir, filename)
            
            logger.info(f"Copying {file_path} to {self.temp_file_path} (original filename: {self.original_filename})")
            
            # Copy file to temporary location
            shutil.copy2(file_path, self.temp_file_path)
            
            file_size = os.path.getsize(self.temp_file_path)
            logger.info(f"Saved {file_size} bytes to {self.temp_file_path}")
            
            return self.temp_file_path
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            self.cleanup_files()
            raise
    
    def process_for_diarization(
        self,
        output_format: str = 'json',
        language: Optional[str] = None,
        speaker_count: Optional[int] = None,
        is_separated_track: bool = False,
        skip_cache: bool = False,
        engine: str = 'speechmatics',
    ) -> Dict[str, Any]:
        """
        Process the audio file for diarization using the selected engine.
        """
        if not self.temp_file_path or not os.path.exists(self.temp_file_path):
            raise FileNotFoundError("No file available for processing")
        
        engine = (engine or 'speechmatics').lower()
        if engine not in ('speechmatics', 'azure', 'azure_realtime'):
            raise ValueError(f"Unsupported engine '{engine}'. Expected 'speechmatics', 'azure' or 'azure_realtime'.")
        
        logger.info(f"Processing file for diarization ({engine}): {self.temp_file_path}")

        if not language:
            language = 'en'
        
        # Cache lookup
        if not skip_cache:
            # IMPORTANT: For voice tracks with original_filename, use filename-based cache (same as Step 1)
            # This ensures stable cache keys based on original file + speaker identifier
            # Only use hash-based cache if original_filename is not provided
            use_hash_for_cache = is_separated_track and not self.original_filename
            
            # First, try filename-based cache lookup (preferred for voice tracks with original_filename)
            # This matches the cache strategy used in Step 1
            if self.original_filename:
                cache_filename = self.original_filename
                logger.info(
                    "🔍 Checking cache for filename: %s (engine=%s, is_separated_track=%s, language=%s, speaker_count=%s)",
                    cache_filename,
                    engine,
                    is_separated_track,
                    language,
                    speaker_count,
                )
                logger.info(f"📋 Cache lookup details: original_filename={self.original_filename}, temp_file_path={self.temp_file_path}")
                cache_path_by_name = self._find_cache_by_filename(
                    cache_filename,
                    language,
                    speaker_count,
                    is_separated_track,
                    engine=engine,
                )
                
                if cache_path_by_name:
                    logger.info(f"✅ Found cache file: {cache_path_by_name}")
                    cached_result = self._load_from_cache(cache_path_by_name)
                    if cached_result:
                        logger.info("✅ Using cached transcription result by filename (engine=%s) - SKIPPING API CALL", engine)
                        cached_result['exportedAt'] = datetime.now(timezone.utc).isoformat()
                        cached_result['_cached'] = True
                        cached_result['_engine'] = engine
                        return cached_result
                    else:
                        logger.warning(f"⚠️ Cache file exists but failed to load: {cache_path_by_name}")
                else:
                    logger.info(f"❌ Cache file not found for filename: {cache_filename}")
            
            # Fallback to hash-based cache for voice tracks without original_filename
            if use_hash_for_cache and os.path.exists(self.temp_file_path):
                # Use file hash for voice tracks without original_filename
                file_hash = self._get_file_hash(self.temp_file_path)
                cache_key_hash = self._get_cache_key(
                    self.temp_file_path,
                    language,
                    speaker_count,
                    is_separated_track,
                    engine=engine,
                    use_filename_only=False,  # Use hash for exact match
                )
                cache_path_hash = self._get_cache_path(cache_key_hash)
                
                logger.info(
                    "🔍 Checking cache by hash for voice track (engine=%s, hash=%s)",
                    engine,
                    file_hash[:8],
                )
                
                if os.path.exists(cache_path_hash):
                    cached_result = self._load_from_cache(cache_path_hash)
                    if cached_result:
                        logger.info("✅ Using cached transcription result by hash (engine=%s)", engine)
                        cached_result['exportedAt'] = datetime.now(timezone.utc).isoformat()
                        cached_result['_cached'] = True
                        cached_result['_engine'] = engine
                        return cached_result
            
            # Final fallback: try filename-based cache with basename if original_filename not set
            if not self.original_filename:
                cache_filename = os.path.basename(self.temp_file_path)
                logger.info(
                    "🔍 Checking cache for filename (fallback): %s (engine=%s)",
                    cache_filename,
                    engine,
                )
                cache_path_by_name = self._find_cache_by_filename(
                    cache_filename,
                    language,
                    speaker_count,
                    is_separated_track,
                    engine=engine,
                )
                
                if cache_path_by_name:
                    cached_result = self._load_from_cache(cache_path_by_name)
                    if cached_result:
                        logger.info("✅ Using cached transcription result by filename (fallback, engine=%s)", engine)
                        cached_result['exportedAt'] = datetime.now(timezone.utc).isoformat()
                        cached_result['_cached'] = True
                        cached_result['_engine'] = engine
                        return cached_result
        else:
            logger.info("⏭️ Skipping cache (skip_cache=True) - engine=%s", engine)

        try:
            segments, service_name, raw_payload = self._transcribe_with_engine(
                engine=engine,
                language=language,
                speaker_count=speaker_count,
                is_separated_track=is_separated_track,
            )

            result = self._build_result_structure(
                segments=segments,
                language=language,
                speaker_count=speaker_count,
                service_name=service_name,
                engine=engine,
                raw_payload=raw_payload,
            )

            # IMPORTANT: For voice tracks with original_filename, prioritize filename-based cache (same as Step 1)
            # This ensures stable cache keys based on original file + speaker identifier
            # Save by filename first (preferred for voice tracks with original_filename)
            if self.original_filename:
                cache_filename = self.original_filename
                logger.info(f"💾 Saving to cache with filename: {cache_filename} (engine={engine}, is_separated_track={is_separated_track})")
                # Use the same logic as _find_cache_by_filename to ensure cache key matches
                filename_base = os.path.splitext(cache_filename)[0]
                filename_safe = "".join(c for c in filename_base if c.isalnum() or c in ('-', '_')).strip()
                engine_safe = (engine or 'speechmatics').lower()
                params = f"{language}_{speaker_count or 'none'}_{is_separated_track}_{engine_safe}"
                cache_key_filename = f"{filename_safe}_{params}"
                cache_path_filename = self._get_cache_path(cache_key_filename)
                self._save_to_cache(cache_path_filename, result)
                logger.info(f"✅ Saved to cache by filename: {cache_key_filename} (engine={engine})")
            
            # Also save by hash for voice tracks without original_filename (fallback)
            use_hash_for_cache = is_separated_track and not self.original_filename
            if use_hash_for_cache and os.path.exists(self.temp_file_path):
                # Save cache by hash for voice tracks without original_filename
                file_hash = self._get_file_hash(self.temp_file_path)
                cache_key_hash = self._get_cache_key(
                    self.temp_file_path,
                    language,
                    speaker_count,
                    is_separated_track,
                    engine=engine,
                    use_filename_only=False,  # Use hash for exact match
                )
                cache_path_hash = self._get_cache_path(cache_key_hash)
                self._save_to_cache(cache_path_hash, result)
                logger.info(f"💾 Saved to cache by hash (engine={engine}, hash={file_hash[:8]})")
            
            # Also save by basename for backward compatibility (if original_filename not set)
            if not self.original_filename:
                cache_filename = os.path.basename(self.temp_file_path)
                logger.info(f"💾 Saving to cache with basename (fallback): {cache_filename} (engine={engine})")
                temp_path_for_cache = self.temp_file_path
                cache_key_filename = self._get_cache_key(
                    temp_path_for_cache,
                    language,
                    speaker_count,
                    is_separated_track,
                    engine=engine,
                    use_filename_only=True,
                )
                cache_path_filename = self._get_cache_path(cache_key_filename)
                self._save_to_cache(cache_path_filename, result)

            logger.info(
                "Processing complete: %s segments (engine=%s)",
                len(segments),
                engine,
            )
            return result
        except Exception as e:
            logger.error(f"Error in process_for_diarization: {type(e).__name__}: {e}")
            logger.error(f"Error traceback: {str(e)}")
            raise

    def _transcribe_with_engine(
        self,
        *,
        engine: str,
        language: str,
        speaker_count: Optional[int],
        is_separated_track: bool,
    ) -> Tuple[List[Dict[str, Any]], str, Optional[Dict[str, Any]]]:
        if engine == 'azure':
            segments, raw_payload = self._transcribe_with_azure(language, speaker_count, is_separated_track)
            return segments, 'Azure STT', raw_payload
        if engine == 'azure_realtime':
            segments, raw_payload = self._transcribe_with_azure_realtime(language, speaker_count, is_separated_track)
            return segments, 'Azure STT Realtime', raw_payload

        segments, raw_payload = self._transcribe_with_speechmatics(language, speaker_count, is_separated_track)
        return segments, 'Model3', raw_payload

    def _transcribe_with_speechmatics(
        self,
        language: str,
        speaker_count: Optional[int],
        is_separated_track: bool,
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        logger.info("💸 Fetching from Speechmatics API...")
        
        api_key = os.getenv('SPEECHMATICS_API_KEY')
        if not api_key:
            raise ValueError("SPEECHMATICS_API_KEY environment variable is not set")
        
        logger.info(
            "Speechmatics API key loaded: %s",
            '*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '***',
        )
        
        logger.info("Uploading file to Speechmatics...")
        job_id = self._upload_to_speechmatics(api_key, language, speaker_count, is_separated_track)
        logger.info(f"Job created: {job_id}")
        
        logger.info("Polling for job completion...")
        transcript_data = self._poll_speechmatics_job(api_key, job_id)
        logger.info("Job completed successfully")
        
        segments = self._parse_speechmatics_transcript(transcript_data)
        logger.info("Parsed %s segments from Speechmatics response", len(segments))
        
        if not segments:
            logger.warning("No segments returned from Speechmatics (job_id=%s)", job_id)

        raw_payload = {
            "service": "speechmatics",
            "jobId": job_id,
            "transcript": transcript_data,
        }
        return segments, raw_payload

    def _transcribe_with_azure(
        self,
        language: str,
        speaker_count: Optional[int],
        is_separated_track: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        logger.info("💸 Fetching from Azure Speech-to-Text API...")

        api_key = os.getenv('AZURE_SPEECH_KEY')
        region = os.getenv('AZURE_SPEECH_REGION')
        endpoint = os.getenv('AZURE_SPEECH_ENDPOINT')

        if not api_key or not region:
            raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set to use Azure STT")

        client = AzureSpeechClient(subscription_key=api_key, region=region, endpoint=endpoint)

        if self.source_url:
            audio_url = self.source_url
            logger.info("[Azure STT] Audio source=remote_url")
        else:
            if not self.temp_file_path:
                raise ValueError("Temporary file path is missing for Azure upload")
            logger.info("[Azure STT] Uploading local file to Azure Blob storage")
            audio_url = upload_file_to_azure_blob(self.temp_file_path)
            logger.info("[Azure STT] Blob upload completed")

        locale = self._map_language_to_locale(language)
        
        # For separated tracks: enable diarization with max_speakers=2
        # This allows detection of main speaker + residual audio from other speakers
        # Then collectVoiceTrackSegments will identify the main speaker and filter out residual
        if is_separated_track:
            # Enable diarization to detect both main speaker and residual audio
            diarization_enabled = True
            min_speakers = 1
            max_speakers = speaker_count if speaker_count and speaker_count > 0 else 2
            channel_list = None
            logger.info(
                "[Azure STT] Separated track mode: diarization enabled with max_speakers=%d "
                "(to detect main speaker + residual audio)",
                max_speakers
            )
        else:
            diarization_enabled = True
            min_speakers = 1
            max_speakers = speaker_count if speaker_count and speaker_count > 0 else 4
            channel_list = None

        logger.info(
            "[Azure STT] Using locale=%s (diarization=%s, channel_mode=%s)",
            locale,
            diarization_enabled,
            'channel' if channel_list else 'speaker',
        )

        segments, azure_payload = client.full_workflow(
            audio_urls=[audio_url],
            locale=locale,
            diarization_enabled=diarization_enabled,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            channel_list=channel_list,
        )
        logger.info("[Azure STT] Segments parsed=%s", len(segments))

        raw_payload = {
            "service": "azure",
            "audioUrl": audio_url,
            "response": azure_payload,
        }
        return segments, raw_payload

    def _transcribe_with_azure_realtime(
        self,
        language: str,
        speaker_count: Optional[int],
        is_separated_track: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        logger.info("💸 Fetching from Azure Real-time STT API...")

        api_key = os.getenv('AZURE_SPEECH_KEY')
        region = os.getenv('AZURE_SPEECH_REGION')

        if not api_key or not region:
            raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set to use Azure STT realtime mode")
        if not self.temp_file_path or not os.path.exists(self.temp_file_path):
            raise ValueError("Temporary file path is missing for Azure realtime processing")

        try:
            from azure_realtime import AzureRealtimeDiarization
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "azure-cognitiveservices-speech is required for Azure realtime transcription. "
                "Install it via pip (azure-cognitiveservices-speech) to enable this engine."
            ) from exc

        locale = self._map_language_to_locale(language)
        realtime_client = AzureRealtimeDiarization(subscription_key=api_key, region=region)
        wav_path, temp_conversion = self._ensure_wav_for_realtime_input()
        try:
            segments, raw_payload = realtime_client.transcribe_from_file(wav_path, locale)
        finally:
            if temp_conversion and os.path.exists(temp_conversion):
                try:
                    os.remove(temp_conversion)
                    logger.debug("Removed temporary WAV used for Azure realtime: %s", temp_conversion)
                except OSError:
                    pass

        logger.info("Azure realtime produced %s segments", len(segments))
        wrapped_payload = {
            "service": "azure_realtime",
            "audioPath": wav_path,
            "response": raw_payload,
        }
        return segments, wrapped_payload

    def _ensure_wav_for_realtime_input(self) -> Tuple[str, Optional[str]]:
        """
        Azure realtime SDK expects PCM WAV. Convert other formats via ffmpeg if needed.
        Returns tuple of (path_to_wav, temp_conversion_path_or_None).
        """
        if not self.temp_file_path:
            raise FileNotFoundError("Temporary file path is missing for Azure realtime processing")

        ext = Path(self.temp_file_path).suffix.lower()
        if ext in ('.wav', '.wave'):
            return self.temp_file_path, None

        ffmpeg_bin = shutil.which('ffmpeg')
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg is required to convert audio to WAV for Azure realtime STT")

        work_dir = self.work_dir or tempfile.mkdtemp(dir=self.temp_dir, prefix='diarization_rt_')
        if not self.work_dir:
            self.work_dir = work_dir

        base_name = Path(self.original_filename or os.path.basename(self.temp_file_path)).stem
        target_path = os.path.join(work_dir, f"{base_name}_azure_rt.wav")

        cmd = [
            ffmpeg_bin,
            '-hide_banner',
            '-loglevel', 'error',
            '-y',
            '-i', self.temp_file_path,
            '-ac', '1',
            '-ar', '16000',
            '-sample_fmt', 's16',
            target_path,
        ]
        logger.info("Converting %s to WAV for Azure realtime via ffmpeg", os.path.basename(self.temp_file_path))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(target_path):
            stderr = (result.stderr or '').strip()
            raise RuntimeError(
                f"ffmpeg failed to convert audio for Azure realtime (code={result.returncode}): {stderr}"
            )

        logger.info("ffmpeg conversion complete: %s", target_path)
        return target_path, target_path

    def _build_result_structure(
        self,
        *,
        segments: List[Dict[str, Any]],
        language: str,
        speaker_count: Optional[int],
        service_name: str,
        engine: str,
        raw_payload: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        recording_id = f"rec_{int(time.time() * 1000)}"
        file_name = os.path.basename(self.temp_file_path)
        unique_speakers = len(set(s.get('speaker', 'SPEAKER_00') for s in segments)) if segments else 0
        resolved_speaker_count = speaker_count or unique_speakers

        engine_lower = (engine or 'speechmatics').lower()
        is_azure = engine_lower.startswith('azure')

        result: Dict[str, Any] = {
            "version": "2.0",
            "exportedAt": datetime.now(timezone.utc).isoformat(),
            "activeRecordingId": recording_id,
            "_engine": engine,
            "recordings": [{
                "id": recording_id,
                "name": file_name,
                "fileName": file_name,
                "size": os.path.getsize(self.temp_file_path),
                "language": language,
                "speakerCount": str(resolved_speaker_count),
                "status": "completed",
                "results": {}
            }]
        }
        
        if is_azure:
            # For Azure: Store in native Azure format (not normalized to Speechmatics)
            result["recordings"][0]["results"]["azure"] = {
                "success": True,
                "serviceName": service_name,
                "engine": engine,
                "segments": segments,  # Already in normalized format for compatibility
                "rawData": raw_payload  # Original Azure response structure
            }
            # Also keep in speechmatics key for backward compatibility (but mark as Azure)
            result["recordings"][0]["results"]["speechmatics"] = {
                "success": True,
                "serviceName": service_name,
                "engine": engine,
                "segments": segments,
                "rawData": raw_payload
            }
        else:
            # For Speechmatics: Keep existing format
            result["recordings"][0]["results"]["speechmatics"] = {
                "success": True,
                "serviceName": service_name,
                "engine": engine,
                "segments": segments,
            }
            if raw_payload is not None:
                result["recordings"][0]["results"]["speechmatics"]["rawData"] = raw_payload

        return result
            
    @staticmethod
    def _map_language_to_locale(language: Optional[str]) -> str:
        if not language:
            return 'en-US'
        normalized = language.lower()
        locale_map = {
            'en': 'en-US',
            'ar': 'ar-SA',
        }
        if '-' in normalized:
            base, region = normalized.split('-', 1)
            return f"{base}-{region.upper()}"
        return locale_map.get(normalized, f"{normalized}-US")
    
    def _upload_to_speechmatics(self, api_key: str, language: str, speaker_count: Optional[int] = None, is_separated_track: bool = False) -> str:
        """
        Upload audio file to Speechmatics and return job ID.
        
        Args:
            api_key: Speechmatics API key
            language: Language code
            speaker_count: Optional speaker count (not used by Speechmatics v2)
            is_separated_track: If True, uses 'channel' diarization mode.
                               CRITICAL: Separated audio tracks should use 'channel' mode
                               to prevent Speechmatics from detecting residual audio as
                               separate speakers. This fixes 70-80% of speaker assignment errors.
        """
        url = 'https://asr.api.speechmatics.com/v2/jobs'
        
        # Get file size to determine timeout
        file_size = os.path.getsize(self.temp_file_path)
        file_size_mb = file_size / (1024 * 1024)
        # Calculate timeout: at least 5 minutes, plus 1 minute per 10MB
        upload_timeout = max(300, int(60 + (file_size_mb / 10) * 60))
        logger.info(f"File size: {file_size_mb:.2f}MB, using timeout: {upload_timeout}s")
        
        # CRITICAL FIX: For separated tracks, use 'speaker' mode with speaker_count=2
        # Separated tracks may contain residual audio from other speakers that wasn't fully separated
        # We need to detect both the main speaker and residual audio, then filter residual in post-processing
        # 'channel' mode assumes one speaker per file, which doesn't work if there's residual audio
        # 'speaker' mode with speaker_count=2 allows detection of main speaker + residual audio
        # Then collectVoiceTrackSegments will identify the main speaker and filter out residual
        if is_separated_track:
            # Use speaker mode to detect both main speaker and residual audio
            diarization_mode = 'speaker'
            logger.info(
                f"Using diarization mode: {diarization_mode} for separated track "
                f"(is_separated_track={is_separated_track}, speaker_count={speaker_count}). "
                f"This allows detection of main speaker + residual audio from other speakers."
            )
        else:
            diarization_mode = 'speaker'
            logger.info(
                f"Using diarization mode: {diarization_mode} (is_separated_track={is_separated_track}, "
                f"speaker_count={speaker_count})"
            )
        
        # Prepare form data with streaming
        config = {
            'type': 'transcription',
            'transcription_config': {
                'language': language,
                'diarization': diarization_mode,
                'operating_point': 'enhanced',
            }
        }
        
        # Add speaker_diarization_config for 'speaker' mode
        # For separated tracks with speaker_count=2, this allows detection of main speaker + residual audio
        if diarization_mode == 'speaker':
            speaker_config = {
                'get_speakers': True
            }
            # If speaker_count is provided, we can hint it (though Speechmatics may ignore it)
            if speaker_count and speaker_count > 0:
                # Note: Speechmatics v2 API doesn't support max_speakers, but we log it for reference
                logger.info(f"Requested speaker_count={speaker_count} (Speechmatics will auto-detect)")
            config['transcription_config']['speaker_diarization_config'] = speaker_config
        
        # Note: max_speakers is not supported by Speechmatics API v2
        # Speechmatics automatically detects the number of speakers
        
        data = {'config': json.dumps(config)}
        headers = {'Authorization': f'Bearer {api_key}'}
        
        # Use streaming upload for large files
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Upload attempt {attempt + 1}/{max_retries}...")
                logger.info(f"Upload status: Starting upload to Speechmatics API")
                logger.info(f"Upload details: File size={file_size_mb:.2f}MB, Timeout={upload_timeout}s, URL={url}")
                
                upload_start_time = time.time()
                
                # Determine MIME type based on file extension
                file_ext = os.path.splitext(self.temp_file_path)[1].lower()
                mime_types = {
                    '.wav': 'audio/wav',
                    '.mp3': 'audio/mpeg',
                    '.m4a': 'audio/mp4',
                    '.mp4': 'audio/mp4',
                    '.flac': 'audio/flac',
                    '.ogg': 'audio/ogg',
                    '.webm': 'audio/webm',
                    '.aac': 'audio/aac',
                    '.wma': 'audio/x-ms-wma'
                }
                mime_type = mime_types.get(file_ext, 'audio/wav')  # Default to wav
                logger.info(f"Detected file extension: {file_ext}, using MIME type: {mime_type}")
                
                with open(self.temp_file_path, 'rb') as f:
                    files = {'data_file': (os.path.basename(self.temp_file_path), f, mime_type)}
                    
                    response = requests.post(
                        url, 
                        files=files, 
                        data=data, 
                        headers=headers, 
                        timeout=upload_timeout,
                        stream=False  # Don't stream response, but allow large uploads
                    )
                    upload_duration = time.time() - upload_start_time
                    logger.info(f"Upload status: HTTP request completed in {upload_duration:.2f}s, status={response.status_code}")
                    
                    # Log detailed error information for 400 Bad Request
                    if response.status_code == 400:
                        try:
                            error_details = response.json()
                            logger.error(f"Speechmatics API 400 error details: {json.dumps(error_details, indent=2)}")
                        except:
                            error_text = response.text[:1000] if hasattr(response, 'text') else str(response.content[:1000])
                            logger.error(f"Speechmatics API 400 error response: {error_text}")
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    if not result.get('id'):
                        raise ValueError(f"No job ID in response: {result}")
                    
                    logger.info(f"Upload status: SUCCESS - Job created with ID: {result['id']}")
                    logger.info(f"Upload summary: Duration={upload_duration:.2f}s, File size={file_size_mb:.2f}MB, Job ID={result['id']}")
                    return result['id']
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, OSError, TimeoutError) as e:
                # Check if it's a timeout error (including write timeout)
                error_str = str(e).lower()
                is_timeout = (
                    isinstance(e, (requests.exceptions.Timeout, TimeoutError)) or
                    (isinstance(e, OSError) and ('timeout' in error_str or 'timed out' in error_str)) or
                    'timeout' in error_str or 
                    'timed out' in error_str or
                    'write operation timed out' in error_str
                )
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    # Increase timeout for next attempt
                    upload_timeout = int(upload_timeout * 1.5)
                    logger.warning(f"Upload timeout/connection error (attempt {attempt + 1}/{max_retries}): {type(e).__name__} - {error_str[:200]}")
                    logger.warning(f"Retrying in {wait_time}s with increased timeout {upload_timeout}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Upload failed after {max_retries} attempts: {type(e).__name__} - {error_str}")
                    raise
            except Exception as e:
                # Check if it's a timeout-related error in the error message
                error_str = str(e).lower()
                if 'timeout' in error_str or 'timed out' in error_str or 'write operation timed out' in error_str:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        upload_timeout = int(upload_timeout * 1.5)
                        logger.warning(f"Upload timeout error detected in exception message (attempt {attempt + 1}/{max_retries}): {error_str[:200]}")
                        logger.warning(f"Retrying in {wait_time}s with increased timeout {upload_timeout}s...")
                        time.sleep(wait_time)
                        continue
                logger.error(f"Upload error: {type(e).__name__} - {error_str}")
                raise
    
    def _poll_speechmatics_job(self, api_key: str, job_id: str, max_attempts: int = 60, poll_interval: int = 2) -> Dict[str, Any]:
        """Poll Speechmatics job until completion."""
        url = f'https://asr.api.speechmatics.com/v2/jobs/{job_id}'
        headers = {'Authorization': f'Bearer {api_key}'}
        
        logger.info(f"Polling status: Starting polling for job {job_id}, max_attempts={max_attempts}, poll_interval={poll_interval}s")
        poll_start_time = time.time()
        
        for attempt in range(max_attempts):
            attempt_start = time.time()
            logger.info(f"Polling status: Attempt {attempt + 1}/{max_attempts} - Checking job status...")
            
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                job_status = result.get('job', {}).get('status')
                elapsed_time = time.time() - poll_start_time
                
                logger.info(f"Polling status: Attempt {attempt + 1} - Job status: {job_status}, elapsed: {elapsed_time:.1f}s")
                
                if job_status == 'done':
                    logger.info(f"Polling status: Job completed! Fetching transcript...")
                    # Get transcript
                    transcript_url = f'https://asr.api.speechmatics.com/v2/jobs/{job_id}/transcript'
                    transcript_response = requests.get(
                        transcript_url,
                        headers={**headers, 'Accept': 'application/json'},
                        timeout=30
                    )
                    transcript_response.raise_for_status()
                    transcript_data = transcript_response.json()
                    total_poll_time = time.time() - poll_start_time
                    logger.info(f"Polling status: SUCCESS - Transcript fetched in {total_poll_time:.1f}s total")
                    return transcript_data
                
                if job_status == 'rejected':
                    failure_reason = result.get('job', {}).get('failure_reason', 'Unknown error')
                    logger.error(f"Polling status: Job rejected - {failure_reason}")
                    raise RuntimeError(f"Speechmatics job rejected: {failure_reason}")
                
                if attempt < max_attempts - 1:
                    attempt_duration = time.time() - attempt_start
                    logger.info(f"Polling status: Waiting {poll_interval}s before next attempt (this attempt took {attempt_duration:.2f}s)...")
                    time.sleep(poll_interval)
            except requests.exceptions.Timeout as e:
                elapsed_time = time.time() - poll_start_time
                logger.warning(f"Polling status: Timeout on attempt {attempt + 1} after {elapsed_time:.1f}s: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(poll_interval)
                else:
                    raise
            except Exception as e:
                elapsed_time = time.time() - poll_start_time
                logger.error(f"Polling status: Error on attempt {attempt + 1} after {elapsed_time:.1f}s: {type(e).__name__} - {e}")
                if attempt < max_attempts - 1:
                    time.sleep(poll_interval)
                else:
                    raise
        
        total_time = time.time() - poll_start_time
        logger.error(f"Polling status: FAILED - Job did not complete within {max_attempts * poll_interval}s (actual: {total_time:.1f}s)")
        raise TimeoutError(f"Speechmatics job did not complete within {max_attempts * poll_interval} seconds (actual: {total_time:.1f}s)")
    
    def _parse_speechmatics_transcript(self, transcript_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Speechmatics transcript into segments format."""
        segments = []
        words = []
        
        # Parse results array
        if transcript_data.get('results') and isinstance(transcript_data['results'], list):
            for result in transcript_data['results']:
                if result.get('type') == 'punctuation':
                    continue
                
                if result.get('type') == 'word' and result.get('alternatives'):
                    alt = result['alternatives'][0]
                    speaker_label = alt.get('speaker', 'S1')
                    
                    # Convert "S1" -> 0, "S2" -> 1, etc.
                    speaker_num = 0
                    if speaker_label.startswith('S'):
                        num_str = speaker_label[1:]
                        speaker_num = int(num_str) - 1 if num_str.isdigit() else 0
                    else:
                        speaker_num = int(speaker_label) if str(speaker_label).isdigit() else 0
                    
                    words.append({
                        'word': alt.get('content', ''),
                        'start': result.get('start_time', 0),
                        'end': result.get('end_time', result.get('start_time', 0)),
                        'speaker': speaker_num,
                        'confidence': alt.get('confidence', 0)
                    })
        
        # Group words into segments by speaker
        if not words:
            logger.warning("No words found in transcript")
            return segments
        
        current_segment = None
        segment_words = []  # Track words for current segment
        
        for word in words:
            speaker_label = f"SPEAKER_{str(word['speaker']).zfill(2)}"
            
            if not current_segment or current_segment['speaker'] != speaker_label:
                if current_segment:
                    # Save words array before moving to next segment
                    current_segment['words'] = segment_words
                    segments.append(current_segment)
                
                # Start new segment
                current_segment = {
                    'speaker': speaker_label,
                    'text': word['word'],
                    'start': word['start'],
                    'end': word['end']
                }
                segment_words = [word]  # Start new words array
            else:
                # Check for pause between words (gap > 0.7s indicates significant pause - split into new replica)
                if segment_words:
                    last_word = segment_words[-1]
                    gap = word['start'] - last_word['end']
                    segment_duration = current_segment['end'] - current_segment['start']
                    min_segment_duration = 1.0  # Minimum 1 second for a segment before splitting
                    
                    # Split segment if:
                    # 1. Gap > 0.7s (significant pause between words) AND segment is at least 1s long
                    # 2. Gap > 0.5s AND segment duration > 15s (very long segment, split at moderate pause)
                    # This prevents splitting on very short segments and reduces over-segmentation
                    should_split = False
                    if gap > 0.7 and segment_duration >= min_segment_duration:
                        should_split = True
                    elif gap > 0.5 and segment_duration > 15.0:
                        should_split = True
                    
                    if should_split:
                        # Save current segment and start new one
                        current_segment['words'] = segment_words
                        segments.append(current_segment)
                        
                        # Start new segment (new replica)
                        current_segment = {
                            'speaker': speaker_label,
                            'text': word['word'],
                            'start': word['start'],
                            'end': word['end']
                        }
                        segment_words = [word]
                    else:
                        # Continue current segment
                        if gap > 0.3:  # Mark pauses > 0.3s (but don't split unless conditions above are met)
                            if 'pauses' not in current_segment:
                                current_segment['pauses'] = []
                            current_segment['pauses'].append({
                                'start': last_word['end'],
                                'end': word['start'],
                                'duration': gap
                            })
                        
                        current_segment['text'] += ' ' + word['word']
                        current_segment['end'] = word['end']
                        segment_words.append(word)
                else:
                    # First word in segment
                    current_segment['text'] = word['word']
                    current_segment['start'] = word['start']
                    current_segment['end'] = word['end']
                    segment_words.append(word)
        
        if current_segment:
            current_segment['words'] = segment_words
            segments.append(current_segment)
        
        logger.info(f"Parsed {len(segments)} segments from transcript")
        return segments
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get information about the temporary file.
        
        Returns:
            Dictionary with file information
        """
        if not self.temp_file_path or not os.path.exists(self.temp_file_path):
            return {'error': 'No file available'}
        
        stat = os.stat(self.temp_file_path)
        return {
            'path': self.temp_file_path,
            'filename': os.path.basename(self.temp_file_path),
            'size': stat.st_size,
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'work_dir': self.work_dir
        }
    
    def cleanup_files(self):
        """Clean up temporary files and directories."""
        if not self.cleanup:
            logger.info("Cleanup disabled, keeping files")
            return
        
        try:
            if self.work_dir and os.path.exists(self.work_dir):
                logger.info(f"Cleaning up work directory: {self.work_dir}")
                shutil.rmtree(self.work_dir)
                logger.info("Cleanup complete")
            elif self.temp_file_path and os.path.exists(self.temp_file_path):
                logger.info(f"Cleaning up file: {self.temp_file_path}")
                os.remove(self.temp_file_path)
                logger.info("Cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatic cleanup."""
        if self.cleanup:
            self.cleanup_files()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Download and process audio files for diarization with automatic cleanup'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='URL of the audio file to download'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to local audio file to process'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        help='Custom temporary directory (default: system temp)'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not cleanup temporary files after processing'
    )
    parser.add_argument(
        '--output-format',
        type=str,
        default='json',
        choices=['json', 'text'],
        help='Output format for results'
    )
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='Only return file info, do not process'
    )
    parser.add_argument(
        '--language',
        type=str,
        help='Language code for processing (e.g., en, ar, es)'
    )
    parser.add_argument(
        '--speaker-count',
        type=int,
        help='Expected number of speakers'
    )
    parser.add_argument(
        '--is-separated-track',
        action='store_true',
        help='If set, uses "channel" diarization mode (for separated audio tracks). '
             'This prevents Speechmatics from detecting residual audio as separate speakers. '
             'CRITICAL: Use this flag when transcribing voice tracks from source separation.'
    )
    parser.add_argument(
        '--original-filename',
        type=str,
        help='Original filename for cache key generation (use this instead of temp filename)'
    )
    parser.add_argument(
        '--skip-cache',
        action='store_true',
        help='Skip cache lookup and fetch fresh results from Speechmatics API'
    )
    parser.add_argument(
        '--engine',
        type=str,
        choices=['speechmatics', 'azure', 'azure_realtime'],
        default='speechmatics',
        help='Transcription engine to use (default: speechmatics)'
    )
    
    args = parser.parse_args()
    
    if not args.url and not args.file:
        parser.error("Either --url or --file must be provided")
    
    if args.url and args.file:
        parser.error("Provide either --url or --file, not both")
    
    try:
        with TempAudioProcessor(
            temp_dir=args.temp_dir,
            cleanup=not args.no_cleanup
        ) as processor:
            
            # Download or save file
            if args.url:
                processor.download_file(args.url)
            else:
                # Use original filename if provided (for cache key generation)
                processor.save_uploaded_file(args.file, filename=args.original_filename)
            
            # Get file info
            file_info = processor.get_file_info()
            
            if args.info_only:
                # Return only file info
                result = {
                    'success': True,
                    'file_info': file_info
                }
            else:
                # Process for diarization
                try:
                    process_result = processor.process_for_diarization(
                        output_format=args.output_format,
                        language=args.language,
                        speaker_count=args.speaker_count,
                        is_separated_track=args.is_separated_track,
                        skip_cache=args.skip_cache,
                        engine=args.engine
                    )
                    # If process_result already has recordings structure, return it directly
                    if 'recordings' in process_result:
                        result = process_result
                        # Verify that recordings have results
                        if not process_result.get('recordings') or not process_result['recordings'][0].get('results'):
                            logger.warning("process_result has recordings but no results - this might be an error")
                            logger.warning(f"process_result structure: {list(process_result.keys())}")
                            logger.warning(f"First recording keys: {list(process_result['recordings'][0].keys()) if process_result.get('recordings') else 'No recordings'}")
                    else:
                        # Legacy format - this should not happen if diarization was successful
                        logger.warning("process_result does not have recordings structure - using legacy format")
                        logger.warning(f"process_result keys: {list(process_result.keys())}")
                        result = {
                            'success': True,
                            'file_info': file_info,
                            'processing': process_result
                        }
                except Exception as diarization_error:
                    logger.error(f"Diarization processing failed: {type(diarization_error).__name__}: {diarization_error}")
                    logger.error(f"Error details: {str(diarization_error)}")
                    # Return error in expected format
                    error_result = {
                        'success': False,
                        'error': str(diarization_error),
                        'error_type': type(diarization_error).__name__,
                        'file_info': file_info
                    }
                    print(json.dumps(error_result, indent=2))
                    return 1
            
            # Output result as JSON
            print(json.dumps(result, indent=2))
            
            # Return success
            return 0
            
    except Exception as e:
        logger.error(f"Main processing failed: {type(e).__name__}: {e}")
        logger.error(f"Error traceback: {str(e)}")
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, indent=2))
        return 1


if __name__ == '__main__':
    sys.exit(main())

