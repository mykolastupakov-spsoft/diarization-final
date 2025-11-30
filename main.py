#!/usr/bin/env python3
"""
Command line entrypoint for the multi-speaker processing pipeline.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv, find_dotenv, dotenv_values

from processor import ProcessingPipeline, TrackResult, AudioShakeError


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an audio file into speakers, transcribe, and classify roles."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file to process.",
    )
    parser.add_argument(
        "--stems-dir",
        type=str,
        default=None,
        help="Directory to store separated stems (default: auto-created temp dir).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output.",
    )
    parser.add_argument(
        "--audio-url",
        type=str,
        default=None,
        help="Public HTTPS URL to the audio file (required for AudioShake Tasks API).",
    )
    return parser.parse_args()


def serialize_results(results: List[TrackResult]) -> List[Any]:
    return [result.to_dict() for result in results]


def main() -> int:
    args = parse_args()
    
    # Debug: де ми зараз?
    cwd = os.getcwd()
    script_dir = Path(__file__).parent.absolute()
    logger.info(f"Current working directory: {cwd}")
    logger.info(f"Script location: {script_dir}")
    
    # Використовуємо dotenv_values() для надійного читання з файлу (без кешування)
    env_file = find_dotenv()
    if not env_file:
        # Fallback: спробувати в директорії скрипта та поточній
        env_paths_to_try = [
            script_dir / ".env",
            Path(".env"),
            Path.cwd() / ".env",
        ]
        for env_path in env_paths_to_try:
            if env_path.exists() and env_path.is_file():
                env_file = str(env_path)
                break
    
    env_config = {}
    if env_file:
        logger.info(f"Loading .env from: {env_file}")
        # Використовуємо dotenv_values() - завжди читає з файлу, ігнорує кеш
        env_config = dotenv_values(dotenv_path=env_file)
        logger.info(f"Loaded {len(env_config)} variables from .env file using dotenv_values()")
        
        # Очистити старі placeholder значення з os.environ перед завантаженням нових
        keys_to_check = ['AUDIOSHAKE_API_KEY', 'OPENROUTER_API_KEY']
        for key in keys_to_check:
            if key in os.environ:
                old_value = os.environ[key]
                if old_value and ('your_' in old_value.lower() or '_here' in old_value.lower()):
                    logger.info(f"Clearing old placeholder {key} from environment cache")
                    del os.environ[key]
        
        # Завантажити з override=True для перезапису всіх змінних
        load_dotenv(dotenv_path=env_file, override=True, verbose=True)
        
        # Також встановити з dotenv_values для гарантії (перезаписуємо os.environ)
        for key, value in env_config.items():
            if value and value.strip():  # Тільки якщо значення не порожнє
                os.environ[key] = value
                logger.debug(f"Set {key} from dotenv_values (length: {len(value)})")
    else:
        logger.warning(f".env file not found. Tried: {script_dir / '.env'}, {Path('.env').absolute()}")
        logger.info("Attempting to load from environment variables (passed from Node.js)...")
        # Fallback: try to load from environment variables (passed from Node.js)
        load_dotenv(override=False)
    
    # Verify API keys are loaded - пріоритет dotenv_values (читає з файлу), потім os.getenv
    if env_config:
        audioshake_key = env_config.get("AUDIOSHAKE_API_KEY") or os.getenv("AUDIOSHAKE_API_KEY")
        openrouter_key = env_config.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    else:
        audioshake_key = os.getenv("AUDIOSHAKE_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    logger.info(f"AUDIOSHAKE_API_KEY present: {audioshake_key is not None}")
    if audioshake_key:
        # Показуємо тільки перші 20 символів для безпеки
        preview = audioshake_key[:20] + "..." if len(audioshake_key) > 20 else audioshake_key
        logger.info(f"AUDIOSHAKE_API_KEY value: {preview}")
        # Перевірка на placeholder
        if audioshake_key == "your_audioshake_api_key_here" or not audioshake_key.strip():
            error_msg = (
                "AUDIOSHAKE_API_KEY appears to be a placeholder or empty!\n"
                "Please update your .env file with a real API key:\n"
                "1. Open .env file\n"
                "2. Replace 'your_audioshake_api_key_here' with your actual API key\n"
                "3. Make sure there are NO spaces around the = sign\n"
                "4. Format: AUDIOSHAKE_API_KEY=ashke_xxx (no quotes, no spaces)"
            )
            logger.error(error_msg)
            print(json.dumps({
                "success": False,
                "error": "AudioShake API key is not configured",
                "details": "Please set AUDIOSHAKE_API_KEY in your .env file with a real API key (not 'your_audioshake_api_key_here')"
            }, indent=2))
            return 1
    else:
        error_msg = (
            "AUDIOSHAKE_API_KEY is not set!\n"
            "Please add it to your .env file:\n"
            "AUDIOSHAKE_API_KEY=your_actual_key_here"
        )
        logger.error(error_msg)
        print(json.dumps({
            "success": False,
            "error": "AudioShake API key is missing",
            "details": "Please set AUDIOSHAKE_API_KEY in your .env file"
        }, indent=2))
        return 1
    
    logger.info(f"OPENROUTER_API_KEY present: {openrouter_key is not None}")
    if not openrouter_key:
        logger.warning(
            "OPENROUTER_API_KEY is not set. "
            "This is OK if you're using /api/audioshake-tasks endpoint (uses Speechmatics for transcription). "
            "But /api/audioshake-pipeline (legacy) requires OpenRouter for speaker role classification."
        )
    
    # Показати всі змінні з AUDIOSHAKE (для діагностики)
    logger.debug("All environment variables starting with AUDIOSHAKE:")
    for key, value in os.environ.items():
        if 'AUDIOSHAKE' in key.upper():
            preview = value[:20] + "..." if len(value) > 20 else value
            logger.debug(f"  {key}: {preview}")

    pipeline = ProcessingPipeline(stems_dir=args.stems_dir)

    # Отримати URL з параметра або environment variable
    audio_url = args.audio_url or os.getenv("AUDIOSHAKE_AUDIO_URL")
    if not audio_url:
        logger.error("AudioShake Tasks API requires a public HTTPS URL. "
                    "Please provide --audio-url parameter or set AUDIOSHAKE_AUDIO_URL environment variable.")
        print(
            json.dumps(
                {"success": False, "error": "AudioShake Tasks API requires a public HTTPS URL"},
                indent=2,
            )
        )
        return 1

    try:
        start_time = time.time()
        results = pipeline.run(args.audio_path, audio_url=audio_url)
        duration = round(time.time() - start_time, 2)
    except (AudioShakeError, FileNotFoundError, TimeoutError, ValueError) as error:
        logger.error("Pipeline failed: %s", error)
        print(
            json.dumps(
                {"success": False, "error": str(error)},
                indent=2,
            )
        )
        return 1

    metrics = {
        "processing_seconds": duration,
        "track_count": len(results),
    }

    payload = {
        "success": True,
        "results": serialize_results(results),
        "metrics": metrics,
        "output_directory": str(pipeline.stems_dir),
    }
    json_payload = json.dumps(payload, indent=2 if args.pretty else None)
    print(json_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

