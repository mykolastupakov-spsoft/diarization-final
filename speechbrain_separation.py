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
import time
from pathlib import Path

import torch
import pyannote_patch  # noqa: F401  # ensures torchaudio compatibility on Python 3.14+
import torchaudio
import soundfile as sf
import numpy as np
from speechbrain.inference.separation import (
    SepformerSeparation as Separator,
)
import torch.nn.functional as F
try:
    from speechbrain.inference.VAD import VAD
    VAD_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        from speechbrain.pretrained import VAD
        VAD_AVAILABLE = True
    except (ImportError, AttributeError):
        VAD_AVAILABLE = False
        # Log will be done later when needed


def log_error(message):
    print(message, file=sys.stderr)
    sys.stderr.flush()


def log_info(message):
    """Log info message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_error(f"[{timestamp}] {message}")


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


def load_waveform(audio_path: str, target_sample_rate: int = 16000):
    """
    Loads audio as torch tensor (shape [1, samples]) and sample rate.
    Prefers soundfile to avoid torchcodec dependency, falls back to torchaudio.
    Обов'язково ресемплює до target_sample_rate (за замовчуванням 16000 Гц).
    """
    log_info("📁 Завантаження аудіо файлу...")
    log_info(f"   Шлях: {audio_path}")
    
    file_size = os.path.getsize(audio_path)
    file_size_mb = file_size / (1024 * 1024)
    log_info(f"   Розмір файлу: {file_size_mb:.2f} MB ({file_size} байт)")
    
    try:
        load_start = time.time()
        data, sample_rate = sf.read(audio_path, dtype='float32', always_2d=True)
        waveform = torch.from_numpy(data.T)
        load_time = time.time() - load_start
        
        log_info(f"✅ Завантажено через soundfile за {load_time:.2f} сек")
        log_info(f"   Форма хвилі: {waveform.shape}")
        log_info(f"   Sample rate: {sample_rate} Hz")
        log_info(f"   Кількість каналів: {waveform.shape[0]}")
        log_info(f"   Кількість зразків: {waveform.shape[1]}")
        log_info(f"   Тривалість: {waveform.shape[1] / sample_rate:.2f} сек")
    except Exception as sf_error:
        log_error(f"[SpeechBrain] soundfile load failed ({sf_error}), falling back to torchaudio")
        load_start = time.time()
        waveform, sample_rate = torchaudio.load(audio_path)
        load_time = time.time() - load_start
        
        log_info(f"✅ Завантажено через torchaudio за {load_time:.2f} сек")
        log_info(f"   Форма хвилі: {waveform.shape}")
        log_info(f"   Sample rate: {sample_rate} Hz")
    
    # Конвертація в моно, якщо стерео
    if waveform.shape[0] > 1:
        log_info(f"🔄 Конвертація з {waveform.shape[0]} каналів в моно")
        waveform = waveform.mean(dim=0, keepdim=True)
        log_info(f"   Нова форма: {waveform.shape}")
    
    # Обов'язковий ресемплінг до target_sample_rate
    if sample_rate != target_sample_rate:
        log_info(f"🔄 Обов'язковий ресемплінг з {sample_rate}Hz на {target_sample_rate}Hz")
        resample_start = time.time()
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        resample_time = time.time() - resample_start
        log_info(f"✅ Ресемплінг завершено за {resample_time:.2f} сек")
        sample_rate = target_sample_rate
    
    return waveform, sample_rate


def normalize_audio_level(waveform: torch.Tensor, method: str = 'peak', target_level: float = 0.80):
    """
    Нормалізує рівень гучності аудіо для уникнення плутанини спікерів через перепади гучності.
    Використовує більш агресивну нормалізацію для кращого вирівнювання гучності.
    
    Args:
        waveform: Tensor форми [channels, samples] або [1, samples]
        method: Метод нормалізації ('peak' або 'rms')
        target_level: Цільовий рівень (0.0-1.0 для peak, дБ для RMS)
    
    Returns:
        Нормалізований waveform
    """
    if waveform.numel() == 0:
        return waveform
    
    # Зберігаємо оригінальну форму
    original_shape = waveform.shape
    was_2d = waveform.dim() == 2
    
    # Перетворюємо в 1D для обчислень
    if was_2d:
        flat_waveform = waveform.flatten()
    else:
        flat_waveform = waveform
    
    # Обчислюємо статистику
    max_val = torch.abs(flat_waveform).max()
    rms_val = torch.sqrt(torch.mean(flat_waveform ** 2))
    
    if method == 'peak':
        # Peak normalization - вирівнюємо до target_level від максимального значення
        # Більш агресивна нормалізація для кращого вирівнювання
        if max_val > 0:
            scale_factor = target_level / max_val
            normalized = flat_waveform * scale_factor
        else:
            normalized = flat_waveform
    elif method == 'rms':
        # RMS normalization - вирівнюємо RMS до цільового рівня
        # Якщо target_level < 1.0, інтерпретуємо як лінійне значення (0.0-1.0)
        # Якщо target_level >= 1.0, інтерпретуємо як дБ
        if target_level < 1.0:
            target_rms = target_level  # Лінійне значення
        else:
            target_rms = 10 ** (target_level / 20)  # Конвертуємо дБ в лінійну шкалу
        
        if rms_val > 0:
            scale_factor = target_rms / rms_val
            # Обмежуємо масштабування, щоб уникнути кліпінгу (але дозволяємо більш агресивну нормалізацію)
            if scale_factor * max_val > 0.90:
                scale_factor = 0.90 / max_val
            normalized = flat_waveform * scale_factor
        else:
            normalized = flat_waveform
    else:
        # За замовчуванням - peak normalization
        if max_val > 0:
            scale_factor = target_level / max_val
            normalized = flat_waveform * scale_factor
        else:
            normalized = flat_waveform
    
    # Відновлюємо оригінальну форму
    if was_2d:
        normalized = normalized.reshape(original_shape)
    
    return normalized


def apply_spectral_gating(separated_sources, mixture, gate_threshold=0.1, gate_alpha=0.5):
    """
    Застосовує спектральне гейтування для покращення розділення одночасних голосів.
    Видаляє залишки іншого спікера з кожного джерела.
    
    Args:
        separated_sources: Tensor форми [speakers, samples] - розділені джерела
        mixture: Tensor форми [1, samples] - оригінальна суміш
        gate_threshold: Поріг для гейтування (0.0-1.0)
        gate_alpha: Коефіцієнт для м'якого гейтування (0.0-1.0)
    
    Returns:
        Покращені розділені джерела
    """
    if separated_sources.numel() == 0 or mixture.numel() == 0:
        return separated_sources
    
    # Переконуємося, що mixture має форму [1, samples]
    if mixture.dim() == 1:
        mixture = mixture.unsqueeze(0)
    
    # Обчислюємо енергію кожного джерела
    sources_energy = torch.abs(separated_sources)
    mixture_energy = torch.abs(mixture.squeeze(0))
    
    # Обчислюємо відносну енергію кожного джерела відносно суміші
    total_energy = sources_energy.sum(dim=0, keepdim=True) + 1e-8
    relative_energy = sources_energy / total_energy
    
    # Створюємо м'яку маску гейтування
    # Джерела з високою відносною енергією зберігаються, з низькою - приглушуються
    gate_mask = torch.clamp((relative_energy - gate_threshold) / (1.0 - gate_threshold + 1e-8), 0, 1)
    gate_mask = gate_alpha + (1.0 - gate_alpha) * gate_mask
    
    # Застосовуємо маску
    gated_sources = separated_sources * gate_mask
    
    return gated_sources


def enhance_speaker_differences(separated_sources, enhancement_strength=0.3):
    """
    Підсилює відмінності між спікерами для кращого розділення одночасних голосів.
    Використовує спектральне підсилення для підкреслення унікальних характеристик кожного голосу.
    
    Args:
        separated_sources: Tensor форми [speakers, samples] - розділені джерела
        enhancement_strength: Сила підсилення (0.0-1.0)
    
    Returns:
        Покращені розділені джерела
    """
    if separated_sources.numel() == 0 or enhancement_strength <= 0:
        return separated_sources
    
    num_speakers = separated_sources.shape[0]
    if num_speakers < 2:
        return separated_sources
    
    # Обчислюємо середнє значення всіх джерел
    mean_source = separated_sources.mean(dim=0, keepdim=True)
    
    # Віднімаємо середнє від кожного джерела для підсилення відмінностей
    differences = separated_sources - mean_source
    
    # Підсилюємо відмінності
    enhanced = separated_sources + enhancement_strength * differences
    
    # Обмежуємо, щоб уникнути кліпінгу
    max_val = torch.abs(enhanced).max()
    if max_val > 0.95:
        enhanced = enhanced * (0.95 / max_val)
    
    return enhanced


def adaptive_volume_tracking(separated_sources, speaker_energy_history=None, alpha=0.9):
    """
    Адаптивно відстежує та нормалізує гучність кожного спікера для запобігання світчу голосів
    при зміні гучності. Використовує експоненційне згладжування для підтримки консистентності.
    
    Args:
        separated_sources: Tensor форми [speakers, samples] - розділені джерела
        speaker_energy_history: Словник з історією енергії кожного спікера {speaker_idx: energy}
        alpha: Коефіцієнт згладжування (0.0-1.0), вищий = більше збереження історії
    
    Returns:
        Нормалізовані джерела та оновлена історія енергії
    """
    if separated_sources.numel() == 0:
        return separated_sources, speaker_energy_history or {}
    
    num_speakers = separated_sources.shape[0]
    if speaker_energy_history is None:
        speaker_energy_history = {}
    
    # Обчислюємо поточну RMS енергію кожного спікера
    current_energies = {}
    normalized_sources = separated_sources.clone()
    
    for speaker_idx in range(num_speakers):
        speaker_audio = separated_sources[speaker_idx, :]
        current_rms = torch.sqrt(torch.mean(speaker_audio ** 2))
        current_energies[speaker_idx] = current_rms.item()
        
        # Якщо є історія, використовуємо адаптивну нормалізацію
        if speaker_idx in speaker_energy_history:
            # Експоненційне згладжування для оновлення цільової енергії
            target_energy = alpha * speaker_energy_history[speaker_idx] + (1 - alpha) * current_rms.item()
            
            # Нормалізуємо до цільової енергії (якщо поточна значно відрізняється)
            if current_rms > 1e-6 and target_energy > 1e-6:
                # Обчислюємо коефіцієнт нормалізації
                # Використовуємо м'яку нормалізацію, щоб не перекручувати сигнал
                energy_ratio = target_energy / current_rms.item()
                
                # Застосовуємо нормалізацію тільки якщо різниця значна (>20%)
                if abs(energy_ratio - 1.0) > 0.2:
                    # М'яка нормалізація - не повна, щоб зберегти природність
                    normalization_factor = 0.7 + 0.3 * energy_ratio  # 70% оригіналу + 30% нормалізованого
                    normalized_sources[speaker_idx, :] = speaker_audio * normalization_factor
                
                # Оновлюємо історію
                speaker_energy_history[speaker_idx] = target_energy
            else:
                speaker_energy_history[speaker_idx] = current_rms.item()
        else:
            # Перший раз - зберігаємо поточну енергію як базову
            speaker_energy_history[speaker_idx] = current_rms.item()
    
    # Обмежуємо, щоб уникнути кліпінгу
    max_val = torch.abs(normalized_sources).max()
    if max_val > 0.95:
        normalized_sources = normalized_sources * (0.95 / max_val)
    
    return normalized_sources, speaker_energy_history


def apply_dynamic_speaker_gating(separated_sources, mixture, speaker_energy_history=None, 
                                  min_energy_ratio=0.3, gate_strength=0.6):
    """
    Застосовує динамічне гейтування на основі локальної енергії кожного спікера.
    Запобігає світчу голосів, коли один спікер стає тихішим.
    
    Args:
        separated_sources: Tensor форми [speakers, samples] - розділені джерела
        mixture: Tensor форми [1, samples] - оригінальна суміш
        speaker_energy_history: Історія енергії спікерів
        min_energy_ratio: Мінімальне співвідношення енергії для збереження сигналу (0.0-1.0)
        gate_strength: Сила гейтування (0.0-1.0)
    
    Returns:
        Покращені розділені джерела
    """
    if separated_sources.numel() == 0 or mixture.numel() == 0:
        return separated_sources
    
    num_speakers = separated_sources.shape[0]
    if num_speakers < 2:
        return separated_sources
    
    # Переконуємося, що mixture має форму [1, samples]
    if mixture.dim() == 1:
        mixture = mixture.unsqueeze(0)
    
    # Обчислюємо локальну енергію кожного спікера
    sources_energy = torch.abs(separated_sources)
    total_sources_energy = sources_energy.sum(dim=0, keepdim=True) + 1e-8
    
    # Відносна енергія кожного спікера
    relative_energy = sources_energy / total_sources_energy
    
    # Створюємо динамічну маску гейтування
    gated_sources = separated_sources.clone()
    
    for speaker_idx in range(num_speakers):
        speaker_relative_energy = relative_energy[speaker_idx, :]
        
        # Якщо є історія, враховуємо її для стабільності
        if speaker_energy_history and speaker_idx in speaker_energy_history:
            # Обчислюємо локальну RMS енергію
            local_rms = torch.sqrt(torch.mean(separated_sources[speaker_idx, :] ** 2))
            historical_energy = speaker_energy_history[speaker_idx]
            
            # Якщо поточна енергія значно нижча за історичну, застосовуємо гейтування
            if historical_energy > 1e-6:
                energy_ratio = local_rms.item() / historical_energy
                
                # Якщо енергія впала більш ніж на 50%, застосовуємо гейтування
                if energy_ratio < 0.5:
                    # Створюємо маску, яка зберігає сигнал тільки там, де він досить сильний
                    energy_mask = torch.clamp(
                        (speaker_relative_energy - min_energy_ratio) / (1.0 - min_energy_ratio + 1e-8),
                        0, 1
                    )
                    # Комбінуємо з глобальним гейтуванням
                    combined_mask = gate_strength + (1.0 - gate_strength) * energy_mask
                    gated_sources[speaker_idx, :] = separated_sources[speaker_idx, :] * combined_mask
        
        # Застосовуємо базове гейтування на основі відносної енергії
        energy_mask = torch.clamp(
            (speaker_relative_energy - min_energy_ratio) / (1.0 - min_energy_ratio + 1e-8),
            0, 1
        )
        base_mask = gate_strength + (1.0 - gate_strength) * energy_mask
        gated_sources[speaker_idx, :] = gated_sources[speaker_idx, :] * base_mask
    
    return gated_sources


def align_channels(prev_chunk, curr_chunk, overlap_len):
    """
    Fix channel permutation by comparing overlap regions.
    
    Args:
        prev_chunk: Previous chunk tensor [speakers, samples]
        curr_chunk: Current chunk tensor [speakers, samples]
        overlap_len: Length of overlap region in samples
    
    Returns:
        curr_chunk with potentially flipped channels
    """
    if prev_chunk is None or overlap_len <= 0:
        return curr_chunk
    
    # Extract overlap regions
    prev_overlap = prev_chunk[:, -overlap_len:]
    curr_overlap = curr_chunk[:, :overlap_len]
    
    # Calculate L1 distance for direct vs swapped assignment
    direct_dist = (prev_overlap - curr_overlap).abs().sum()
    
    # Flip channels (swap speakers)
    flipped_curr = torch.flip(curr_overlap, dims=[0])
    cross_dist = (prev_overlap - flipped_curr).abs().sum()
    
    # If swapped version is better, flip the entire current chunk
    if cross_dist < direct_dist:
        return torch.flip(curr_chunk, dims=[0])
    
    return curr_chunk


def separate(audio_path: str, output_dir: str, settings: dict = None):
    if settings is None:
        settings = {}
    
    process_start_time = time.time()
    
    log_info("═══════════════════════════════════════════════════════════")
    log_info("🚀 ПОЧАТОК РОЗДІЛЕННЯ ТРЕКІВ (PYTHON)")
    log_info("═══════════════════════════════════════════════════════════")
    
    # 2. Перед початком розділення
    log_info("🔀 ЕТАП 2: ПІДГОТОВКА ДО РОЗДІЛЕННЯ")
    log_info(f"   Стартова точка: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    log_info(f"   Вихідна директорія: {output_dir}")
    
    # Get device from settings or environment
    if settings.get('device'):
        device = settings['device']
    else:
        device = get_device()
    
    cache_dir = os.path.expanduser(
        os.getenv("SPEECHBRAIN_CACHE_DIR", "~/.cache/speechbrain/sepformer-whamr16k")
    )

    log_info(f"   Пристрій: {device}")
    log_info(f"   Директорія кешу: {cache_dir}")
    log_info("   Параметри алгоритму:")
    log_info("     - Модель: SpeechBrain SepFormer WHAMR16k")
    
    # Get settings with defaults
    num_speakers = int(settings.get('numSpeakers', os.getenv("SPEECHBRAIN_NUM_SPEAKERS", "2")))
    target_sample_rate = int(settings.get('sampleRate', os.getenv("SPEECHBRAIN_SAMPLE_RATE", "16000")))
    # Параметри для безпечного chunking (sliding window)
    chunk_size_seconds = float(settings.get('chunkSeconds', os.getenv("SPEECHBRAIN_CHUNK_SECONDS", "10.0")))
    overlap_seconds = float(settings.get('overlapSeconds', os.getenv("SPEECHBRAIN_OVERLAP_SECONDS", "2.0")))
    
    # Quality settings (critical for separation quality)
    # Оптимізовані значення за замовчуванням для кращої якості
    segment_overlap = float(settings.get('segmentOverlap', os.getenv("SPEECHBRAIN_SEGMENT_OVERLAP", "0.5")))  # Більший overlap для кращого зшивання
    min_intersegment_gap = float(settings.get('minIntersegmentGap', os.getenv("SPEECHBRAIN_MIN_INTERSEGMENT_GAP", "0.1")))  # Більший gap для кращого розділення
    strict_mode = settings.get('strictMode', os.getenv("SPEECHBRAIN_STRICT_MODE", "true").lower() == "true")
    vad_threshold = float(settings.get('vadThreshold', os.getenv("SPEECHBRAIN_VAD_THRESHOLD", "0.7")))
    max_speech_duration = float(settings.get('maxSpeechDuration', os.getenv("SPEECHBRAIN_MAX_SPEECH_DURATION", "30")))  # Не обмежуємо, використовуємо chunk size
    
    # Advanced settings
    batch_size = int(settings.get('batchSize', os.getenv("SPEECHBRAIN_BATCH_SIZE", "4")))
    dynamic_batching = settings.get('dynamicBatching', os.getenv("SPEECHBRAIN_DYNAMIC_BATCHING", "false").lower() == "true")
    vad_model = settings.get('vadModel', os.getenv("SPEECHBRAIN_VAD_MODEL", "speechbrain/vad-crdnn-libriparty"))
    diarization_model = settings.get('diarizationModel', os.getenv("SPEECHBRAIN_DIARIZATION_MODEL", "speechbrain/diarization-mfa"))
    
    # Post-processing settings for improving separation of similar voices
    enable_spectral_gating = settings.get('enableSpectralGating', os.getenv("SPEECHBRAIN_ENABLE_SPECTRAL_GATING", "true").lower() == "true")
    spectral_gate_threshold = float(settings.get('spectralGateThreshold', os.getenv("SPEECHBRAIN_SPECTRAL_GATE_THRESHOLD", "0.15")))
    spectral_gate_alpha = float(settings.get('spectralGateAlpha', os.getenv("SPEECHBRAIN_SPECTRAL_GATE_ALPHA", "0.4")))
    enable_speaker_enhancement = settings.get('enableSpeakerEnhancement', os.getenv("SPEECHBRAIN_ENABLE_SPEAKER_ENHANCEMENT", "true").lower() == "true")
    speaker_enhancement_strength = float(settings.get('speakerEnhancementStrength', os.getenv("SPEECHBRAIN_SPEAKER_ENHANCEMENT_STRENGTH", "0.4")))
    
    # Adaptive processing settings for preventing voice switching on volume changes
    enable_adaptive_volume_tracking = settings.get('enableAdaptiveVolumeTracking', os.getenv("SPEECHBRAIN_ENABLE_ADAPTIVE_VOLUME_TRACKING", "true").lower() == "true")
    adaptive_volume_alpha = float(settings.get('adaptiveVolumeAlpha', os.getenv("SPEECHBRAIN_ADAPTIVE_VOLUME_ALPHA", "0.85")))
    enable_dynamic_speaker_gating = settings.get('enableDynamicSpeakerGating', os.getenv("SPEECHBRAIN_ENABLE_DYNAMIC_SPEAKER_GATING", "true").lower() == "true")
    dynamic_gate_min_energy_ratio = float(settings.get('dynamicGateMinEnergyRatio', os.getenv("SPEECHBRAIN_DYNAMIC_GATE_MIN_ENERGY_RATIO", "0.25")))
    dynamic_gate_strength = float(settings.get('dynamicGateStrength', os.getenv("SPEECHBRAIN_DYNAMIC_GATE_STRENGTH", "0.5")))
    
    log_info(f"     - Очікувана кількість спікерів: {num_speakers}")
    log_info(f"     - Target sample rate: {target_sample_rate} Hz")
    log_info(f"     - Розмір чанка (sliding window): {chunk_size_seconds} сек")
    log_info(f"     - Overlap між чанками: {overlap_seconds} сек")
    log_info("")
    log_info("   🎯 Критичні параметри якості розділення:")
    log_info(f"     - Segment Overlap: {segment_overlap} сек (впливає на залишки спікерів)")
    log_info(f"     - Min Intersegment Gap: {min_intersegment_gap} сек (впливає на 'слипання' голосів)")
    log_info(f"     - Strict Mode: {strict_mode} (критично для якості!)")
    log_info(f"     - VAD Threshold: {vad_threshold} (чутливість до голосу)")
    log_info(f"     - Max Speech Duration: {max_speech_duration} сек (максимальна тривалість сегменту)")
    log_info("")
    log_info("   Додаткові параметри:")
    log_info(f"     - Batch Size: {batch_size}")
    log_info(f"     - Dynamic Batching: {dynamic_batching}")
    log_info(f"     - VAD Model: {vad_model}")
    log_info(f"     - Diarization Model: {diarization_model}")
    log_info("")
    log_info("   🎯 Параметри пост-обробки для покращення розділення одночасних голосів:")
    log_info(f"     - Spectral Gating: {enable_spectral_gating} (видалення залишків іншого спікера)")
    if enable_spectral_gating:
        log_info(f"       - Gate Threshold: {spectral_gate_threshold}")
        log_info(f"       - Gate Alpha: {spectral_gate_alpha}")
    log_info(f"     - Speaker Enhancement: {enable_speaker_enhancement} (підсилення відмінностей між голосами)")
    if enable_speaker_enhancement:
        log_info(f"       - Enhancement Strength: {speaker_enhancement_strength}")
    log_info("")
    log_info("   🔄 Адаптивна обробка для запобігання світчу голосів при зміні гучності:")
    log_info(f"     - Adaptive Volume Tracking: {enable_adaptive_volume_tracking} (відстеження та нормалізація гучності)")
    if enable_adaptive_volume_tracking:
        log_info(f"       - Smoothing Alpha: {adaptive_volume_alpha}")
    log_info(f"     - Dynamic Speaker Gating: {enable_dynamic_speaker_gating} (динамічне гейтування на основі енергії)")
    if enable_dynamic_speaker_gating:
        log_info(f"       - Min Energy Ratio: {dynamic_gate_min_energy_ratio}")
        log_info(f"       - Gate Strength: {dynamic_gate_strength}")
    log_info("")

    # Завантаження VAD моделі (якщо доступна та потрібна)
    # Примітка: VAD інтеграція поки що обмежена через складність API
    # Параметри vad_threshold та vad_model зберігаються для майбутньої інтеграції
    vad_model_instance = None
    if VAD_AVAILABLE and vad_threshold > 0:
        try:
            log_info("📦 Спроба завантаження VAD моделі...")
            log_info(f"   VAD Model: {vad_model}")
            log_info(f"   VAD Threshold: {vad_threshold}")
            log_info("   ⚠️  VAD інтеграція поки що в розробці, параметри зберігаються для логування")
            # TODO: Повна інтеграція VAD потребує додаткового тестування
            vad_model_instance = None
        except Exception as e:
            log_error(f"[SpeechBrain] VAD not available: {e}")
            log_error("   Продовжуємо без VAD...")
            vad_model_instance = None
    else:
        if not VAD_AVAILABLE:
            log_info("   ℹ️  VAD не доступна в цій версії SpeechBrain")
        log_info(f"   VAD Threshold: {vad_threshold} (параметр збережено)")

    # Завантаження моделі розділення
    log_info("📦 Завантаження моделі розділення...")
    model_load_start = time.time()
    model = Separator.from_hparams(
        source="speechbrain/sepformer-whamr16k",
        savedir=cache_dir,
        run_opts={"device": device},
    )
    model_load_time = time.time() - model_load_start
    log_info(f"✅ Модель розділення завантажена за {model_load_time:.2f} сек")
    log_info(f"   Кількість спікерів у моделі: {model.hparams.num_spks}")
    log_info("")

    # Завантаження аудіо (з обов'язковим ресемплінгом до target_sample_rate)
    waveform, sample_rate = load_waveform(audio_path, target_sample_rate)
    log_info("")
    
    # Нормалізація гучності для уникнення плутанини спікерів через перепади гучності
    # Більш агресивна нормалізація за замовчуванням (0.80 замість 0.95)
    normalization_method = settings.get('normalizationMethod', os.getenv("SPEECHBRAIN_NORMALIZATION_METHOD", "peak"))
    normalization_level = float(settings.get('normalizationLevel', os.getenv("SPEECHBRAIN_NORMALIZATION_LEVEL", "0.80")))
    
    log_info("🔊 Нормалізація рівня гучності...")
    log_info(f"   Метод: {normalization_method}")
    log_info(f"   Цільовий рівень: {normalization_level}")
    
    normalization_start = time.time()
    waveform_before_norm = waveform.clone()
    max_before = torch.abs(waveform_before_norm).max().item()
    rms_before = torch.sqrt(torch.mean(waveform_before_norm ** 2)).item()
    
    waveform = normalize_audio_level(waveform, method=normalization_method, target_level=normalization_level)
    
    max_after = torch.abs(waveform).max().item()
    rms_after = torch.sqrt(torch.mean(waveform ** 2)).item()
    normalization_time = time.time() - normalization_start
    
    log_info(f"✅ Нормалізація завершена за {normalization_time:.2f} сек")
    log_info(f"   Peak до: {max_before:.4f}, після: {max_after:.4f}")
    log_info(f"   RMS до: {rms_before:.4f}, після: {rms_after:.4f}")
    log_info("")
    
    # Зберігаємо model_load_time для підсумку
    model_load_time_for_summary = model_load_time

    # Застосування VAD (якщо доступна)
    if vad_model_instance is not None:
        log_info("🎤 Застосування VAD для виявлення активних сегментів...")
        vad_start = time.time()
        try:
            # VAD повертає маску активності
            vad_boundaries = vad_model_instance(waveform.squeeze().numpy(), sample_rate)
            vad_time = time.time() - vad_start
            log_info(f"✅ VAD завершено за {vad_time:.2f} сек")
            log_info(f"   Знайдено {len(vad_boundaries)} активних сегментів")
            
            # Фільтруємо сегменти за threshold
            filtered_segments = []
            for start, end in vad_boundaries:
                # Перевіряємо confidence (якщо доступно) або використовуємо threshold
                # Для простоти, приймаємо всі сегменти, які VAD виявив
                if end - start >= 0.1:  # Мінімальна тривалість 100мс
                    filtered_segments.append((start, end))
            
            log_info(f"   Після фільтрації: {len(filtered_segments)} сегментів")
        except Exception as e:
            log_error(f"[SpeechBrain] VAD processing failed: {e}")
            log_error("   Продовжуємо без VAD фільтрації...")
            vad_boundaries = None
    else:
        vad_boundaries = None

    total_samples = waveform.shape[1]
    
    # Параметри для sliding window chunking
    chunk_size_samples = int(chunk_size_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    step_size_samples = chunk_size_samples - overlap_samples  # Крок між чанками
    
    # Store final sample rate for saving files
    final_sample_rate = sample_rate

    log_info(f"📊 Параметри хвилі перед розділенням:")
    log_info(f"   Форма: {waveform.shape}")
    log_info(f"   Тип даних: {waveform.dtype}")
    log_info(f"   Загальна кількість зразків: {total_samples}")
    log_info(f"   Розмір чанка: {chunk_size_samples} зразків ({chunk_size_seconds} сек)")
    log_info(f"   Overlap: {overlap_samples} зразків ({overlap_seconds} сек)")
    log_info(f"   Step size: {step_size_samples} зразків ({step_size_samples / sample_rate:.2f} сек)")
    if vad_boundaries:
        log_info(f"   VAD активні сегменти: {len(vad_boundaries)}")
    log_info("")

    # 3. Під час аналізу аудіо
    log_info("🔀 ЕТАП 3: АНАЛІЗ АУДІО ТА РОЗДІЛЕННЯ")
    
    def separate_chunk(chunk_tensor: torch.Tensor, chunk_index: int = None, total_chunks: int = None):
        chunk_start_time = time.time()
        
        # Нормалізуємо форму тензора до [channels, samples]
        original_shape = chunk_tensor.shape
        log_info(f"   Початкова форма тензора: {original_shape}")
        
        # Видаляємо всі зайві виміри, поки не отримаємо 2D
        while chunk_tensor.dim() > 2:
            chunk_tensor = chunk_tensor.squeeze(0)
        
        # Переконуємося, що маємо форму [channels, samples]
        if chunk_tensor.dim() == 1:
            # Якщо 1D, додаємо channel dimension
            chunk_tensor = chunk_tensor.unsqueeze(0)  # [1, samples]
        elif chunk_tensor.dim() != 2:
            raise ValueError(f"Unexpected tensor dimension after normalization: {chunk_tensor.dim()}, shape: {chunk_tensor.shape}")
        
        # Переконуємося, що маємо один канал (моно)
        if chunk_tensor.shape[0] > 1:
            # Якщо багато каналів, об'єднуємо в моно
            chunk_tensor = chunk_tensor.mean(dim=0, keepdim=True)
        
        chunk_size = chunk_tensor.shape[1]
        chunk_duration = chunk_size / sample_rate
        
        if chunk_index is not None:
            log_info(f"📦 Обробка сегмента {chunk_index + 1}/{total_chunks}")
        else:
            log_info(f"📦 Обробка сегмента")
        log_info(f"   Розмір сегмента: {chunk_size} зразків ({chunk_duration:.2f} сек)")
        log_info(f"   Нормалізована форма: {chunk_tensor.shape} (очікується [1, samples])")
        
        # Переконуємося, що форма правильна перед передачею в модель
        if chunk_tensor.shape[0] != 1:
            raise ValueError(f"Expected 1 channel, got {chunk_tensor.shape[0]} channels. Shape: {chunk_tensor.shape}")
        
        chunk_tensor = chunk_tensor.to(device)
        recognition_start = time.time()
        
        # separate_batch очікує [batch, channels, samples] або [channels, samples]
        # У нас зараз [1, samples] = [channels, samples]
        # Спробуємо передати як [batch, channels, samples] = [1, 1, samples]
        with torch.no_grad():
            # Переконуємося, що маємо правильну форму [channels, samples]
            log_info(f"   Форма перед batch: {chunk_tensor.shape} (очікується [1, samples])")
            
            if chunk_tensor.dim() != 2:
                raise ValueError(f"Expected 2D tensor [channels, samples] for model, got {chunk_tensor.dim()}D. Shape: {chunk_tensor.shape}")
            
            if chunk_tensor.shape[0] != 1:
                raise ValueError(f"Expected 1 channel, got {chunk_tensor.shape[0]} channels. Shape: {chunk_tensor.shape}")
            
            # separate_batch може приймати [channels, samples] і сам додасть batch dimension
            # АБО [batch, channels, samples] якщо вже є batch dimension
            # Спробуємо передати просто [channels, samples] = [1, samples]
            # Якщо це не спрацює, модель сама додасть batch dimension
            log_info(f"   Передаємо в separate_batch форму: {chunk_tensor.shape}")
            
            # Викликаємо separate_batch з [channels, samples] - він сам обробить batch
            result = model.separate_batch(chunk_tensor)
            
            # Результат має форму [batch, samples, speakers] або [samples, speakers]
            log_info(f"   Форма результату від моделі: {result.shape}")
            
            # Нормалізуємо форму до [speakers, samples]
            if result.dim() == 3:
                # [batch, samples, speakers] -> [samples, speakers]
                result = result[0]
            elif result.dim() == 2:
                # [samples, speakers] - вже правильна форма
                pass
            else:
                raise ValueError(f"Unexpected result dimension: {result.dim()}, shape: {result.shape}")
            
            # Транспонуємо з [samples, speakers] в [speakers, samples]
            # SpeechBrain separate_batch повертає [samples, speakers] або [batch, samples, speakers]
            # Нам потрібно [speakers, samples]
            if result.shape[1] == num_speakers and result.shape[0] != num_speakers:
                # Якщо другий вимір = num_speakers, а перший != num_speakers
                # то це [samples, speakers], потрібно транспонувати
                result = result.transpose(0, 1)  # [samples, speakers] -> [speakers, samples]
                log_info(f"   Після транспонування: {result.shape} (очікується [speakers, samples])")
            elif result.shape[0] == num_speakers:
                # Вже правильна форма [speakers, samples]
                log_info(f"   Форма вже правильна: {result.shape} (очікується [speakers, samples])")
            else:
                # Якщо не можемо визначити, спробуємо транспонувати якщо другий вимір = num_speakers
                if result.shape[1] == num_speakers:
                    result = result.transpose(0, 1)
                    log_info(f"   Після транспонування (fallback): {result.shape}")
                else:
                    log_info(f"   ⚠️  Неочікувана форма: {result.shape}, num_speakers: {num_speakers}")
                    # Спробуємо транспонувати навпаки
                    if result.shape[0] == num_speakers:
                        log_info(f"   Форма вже правильна (fallback): {result.shape}")
                    else:
                        raise ValueError(f"Cannot determine correct shape. Result: {result.shape}, num_speakers: {num_speakers}")
        
        recognition_time = time.time() - recognition_start
        result_cpu = result.cpu()
        
        log_info(f"   Час розпізнавання спікерів: {recognition_time:.2f} сек")
        log_info(f"   Фінальна форма результату: {result_cpu.shape}")
        
        # Переконуємося, що форма [speakers, samples]
        if result_cpu.dim() != 2:
            raise ValueError(f"Expected 2D result [speakers, samples], got {result_cpu.dim()}D. Shape: {result_cpu.shape}")
        if result_cpu.shape[0] != num_speakers:
            raise ValueError(f"Expected {num_speakers} speakers in first dimension, got {result_cpu.shape[0]}. Shape: {result_cpu.shape}")
        
        return result_cpu

    separation_start = time.time()
    
    log_info(f"📦 Параметри sliding window chunking:")
    log_info(f"   Chunk size: {chunk_size_samples} зразків ({chunk_size_seconds} сек)")
    log_info(f"   Overlap: {overlap_samples} зразків ({overlap_seconds} сек)")
    log_info(f"   Step size: {step_size_samples} зразків ({step_size_samples / sample_rate:.2f} сек)")
    log_info("")
    
    # Використовуємо sliding window для всіх файлів (навіть коротких)
    if total_samples > chunk_size_samples:
        # Розрахунок кількості чанків
        num_chunks = (total_samples - overlap_samples + step_size_samples - 1) // step_size_samples
        if num_chunks == 0:
            num_chunks = 1
        
        log_info(f"📦 Обробка в {num_chunks} чанках (sliding window)")
        log_info(f"   Загальна тривалість: {total_samples / sample_rate:.2f} сек")
        log_info("")
        
        # Створюємо вікно Ханна для плавного зшивання
        hann_window = torch.hann_window(overlap_samples * 2)
        hann_left = hann_window[:overlap_samples]  # Для початку overlap
        hann_right = hann_window[overlap_samples:]  # Для кінця overlap
        
        # Буфер для накопичення результатів
        output_buffer = torch.zeros((num_speakers, total_samples))
        prev_chunk_result = None
        
        # Історія енергії спікерів для адаптивної обробки
        speaker_energy_history = {}
        
        for chunk_idx in range(num_chunks):
            # Визначаємо позиції чанка
            start = chunk_idx * step_size_samples
            end = min(start + chunk_size_samples, total_samples)
            
            # Витягуємо chunk
            chunk = waveform[:, start:end]
            
            # Переконуємося, що chunk має форму [1, samples]
            if chunk.dim() > 2:
                chunk = chunk.squeeze(0)
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            if chunk.shape[0] > 1:
                chunk = chunk.mean(dim=0, keepdim=True)
            
            log_info(f"📦 Обробка чанка {chunk_idx + 1}/{num_chunks}")
            log_info(f"   Позиція: {start}-{end} зразків ({start/sample_rate:.2f}-{end/sample_rate:.2f} сек)")
            
            # Локальна нормалізація чанка для кращого вирівнювання гучності
            # Це допомагає уникнути перепадів гучності між чанками
            chunk = normalize_audio_level(chunk, method=normalization_method, target_level=normalization_level)
            
            # Обробка чанка в режимі no_grad для економії пам'яті
            with torch.no_grad():
                chunk_result = separate_chunk(chunk, chunk_idx, num_chunks)
            
            # Пост-обробка для покращення розділення одночасних голосів
            if enable_spectral_gating:
                log_info("   🔧 Застосування спектрального гейтування...")
                chunk_result = apply_spectral_gating(
                    chunk_result, 
                    chunk, 
                    gate_threshold=spectral_gate_threshold,
                    gate_alpha=spectral_gate_alpha
                )
            
            if enable_speaker_enhancement:
                log_info("   🔧 Підсилення відмінностей між спікерами...")
                chunk_result = enhance_speaker_differences(
                    chunk_result,
                    enhancement_strength=speaker_enhancement_strength
                )
            
            # Адаптивна обробка для запобігання світчу голосів при зміні гучності
            if enable_adaptive_volume_tracking:
                log_info("   🔄 Адаптивне відстеження гучності спікерів...")
                chunk_result, speaker_energy_history = adaptive_volume_tracking(
                    chunk_result,
                    speaker_energy_history=speaker_energy_history,
                    alpha=adaptive_volume_alpha
                )
            
            if enable_dynamic_speaker_gating:
                log_info("   🔧 Динамічне гейтування спікерів...")
                chunk_result = apply_dynamic_speaker_gating(
                    chunk_result,
                    chunk,
                    speaker_energy_history=speaker_energy_history,
                    min_energy_ratio=dynamic_gate_min_energy_ratio,
                    gate_strength=dynamic_gate_strength
                )
            
            # Вирівнювання каналів (якщо не перший чанк)
            if prev_chunk_result is not None and overlap_samples > 0:
                chunk_result = align_channels(prev_chunk_result, chunk_result, overlap_samples)
            
            # Overlap-Add зшивання з вікном Ханна
            chunk_samples = chunk_result.shape[1]
            actual_end = min(start + chunk_samples, total_samples)
            actual_chunk_samples = actual_end - start
            
            for speaker_idx in range(num_speakers):
                chunk_data = chunk_result[speaker_idx, :actual_chunk_samples]
                
                # Застосовуємо вікно Ханна для overlap regions
                if chunk_idx > 0 and overlap_samples > 0:
                    # Fade in на початку (використовуємо праву частину вікна)
                    fade_len = min(overlap_samples, actual_chunk_samples)
                    if fade_len > 0:
                        chunk_data[:fade_len] *= hann_right[:fade_len]
                
                if chunk_idx < num_chunks - 1 and overlap_samples > 0:
                    # Fade out в кінці (використовуємо ліву частину вікна)
                    fade_start = max(0, actual_chunk_samples - overlap_samples)
                    fade_len = actual_chunk_samples - fade_start
                    if fade_len > 0:
                        chunk_data[fade_start:] *= hann_left[-fade_len:]
                
                # Додаємо до буфера
                output_buffer[speaker_idx, start:actual_end] += chunk_data
            
            prev_chunk_result = chunk_result
            log_info("")
        
        est_sources = output_buffer
        log_info("✅ Sliding window обробка завершена")
    else:
        log_info("📦 Обробка як один сегмент (файл занадто короткий для chunking)")
        log_info("")
        # Додаткова нормалізація перед обробкою (навіть для коротких файлів)
        waveform = normalize_audio_level(waveform, method=normalization_method, target_level=normalization_level)
        with torch.no_grad():
            est_sources = separate_chunk(waveform)
        
        # Пост-обробка для покращення розділення одночасних голосів
        if enable_spectral_gating:
            log_info("🔧 Застосування спектрального гейтування...")
            est_sources = apply_spectral_gating(
                est_sources,
                waveform,
                gate_threshold=spectral_gate_threshold,
                gate_alpha=spectral_gate_alpha
            )
        
        if enable_speaker_enhancement:
            log_info("🔧 Підсилення відмінностей між спікерами...")
            est_sources = enhance_speaker_differences(
                est_sources,
                enhancement_strength=speaker_enhancement_strength
            )
    
    separation_time = time.time() - separation_start
    log_info("")
    log_info(f"✅ Розділення завершено за {separation_time:.2f} сек")
    log_info("")

    # Обробка форми результату
    log_info("🔄 Обробка форми результату...")
    log_info(f"   Початкова форма: {est_sources.shape}")
    log_info(f"   Кількість вимірів: {est_sources.dim()}")
    
    if est_sources.dim() == 3:
        est_sources = est_sources[0]  # [time, num_speakers]
        log_info(f"   Після зменшення вимірів: {est_sources.shape}")

    if est_sources.dim() == 2:
        if est_sources.shape[0] == model.hparams.num_spks:
            # shape [num_speakers, time]
            sources_tensor = est_sources
            log_info(f"   Форма [num_speakers, time]: {sources_tensor.shape}")
        elif est_sources.shape[1] == model.hparams.num_spks:
            sources_tensor = est_sources.transpose(0, 1)
            log_info(f"   Транспоновано до [num_speakers, time]: {sources_tensor.shape}")
        else:
            error_msg = f"Unexpected est_sources shape: {est_sources.shape}"
            log_info(f"❌ {error_msg}")
            raise ValueError(error_msg)
    else:
        error_msg = f"Unsupported est_sources dimension: {est_sources.dim()}"
        log_info(f"❌ {error_msg}")
        raise ValueError(error_msg)

    if est_sources is None or len(est_sources) == 0:
        log_info("❌ Спікери не виявлені в аудіо")
        return {
            "success": False,
            "error": "No speakers detected in audio",
        }

    log_info(f"✅ Знайдено {len(sources_tensor)} джерел")
    log_info("")

    # 4. Після завершення розділення
    log_info("✅ ЕТАП 4: ЗАВЕРШЕННЯ РОЗДІЛЕННЯ")
    log_info(f"   Кількість отриманих фрагментів: {len(sources_tensor)}")
    log_info("")

    # 5. При збереженні результатів
    log_info("💾 ЕТАП 5: ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
    save_start_time = time.time()
    
    speakers = []
    timeline = []

    for idx, source in enumerate(sources_tensor):
        speaker_name = f"SPEAKER_{idx:02d}"
        output_path = os.path.join(output_dir, f"{speaker_name}.wav")
        
        log_info(f"💾 Збереження {idx + 1}/{len(sources_tensor)}: {speaker_name}")
        log_info(f"   Шлях: {output_path}")

        source_np = source.squeeze().numpy()
        source_size = source_np.nbytes
        source_size_mb = source_size / (1024 * 1024)
        log_info(f"   Розмір даних: {source_size_mb:.2f} MB ({source_size} байт)")
        
        write_start = time.time()
        try:
            sf.write(output_path, source_np, final_sample_rate)
            write_time = time.time() - write_start
            
            # Перевірка збереженого файлу
            if os.path.exists(output_path):
                saved_file_size = os.path.getsize(output_path)
                saved_file_size_mb = saved_file_size / (1024 * 1024)
                log_info(f"   ✅ Файл збережено за {write_time:.2f} сек")
                log_info(f"   Розмір файлу: {saved_file_size_mb:.2f} MB ({saved_file_size} байт)")
                log_info(f"   Статус запису: ✅ Успішно")
            else:
                log_info(f"   ❌ Файл не знайдено після запису")
                log_info(f"   Статус запису: ❌ Не вдалося")
        except Exception as e:
            log_info(f"   ❌ Помилка запису: {str(e)}")
            log_info(f"   Статус запису: ❌ Не вдалося")
            raise

        duration = source.shape[-1] / sample_rate
        log_info(f"   Тривалість: {duration:.2f} сек")
        log_info("")

        speakers.append(
            {
                "name": speaker_name,
                "format": "wav",
                "local_path": output_path,
                "isBackground": False,
            }
        )

        timeline.append(
            {
                "speaker": speaker_name,
                "start": 0.0,
                "end": round(duration, 2),
                "duration": round(duration, 2),
            }
        )
    
    save_time = time.time() - save_start_time
    log_info(f"✅ Всі файли збережено за {save_time:.2f} сек")
    log_info("")
    
    total_time = time.time() - process_start_time
    log_info("═══════════════════════════════════════════════════════════")
    log_info("📊 ПІДСУМОК ПРОЦЕСУ")
    log_info("═══════════════════════════════════════════════════════════")
    log_info(f"   Загальний час обробки: {total_time:.2f} сек")
    log_info(f"   Час завантаження моделі: {model_load_time_for_summary:.2f} сек")
    log_info(f"   Час розділення: {separation_time:.2f} сек")
    log_info(f"   Час збереження: {save_time:.2f} сек")
    log_info(f"   Кількість фрагментів: {len(speakers)}")
    log_info(f"   Статус: ✅ Успішно завершено")
    log_info("═══════════════════════════════════════════════════════════")

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
                    "error": "Usage: python speechbrain_separation.py <audio_path> <output_dir> [settings_json]",
                }
            )
        )
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = ensure_output_dir(sys.argv[2])
    
    # Parse settings from JSON (if provided)
    settings = {}
    if len(sys.argv) >= 4:
        try:
            settings = json.loads(sys.argv[3])
        except (json.JSONDecodeError, ValueError) as e:
            log_error(f"[SpeechBrain] Warning: Failed to parse settings JSON: {e}")
            settings = {}

    if not os.path.isfile(audio_path):
        print(
            json.dumps(
                {"success": False, "error": f"Audio file not found: {audio_path}"}
            )
        )
        sys.exit(1)

    try:
        result = separate(audio_path, output_dir, settings)
        print(json.dumps(result))
        sys.exit(0 if result.get("success") else 1)
    except Exception as exc:
        # 6. У випадку помилок
        import traceback
        log_info("")
        log_info("═══════════════════════════════════════════════════════════")
        log_info("❌ ПОМИЛКА ПРОЦЕСУ")
        log_info("═══════════════════════════════════════════════════════════")
        log_info(f"   Код помилки: {type(exc).__name__}")
        log_info(f"   Опис: {str(exc)}")
        log_info("   Стек трасування:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                log_info(f"      {line}")
        log_info("═══════════════════════════════════════════════════════════")
        
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

