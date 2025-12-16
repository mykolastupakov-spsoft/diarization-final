#!/usr/bin/env python3
"""
Діагностичний скрипт для тестування діаризації на файлі speaker_0.wav
"""
import os
import sys
import json

# Додаємо поточну директорію до шляху
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Імпортуємо функції з app_ios_shortcuts.py
from app_ios_shortcuts import (
    extract_speaker_embeddings,
    diarize_audio,
    transcribe_audio,
    combine_diarization_and_transcription
)

def test_diarization(audio_path):
    """Тестує діаризацію на заданому файлі"""
    print(f"🔍 Testing diarization on: {audio_path}")
    print("=" * 80)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    # Тест 1: PyAnnote діаризація
    print("\n📊 TEST 1: PyAnnote Diarization")
    print("-" * 80)
    try:
        import pyannote_patch  # noqa: F401
        from pyannote.audio import Pipeline
        import torch
        import torchaudio
        import soundfile as sf
        
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            print("⚠️  HUGGINGFACE_TOKEN not set, skipping PyAnnote test")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📦 Loading PyAnnote pipeline on {device}...")
            
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            pipeline.to(device)
            
            # Завантажуємо аудіо
            try:
                data, sample_rate = sf.read(audio_path, dtype='float32')
                if len(data.shape) == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0).float()
                else:
                    waveform = torch.from_numpy(data).transpose(0, 1).float()
            except Exception as load_error:
                print(f"⚠️  soundfile failed: {load_error}, trying torchaudio...")
                waveform, sample_rate = torchaudio.load(audio_path)
            
            # Конвертуємо в mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample до 16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Запускаємо діаризацію
            print("🎯 Running PyAnnote diarization...")
            diarization = pipeline({
                "waveform": waveform,
                "sample_rate": sample_rate
            })
            
            # Конвертуємо результат
            diarization_segments = []
            speaker_map = {}
            next_speaker_id = 0
            
            # Збираємо всі унікальні спікерів
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_map:
                    speaker_map[speaker] = next_speaker_id
                    next_speaker_id += 1
            
            # Створюємо сегменти
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_id = speaker_map[speaker]
                diarization_segments.append({
                    'speaker': speaker_id,
                    'start': round(turn.start, 2),
                    'end': round(turn.end, 2)
                })
            
            # Сортуємо за часом
            diarization_segments.sort(key=lambda x: x['start'])
            
            print(f"✅ PyAnnote found {len(diarization_segments)} segments from {len(speaker_map)} speakers")
            print(f"   Speaker mapping: {speaker_map}")
            print(f"\n📋 Diarization segments:")
            for seg in diarization_segments[:20]:  # Перші 20 сегментів
                print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}")
            
            # Підраховуємо тривалість для кожного спікера
            speaker_durations = {}
            for seg in diarization_segments:
                speaker = seg['speaker']
                duration = seg['end'] - seg['start']
                if speaker not in speaker_durations:
                    speaker_durations[speaker] = 0
                speaker_durations[speaker] += duration
            
            print(f"\n📊 Speaker durations:")
            for speaker, dur in sorted(speaker_durations.items()):
                print(f"   Speaker {speaker}: {dur:.2f}s")
            
    except Exception as e:
        print(f"❌ PyAnnote test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 2: SpeechBrain діаризація
    print("\n📊 TEST 2: SpeechBrain Diarization")
    print("-" * 80)
    try:
        print("🔄 Extracting embeddings...")
        embeddings, timestamps = extract_speaker_embeddings(
            audio_path,
            segment_duration=1.5,
            overlap=0.5
        )
        
        if embeddings is None or len(embeddings) == 0:
            print("❌ Failed to extract embeddings")
        else:
            print(f"✅ Extracted {len(embeddings)} embeddings")
            
            # Виконуємо діаризацію
            print("🎯 Running SpeechBrain diarization...")
            diarization_segments_sb = diarize_audio(embeddings, timestamps, num_speakers=2)
            
            if not diarization_segments_sb:
                print("❌ SpeechBrain diarization failed")
            else:
                print(f"✅ SpeechBrain found {len(diarization_segments_sb)} segments")
                print(f"\n📋 Diarization segments:")
                for seg in diarization_segments_sb[:20]:  # Перші 20 сегментів
                    print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}")
                
                # Підраховуємо тривалість для кожного спікера
                speaker_durations_sb = {}
                for seg in diarization_segments_sb:
                    speaker = seg['speaker']
                    duration = seg['end'] - seg['start']
                    if speaker not in speaker_durations_sb:
                        speaker_durations_sb[speaker] = 0
                    speaker_durations_sb[speaker] += duration
                
                print(f"\n📊 Speaker durations:")
                for speaker, dur in sorted(speaker_durations_sb.items()):
                    print(f"   Speaker {speaker}: {dur:.2f}s")
    
    except Exception as e:
        print(f"❌ SpeechBrain test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Тест 3: Транскрипція + об'єднання
    print("\n📊 TEST 3: Transcription + Combination")
    print("-" * 80)
    try:
        print("🎤 Transcribing audio...")
        transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None)
        
        if not words:
            print("❌ No words transcribed")
        else:
            print(f"✅ Transcribed {len(words)} words")
            print(f"\n📋 First 20 words:")
            for word in words[:20]:
                print(f"   [{word['start']:.2f}s - {word['end']:.2f}s] '{word['word']}'")
            
            # Використовуємо PyAnnote діаризацію якщо доступна
            if 'diarization_segments' in locals() and diarization_segments:
                print(f"\n🔗 Combining transcription with PyAnnote diarization...")
                combined = combine_diarization_and_transcription(diarization_segments, words)
                
                print(f"✅ Combined into {len(combined)} segments")
                print(f"\n📋 Combined segments:")
                for seg in combined[:20]:
                    print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}: {seg['text'][:50]}")
                
                # Підраховуємо слова по спікерах
                speaker_word_counts = {}
                for seg in combined:
                    speaker = seg['speaker']
                    word_count = len(seg['text'].split())
                    if speaker not in speaker_word_counts:
                        speaker_word_counts[speaker] = 0
                    speaker_word_counts[speaker] += word_count
                
                print(f"\n📊 Word distribution by speaker:")
                for speaker, count in sorted(speaker_word_counts.items()):
                    print(f"   Speaker {speaker}: {count} words")
    
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ Diagnostics complete")

if __name__ == "__main__":
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    test_diarization(test_file)

