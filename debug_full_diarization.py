#!/usr/bin/env python3
"""
Повна діагностика діаризації та транскрипції
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import (
    extract_speaker_embeddings,
    diarize_audio,
    transcribe_audio,
    combine_diarization_and_transcription
)

def full_diagnostic(audio_path):
    """Повна діагностика всіх кроків"""
    print("=" * 80)
    print(f"🔍 FULL DIAGNOSTIC: {audio_path}")
    print("=" * 80)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    # КРОК 1: Діаризація
    print("\n" + "=" * 80)
    print("STEP 1: DIARIZATION")
    print("=" * 80)
    
    embeddings, timestamps = extract_speaker_embeddings(
        audio_path,
        segment_duration=1.5,
        overlap=0.5
    )
    
    print(f"✅ Extracted {len(embeddings)} embeddings")
    
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    
    print(f"\n📊 Diarization segments ({len(diarization_segments)} total):")
    for i, seg in enumerate(diarization_segments):
        print(f"   {i+1}. [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']} (duration: {seg['end']-seg['start']:.2f}s)")
    
    # Підраховуємо тривалість
    speaker_durations = {}
    for seg in diarization_segments:
        speaker = seg['speaker']
        duration = seg['end'] - seg['start']
        if speaker not in speaker_durations:
            speaker_durations[speaker] = 0
        speaker_durations[speaker] += duration
    
    print(f"\n📊 Speaker durations from diarization:")
    for speaker, dur in sorted(speaker_durations.items()):
        print(f"   Speaker {speaker}: {dur:.2f}s")
    
    # КРОК 2: Транскрипція
    print("\n" + "=" * 80)
    print("STEP 2: TRANSCRIPTION")
    print("=" * 80)
    
    transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None)
    
    print(f"✅ Transcribed {len(words)} words")
    print(f"\n📋 First 30 words with timestamps:")
    for i, word in enumerate(words[:30]):
        print(f"   {i+1}. [{word['start']:.2f}s - {word['end']:.2f}s] '{word['word']}'")
    
    # КРОК 3: Об'єднання
    print("\n" + "=" * 80)
    print("STEP 3: COMBINING DIARIZATION + TRANSCRIPTION")
    print("=" * 80)
    
    combined = combine_diarization_and_transcription(diarization_segments, words)
    
    print(f"✅ Combined into {len(combined)} segments")
    print(f"\n📋 Combined segments:")
    for i, seg in enumerate(combined):
        speakers_in_seg = set()
        # Перевіряємо, які спікери є в діаризації для цього часу
        for diar_seg in diarization_segments:
            if (seg['start'] < diar_seg['end'] and seg['end'] > diar_seg['start']):
                speakers_in_seg.add(diar_seg['speaker'])
        
        diar_speakers = f" (diarization has: {sorted(speakers_in_seg)})" if speakers_in_seg else ""
        print(f"   {i+1}. [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}{diar_speakers}: {seg['text'][:70]}")
    
    # Аналіз проблеми
    print("\n" + "=" * 80)
    print("STEP 4: PROBLEM ANALYSIS")
    print("=" * 80)
    
    speakers_in_combined = set(seg['speaker'] for seg in combined)
    speakers_in_diarization = set(seg['speaker'] for seg in diarization_segments)
    
    print(f"📊 Speakers in diarization: {sorted(speakers_in_diarization)}")
    print(f"📊 Speakers in combined transcription: {sorted(speakers_in_combined)}")
    
    if speakers_in_diarization != speakers_in_combined:
        print(f"\n⚠️  PROBLEM: Speakers mismatch!")
        print(f"   Diarization has: {sorted(speakers_in_diarization)}")
        print(f"   Combined has: {sorted(speakers_in_combined)}")
        
        missing = speakers_in_diarization - speakers_in_combined
        if missing:
            print(f"   ❌ Missing in combined: {sorted(missing)}")
            
            # Знаходимо слова, які мають бути від відсутніх спікерів
            print(f"\n   🔍 Finding words that should be from missing speakers:")
            for missing_speaker in missing:
                # Знаходимо сегменти діаризації для цього спікера
                missing_segments = [s for s in diarization_segments if s['speaker'] == missing_speaker]
                print(f"\n   Speaker {missing_speaker} diarization segments:")
                for seg in missing_segments:
                    # Знаходимо слова в цьому часовому діапазоні
                    words_in_range = [w for w in words if w['start'] < seg['end'] and w['end'] > seg['start']]
                    print(f"      [{seg['start']:.2f}s - {seg['end']:.2f}s]: {len(words_in_range)} words")
                    if words_in_range:
                        print(f"         Words: {[w['word'] for w in words_in_range[:10]]}")
                    
                    # Перевіряємо, який спікер призначений цим словам в combined
                    for comb_seg in combined:
                        if comb_seg['start'] < seg['end'] and comb_seg['end'] > seg['start']:
                            print(f"         → Assigned to Speaker {comb_seg['speaker']} in combined: '{comb_seg['text'][:50]}'")
    
    # Підраховуємо слова по спікерах
    speaker_word_counts = {}
    for seg in combined:
        speaker = seg['speaker']
        word_count = len(seg['text'].split())
        if speaker not in speaker_word_counts:
            speaker_word_counts[speaker] = 0
        speaker_word_counts[speaker] += word_count
    
    print(f"\n📊 Word distribution by speaker in combined:")
    for speaker, count in sorted(speaker_word_counts.items()):
        print(f"   Speaker {speaker}: {count} words")
    
    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    full_diagnostic(test_file)

