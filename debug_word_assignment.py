#!/usr/bin/env python3
"""
Детальна діагностика призначення слів спікерам
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import (
    extract_speaker_embeddings,
    diarize_audio,
    transcribe_audio
)

def debug_word_assignment(audio_path):
    """Детальна діагностика призначення конкретних слів"""
    print("=" * 80)
    print(f"🔍 WORD ASSIGNMENT DEBUG: {audio_path}")
    print("=" * 80)
    
    # Діаризація
    embeddings, timestamps = extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5)
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    
    # Транскрипція
    transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None)
    
    print(f"\n📊 Diarization segments:")
    for seg in sorted_diar_segments:
        print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']}")
    
    # Тестуємо конкретні проблемні слова
    problem_words = [
        {'word': 'Hey,', 'start': 2.18, 'end': 2.90},
        {'word': 'Try', 'start': 8.42, 'end': 8.64},
        {'word': 'Try', 'start': 65.98, 'end': 66.18},  # Приблизно
    ]
    
    print(f"\n🔍 Testing word assignment logic:")
    for word_info in problem_words:
        word_start = word_info['start']
        word_end = word_info['end']
        word_center = (word_start + word_end) / 2.0
        word_duration = word_end - word_start
        
        print(f"\n   Word: '{word_info['word']}' at [{word_start:.2f}s - {word_end:.2f}s]")
        
        # Знаходимо всі сегменти, що перетинаються
        overlapping_segments = []
        for diar_seg in sorted_diar_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            
            overlap_start = max(word_start, diar_start)
            overlap_end = min(word_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                overlap_ratio = overlap / word_duration if word_duration > 0 else 0
                diar_center = (diar_start + diar_end) / 2.0
                center_distance = abs(word_center - diar_center)
                fully_contained = (word_start >= diar_start and word_end <= diar_end)
                
                overlapping_segments.append({
                    'speaker': diar_seg['speaker'],
                    'overlap': overlap,
                    'overlap_ratio': overlap_ratio,
                    'center_distance': center_distance,
                    'fully_contained': fully_contained,
                    'segment': f"[{diar_start:.2f}s - {diar_end:.2f}s]"
                })
                
                print(f"      Overlaps with Speaker {diar_seg['speaker']} {diar_seg}: overlap={overlap:.3f}s ({overlap_ratio*100:.1f}%), "
                      f"center_dist={center_distance:.3f}s, fully_contained={fully_contained}")
        
        if overlapping_segments:
            # Симулюємо логіку вибору
            fully_contained = [s for s in overlapping_segments if s['fully_contained']]
            if fully_contained:
                best = max(fully_contained, key=lambda x: x['overlap_ratio'])
                print(f"      → Selected: Speaker {best['speaker']} (fully contained, overlap_ratio={best['overlap_ratio']:.3f})")
            else:
                best = max(overlapping_segments, key=lambda x: (x['overlap_ratio'], -x['center_distance']))
                print(f"      → Selected: Speaker {best['speaker']} (overlap_ratio={best['overlap_ratio']:.3f}, center_dist={best['center_distance']:.3f})")
        else:
            print(f"      → No overlapping segments found!")

if __name__ == "__main__":
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    debug_word_assignment(test_file)

