#!/usr/bin/env python3
"""
Діагностика ТРІЗ логіки для слова "dropping."
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from app_ios_shortcuts import (
    extract_speaker_embeddings,
    diarize_audio,
    transcribe_audio,
    combine_diarization_and_transcription
)

def debug_triz_logic():
    """Діагностика ТРІЗ логіки"""
    audio_path = "audio examples/detecting main speakers/speaker_0.wav"
    
    # Діаризація
    embeddings, timestamps = extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5)
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    
    # Транскрипція
    transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None)
    
    # Знаходимо слово "dropping."
    dropping_word = None
    for word in words:
        if "dropping" in word['word'].lower():
            dropping_word = word
            break
    
    if not dropping_word:
        print("❌ Word 'dropping' not found")
        return
    
    print("=" * 80)
    print(f"TESTING TRIZ LOGIC FOR: 'dropping.' at [{dropping_word['start']:.2f}s - {dropping_word['end']:.2f}s]")
    print("=" * 80)
    
    # Симулюємо логіку з combine_diarization_and_transcription
    word_start = dropping_word['start']
    word_end = dropping_word['end']
    word_center = (word_start + word_end) / 2.0
    word_duration = word_end - word_start
    
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    
    # Знаходимо overlapping segments
    overlapping_segments = []
    for diar_seg in sorted_diar_segments:
        diar_start = diar_seg['start']
        diar_end = diar_seg['end']
        segment_duration = diar_end - diar_start
        
        overlap_start = max(word_start, diar_start)
        overlap_end = min(word_end, diar_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > 0:
            overlap_ratio = overlap / word_duration if word_duration > 0 else 0
            diar_center = (diar_start + diar_end) / 2.0
            center_distance = abs(word_center - diar_center)
            fully_contained = (word_start >= diar_start and word_end <= diar_end)
            is_suspicious_large = segment_duration > 10.0
            
            overlapping_segments.append({
                'segment': diar_seg,
                'speaker': diar_seg['speaker'],
                'overlap_ratio': overlap_ratio,
                'center_distance': center_distance,
                'fully_contained': fully_contained,
                'segment_duration': segment_duration,
                'is_suspicious_large': is_suspicious_large
            })
    
    print(f"\n📊 Overlapping segments:")
    for seg in overlapping_segments:
        print(f"   Speaker {seg['speaker']}: center_dist={seg['center_distance']:.3f}s, "
              f"overlap={seg['overlap_ratio']*100:.1f}%, "
              f"segment_start={seg['segment']['start']:.2f}s, "
              f"suspicious={seg['is_suspicious_large']}")
    
    # Перевіряємо ТРІЗ логіку для слів на початку файлу
    if word_start < 1.0:
        print(f"\n🔍 Checking nearby other speaker segments (word_start={word_start:.2f}s < 1.0s):")
        
        nearby_other_speaker_segments = []
        current_speakers = set(s['speaker'] for s in overlapping_segments)
        
        for diar_seg in sorted_diar_segments:
            distance_to_start = diar_seg['start'] - word_end
            if distance_to_start >= 0 and distance_to_start < 0.5:
                if diar_seg['speaker'] not in current_speakers:
                    nearby_other_speaker_segments.append({
                        'segment': diar_seg,
                        'speaker': diar_seg['speaker'],
                        'distance_to_start': distance_to_start,
                        'segment_start': diar_seg['start'],
                        'segment_end': diar_seg['end']
                    })
                    print(f"   Found nearby segment: Speaker {diar_seg['speaker']} "
                          f"[{diar_seg['start']:.2f}s - {diar_seg['end']:.2f}s], "
                          f"distance={distance_to_start:.3f}s")
        
        if nearby_other_speaker_segments:
            closest_other = min(nearby_other_speaker_segments, key=lambda x: x['distance_to_start'])
            closest_current_seg = min(overlapping_segments, key=lambda x: x['center_distance'])
            
            print(f"\n   Closest other speaker segment: Speaker {closest_other['speaker']}, "
                  f"distance={closest_other['distance_to_start']:.3f}s")
            print(f"   Closest current segment: Speaker {closest_current_seg['speaker']}, "
                  f"center_dist={closest_current_seg['center_distance']:.3f}s, "
                  f"segment_start={closest_current_seg['segment']['start']:.2f}s")
            
            current_seg_starts_at_zero = closest_current_seg['segment']['start'] < 0.1
            print(f"   Current seg starts at zero: {current_seg_starts_at_zero}")
            print(f"   Current seg is suspicious: {closest_current_seg['is_suspicious_large']}")
            print(f"   Current seg center_distance > 0.2: {closest_current_seg['center_distance'] > 0.2}")
            
            if closest_other['distance_to_start'] < 0.5:
                condition1 = current_seg_starts_at_zero
                condition2 = closest_current_seg['is_suspicious_large']
                condition3 = closest_current_seg['center_distance'] > 0.2
                
                print(f"\n   Conditions for using other speaker:")
                print(f"      distance < 0.5: {closest_other['distance_to_start'] < 0.5} ✅")
                print(f"      starts_at_zero: {condition1}")
                print(f"      is_suspicious: {condition2}")
                print(f"      center_dist > 0.2: {condition3}")
                print(f"      Combined: {condition1 or condition2 or condition3}")
                
                if condition1 or condition2 or condition3:
                    print(f"   → Would use Speaker {closest_other['speaker']}")
                else:
                    print(f"   → Would use Speaker {closest_current_seg['speaker']} (current)")
        else:
            print(f"   No nearby other speaker segments found")
    
    # Тестуємо реальну функцію
    print(f"\n" + "=" * 80)
    print("TESTING REAL combine_diarization_and_transcription FUNCTION")
    print("=" * 80)
    
    combined = combine_diarization_and_transcription(diarization_segments, words)
    
    # Знаходимо сегмент з "dropping."
    for seg in combined:
        if "dropping" in seg.get('text', '').lower():
            print(f"\n   Found in combined: [{seg['start']:.2f}s - {seg['end']:.2f}s] "
                  f"Speaker {seg['speaker']}: {seg['text']}")
            break

if __name__ == "__main__":
    debug_triz_logic()

