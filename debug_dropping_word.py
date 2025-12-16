#!/usr/bin/env python3
"""
Детальна діагностика слова "dropping."
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

def debug_dropping():
    """Детальна діагностика слова 'dropping.'"""
    audio_path = "audio examples/detecting main speakers/speaker_0.wav"
    
    # Діаризація
    embeddings, timestamps = extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5)
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    
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
    
    word_start = dropping_word['start']
    word_end = dropping_word['end']
    word_center = (word_start + word_end) / 2.0
    word_duration = word_end - word_start
    
    print("=" * 80)
    print(f"DEBUGGING WORD: 'dropping.' at [{word_start:.2f}s - {word_end:.2f}s]")
    print("=" * 80)
    
    print(f"\n📊 Word info:")
    print(f"   Start: {word_start:.2f}s")
    print(f"   End: {word_end:.2f}s")
    print(f"   Center: {word_center:.2f}s")
    print(f"   Duration: {word_duration:.2f}s")
    
    print(f"\n📊 All diarization segments:")
    for i, seg in enumerate(sorted_diar_segments):
        seg_duration = seg['end'] - seg['start']
        is_suspicious = seg_duration > 10.0
        marker = " ⚠️ SUSPICIOUS" if is_suspicious else ""
        print(f"   {i+1}. [{seg['start']:.2f}s - {seg['end']:.2f}s] Speaker {seg['speaker']} (duration: {seg_duration:.2f}s){marker}")
    
    print(f"\n🔍 Finding overlapping segments:")
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
                'overlap': overlap,
                'overlap_ratio': overlap_ratio,
                'center_distance': center_distance,
                'fully_contained': fully_contained,
                'segment_duration': segment_duration,
                'is_suspicious_large': is_suspicious_large
            })
            
            print(f"\n   Speaker {diar_seg['speaker']} [{diar_start:.2f}s - {diar_end:.2f}s]:")
            print(f"      Overlap: {overlap:.3f}s ({overlap_ratio*100:.1f}%)")
            print(f"      Center distance: {center_distance:.3f}s")
            print(f"      Fully contained: {fully_contained}")
            print(f"      Segment duration: {segment_duration:.2f}s")
            print(f"      Is suspicious large: {is_suspicious_large}")
    
    if not overlapping_segments:
        print(f"\n   ⚠️  No overlapping segments found!")
        print(f"   This means the word is in a gap between diarization segments")
        return
    
    print(f"\n🎯 Applying TRIZ selection logic:")
    
    # Пріоритет 1: Найближчі сегменти
    close_segments = [
        s for s in overlapping_segments 
        if s['center_distance'] < 0.5 and s['overlap_ratio'] > 0.05
    ]
    
    print(f"\n   Priority 1: Close segments (center_distance <0.5s, overlap_ratio >5%):")
    if close_segments:
        for seg in close_segments:
            print(f"      Speaker {seg['speaker']}: center_dist={seg['center_distance']:.3f}s, overlap={seg['overlap_ratio']*100:.1f}%, suspicious={seg['is_suspicious_large']}")
        
        non_suspicious_close = [s for s in close_segments if not s['is_suspicious_large']]
        if non_suspicious_close:
            best_seg = min(non_suspicious_close, key=lambda x: x['center_distance'])
            print(f"      → Selected: Speaker {best_seg['speaker']} (non-suspicious, closest)")
        else:
            best_seg = min(close_segments, key=lambda x: x['center_distance'])
            print(f"      → Selected: Speaker {best_seg['speaker']} (closest, but suspicious)")
    else:
        print(f"      None found")
        
        # Пріоритет 2: Непідозрілі сегменти
        non_suspicious = [s for s in overlapping_segments if not s['is_suspicious_large']]
        print(f"\n   Priority 2: Non-suspicious segments:")
        if non_suspicious:
            for seg in non_suspicious:
                print(f"      Speaker {seg['speaker']}: center_dist={seg['center_distance']:.3f}s, overlap={seg['overlap_ratio']*100:.1f}%")
            
            valid_non_suspicious = [s for s in non_suspicious if s['overlap_ratio'] > 0.05]
            if valid_non_suspicious:
                best_seg = min(valid_non_suspicious, key=lambda x: x['center_distance'])
                print(f"      → Selected: Speaker {best_seg['speaker']} (non-suspicious, valid overlap)")
            else:
                best_seg = min(non_suspicious, key=lambda x: x['center_distance'])
                print(f"      → Selected: Speaker {best_seg['speaker']} (non-suspicious, closest)")
        else:
            print(f"      None found")
            # Пріоритет 3: Всі сегменти
            best_seg = min(overlapping_segments, key=lambda x: x['center_distance'])
            print(f"      → Selected: Speaker {best_seg['speaker']} (closest overall)")

if __name__ == "__main__":
    debug_dropping()

