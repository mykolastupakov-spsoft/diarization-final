#!/usr/bin/env python3
"""
Діагностика: чому слова не призначаються спікеру 1
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

def debug_speaker1():
    """Детальна діагностика призначення слів спікеру 1"""
    audio_path = "audio examples/detecting main speakers/speaker_0.wav"
    
    # Діаризація
    embeddings, timestamps = extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5)
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    
    # Транскрипція
    transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None)
    
    print("=" * 80)
    print("SPEAKER 1 SEGMENTS ANALYSIS")
    print("=" * 80)
    
    speaker1_segments = [s for s in sorted_diar_segments if s['speaker'] == 1]
    print(f"\n📊 Speaker 1 has {len(speaker1_segments)} segments:")
    for seg in speaker1_segments:
        print(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] (duration: {seg['end']-seg['start']:.2f}s)")
        
        # Знаходимо слова в цьому діапазоні
        words_in_range = [w for w in words if w['start'] < seg['end'] and w['end'] > seg['start']]
        print(f"      Words in this range: {len(words_in_range)}")
        if words_in_range:
            print(f"      Words: {[w['word'] for w in words_in_range]}")
            
            # Перевіряємо, який спікер призначений цим словам
            combined = combine_diarization_and_transcription(diarization_segments, words)
            for word in words_in_range:
                # Знаходимо сегмент в combined, який містить це слово
                for comb_seg in combined:
                    if comb_seg['start'] <= word['start'] and comb_seg['end'] >= word['end']:
                        print(f"         '{word['word']}' [{word['start']:.2f}s-{word['end']:.2f}s] → Assigned to Speaker {comb_seg['speaker']} in combined")
                        break
    
    print("\n" + "=" * 80)
    print("TESTING WORD ASSIGNMENT FOR SPEAKER 1 WORDS")
    print("=" * 80)
    
    # Тестуємо конкретні слова, які мають бути від спікера 1
    test_words = [
        {'word': 'Hey,', 'start': 2.18, 'end': 2.90},
    ]
    
    for word_info in test_words:
        word_start = word_info['start']
        word_end = word_info['end']
        word_center = (word_start + word_end) / 2.0
        word_duration = word_end - word_start
        
        print(f"\n   Testing word: '{word_info['word']}' at [{word_start:.2f}s - {word_end:.2f}s]")
        
        # Знаходимо всі сегменти, що перетинаються
        overlapping = []
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
                
                overlapping.append({
                    'speaker': diar_seg['speaker'],
                    'overlap_ratio': overlap_ratio,
                    'center_distance': center_distance,
                    'fully_contained': fully_contained,
                    'segment': f"[{diar_start:.2f}s - {diar_end:.2f}s]"
                })
        
        print(f"      Overlapping segments:")
        for ov in overlapping:
            print(f"         Speaker {ov['speaker']} {ov['segment']}: overlap={ov['overlap_ratio']*100:.1f}%, "
                  f"center_dist={ov['center_distance']:.3f}s, fully_contained={ov['fully_contained']}")
        
        if overlapping:
            # Симулюємо вибір
            fully_contained = [s for s in overlapping if s['fully_contained']]
            if fully_contained:
                best = max(fully_contained, key=lambda x: x['overlap_ratio'])
                print(f"      → Would select: Speaker {best['speaker']} (fully contained)")
            else:
                best = max(overlapping, key=lambda x: (x['overlap_ratio'], -x['center_distance']))
                print(f"      → Would select: Speaker {best['speaker']} (best overlap)")

if __name__ == "__main__":
    debug_speaker1()

