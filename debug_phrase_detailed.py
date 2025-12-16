#!/usr/bin/env python3
"""
Детальна діагностика фрази "I can't do this." - перевірка сегментів спікера 1 поблизу
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

def debug_phrase_detailed(audio_path, phrase="I can't do this"):
    """Детальна діагностика фрази з перевіркою близьких сегментів"""
    print(f"🔍 Detailed analysis of phrase: '{phrase}'")
    print("=" * 80)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    # Діаризація
    embeddings, timestamps = extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5)
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    
    # Транскрипція
    transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None)
    
    # Знаходимо слова фрази
    phrase_words = [w for w in words if any(pw.lower() in w['word'].lower() for pw in phrase.split())]
    if not phrase_words:
        print(f"❌ Phrase not found")
        return
    
    # Знаходимо точні слова фрази "I can't do this"
    exact_words = []
    for i, word in enumerate(words):
        if i < len(words) - 3:
            if (word['word'].lower().startswith('i') and 
                words[i+1]['word'].lower().startswith("can't") and
                words[i+2]['word'].lower().startswith('do') and
                words[i+3]['word'].lower().startswith('this')):
                exact_words = words[i:i+4]
                break
    
    if not exact_words:
        print(f"❌ Exact phrase not found")
        return
    
    phrase_start = exact_words[0]['start']
    phrase_end = exact_words[-1]['end']
    phrase_center = (phrase_start + phrase_end) / 2.0
    
    print(f"\n📊 Phrase: '{' '.join(w['word'] for w in exact_words)}'")
    print(f"   Timing: [{phrase_start:.2f}s - {phrase_end:.2f}s], center: {phrase_center:.2f}s")
    
    # Знаходимо всі сегменти спікера 1 поблизу (навіть якщо не перетинаються)
    print(f"\n📊 Speaker 1 segments near the phrase:")
    nearby_speaker1_segments = []
    for diar_seg in sorted_diar_segments:
        if diar_seg['speaker'] == 1:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            diar_center = (diar_start + diar_end) / 2.0
            
            # Відстань від центру фрази до центру сегмента
            center_distance = abs(phrase_center - diar_center)
            
            # Відстань від кінця сегмента до початку фрази (якщо сегмент перед фразою)
            distance_before = phrase_start - diar_end if phrase_start > diar_end else float('inf')
            
            # Відстань від кінця фрази до початку сегмента (якщо сегмент після фрази)
            distance_after = diar_start - phrase_end if diar_start > phrase_end else float('inf')
            
            # Перевіряємо чи сегмент близький (<3s)
            if center_distance < 3.0 or distance_before < 3.0 or distance_after < 3.0:
                nearby_speaker1_segments.append({
                    'segment': diar_seg,
                    'center_distance': center_distance,
                    'distance_before': distance_before,
                    'distance_after': distance_after,
                    'overlaps': phrase_start < diar_end and phrase_end > diar_start
                })
                
                overlap_status = "OVERLAPS" if (phrase_start < diar_end and phrase_end > diar_start) else "NEARBY"
                print(f"   [{diar_start:.2f}s - {diar_end:.2f}s] {overlap_status}: "
                      f"center_dist={center_distance:.2f}s, "
                      f"before={distance_before:.2f}s, "
                      f"after={distance_after:.2f}s")
    
    # Знаходимо сегменти спікера 0, що перетинаються
    print(f"\n📊 Speaker 0 segments overlapping with phrase:")
    overlapping_speaker0 = []
    for diar_seg in sorted_diar_segments:
        if diar_seg['speaker'] == 0:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            
            if phrase_start < diar_end and phrase_end > diar_start:
                overlap_start = max(phrase_start, diar_start)
                overlap_end = min(phrase_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)
                overlap_ratio = overlap / (phrase_end - phrase_start) if (phrase_end - phrase_start) > 0 else 0
                
                diar_center = (diar_start + diar_end) / 2.0
                center_distance = abs(phrase_center - diar_center)
                is_suspicious = (diar_end - diar_start) > 10.0
                
                overlapping_speaker0.append({
                    'segment': diar_seg,
                    'overlap_ratio': overlap_ratio,
                    'center_distance': center_distance,
                    'is_suspicious': is_suspicious
                })
                
                suspicious_marker = " ⚠️ SUSPICIOUS" if is_suspicious else ""
                print(f"   [{diar_start:.2f}s - {diar_end:.2f}s]{suspicious_marker}: "
                      f"overlap={overlap_ratio*100:.1f}%, "
                      f"center_dist={center_distance:.2f}s")
    
    # Висновок
    print(f"\n💡 Analysis:")
    if nearby_speaker1_segments:
        closest_s1 = min(nearby_speaker1_segments, key=lambda x: min(x['center_distance'], x['distance_before'], x['distance_after']))
        print(f"   Closest Speaker 1 segment: {min(closest_s1['center_distance'], closest_s1['distance_before'], closest_s1['distance_after']):.2f}s away")
    
    if overlapping_speaker0:
        closest_s0 = min(overlapping_speaker0, key=lambda x: x['center_distance'])
        print(f"   Closest Speaker 0 segment: center_dist={closest_s0['center_distance']:.2f}s, "
              f"overlap={closest_s0['overlap_ratio']*100:.1f}%, "
              f"suspicious={closest_s0['is_suspicious']}")
        
        if closest_s0['is_suspicious'] and nearby_speaker1_segments:
            min_s1_dist = min(min(s['center_distance'], s['distance_before'], s['distance_after']) 
                            for s in nearby_speaker1_segments)
            if min_s1_dist < 2.0:
                print(f"   ⚠️  PROBLEM: Suspicious large Speaker 0 segment 'swallows' phrase, "
                      f"but Speaker 1 segment is very close ({min_s1_dist:.2f}s)!")

if __name__ == "__main__":
    test_file = "audio examples/detecting main speakers/speaker_0.wav"
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    
    debug_phrase_detailed(test_file)

