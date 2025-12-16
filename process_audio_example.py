#!/usr/bin/env python3
"""
Скрипт для обробки аудіо прикладу:
1. Діаризація (SpeechBrain)
2. Розділення на одноголосі треки
3. Транскрипція оригінального файлу
4. Транскрипція кожного треку
5. Збереження результатів
"""

import os
import sys
import json
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
import warnings

# Патч для torchaudio сумісності
exec(open('patch_torchaudio.py').read())

from speechbrain.pretrained import SpeakerRecognition
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
import whisper
import copy

warnings.filterwarnings("ignore")

# Імпортуємо функції з app_ios_shortcuts.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Глобальні змінні для моделей
speaker_model = None
whisper_model = None

def load_models():
    """Завантажує моделі SpeechBrain та Whisper"""
    global speaker_model, whisper_model
    
    if speaker_model is None:
        print("🔄 Loading SpeechBrain speaker recognition model...")
        try:
            model_path = "pretrained_models/spkrec-ecapa-voxceleb"
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "hyperparams.yaml")):
                print(f"📂 Loading from local directory: {model_path}")
                speaker_model = SpeakerRecognition.from_hparams(
                    source=model_path,
                    savedir=model_path
                )
            else:
                print("🌐 Loading from HuggingFace...")
                speaker_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
            print("✅ SpeechBrain model loaded")
        except Exception as e:
            print(f"❌ Error loading SpeechBrain model: {e}")
            raise
    
    if whisper_model is None:
        print("🔄 Loading Whisper model (small)...")
        try:
            cache_dir = os.path.expanduser("~/.cache/whisper")
            whisper_model = whisper.load_model("small", download_root=cache_dir)
            print("✅ Whisper model loaded")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            raise

def extract_speaker_embeddings(audio_path, segment_duration=2.5, overlap=0.4):
    """Витягує embeddings для діаризації"""
    print(f"🎤 Extracting embeddings from {audio_path}...")
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        print(f"📊 Audio: {duration:.2f}s, {sr}Hz")
        
        segment_samples = int(segment_duration * sr)
        stride_samples = int(segment_samples * (1 - overlap))
        
        embeddings = []
        timestamps = []
        segments_processed = 0
        
        max_start = len(audio) - segment_samples
        if max_start < 0:
            max_start = 0
        
        for start_sample in range(0, max_start + 1, stride_samples):
            end_sample = min(start_sample + segment_samples, len(audio))
            segment = audio[start_sample:end_sample]
            
            if len(segment) == 0:
                continue
            
            try:
                model_device = next(speaker_model.parameters()).device
                segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(model_device)
                
                embedding = speaker_model.encode_batch(segment_tensor, normalize=False)
                embedding = embedding.squeeze().cpu().detach().numpy()
                
                if embedding is not None and len(embedding) > 0:
                    embeddings.append(embedding)
                    start_time = start_sample / sr
                    end_time = end_sample / sr
                    timestamps.append((start_time, min(end_time, duration)))
                    segments_processed += 1
            except Exception as e:
                print(f"⚠️  Error extracting embedding: {e}")
                continue
        
        print(f"✅ Extracted {len(embeddings)} embeddings")
        return np.array(embeddings), timestamps
    
    except Exception as e:
        print(f"❌ Error in extract_speaker_embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def diarize_audio(embeddings, timestamps, num_speakers=2):
    """Виконує діаризацію через spectral clustering"""
    print(f"🔍 Performing diarization for {num_speakers} speakers...")
    
    if embeddings is None or len(embeddings) < 2:
        print("❌ Not enough embeddings for diarization")
        return []
    
    try:
        from sklearn.preprocessing import normalize
        embeddings_normalized = normalize(embeddings, norm='l2')
        
        distances = pdist(embeddings_normalized, metric='cosine')
        distance_matrix = squareform(distances)
        
        mean_dist = np.mean(distances)
        scale = mean_dist if mean_dist > 0.01 else 0.1
        similarity_matrix = np.exp(-distance_matrix / scale)
        
        clustering = SpectralClustering(
            n_clusters=num_speakers,
            affinity='precomputed',
            random_state=42,
            assign_labels='kmeans',
            n_init=10
        )
        labels = clustering.fit_predict(similarity_matrix)
        
        # Зливаємо сусідні сегменти одного спікера
        segments = []
        current_speaker = None
        current_start = None
        
        for i, (label, (start, end)) in enumerate(zip(labels, timestamps)):
            if label != current_speaker:
                if current_speaker is not None:
                    segments.append({
                        'speaker': int(current_speaker),
                        'start': round(current_start, 2),
                        'end': round(timestamps[i-1][1], 2)
                    })
                current_speaker = label
                current_start = start
        
        if current_speaker is not None:
            segments.append({
                'speaker': int(current_speaker),
                'start': round(current_start, 2),
                'end': round(timestamps[-1][1], 2)
            })
        
        print(f"✅ Created {len(segments)} diarization segments")
        return segments
    
    except Exception as e:
        print(f"❌ Error in diarize_audio: {e}")
        import traceback
        traceback.print_exc()
        return []

def transcribe_audio(audio_path, language=None):
    """Транскрибує аудіо за допомогою Whisper"""
    print(f"📝 Transcribing {audio_path}...")
    try:
        result = whisper_model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            beam_size=3,
            fp16=torch.cuda.is_available(),
            verbose=True
        )
        
        transcription = result['text']
        segments = result['segments']
        words = []
        
        for seg in segments:
            for word_info in seg.get('words', []):
                words.append({
                    'word': word_info['word'],
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        
        print(f"✅ Transcription: {len(transcription)} chars, {len(segments)} segments")
        return transcription, segments, words
    
    except Exception as e:
        print(f"❌ Error in transcribe_audio: {e}")
        import traceback
        traceback.print_exc()
        return None, [], []

def extract_single_speaker_audio(audio_path, speaker_segments, output_path):
    """Витягує сегменти одного спікера з аудіо файлу"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        speaker_audio_segments = []
        for seg in speaker_segments:
            start_time = max(0, seg['start'])
            end_time = min(duration, seg['end'])
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if start_sample < len(audio) and end_sample <= len(audio) and start_sample < end_sample:
                segment_audio = audio[start_sample:end_sample]
                speaker_audio_segments.append(segment_audio)
        
        if not speaker_audio_segments:
            return None
        
        combined_audio = np.concatenate(speaker_audio_segments)
        sf.write(output_path, combined_audio, sr)
        
        print(f"✅ Extracted speaker audio: {len(combined_audio)/sr:.2f}s → {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ Error in extract_single_speaker_audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def format_dialogue(segments, speaker_label_prefix="Speaker"):
    """Форматує сегменти у читабельний діалог"""
    lines = []
    for seg in segments:
        start_time = seg.get('start', 0)
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        speaker = seg.get('speaker', 0)
        text = seg.get('text', '').strip()
        
        if text:
            lines.append(f"{time_str} {speaker_label_prefix} {speaker}: {text}")
    
    return "\n".join(lines)

def main():
    # Шляхи
    input_file = "audio examples/Screen Recording 2025-12-05 at 07.29.15.m4a"
    output_dir = "Audio Examples/detecting main speakers"
    
    # Створюємо вихідну папку
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🎵 Processing audio example")
    print("=" * 60)
    
    # 1. Завантажуємо моделі
    load_models()
    
    # 2. Діаризація
    print("\n📊 Step 1: Diarization...")
    embeddings, timestamps = extract_speaker_embeddings(input_file)
    if embeddings is None:
        print("❌ Failed to extract embeddings")
        return
    
    diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
    if not diarization_segments:
        print("❌ Diarization failed")
        return
    
    # Зберігаємо оригінальні segments
    original_diarization_segments = copy.deepcopy(diarization_segments)
    
    # 3. Транскрипція оригінального файлу
    print("\n📝 Step 2: Transcribing original file...")
    original_transcription, original_segments, original_words = transcribe_audio(input_file)
    
    # Об'єднуємо діаризацію з транскрипцією
    print("\n🔗 Step 3: Combining diarization with transcription...")
    combined_segments = []
    
    for diar_seg in diarization_segments:
        # Знаходимо слова, які потрапляють в цей сегмент
        segment_words = []
        for word in original_words:
            word_start = word['start']
            word_end = word['end']
            if word_start >= diar_seg['start'] and word_end <= diar_seg['end']:
                segment_words.append(word['word'])
        
        text = ' '.join(segment_words).strip()
        if not text:
            # Якщо немає слів, шукаємо найближчий сегмент транскрипції
            for seg in original_segments:
                seg_start = seg.get('start', 0)
                seg_end = seg.get('end', 0)
                if seg_start >= diar_seg['start'] and seg_end <= diar_seg['end']:
                    text = seg.get('text', '').strip()
                    break
        
        combined_segments.append({
            'speaker': diar_seg['speaker'],
            'start': diar_seg['start'],
            'end': diar_seg['end'],
            'text': text
        })
    
    # 4. Розділення на одноголосі файли
    print("\n🔀 Step 4: Splitting into single-speaker files...")
    speakers_segments = {}
    for seg in original_diarization_segments:
        speaker = seg.get('speaker', 0)
        if speaker not in speakers_segments:
            speakers_segments[speaker] = []
        speakers_segments[speaker].append(seg)
    
    speaker_files = {}
    for speaker, segments in speakers_segments.items():
        segments_sorted = sorted(segments, key=lambda x: x['start'])
        output_path = os.path.join(output_dir, f"speaker_{speaker}.wav")
        extract_single_speaker_audio(input_file, segments_sorted, output_path)
        speaker_files[speaker] = {
            'path': output_path,
            'segments': segments_sorted
        }
    
    # 5. Транскрипція одноголосих файлів
    print("\n📝 Step 5: Transcribing single-speaker files...")
    speaker_transcriptions = {}
    for speaker, file_info in speaker_files.items():
        print(f"\n🎤 Transcribing speaker {speaker}...")
        transcription, segments, words = transcribe_audio(file_info['path'])
        
        if transcription:
            # Об'єднуємо з таймстемпами з діаризації
            speaker_combined = []
            diar_segments = file_info['segments']
            
            # Визначаємо offset для таймстемпів (перший сегмент діаризації)
            first_diar_start = diar_segments[0]['start'] if diar_segments else 0
            
            # Мапімо слова з транскрипції на сегменти діаризації
            for diar_seg in diar_segments:
                # Знаходимо слова, які потрапляють в цей сегмент
                segment_words = []
                for word in words:
                    # Таймстемпи в одноголосому файлі відносні (починаються з 0)
                    # Потрібно перетворити їх на абсолютні таймстемпи
                    word_start_absolute = first_diar_start + word['start']
                    word_end_absolute = first_diar_start + word['end']
                    
                    # Перевіряємо, чи слово потрапляє в сегмент діаризації
                    if word_start_absolute >= diar_seg['start'] and word_end_absolute <= diar_seg['end']:
                        segment_words.append(word['word'])
                
                text = ' '.join(segment_words).strip()
                
                # Якщо немає слів, шукаємо найближчий сегмент транскрипції
                if not text and segments:
                    for seg in segments:
                        seg_start_absolute = first_diar_start + seg.get('start', 0)
                        seg_end_absolute = first_diar_start + seg.get('end', 0)
                        # Перевіряємо перекриття
                        overlap_start = max(seg_start_absolute, diar_seg['start'])
                        overlap_end = min(seg_end_absolute, diar_seg['end'])
                        if overlap_end > overlap_start:
                            text = seg.get('text', '').strip()
                            break
                
                if text:  # Додаємо тільки якщо є текст
                    speaker_combined.append({
                        'speaker': speaker,
                        'start': diar_seg['start'],
                        'end': diar_seg['end'],
                        'text': text
                    })
            
            speaker_transcriptions[speaker] = {
                'transcription': transcription,
                'segments': speaker_combined
            }
    
    # 6. Збереження результатів
    print("\n💾 Step 6: Saving results...")
    
    # Оригінальний транскрипт
    original_dialogue = format_dialogue(combined_segments)
    original_path = os.path.join(output_dir, "original_dialogue.txt")
    with open(original_path, 'w', encoding='utf-8') as f:
        f.write(original_dialogue)
    print(f"✅ Saved: {original_path}")
    
    # Транскрипти одноголосих файлів
    for speaker, info in speaker_transcriptions.items():
        dialogue = format_dialogue(info['segments'], speaker_label_prefix="Speaker")
        transcript_path = os.path.join(output_dir, f"speaker_{speaker}_transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(dialogue)
        print(f"✅ Saved: {transcript_path}")
    
    # JSON з метаданими
    metadata = {
        'original_file': input_file,
        'num_speakers': len(speaker_files),
        'original_diarization_segments': original_diarization_segments,
        'combined_segments': combined_segments,
        'speaker_files': {
            speaker: {
                'path': info['path'],
                'num_segments': len(info['segments']),
                'total_duration': sum(seg['end'] - seg['start'] for seg in info['segments'])
            }
            for speaker, info in speaker_files.items()
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("✅ Processing completed!")
    print("=" * 60)
    print(f"\n📁 Results saved in: {output_dir}")
    print("\nFiles created:")
    print(f"  - speaker_0.wav")
    print(f"  - speaker_1.wav")
    print(f"  - original_dialogue.txt")
    print(f"  - speaker_0_transcript.txt")
    print(f"  - speaker_1_transcript.txt")
    print(f"  - metadata.json")

if __name__ == "__main__":
    main()

