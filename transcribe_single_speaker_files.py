#!/usr/bin/env python3
"""
Транскрибує одноголосі файли з діаризацією, щоб виявити залишки другого спікера
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
        
        # Автоматично визначаємо кількість спікерів
        from sklearn.metrics import silhouette_score
        best_k = 1
        best_score = -1
        
        max_k = min(3, max(2, len(embeddings) // 3))
        for k in range(1, max_k + 1):
            try:
                test_clustering = SpectralClustering(
                    n_clusters=k,
                    affinity='precomputed',
                    random_state=42,
                    assign_labels='kmeans',
                    n_init=5
                )
                test_labels = test_clustering.fit_predict(similarity_matrix)
                
                if len(np.unique(test_labels)) > 1:
                    score = silhouette_score(embeddings_normalized, test_labels, metric='cosine')
                    print(f"   k={k}: silhouette_score={score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception as e:
                print(f"   k={k}: error - {e}")
                continue
        
        num_speakers = best_k
        print(f"🔍 Auto-detected {num_speakers} speaker(s) (best silhouette_score={best_score:.4f})")
        
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
        unique_speakers = set(seg.get('speaker', 0) for seg in segments)
        print(f"📊 Found {len(unique_speakers)} unique speaker(s): {sorted(unique_speakers)}")
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

def combine_diarization_and_transcription(diarization_segments, words):
    """Об'єднує результати діаризації та транскрипції"""
    if not words:
        print("⚠️  No words provided for combination")
        return []
    
    if not diarization_segments:
        print("⚠️  No diarization segments, using single speaker")
        # Якщо немає діаризації, повертаємо транскрипцію з одним спікером
        combined = []
        current_start = words[0]['start']
        current_words = []
        
        for word in words:
            if not word['word'].strip():
                continue
            current_words.append(word['word'])
            if word['end'] - current_start > 1.0:  # Сегменти по 1 секунді
                combined.append({
                    'speaker': 0,
                    'start': round(current_start, 2),
                    'end': round(word['end'], 2),
                    'text': ' '.join(current_words).strip()
                })
                current_start = word['start']
                current_words = []
        
        if current_words:
            combined.append({
                'speaker': 0,
                'start': round(current_start, 2),
                'end': round(words[-1]['end'], 2),
                'text': ' '.join(current_words).strip()
            })
        return combined
    
    print(f"🔗 Combining {len(words)} words with {len(diarization_segments)} diarization segments")
    
    # Для кожного слова знаходимо найкраще перекриття з сегментами діаризації
    word_speakers = []
    for word in words:
        word_start = word['start']
        word_end = word['end']
        word_text = word['word']
        
        if not word_text.strip():
            continue
        
        best_overlap = 0
        best_speaker = 0
        
        for diar_seg in diarization_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            
            overlap_start = max(word_start, diar_start)
            overlap_end = min(word_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = diar_seg['speaker']
        
        word_speakers.append({
            'word': word_text,
            'start': word_start,
            'end': word_end,
            'speaker': best_speaker
        })
    
    print(f"✅ Mapped {len(word_speakers)} words to speakers")
    
    # Групуємо слова по спікерах та сегментах
    combined = []
    current_speaker = None
    current_start = None
    current_words = []
    
    for word_info in word_speakers:
        if word_info['speaker'] != current_speaker:
            if current_speaker is not None and current_words:
                combined.append({
                    'speaker': current_speaker,
                    'start': round(current_start, 2),
                    'end': round(word_info['start'], 2),
                    'text': ' '.join(current_words).strip()
                })
            current_speaker = word_info['speaker']
            current_start = word_info['start']
            current_words = [word_info['word']]
        else:
            current_words.append(word_info['word'])
    
    if current_words:
        combined.append({
            'speaker': current_speaker if current_speaker is not None else 0,
            'start': round(current_start, 2) if current_start is not None else 0,
            'end': round(word_speakers[-1]['end'], 2) if word_speakers else 0,
            'text': ' '.join(current_words).strip()
        })
    
    print(f"✅ Created {len(combined)} combined segments")
    return combined

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
    input_dir = "Audio Examples/detecting main speakers"
    output_dir = "Audio Examples/detecting main speakers"
    
    speaker_files = [
        os.path.join(input_dir, "speaker_0.wav"),
        os.path.join(input_dir, "speaker_1.wav")
    ]
    
    print("=" * 60)
    print("🎵 Transcribing single-speaker files with diarization")
    print("=" * 60)
    
    # 1. Завантажуємо моделі
    load_models()
    
    results = {}
    
    # 2. Обробляємо кожен одноголосий файл
    for speaker_file in speaker_files:
        if not os.path.exists(speaker_file):
            print(f"⚠️  File not found: {speaker_file}")
            continue
        
        speaker_name = os.path.basename(speaker_file).replace('.wav', '')
        print(f"\n{'='*60}")
        print(f"📁 Processing: {speaker_name}")
        print(f"{'='*60}")
        
        # 2.1. Діаризація
        print(f"\n📊 Step 1: Diarization for {speaker_name}...")
        embeddings, timestamps = extract_speaker_embeddings(speaker_file)
        if embeddings is None:
            print(f"❌ Failed to extract embeddings from {speaker_name}")
            continue
        
        diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=2)
        
        # 2.2. Транскрипція
        print(f"\n📝 Step 2: Transcribing {speaker_name}...")
        transcription, segments, words = transcribe_audio(speaker_file)
        
        if not transcription:
            print(f"❌ Failed to transcribe {speaker_name}")
            continue
        
        # 2.3. Об'єднання діаризації з транскрипцією
        print(f"\n🔗 Step 3: Combining diarization with transcription for {speaker_name}...")
        # Якщо немає words, використовуємо segments
        if not words:
            print("⚠️  No words from Whisper, using segments instead")
            # Створюємо words з segments
            words = []
            for seg in segments:
                seg_start = seg.get('start', 0)
                seg_end = seg.get('end', 0)
                seg_text = seg.get('text', '').strip()
                if seg_text:
                    # Розбиваємо текст на слова (приблизно)
                    word_list = seg_text.split()
                    if word_list:
                        word_duration = (seg_end - seg_start) / len(word_list)
                        for i, word_text in enumerate(word_list):
                            words.append({
                                'word': word_text,
                                'start': seg_start + i * word_duration,
                                'end': seg_start + (i + 1) * word_duration
                            })
        
        combined_segments = combine_diarization_and_transcription(diarization_segments, words)
        
        # 2.4. Збереження результатів
        print(f"\n💾 Step 4: Saving results for {speaker_name}...")
        
        # Транскрипт з діаризацією
        dialogue = format_dialogue(combined_segments)
        transcript_path = os.path.join(output_dir, f"{speaker_name}_with_diarization.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(dialogue)
        print(f"✅ Saved: {transcript_path}")
        
        # JSON з метаданими
        unique_speakers = sorted(set(seg.get('speaker', 0) for seg in combined_segments))
        metadata = {
            'file': speaker_file,
            'num_speakers_detected': len(unique_speakers),
            'speakers': unique_speakers,
            'diarization_segments': diarization_segments,
            'combined_segments': combined_segments,
            'transcription': transcription
        }
        
        metadata_path = os.path.join(output_dir, f"{speaker_name}_diarization_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved: {metadata_path}")
        
        results[speaker_name] = {
            'num_speakers': len(unique_speakers),
            'speakers': unique_speakers,
            'num_segments': len(combined_segments)
        }
    
    # 3. Підсумок
    print("\n" + "=" * 60)
    print("✅ Processing completed!")
    print("=" * 60)
    print("\n📊 Summary:")
    for speaker_name, info in results.items():
        print(f"  {speaker_name}:")
        print(f"    - Speakers detected: {info['num_speakers']} ({info['speakers']})")
        print(f"    - Segments: {info['num_segments']}")

if __name__ == "__main__":
    main()

