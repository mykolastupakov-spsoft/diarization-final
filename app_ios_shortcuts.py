#!/usr/bin/env python3
"""
Flask сервер для iOS Shortcuts: SpeechBrain діаризація + Whisper транскрипція
Окремий файл для уникнення конфліктів з іншими процесами
"""

import os
import sys
import json
import base64
import numpy as np
import torch
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, send_file, send_from_directory
import time
from werkzeug.utils import secure_filename
import threading
import uuid
from datetime import datetime, timedelta

# Патч для torchaudio сумісності з speechbrain (завантажуємо ДО імпорту speechbrain)
exec(open('patch_torchaudio.py').read())

from speechbrain.pretrained import SpeakerRecognition
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, squareform
import whisper
import warnings
from pathlib import Path
import requests

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Middleware для гарантії правильних CORS заголовків для всіх запитів з браузера
@app.after_request
def after_request(response):
    """Додає CORS заголовки до всіх відповідей"""
    # Встановлюємо заголовки тільки якщо їх ще немає (щоб уникнути дублювання)
    if 'Access-Control-Allow-Origin' not in response.headers:
        response.headers['Access-Control-Allow-Origin'] = '*'
    if 'Access-Control-Allow-Methods' not in response.headers:
        response.headers['Access-Control-Allow-Methods'] = 'POST'
    if 'Access-Control-Allow-Headers' not in response.headers:
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Обробка OPTIONS запитів (preflight)
@app.before_request
def handle_preflight():
    """Обробка preflight OPTIONS запитів"""
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers['Access-Control-Allow-Origin'] = '*'
        # Для preflight потрібно повертати OPTIONS, але також POST для дозволу
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size

# Константи для iOS Shortcuts API
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac'}
PROCESSING_TIMEOUT = 300  # 5 хвилин

# Дозволи для завантажень
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Асинхронна обробка: словник для зберігання статусів завдань
jobs = {}  # {job_id: {'status': 'pending'|'processing'|'completed'|'failed', 'result': {...}, 'error': '...', 'created_at': datetime}}
jobs_lock = threading.Lock()

# Очищення старих завдань (старіше 1 години)
def cleanup_old_jobs():
    """Фоновий потік для очищення старих завдань"""
    while True:
        time.sleep(3600)  # Кожну годину
        with jobs_lock:
            now = datetime.now()
            expired = [job_id for job_id, job in jobs.items() 
                      if now - job['created_at'] > timedelta(hours=1)]
            for job_id in expired:
                del jobs[job_id]
                print(f"🧹 Cleaned up expired job: {job_id}")

# Запускаємо очищення в фоні
cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
cleanup_thread.start()

# Глобальні змінні для моделей (завантажуються один раз)
speaker_model = None
whisper_model = None

def load_models():
    """Завантажує моделі SpeechBrain та Whisper один раз при старті"""
    global speaker_model, whisper_model
    
    if speaker_model is None:
        print("🔄 Loading SpeechBrain speaker recognition model...")
        try:
            # Спробуємо завантажити з локальної папки
            model_path = "pretrained_models/spkrec-ecapa-voxceleb"
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "hyperparams.yaml")):
                print(f"📂 Loading from local directory: {model_path}")
                speaker_model = SpeakerRecognition.from_hparams(
                    source=model_path,
                    savedir=model_path
                )
            else:
                # Якщо локальної моделі немає, завантажуємо з HuggingFace
                print("🌐 Loading from HuggingFace...")
                speaker_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
            print("✅ SpeechBrain model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading SpeechBrain model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    if whisper_model is None:
        print("🔄 Loading Whisper model...")
        try:
            # Використовуємо large-v3 - найпотужнішу модель Whisper для кращої якості
            # Підтримується на системах з 48 ГБ+ ОЗУ
            # Модель займає ~3-4 ГБ в пам'яті
            model_size = os.environ.get('WHISPER_MODEL_SIZE', 'small')
            
            # Перевіряємо, чи модель вже в кеші
            cache_dir = os.path.expanduser("~/.cache/whisper")
            model_path = os.path.join(cache_dir, f"{model_size}.pt")
            
            if os.path.exists(model_path):
                file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"📦 Loading Whisper {model_size} model from cache ({file_size_mb:.1f} MB)...")
                print(f"   📂 Cache location: {model_path}")
            else:
                print(f"📦 Loading Whisper {model_size} model (downloading to cache first time)...")
            
            # Whisper автоматично використовує кеш з ~/.cache/whisper/
            whisper_model = whisper.load_model(model_size, download_root=cache_dir)
            print(f"✅ Whisper model ({model_size}) loaded successfully!")
            print(f"   💾 Model size: ~3-4 GB in memory")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            print(f"   💡 If you have less RAM, try: WHISPER_MODEL_SIZE=medium")
            raise

# Завантажуємо моделі при старті в окремому потоці (щоб не блокувати запуск сервера)
def load_models_background():
    """Завантажує моделі в фоні"""
    import sys
    try:
        print("🔄 Starting background model loading...", flush=True)
        sys.stdout.flush()
        load_models()
        print("✅ All models loaded successfully in background!", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"⚠️  Warning: Could not load models at startup: {e}", flush=True)
        print("   Models will be loaded on first request", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

# Запускаємо завантаження моделей в фоні
print("🚀 Starting model loading thread...", flush=True)
model_loading_thread = threading.Thread(target=load_models_background, daemon=True)
model_loading_thread.start()


def extract_speaker_embeddings(audio_path, segment_duration=1.5, overlap=0.5):
    """
    Витягує ембеддинги спікера для сегментів аудіо.
    
    Args:
        audio_path: шлях до аудіофайлу
        segment_duration: довжина сегмента в секундах
        overlap: перекриття між сегментами (0-1)
    
    Returns:
        embeddings: матриця ембедингів (N, 192)
        timestamps: список (start, end) для кожного сегмента
    """
    global speaker_model
    
    if speaker_model is None:
        load_models()
    
    try:
        # Завантажуємо аудіо
        print(f"📂 Loading audio from: {audio_path}")
        import sys
        sys.stdout.flush()
        
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"⏱️  Audio duration: {duration:.2f} seconds, sample rate: {sr} Hz, samples: {len(audio)}")
        sys.stdout.flush()
        
        # Перевірка мінімальної довжини
        min_duration = 0.5  # Мінімум 0.5 секунди
        if duration < min_duration:
            print(f"⚠️  Audio too short ({duration:.2f}s < {min_duration}s), using entire audio as single segment")
            # Використовуємо все аудіо як один сегмент
            embedding = None
            try:
                # Спробуємо через прямий доступ до embedding_model
                segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                wav_lens = torch.tensor([duration], dtype=torch.float32)
                
                with torch.no_grad():
                    if hasattr(speaker_model, 'mods') and hasattr(speaker_model.mods, 'encoder'):
                        features = speaker_model.mods.encoder(segment_tensor, wav_lens=wav_lens)
                        if hasattr(speaker_model.mods, 'embedding_model'):
                            embedding = speaker_model.mods.embedding_model(features, wav_lens=wav_lens)
                        else:
                            embedding = features
                        embedding = embedding.squeeze().cpu().detach().numpy()
            except Exception as e1:
                try:
                    # Fallback до encode_batch
                    segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    embedding = speaker_model.encode_batch(segment_tensor).squeeze().cpu().detach().numpy()
                except Exception as e2:
                    print(f"❌ Error processing short audio: {e1}, {e2}")
                    return None, []
            
            if embedding is not None and len(embedding) > 0:
                return np.array([embedding]), [(0.0, duration)]
            else:
                return None, []
        
        embeddings = []
        timestamps = []
        
        # Ковзні вікна
        segment_samples = int(segment_duration * sr)
        stride_samples = int(segment_duration * (1 - overlap) * sr)
        
        print(f"🔍 Processing with segment_duration={segment_duration}s, overlap={overlap}, segment_samples={segment_samples}, stride_samples={stride_samples}")
        
        # Якщо аудіо коротше за один сегмент, використовуємо все аудіо
        if len(audio) < segment_samples:
            print(f"⚠️  Audio shorter than segment ({len(audio)} < {segment_samples}), using entire audio")
            embedding = None
            try:
                # Спробуємо через прямий доступ до embedding_model
                segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                wav_lens = torch.tensor([duration], dtype=torch.float32)
                
                with torch.no_grad():
                    if hasattr(speaker_model, 'mods') and hasattr(speaker_model.mods, 'encoder'):
                        features = speaker_model.mods.encoder(segment_tensor, wav_lens=wav_lens)
                        if hasattr(speaker_model.mods, 'embedding_model'):
                            embedding = speaker_model.mods.embedding_model(features, wav_lens=wav_lens)
                        else:
                            embedding = features
                        embedding = embedding.squeeze().cpu().detach().numpy()
            except Exception as e1:
                try:
                    # Fallback до encode_batch
                    segment_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    embedding = speaker_model.encode_batch(segment_tensor).squeeze().cpu().detach().numpy()
                except Exception as e2:
                    print(f"❌ Error processing short audio segment: {e1}, {e2}")
                    return None, []
            
            if embedding is not None and len(embedding) > 0:
                return np.array([embedding]), [(0.0, duration)]
            else:
                return None, []
        
        # Обробка сегментів
        max_start = len(audio) - segment_samples
        if max_start < 0:
            max_start = 0
        
        segments_processed = 0
        for start_sample in range(0, max_start + 1, stride_samples):
            end_sample = min(start_sample + segment_samples, len(audio))
            segment = audio[start_sample:end_sample]
            
            # Перевірка, що сегмент не порожній
            if len(segment) == 0:
                continue
            
            # Витягуємо ембеддинг через SpeechBrain
            # Використовуємо classify_file з тимчасовим файлом (найнадійніший метод)
            embedding = None
            tmp_path = None
            
            try:
                # Використовуємо encode_batch напряму з тензором (обходимо torchaudio/torchcodec)
                # Конвертуємо сегмент у тензор у правильному форматі для SpeechBrain
                segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # [1, samples]
                
                # SpeechBrain очікує формат [batch, channels, samples] або [batch, samples]
                # Спробуємо обидва варіанти
                try:
                    # Варіант 1: [1, 1, samples] - з каналом
                    segment_tensor_with_channel = segment_tensor.unsqueeze(0)  # [1, 1, samples]
                    embedding_tensor = speaker_model.encode_batch(segment_tensor_with_channel)
                    embedding = embedding_tensor.squeeze().cpu().detach().numpy()
                except Exception as e1:
                    try:
                        # Варіант 2: [1, samples] - без каналу
                        embedding_tensor = speaker_model.encode_batch(segment_tensor)
                        embedding = embedding_tensor.squeeze().cpu().detach().numpy()
                    except Exception as e2:
                        # Варіант 3: через прямий доступ до encoder
                        try:
                            wav_lens = torch.tensor([len(segment) / sr], dtype=torch.float32)
                            with torch.no_grad():
                                if hasattr(speaker_model, 'mods') and hasattr(speaker_model.mods, 'encoder'):
                                    features = speaker_model.mods.encoder(segment_tensor, wav_lens=wav_lens)
                                    if hasattr(speaker_model.mods, 'embedding_model'):
                                        embedding_tensor = speaker_model.mods.embedding_model(features, wav_lens=wav_lens)
                                    else:
                                        embedding_tensor = features
                                    embedding = embedding_tensor.squeeze().cpu().detach().numpy()
                                else:
                                    raise Exception("No encoder found in model")
                        except Exception as e3:
                            print(f"❌ All embedding extraction methods failed: {e1}, {e2}, {e3}")
                            embedding = None
                
                if embedding is not None and len(embedding) > 0:
                    if segments_processed < 3:  # Логуємо перші 3 успішні
                        print(f"✅ Extracted embedding for segment at {start_sample}, shape: {embedding.shape}")
            except Exception as e:
                if start_sample == 0:
                    print(f"❌ Embedding extraction failed for first segment: {e}")
                    import traceback
                    traceback.print_exc()
                embedding = None
            
            if embedding is not None and len(embedding) > 0:
                embeddings.append(embedding)
                
                start_time = start_sample / sr
                end_time = end_sample / sr
                timestamps.append((start_time, min(end_time, duration)))
                segments_processed += 1
                if segments_processed <= 3:  # Логуємо перші 3 успішні
                    print(f"✅ Extracted embedding for segment at {start_sample}, shape: {embedding.shape}")
            else:
                if start_sample == 0:
                    print(f"⚠️  No embedding extracted for first segment at {start_sample}")
                continue
        
        print(f"✅ Processed {segments_processed} segments, extracted {len(embeddings)} embeddings")
        sys.stdout.flush()
        
        if len(embeddings) == 0:
            print("❌ No embeddings extracted!")
            print(f"   Audio info: duration={duration:.2f}s, samples={len(audio)}, sr={sr}Hz")
            print(f"   Segment params: segment_duration={segment_duration}s, overlap={overlap}")
            print(f"   Segment samples: {segment_samples}, stride_samples={stride_samples}")
            sys.stdout.flush()
            return None, []
        
        print(f"✅ Returning {len(embeddings)} embeddings with shapes: {[e.shape for e in embeddings[:3]]}")
        sys.stdout.flush()
        return np.array(embeddings), timestamps
    
    except Exception as e:
        print(f"❌ Error in extract_speaker_embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None, []


def diarize_audio(embeddings, timestamps, num_speakers=None):
    """
    Виконує діаризацію через spectral clustering на ембедингах.
    
    Args:
        embeddings: матриця ембедингів (N, 192)
        timestamps: список (start, end) для кожного сегмента
        num_speakers: кількість спікерів (якщо None, автоматично визначається)
    
    Returns:
        segments: список {'speaker': int, 'start': float, 'end': float}
    """
    if embeddings is None:
        print("❌ No embeddings provided for diarization")
        return []
    
    if len(embeddings) < 2:
        print(f"⚠️  Only {len(embeddings)} embedding(s) available, need at least 2 for clustering")
        # Якщо тільки один сегмент, повертаємо його як одного спікера
        if len(embeddings) == 1 and timestamps:
            return [{
                'speaker': 0,
                'start': round(timestamps[0][0], 2),
                'end': round(timestamps[0][1], 2)
            }]
        return []
    
    try:
        # Нормалізуємо ембеддинги (L2 нормалізація)
        from sklearn.preprocessing import normalize
        embeddings_normalized = normalize(embeddings, norm='l2')
        
        # Обчислюємо косинусну відстань між ембедингами
        distances = pdist(embeddings_normalized, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Діагностика: перевіряємо розподіл відстаней
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        print(f"📊 Distance stats: mean={mean_dist:.4f}, std={std_dist:.4f}, min={np.min(distances):.4f}, max={np.max(distances):.4f}")
        
        # Створюємо similarity matrix для кластеризації
        if std_dist < 1e-6:
            print(f"⚠️  All distances are nearly identical, using uniform similarity")
            similarity_matrix = np.ones_like(distance_matrix) * 0.5
        else:
            # Використовуємо адаптивне масштабування
            scale = mean_dist if mean_dist > 0.01 else 0.1
            similarity_matrix = np.exp(-distance_matrix / scale)
        
        # Визначаємо кількість спікерів автоматично, якщо не задано
        if num_speakers is None:
            # Використовуємо ельбовий метод для визначення оптимальної кількості кластерів
            from sklearn.metrics import silhouette_score
            
            best_k = 2
            best_score = -1
            
            # Перевіряємо k від 2 до min(5, кількість_сегментів/3)
            max_k = min(5, max(2, len(embeddings) // 3))
            
            for k in range(2, max_k + 1):
                try:
                    test_clustering = SpectralClustering(
                        n_clusters=k,
                        affinity='precomputed',
                        random_state=42,
                        assign_labels='kmeans',
                        n_init=5
                    )
                    test_labels = test_clustering.fit_predict(similarity_matrix)
                    
                    # Обчислюємо silhouette score (потребує принаймні 2 кластери)
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
            print(f"🔍 Auto-detected {num_speakers} speakers (best silhouette_score={best_score:.4f})")
            
            # КРИТИЧНО: Не примусово встановлюємо 2 спікерів, якщо файл одноголосий!
            # Якщо всі сегменти дуже схожі (mean_dist < 0.05), це означає один спікер
            if mean_dist < 0.05:
                if num_speakers > 1:
                    print(f"⚠️  Very low distance ({mean_dist:.4f}), but detected {num_speakers} speakers - likely single speaker, forcing 1")
                    num_speakers = 1
                else:
                    print(f"✅ Very low distance ({mean_dist:.4f}) confirms single speaker")
        
        # Перевіряємо, чи достатньо сегментів для кластеризації
        if len(embeddings) < num_speakers:
            print(f"⚠️  Not enough segments ({len(embeddings)}) for {num_speakers} speakers, using {len(embeddings)}")
            num_speakers = len(embeddings)
        
        # Спробуємо різні алгоритми кластеризації
        # Спочатку Spectral clustering
        labels = None
        try:
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans',
                n_init=10  # Більше спроб для кращої стабільності
            )
            labels = clustering.fit_predict(similarity_matrix)
            print(f"✅ Used SpectralClustering")
        except Exception as e:
            print(f"⚠️  Spectral clustering failed: {e}, trying AgglomerativeClustering")
            # Fallback до AgglomerativeClustering
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='average',
                affinity='precomputed'
            )
            labels = clustering.fit_predict(similarity_matrix)
            print(f"✅ Used AgglomerativeClustering")
        
        if labels is None:
            print("❌ Clustering failed completely")
            return []
        
        # Діагностика: перевіряємо розподіл лейблів
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"📊 Clustering result: {len(unique_labels)} unique speakers found")
        for label, count in zip(unique_labels, counts):
            print(f"   Speaker {label}: {count} segments ({count/len(labels)*100:.1f}%)")
        
        # КРИТИЧНО: Якщо один спікер займає >90% сегментів, це одноголосий файл
        # Але тільки якщо середня відстань не дуже висока (якщо висока, можуть бути різні спікери)
        if len(unique_labels) > 1:
            max_count = max(counts)
            max_ratio = max_count / len(labels)
            
            # Перевіряємо тривалість сегментів спікера 1
            speaker1_count = counts[1] if len(counts) > 1 else 0
            speaker1_ratio = speaker1_count / len(labels) if len(labels) > 0 else 0
            
            # Якщо спікер 1 займає <15% сегментів І середня відстань висока (>0.5), це одноголосий файл
            # (висока відстань означає, що сегменти різні, але це один спікер з різними інтонаціями)
            if max_ratio > 0.90 and speaker1_ratio < 0.15 and mean_dist > 0.5:
                print(f"⚠️  One speaker has {max_ratio*100:.1f}% of segments, other has only {speaker1_ratio*100:.1f}% (mean_dist={mean_dist:.4f}) - likely single speaker file, forcing all to speaker 0")
                labels = np.zeros_like(labels)
                unique_labels = [0]
                counts = [len(labels)]
        
        # Якщо всі сегменти одного спікера, спробуємо інший підхід (тільки якщо num_speakers > 1)
        if len(unique_labels) == 1 and num_speakers > 1:
            print(f"⚠️  All segments assigned to one speaker, trying alternative clustering...")
            # Спробуємо використати відстані безпосередньо
            from sklearn.cluster import AgglomerativeClustering
            clustering_alt = AgglomerativeClustering(
                n_clusters=num_speakers,
                linkage='ward',
                metric='euclidean'
            )
            labels_alt = clustering_alt.fit_predict(embeddings_normalized)
            unique_alt, counts_alt = np.unique(labels_alt, return_counts=True)
            if len(unique_alt) > 1:
                print(f"✅ Alternative clustering found {len(unique_alt)} speakers")
                labels = labels_alt
                unique_labels, counts = unique_alt, counts_alt
        
        # КРИТИЧНО: Зливаємо сусідні сегменти одного спікера, але НЕ через інші спікери
        # Створюємо сегменти з урахуванням того, що між сегментами одного спікера можуть бути інші спікери
        segments = []
        current_speaker = None
        current_start = None
        
        for i, (label, (start, end)) in enumerate(zip(labels, timestamps)):
            if label != current_speaker:
                # Якщо змінився спікер, зберігаємо поточний сегмент
                if current_speaker is not None:
                    segments.append({
                        'speaker': int(current_speaker),
                        'start': round(current_start, 2),
                        'end': round(timestamps[i-1][1], 2)
                    })
                # Починаємо новий сегмент
                current_speaker = label
                current_start = start
            # Якщо спікер той самий, продовжуємо сегмент (не зберігаємо, поки не зміниться)
        
        # Додаємо останній сегмент
        if current_speaker is not None:
            segments.append({
                'speaker': int(current_speaker),
                'start': round(current_start, 2),
                'end': round(timestamps[-1][1], 2)
            })
        
        # КРИТИЧНО: Виправляємо перекриття сегментів різних спікерів
        # Простий підхід: якщо сегменти перекриваються, розбиваємо їх на межі перекриття
        # Але НЕ додаємо оригінальні сегменти знову - це створює дублікати!
        
        # Спочатку збираємо всі точки перетину
        split_points = set()
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i != j and seg1['speaker'] != seg2['speaker']:
                    # Перевіряємо перекриття
                    if seg1['start'] < seg2['end'] and seg1['end'] > seg2['start']:
                        # Додаємо точки початку та кінця перекриття
                        overlap_start = max(seg1['start'], seg2['start'])
                        overlap_end = min(seg1['end'], seg2['end'])
                        split_points.add(round(overlap_start, 2))
                        split_points.add(round(overlap_end, 2))
        
        # Якщо є точки розбиття, розбиваємо сегменти
        if split_points:
            split_points = sorted(split_points)
            fixed_segments = []
            
            for seg in segments:
                seg_start = seg['start']
                seg_end = seg['end']
                seg_speaker = seg['speaker']
                
                # Знаходимо всі точки розбиття в межах цього сегмента
                points_in_segment = [p for p in split_points if seg_start < p < seg_end]
                
                if points_in_segment:
                    # Розбиваємо сегмент на частини
                    all_points = [seg_start] + points_in_segment + [seg_end]
                    for k in range(len(all_points) - 1):
                        part_start = all_points[k]
                        part_end = all_points[k + 1]
                        
                        # Перевіряємо, чи ця частина не перекривається з іншим спікером
                        # Якщо перекривається значно (>50%), пропускаємо цю частину
                        overlaps_with_other = False
                        for other_seg in segments:
                            if other_seg['speaker'] != seg_speaker:
                                if part_start < other_seg['end'] and part_end > other_seg['start']:
                                    overlap_size = min(part_end, other_seg['end']) - max(part_start, other_seg['start'])
                                    part_size = part_end - part_start
                                    if overlap_size > part_size * 0.5:  # Більше 50% перекриття
                                        overlaps_with_other = True
                                        break
                        
                        if not overlaps_with_other and part_end > part_start:
                            fixed_segments.append({
                                'speaker': seg_speaker,
                                'start': round(part_start, 2),
                                'end': round(part_end, 2)
                            })
                else:
                    # Немає точок розбиття, додаємо як є
                    fixed_segments.append(seg)
            
            # ВАЖЛИВО: НЕ додаємо оригінальні сегменти знову - це створює дублікати!
            # Всі сегменти вже оброблені вище
            segments = fixed_segments
        
        # Сортуємо за часом
        segments = sorted(segments, key=lambda x: x['start'])
        
        print(f"✅ Created {len(segments)} diarization segments")
        return segments
    
    except Exception as e:
        print(f"❌ Error in diarize_audio: {e}")
        import traceback
        traceback.print_exc()
        return []


def transcribe_with_speechmatics(audio_path, language='en'):
    """
    Транскрибує аудіо за допомогою Speechmatics API з word timestamps.
    
    Args:
        audio_path: шлях до аудіофайлу
        language: код мови (наприклад, 'uk', 'en', 'ar')
    
    Returns:
        transcription: текст транскрипції
        segments: список сегментів з текстом та часовими мітками
        words: список слів з timestamps для точного матчингу
    """
    import sys
    from transcribe_with_speechmatics import upload_to_speechmatics, poll_speechmatics_job
    
    api_key = os.getenv('SPEECHMATICS_API_KEY')
    if not api_key:
        raise ValueError("SPEECHMATICS_API_KEY environment variable is not set")
    
    print(f"🎤 Transcribing with Speechmatics: {audio_path}")
    print(f"   Language: {language}")
    sys.stdout.flush()
    
    # Завантажуємо файл
    job_id = upload_to_speechmatics(api_key, audio_path, language, is_separated_track=True)
    
    # Очікуємо завершення
    transcript_data = poll_speechmatics_job(api_key, job_id)
    
    # Парсимо результат
    words = []
    if transcript_data.get('results') and isinstance(transcript_data['results'], list):
        for result in transcript_data['results']:
            if result.get('type') == 'punctuation':
                continue
            
            if result.get('type') == 'word' and result.get('alternatives'):
                alt = result['alternatives'][0]
                speaker_label = alt.get('speaker', 'S1')
                
                # Convert "S1" -> 0, "S2" -> 1, etc.
                speaker_num = 0
                if speaker_label.startswith('S'):
                    num_str = speaker_label[1:]
                    speaker_num = int(num_str) - 1 if num_str.isdigit() else 0
                else:
                    speaker_num = int(speaker_label) if str(speaker_label).isdigit() else 0
                
                words.append({
                    'word': alt.get('content', ''),
                    'start': result.get('start_time', 0),
                    'end': result.get('end_time', result.get('start_time', 0)),
                    'speaker': speaker_num
                })
    
    # Формуємо сегменти та текст
    segments = []
    transcription_parts = []
    
    if words:
        current_speaker = None
        current_start = None
        current_words = []
        
        for word_info in words:
            speaker = word_info.get('speaker', 0)
            word_start = word_info.get('start', 0)
            word_text = word_info.get('word', '').strip()
            
            if not word_text:
                continue
            
            if speaker != current_speaker:
                if current_speaker is not None and current_words:
                    text = ' '.join(current_words).strip()
                    segments.append({
                        'speaker': current_speaker,
                        'start': round(current_start, 2),
                        'end': round(word_start, 2),
                        'text': text
                    })
                    transcription_parts.append(text)
                current_speaker = speaker
                current_start = word_start
                current_words = [word_text]
            else:
                current_words.append(word_text)
        
        if current_words:
            text = ' '.join(current_words).strip()
            segments.append({
                'speaker': current_speaker if current_speaker is not None else 0,
                'start': round(current_start, 2) if current_start is not None else 0,
                'end': round(words[-1].get('end', 0), 2) if words else 0,
                'text': text
            })
            transcription_parts.append(text)
    
    transcription = ' '.join(transcription_parts)
    
    print(f"✅ Speechmatics transcription completed: {len(segments)} segments, {len(words)} words")
    sys.stdout.flush()
    
    return transcription, segments, words


def transcribe_with_azure(audio_path, language='en-US'):
    """
    Транскрибує аудіо за допомогою Azure Speech Services з word timestamps.
    
    Args:
        audio_path: шлях до аудіофайлу
        language: код мови (наприклад, 'uk-UA', 'en-US', 'ar-SA')
    
    Returns:
        transcription: текст транскрипції
        segments: список сегментів з текстом та часовими мітками
        words: список слів з timestamps для точного матчингу
    """
    import sys
    from azure_stt import AzureSpeechClient
    
    subscription_key = os.getenv('AZURE_SPEECH_KEY')
    region = os.getenv('AZURE_SPEECH_REGION')
    
    if not subscription_key or not region:
        raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables are required")
    
    print(f"🎤 Transcribing with Azure: {audio_path}")
    print(f"   Language: {language}")
    sys.stdout.flush()
    
    # Azure потребує файл у хмарі (Blob Storage) або можна використати локальний файл через SAS URL
    # Для спрощення, використаємо локальний файл (Azure підтримує це через file://)
    # Але краще використати Azure Blob Storage або завантажити файл
    
    # Тимчасово: використовуємо Azure Speech SDK для локальних файлів
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ImportError("azure-cognitiveservices-speech is required. Install: pip install azure-cognitiveservices-speech")
    
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_recognition_language = language
    speech_config.request_word_level_timestamps()
    
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    
    # Використовуємо ConversationTranscriber для діаризації
    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
    
    segments = []
    words = []
    transcription_parts = []
    
    done = False
    error_holder = {}
    
    def recognized_cb(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            start_time = evt.result.offset / 10000000.0  # Convert from 100-nanosecond units
            duration = evt.result.duration / 10000000.0
            end_time = start_time + duration
            
            # Azure повертає speaker_id в evt.result.speaker_id (якщо доступно)
            speaker = getattr(evt.result, 'speaker_id', 0)
            if isinstance(speaker, str) and speaker.startswith('Guest'):
                speaker = int(speaker.replace('Guest', '')) if speaker.replace('Guest', '').isdigit() else 0
            
            segments.append({
                'speaker': speaker,
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'text': text.strip()
            })
            transcription_parts.append(text.strip())
            
            # Розбиваємо на слова (Azure не завжди повертає word-level timestamps в цьому API)
            # Для простоти, розбиваємо текст на слова з приблизними timestamps
            word_list = text.strip().split()
            if word_list:
                word_duration = duration / len(word_list)
                for i, word in enumerate(word_list):
                    word_start = start_time + (i * word_duration)
                    word_end = word_start + word_duration
                    words.append({
                        'word': word,
                        'start': round(word_start, 2),
                        'end': round(word_end, 2),
                        'speaker': speaker
                    })
    
    def canceled_cb(evt):
        error_holder['error'] = evt.error_details
        done = True
    
    def stop_cb(evt):
        nonlocal done
        done = True
    
    transcriber.transcribed.connect(recognized_cb)
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(canceled_cb)
    
    transcriber.start_transcribing_async().wait()
    
    # Очікуємо завершення
    import time
    timeout = 300  # 5 хвилин
    start_time = time.time()
    while not done and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    transcriber.stop_transcribing_async().wait()
    
    if 'error' in error_holder:
        raise RuntimeError(f"Azure transcription error: {error_holder['error']}")
    
    transcription = ' '.join(transcription_parts)
    
    print(f"✅ Azure transcription completed: {len(segments)} segments, {len(words)} words")
    sys.stdout.flush()
    
    return transcription, segments, words


def transcribe_audio(audio_path, language=None, transcription_provider='whisper'):
    """
    Транскрибує аудіо за допомогою вибраного провайдера з word timestamps.
    
    Args:
        audio_path: шлях до аудіофайлу
        language: код мови (наприклад, 'uk', 'en', 'ar') або None для авто-визначення
        transcription_provider: провайдер транскрипції ('whisper', 'azure', 'speechmatics')
    
    Returns:
        transcription: текст транскрипції
        segments: список сегментів з текстом та часовими мітками
        words: список слів з timestamps для точного матчингу
    """
    if transcription_provider == 'speechmatics':
        # Speechmatics використовує формат 'en', 'uk', 'ar'
        lang_map = {'en': 'en', 'uk': 'uk', 'ar': 'ar', 'en-US': 'en', 'uk-UA': 'uk', 'ar-SA': 'ar'}
        speechmatics_lang = lang_map.get(language, language or 'en')
        return transcribe_with_speechmatics(audio_path, speechmatics_lang)
    elif transcription_provider == 'azure':
        # Azure використовує формат 'en-US', 'uk-UA', 'ar-SA'
        lang_map = {'en': 'en-US', 'uk': 'uk-UA', 'ar': 'ar-SA'}
        azure_lang = lang_map.get(language, language or 'en-US')
        return transcribe_with_azure(audio_path, azure_lang)
    else:
        # Whisper (за замовчуванням)
        return transcribe_audio_whisper(audio_path, language)


def transcribe_audio_whisper(audio_path, language=None):
    """
    Транскрибує аудіо за допомогою Whisper з word timestamps.
    
    Args:
        audio_path: шлях до аудіофайлу
        language: код мови (наприклад, 'uk', 'en', 'ar') або None для авто-визначення
    
    Returns:
        transcription: текст транскрипції
        segments: список сегментів з текстом та часовими мітками
        words: список слів з timestamps для точного матчингу
    """
    import sys
    global whisper_model
    
    if whisper_model is None:
        load_models()
    
    try:
        # Отримуємо тривалість аудіо для оцінки часу обробки
        try:
            import librosa
            audio_duration = librosa.get_duration(path=audio_path)
        except:
            audio_duration = 0
        
        print(f"🎤 Transcribing audio: {audio_path}")
        if audio_duration > 0:
            print(f"   Audio duration: {audio_duration:.2f} seconds ({audio_duration/60:.1f} minutes)")
        
        # Налаштування для транскрипції (оптимізовано для швидкості)
        import torch
        device = next(whisper_model.parameters()).device
        use_fp16 = device.type == 'cuda'  # fp16 тільки на GPU, на CPU може бути повільніше
        
        # Для large моделі використовуємо більш агресивні параметри для кращого розпізнавання
        model_size = os.environ.get('WHISPER_MODEL_SIZE', 'small')
        is_large_model = model_size in ['large', 'large-v2', 'large-v3']
        
        transcribe_options = {
            'word_timestamps': True,
            'verbose': True,  # Увімкнуто для прогресу
            'task': 'transcribe',  # Завжди транскрибуємо, не перекладаємо
            'fp16': use_fp16,  # fp16 на GPU для швидкості, fp32 на CPU
            'temperature': 0.0,  # Менше випадковості = більш стабільний результат
            'best_of': 2 if is_large_model else 1,  # Для large моделі - більше варіантів
            'beam_size': 5 if is_large_model else 3,  # Для large моделі - більший beam для кращої якості
            'compression_ratio_threshold': 2.4,  # Фільтр повторень
            'logprob_threshold': -1.0,  # Фільтр низької впевненості
            'no_speech_threshold': 0.5 if is_large_model else 0.6  # Для large моделі - менший поріг тиші (більше розпізнає)
        }
        
        print(f"⚙️  Whisper settings: fp16={use_fp16}, beam_size={transcribe_options['beam_size']}, device={device.type}")
        
        if language:
            transcribe_options['language'] = language
            print(f"🌐 Using specified language: {language}")
        else:
            # Автоматичне визначення мови - Whisper зробить це автоматично
            print(f"🌐 Auto-detecting language (Whisper will detect automatically)")
            print(f"   💡 Tip: Specify 'language=uk' for Ukrainian to improve accuracy")
        
        # Транскрибуємо з детальними сегментами та word timestamps
        import time
        start_time = time.time()
        print(f"⏱️  Starting Whisper transcription (this may take a while for long audio)...")
        print(f"   Estimated time: ~{audio_duration * 0.5:.1f} seconds (rough estimate)")
        sys.stdout.flush()
        
        result = whisper_model.transcribe(
            audio_path,
            **transcribe_options
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Whisper transcription completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"   Processing speed: {audio_duration/elapsed_time:.2f}x real-time")
        sys.stdout.flush()
        
        detected_lang = result.get('language', 'unknown')
        print(f"🌐 Detected language: {detected_lang}")
        
        transcription = result.get("text", "")
        print(f"📝 Transcription text length: {len(transcription) if transcription else 0} characters")
        print(f"📝 Transcription preview: {transcription[:200] if transcription else 'EMPTY'}")
        
        segments = []
        words = []
        
        # Формуємо сегменти з текстом та збираємо всі слова
        for seg in result["segments"]:
            segments.append({
                'start': round(seg['start'], 2),
                'end': round(seg['end'], 2),
                'text': seg['text'].strip()
            })
            
            # Збираємо слова з timestamps
            if 'words' in seg:
                for word_info in seg['words']:
                    words.append({
                        'word': word_info.get('word', '').strip(),
                        'start': round(word_info.get('start', 0), 2),
                        'end': round(word_info.get('end', 0), 2)
                    })
        
        print(f"✅ Transcribed {len(segments)} segments, language: {detected_lang}")
        return transcription, segments, words
    
    except Exception as e:
        print(f"❌ Error in transcribe_audio_whisper: {e}")
        import traceback
        traceback.print_exc()
        return "", [], []


def transcribe_with_speechmatics(audio_path, language='en'):
    """
    Транскрибує аудіо за допомогою Speechmatics API з word timestamps.
    
    Args:
        audio_path: шлях до аудіофайлу
        language: код мови (наприклад, 'uk', 'en', 'ar')
    
    Returns:
        transcription: текст транскрипції
        segments: список сегментів з текстом та часовими мітками
        words: список слів з timestamps для точного матчингу
    """
    import sys
    from transcribe_with_speechmatics import upload_to_speechmatics, poll_speechmatics_job
    
    api_key = os.getenv('SPEECHMATICS_API_KEY')
    if not api_key:
        raise ValueError("SPEECHMATICS_API_KEY environment variable is not set")
    
    print(f"🎤 Transcribing with Speechmatics: {audio_path}")
    print(f"   Language: {language}")
    sys.stdout.flush()
    
    # Завантажуємо файл
    job_id = upload_to_speechmatics(api_key, audio_path, language, is_separated_track=True)
    
    # Очікуємо завершення
    transcript_data = poll_speechmatics_job(api_key, job_id)
    
    # Парсимо результат
    words = []
    if transcript_data.get('results') and isinstance(transcript_data['results'], list):
        for result in transcript_data['results']:
            if result.get('type') == 'punctuation':
                continue
            
            if result.get('type') == 'word' and result.get('alternatives'):
                alt = result['alternatives'][0]
                speaker_label = alt.get('speaker', 'S1')
                
                # Convert "S1" -> 0, "S2" -> 1, etc.
                speaker_num = 0
                if speaker_label.startswith('S'):
                    num_str = speaker_label[1:]
                    speaker_num = int(num_str) - 1 if num_str.isdigit() else 0
                else:
                    speaker_num = int(speaker_label) if str(speaker_label).isdigit() else 0
                
                words.append({
                    'word': alt.get('content', ''),
                    'start': result.get('start_time', 0),
                    'end': result.get('end_time', result.get('start_time', 0)),
                    'speaker': speaker_num
                })
    
    # Формуємо сегменти та текст
    segments = []
    transcription_parts = []
    
    if words:
        current_speaker = None
        current_start = None
        current_words = []
        
        for word_info in words:
            speaker = word_info.get('speaker', 0)
            word_start = word_info.get('start', 0)
            word_text = word_info.get('word', '').strip()
            
            if not word_text:
                continue
            
            if speaker != current_speaker:
                if current_speaker is not None and current_words:
                    text = ' '.join(current_words).strip()
                    segments.append({
                        'speaker': current_speaker,
                        'start': round(current_start, 2),
                        'end': round(word_start, 2),
                        'text': text
                    })
                    transcription_parts.append(text)
                current_speaker = speaker
                current_start = word_start
                current_words = [word_text]
            else:
                current_words.append(word_text)
        
        if current_words:
            text = ' '.join(current_words).strip()
            segments.append({
                'speaker': current_speaker if current_speaker is not None else 0,
                'start': round(current_start, 2) if current_start is not None else 0,
                'end': round(words[-1].get('end', 0), 2) if words else 0,
                'text': text
            })
            transcription_parts.append(text)
    
    transcription = ' '.join(transcription_parts)
    
    print(f"✅ Speechmatics transcription completed: {len(segments)} segments, {len(words)} words")
    sys.stdout.flush()
    
    return transcription, segments, words


def transcribe_with_azure(audio_path, language='en-US'):
    """
    Транскрибує аудіо за допомогою Azure Speech Services з word timestamps.
    
    Args:
        audio_path: шлях до аудіофайлу
        language: код мови (наприклад, 'uk-UA', 'en-US', 'ar-SA')
    
    Returns:
        transcription: текст транскрипції
        segments: список сегментів з текстом та часовими мітками
        words: список слів з timestamps для точного матчингу
    """
    import sys
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ImportError("azure-cognitiveservices-speech is required. Install: pip install azure-cognitiveservices-speech")
    
    subscription_key = os.getenv('AZURE_SPEECH_KEY')
    region = os.getenv('AZURE_SPEECH_REGION')
    
    if not subscription_key or not region:
        raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables are required")
    
    print(f"🎤 Transcribing with Azure: {audio_path}")
    print(f"   Language: {language}")
    sys.stdout.flush()
    
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
    speech_config.speech_recognition_language = language
    speech_config.request_word_level_timestamps()
    
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    
    # Використовуємо ConversationTranscriber для діаризації
    transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config, audio_config=audio_config)
    
    segments = []
    words = []
    transcription_parts = []
    
    done = False
    error_holder = {}
    
    def recognized_cb(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text
            start_time = evt.result.offset / 10000000.0  # Convert from 100-nanosecond units
            duration = evt.result.duration / 10000000.0
            end_time = start_time + duration
            
            # Azure повертає speaker_id в evt.result.speaker_id (якщо доступно)
            speaker = getattr(evt.result, 'speaker_id', 0)
            if isinstance(speaker, str):
                if speaker.startswith('Guest'):
                    speaker = int(speaker.replace('Guest', '')) if speaker.replace('Guest', '').isdigit() else 0
                elif speaker.startswith('SPEAKER_'):
                    speaker = int(speaker.replace('SPEAKER_', '')) if speaker.replace('SPEAKER_', '').isdigit() else 0
            
            segments.append({
                'speaker': speaker,
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'text': text.strip()
            })
            transcription_parts.append(text.strip())
            
            # Розбиваємо на слова з приблизними timestamps
            word_list = text.strip().split()
            if word_list:
                word_duration = duration / len(word_list)
                for i, word in enumerate(word_list):
                    word_start = start_time + (i * word_duration)
                    word_end = word_start + word_duration
                    words.append({
                        'word': word,
                        'start': round(word_start, 2),
                        'end': round(word_end, 2),
                        'speaker': speaker
                    })
    
    def canceled_cb(evt):
        error_holder['error'] = evt.error_details
        nonlocal done
        done = True
    
    def stop_cb(evt):
        nonlocal done
        done = True
    
    transcriber.transcribed.connect(recognized_cb)
    transcriber.session_stopped.connect(stop_cb)
    transcriber.canceled.connect(canceled_cb)
    
    transcriber.start_transcribing_async().wait()
    
    # Очікуємо завершення
    import time
    timeout = 300  # 5 хвилин
    start_time = time.time()
    while not done and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    transcriber.stop_transcribing_async().wait()
    
    if 'error' in error_holder:
        raise RuntimeError(f"Azure transcription error: {error_holder['error']}")
    
    transcription = ' '.join(transcription_parts)
    
    print(f"✅ Azure transcription completed: {len(segments)} segments, {len(words)} words")
    sys.stdout.flush()
    
    return transcription, segments, words


def clean_punctuation(text):
    """Очищає пунктуацію з початку та кінця тексту"""
    import string
    if not text:
        return text
    # Видаляємо пунктуацію з початку та кінця
    text = text.strip()
    while text and text[0] in string.punctuation:
        text = text[1:].strip()
    while text and text[-1] in string.punctuation:
        text = text[:-1].strip()
    return text


def get_model_id(mode='smart'):
    """
    Отримує ID моделі для заданого режиму (аналог getModelId з server.js)
    
    Args:
        mode: Режим LLM ('local', 'fast', 'smart', 'smart-2', 'test', 'test2')
    
    Returns:
        str: ID моделі
    """
    if mode == 'local':
        return os.getenv('LOCAL_LLM_MODEL') or 'openai/gpt-oss-20b'
    elif mode == 'test':
        return os.getenv('TEST_MODEL_ID') or os.getenv('OPENROUTER_TEST_MODEL_ID') or 'google/gemma-3-4b'
    elif mode == 'test2':
        return os.getenv('TEST2_MODEL_ID') or os.getenv('OPENROUTER_TEST2_MODEL_ID') or 'llama-3.2-1b-instruct'
    elif mode == 'fast':
        return os.getenv('FAST_MODEL_ID') or os.getenv('OPENROUTER_FAST_MODEL_ID') or 'gpt-oss-120b'
    elif mode == 'smart-2' or mode == 'smart2':
        return os.getenv('SMART_2_MODEL_ID') or os.getenv('OPENROUTER_SMART_2_MODEL_ID') or 'google/gemini-3-pro-preview'
    else:
        # Default to 'smart'
        return os.getenv('SMART_MODEL_ID') or os.getenv('OPENROUTER_SMART_MODEL_ID') or 'google/gemini-3.0-pro'


def call_llm_for_segment_splitting(segment, all_segments_context=None, mode='local'):
    """
    Використовує LLM для визначення, чи потрібно розділити сегмент на частини,
    якщо він містить питання + відповідь від різних спікерів.
    
    Args:
        segment: Сегмент для аналізу {'speaker': int, 'start': float, 'end': float, 'text': str}
        all_segments_context: Список всіх сегментів для контексту діалогу
        mode: Режим LLM ('local', 'fast', 'smart', 'smart-2')
    
    Returns:
        dict or None: {
            'should_split': bool,
            'parts': [
                {'text': str, 'speaker': int, 'start': float, 'end': float},
                ...
            ]
        } або None якщо LLM недоступний
    """
    import sys
    
    # Визначаємо, чи це локальний LLM
    use_local_llm = mode == 'local' or mode == 'test' or mode == 'test2'
    
    # Отримуємо модель на основі режиму
    llm_model = get_model_id(mode)
    
    # Визначаємо API URL та ключ
    if use_local_llm:
        llm_api_url = os.getenv('LOCAL_LLM_BASE_URL') or 'http://127.0.0.1:3001'
        llm_api_key = os.getenv('LOCAL_LLM_API_KEY') or ''
    else:
        llm_api_url = 'https://openrouter.ai/api/v1/chat/completions'
        llm_api_key = os.getenv('OPENROUTER_API_KEY') or ''
    
    # Перевіряємо, чи налаштований LLM
    if use_local_llm:
        if not llm_api_url:
            return None
    else:
        if not llm_api_key:
            return None
    
    # Формуємо контекст діалогу
    context_segments = ""
    if all_segments_context:
        for i, seg in enumerate(all_segments_context):
            context_segments += f"\n{i+1}. [{seg['start']:.2f}s-{seg['end']:.2f}s] Спікер {seg['speaker']}: \"{seg.get('text', '')}\""
    
    # Визначаємо основного спікера
    speaker_word_counts = {}
    if all_segments_context:
        for seg in all_segments_context:
            speaker = seg['speaker']
            word_count = len(seg.get('text', '').split())
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
    main_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0] if speaker_word_counts else 0
    
    system_prompt = """Ти експерт з аналізу діалогів. Твоя задача - визначити, чи один сегмент містить питання + відповідь від РІЗНИХ спікерів, і якщо так, розділити його на частини.

КРИТИЧНО ВАЖЛИВО:
- Якщо після питання йде відповідь, яка починається з "Uh", "Um", "Well", "Yes", "No", або містить короткі фрази типу "per second", "per minute" - це ЗАВЖДИ відповідь від ІНШОГО спікера
- Питання зазвичай задає основний спікер (той, хто веде діалог, має більше слів)
- Відповіді на питання зазвичай належать іншому спікеру (не основному)

ПРИКЛАДИ РОЗДІЛЕННЯ:

1. "What speed does it show? Uh, per second."
   → РОЗДІЛИТИ на:
   * "What speed does it show?" (основний спікер, хто задає питання)
   * "Uh, per second." (інший спікер, хто відповідає)

2. "Can you try to reset? Yes, I did."
   → РОЗДІЛИТИ на:
   * "Can you try to reset?" (основний спікер)
   * "Yes, I did." (інший спікер)

3. "Did you check the settings? Well, I think so."
   → РОЗДІЛИТИ на:
   * "Did you check the settings?" (основний спікер)
   * "Well, I think so." (інший спікер)

ПРАВИЛА ВИЯВЛЕННЯ:

1. ПИТАННЯ:
   - Містить "?"
   - Або починається з "What", "How", "Why", "When", "Where", "Did you", "Can you", "Do you", "Try to"
   - Зазвичай задає основний спікер

2. ВІДПОВІДІ:
   - Починаються з "Uh", "Um", "Well", "Yes", "No", "Yeah", "Sure", "Okay"
   - Або містять короткі фрази типу "per second", "per minute", "I did", "I do"
   - Зазвичай належать іншому спікеру (не основному)

3. КОНТЕКСТ:
   - Основний спікер - той, хто має більше слів у діалозі
   - Основний спікер зазвичай задає питання та дає інструкції
   - Інший спікер зазвичай відповідає на питання

4. КОЛИ НЕ РОЗДІЛЯТИ:
   - Якщо це риторичне питання з відповіддю від того ж спікера
   - Якщо це самоперепитування (спікер сам собі відповідає)

Поверни JSON у форматі:
{
  "should_split": true/false,
  "parts": [
    {"text": "перша частина", "speaker": 0 або 1},
    {"text": "друга частина", "speaker": 0 або 1}
  ]
}

ВАЖЛИВО: Якщо бачиш питання + відповідь з "Uh", "Um", "per second" тощо - ЗАВЖДИ розділяй!"""

    user_prompt = f"""Аналізуй цей сегмент і визнач, чи потрібно його розділити:

Сегмент для аналізу:
- Спікер (поточний): {segment['speaker']}
- Текст: "{segment.get('text', '')}"
- Час: {segment['start']:.2f}s - {segment['end']:.2f}s

Контекст діалогу (всі сегменти):{context_segments}

Основний спікер (більше слів): {main_speaker}
Розподіл слів по спікерах: {speaker_word_counts}

Визнач, чи цей сегмент містить питання + відповідь від різних спікерів. Якщо так, розділи його на частини та визнач правильних спікерів для кожної частини.

Поверни ТІЛЬКИ JSON без додаткових пояснень."""

    try:
        # Формуємо URL для локального LLM
        if use_local_llm:
            if not llm_api_url.startswith('http'):
                llm_api_url = f"http://{llm_api_url}"
            llm_api_url = llm_api_url.rstrip('/')
            if not llm_api_url.endswith('/v1/chat/completions'):
                llm_api_url = f"{llm_api_url}/v1/chat/completions"
        
        # Формуємо заголовки
        headers = {"Content-Type": "application/json"}
        if llm_api_key:
            headers["Authorization"] = f"Bearer {llm_api_key}"
        if not use_local_llm:
            headers["HTTP-Referer"] = os.getenv('APP_URL', 'http://localhost:5005')
            headers["X-Title"] = "Segment Splitting"
        
        # Формуємо payload
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": 500
        }
        
        # Викликаємо LLM
        timeout = 30 if use_local_llm else 10
        print(f"📤 [LLM Split] Відправляємо запит для розділення сегмента: '{segment.get('text', '')[:50]}...'")
        sys.stdout.flush()
        
        response = requests.post(llm_api_url, json=payload, headers=headers, timeout=timeout)
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            print(f"📝 [LLM Split] Відповідь LLM: {content[:200]}...")
            sys.stdout.flush()
            
            # Парсимо JSON відповідь
            try:
                import json
                # Видаляємо markdown code blocks якщо є
                if content.startswith('```'):
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                content = content.strip()
                
                result = json.loads(content)
                
                if result.get('should_split') and result.get('parts'):
                    # Додаємо timestamps до частин (приблизно, на основі довжини тексту)
                    total_duration = segment['end'] - segment['start']
                    total_text_length = len(segment.get('text', ''))
                    
                    parts_with_timestamps = []
                    current_time = segment['start']
                    
                    for i, part in enumerate(result['parts']):
                        part_text = part.get('text', '').strip()
                        if not part_text:
                            continue
                        
                        # Приблизна тривалість на основі довжини тексту
                        if total_text_length > 0:
                            part_ratio = len(part_text) / total_text_length
                        else:
                            part_ratio = 1.0 / len(result['parts'])
                        
                        part_duration = total_duration * part_ratio
                        part_start = current_time
                        part_end = current_time + part_duration
                        
                        parts_with_timestamps.append({
                            'text': part_text,
                            'speaker': int(part.get('speaker', segment['speaker'])),
                            'start': round(part_start, 2),
                            'end': round(part_end, 2)
                        })
                        
                        current_time = part_end
                    
                    print(f"✅ [LLM Split] Розділено на {len(parts_with_timestamps)} частин")
                    sys.stdout.flush()
                    
                    return {
                        'should_split': True,
                        'parts': parts_with_timestamps
                    }
                else:
                    print(f"ℹ️ [LLM Split] LLM визначив, що розділення не потрібно")
                    sys.stdout.flush()
                    return {
                        'should_split': False,
                        'parts': []
                    }
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"⚠️ [LLM Split] Помилка парсингу JSON: {e}, відповідь: {content[:200]}")
                sys.stdout.flush()
                return None
        else:
            print(f"⚠️ [LLM Split] LLM API повернув статус {response.status_code}: {response.text[:200]}")
            sys.stdout.flush()
            return None
            
    except requests.exceptions.Timeout:
        print(f"⚠️ [LLM Split] LLM API timeout")
        sys.stdout.flush()
        return None
    except Exception as e:
        print(f"⚠️ [LLM Split] Помилка запиту: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None


def call_llm_for_speaker_correction(prev_seg, current_seg, gap_to_prev, all_segments_context=None, mode='local'):
    """
    Використовує LLM як посередника для визначення правильного спікера для поточного сегмента.
    
    Args:
        prev_seg: Попередній сегмент {'speaker': int, 'start': float, 'end': float, 'text': str}
        current_seg: Поточний сегмент {'speaker': int, 'start': float, 'end': float, 'text': str}
        gap_to_prev: Gap між сегментами в секундах
        all_segments_context: Опціонально, список всіх сегментів для контексту
        mode: Режим LLM ('local', 'fast', 'smart', 'smart-2', 'test', 'test2')
    
    Returns:
        int or None: Правильний спікер для поточного сегмента, або None якщо LLM недоступний
    """
    # Визначаємо, чи це локальний LLM
    use_local_llm = mode == 'local' or mode == 'test' or mode == 'test2'
    
    # Отримуємо модель на основі режиму
    llm_model = get_model_id(mode)
    
    # Визначаємо API URL та ключ
    if use_local_llm:
        llm_api_url = os.getenv('LOCAL_LLM_BASE_URL') or 'http://127.0.0.1:3001'
        llm_api_key = os.getenv('LOCAL_LLM_API_KEY') or ''
    else:
        llm_api_url = 'https://openrouter.ai/api/v1/chat/completions'
        llm_api_key = os.getenv('OPENROUTER_API_KEY') or ''
    
    # Перевіряємо, чи налаштований LLM
    if use_local_llm:
        if not llm_api_url:
            print(f"⚠️ LLM недоступний: LOCAL_LLM_BASE_URL не налаштований")
            sys.stdout.flush()
            return None
        print(f"🔍 [LLM] Викликаємо локальний LLM: {llm_api_url}, модель: {llm_model}")
    else:
        if not llm_api_key:
            print(f"⚠️ LLM недоступний: OPENROUTER_API_KEY не налаштований")
            sys.stdout.flush()
            return None
        print(f"🔍 [LLM] Викликаємо OpenRouter LLM: {llm_api_url}, модель: {llm_model}")
    
    sys.stdout.flush()
    
    # Формуємо промпт для LLM
    prev_duration = prev_seg['end'] - prev_seg['start']
    current_duration = current_seg['end'] - current_seg['start']
    
    context_info = ""
    if all_segments_context:
        # Додаємо контекст про розподіл спікерів
        speaker_word_counts = {}
        for seg in all_segments_context:
            speaker = seg['speaker']
            word_count = len(seg.get('text', '').split())
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
        
        main_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0] if speaker_word_counts else 0
        context_info = f"\n\nКонтекст діалогу:\n- Основний спікер (більше слів): {main_speaker}\n- Розподіл слів по спікерах: {speaker_word_counts}"
    
    system_prompt = """Ти експерт з діаризації спікерів. Твоя задача - визначити правильний спікер для поточного сегмента на основі контексту діалогу та мовних паттернів.

ВАЖЛИВІ ПРАВИЛА ДЛЯ АНАЛІЗУ:

1. КОРОТКІ РЕПЛІКИ ТА ПИТАННЯ:
   - Якщо попередній сегмент короткий (<1 секунда) і належить одному спікеру,
   - А поточний сегмент є питанням або інструкцією (починається з "Hey", "Did you", "Can you", "Try to", "What", "How", "Why", "When", "Where", "You should", "You can", "You need", містить "?"),
   - І gap між сегментами <3 секунди,
   - То поточний сегмент має належати ОСНОВНОМУ спікеру (тому, хто веде діалог, задає питання, дає інструкції).

2. СЛОВА НА ПОЧАТКУ ФАЙЛУ:
   - Якщо слово знаходиться на початку файлу (<3 секунди від початку),
   - І є питанням/інструкцією (починається з "Hey", "Did", "Can", "Try", "What", "How", "Why", "When", "Where", "You"),
   - То воно має належати основному спікеру.

3. КОНТЕКСТНІ ЗВ'ЯЗКИ:
   - Якщо попереднє і наступне слова належать одному спікеру, і gap <2 секунди,
   - Поточне слово теж має належати тому спікеру (це вирішує проблему з фразами між репліками одного спікера).

4. БІДИРЕКЦІЙНИЙ КОНТЕКСТ:
   - Якщо слово оточене словами одного спікера з обох боків (gap <2 секунди),
   - Воно має належати тому ж спікеру.

5. ВИЗНАЧЕННЯ ОСНОВНОГО СПІКЕРА:
   - Основний спікер - це той, хто веде діалог, задає питання, дає інструкції, має більше слів у діалозі.
   - Він зазвичай продовжує розмову після коротких реплік іншого спікера.

6. ЗАПЕРЕЧЕННЯ ТА ВИПРАВЛЕННЯ:
   - Якщо попередній сегмент містить питання (наприклад, "What speed does it show?") або репліку з інформацією (наприклад, "Uh, per second."),
   - А поточний сегмент починається з "No" або подібних заперечень (наприклад, "No, it should be 200."),
   - То це ЗАЗВИЧАЙ відповідь від ІНШОГО спікера, який виправляє або уточнює інформацію.
   - ПРИКЛАД: "What speed does it show?" (спікер 0) → "Uh, per second." (спікер 1) → "No, it should be 200." (спікер 0 або 1?)
   - В такому випадку "No, it should be 200" зазвичай належить тому ж спікеру, що і попередня відповідь ("Uh, per second"), або іншому спікеру, який виправляє.
   - АНАЛІЗУЙ КОНТЕКСТ: якщо попередня відповідь була від неосновного спікера, то "No" може бути від основного спікера (виправлення), або від того ж неосновного (уточнення).

7. ОСОБЛИВІ ВИПАДКИ:
   - Фрази типу "Hey, did you try to reset your modem?" після короткої репліки іншого спікера належать основному спікеру (продовження діалогу).

АНАЛІЗУЙ КОНТЕКСТ:
- Розподіл слів по спікерах (хто говорить більше)
- Хто задає питання та дає інструкції
- Хто продовжує діалог після коротких реплік
- Тривалість та gap між сегментами
- Мовні паттерни (питання, інструкції, відповіді)

Поверни ТІЛЬКИ номер спікера (0 або 1) без додаткових пояснень."""

    # Перевіряємо, чи є питання в попередньому сегменті
    prev_text = prev_seg.get('text', '').strip()
    has_question = '?' in prev_text
    
    # Перевіряємо, чи поточний сегмент починається з заперечення
    current_text = current_seg.get('text', '').strip().lower()
    starts_with_negation = any(
        current_text.startswith(neg) for neg in ['no,', 'no ', 'nope,', 'nope ', 'nah,', 'nah ', 'not,', 'not ']
    )
    
    # Додаємо спеціальну інформацію для випадків з запереченнями
    negation_info = ""
    if has_question and starts_with_negation:
        negation_info = "\n\n⚠️ ВАЖЛИВО: Попередній сегмент містить питання, а поточний починається з заперечення (No/Nope/Nah/Not).\nЦе зазвичай означає, що поточний сегмент - це відповідь/виправлення від ІНШОГО спікера, але проаналізуй контекст діалогу, щоб визначити правильного спікера."
    
    user_prompt = f"""Попередній сегмент:
- Спікер: {prev_seg['speaker']}
- Текст: "{prev_text}"
- Тривалість: {prev_duration:.2f} секунд
- Час: {prev_seg['start']:.2f}s - {prev_seg['end']:.2f}s
- Містить питання: {'Так' if has_question else 'Ні'}

Поточний сегмент:
- Спікер (поточний): {current_seg['speaker']}
- Текст: "{current_seg.get('text', '')}"
- Тривалість: {current_duration:.2f} секунд
- Час: {current_seg['start']:.2f}s - {current_seg['end']:.2f}s
- Починається з заперечення: {'Так' if starts_with_negation else 'Ні'}

Gap між сегментами: {gap_to_prev:.2f} секунд{context_info}{negation_info}

Визнач правильний спікер для поточного сегмента на основі контексту діалогу. Поверни ТІЛЬКИ номер спікера (0 або 1)."""

    try:
        # Формуємо URL для локального LLM
        if use_local_llm:
            # Для локального LLM додаємо /v1/chat/completions якщо потрібно
            if not llm_api_url.startswith('http'):
                llm_api_url = f"http://{llm_api_url}"
            llm_api_url = llm_api_url.rstrip('/')
            if not llm_api_url.endswith('/v1/chat/completions'):
                llm_api_url = f"{llm_api_url}/v1/chat/completions"
        
        # Формуємо заголовки
        headers = {"Content-Type": "application/json"}
        if llm_api_key:
            headers["Authorization"] = f"Bearer {llm_api_key}"
        if not use_local_llm:
            headers["HTTP-Referer"] = os.getenv('APP_URL', 'http://localhost:5005')
            headers["X-Title"] = "Speaker Correction"
        
        # Формуємо payload
        payload = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
            "max_tokens": 10  # Потрібно тільки номер спікера
        }
        
        # Викликаємо LLM
        timeout = 30 if use_local_llm else 10
        print(f"📤 [LLM] Відправляємо запит до LLM...")
        sys.stdout.flush()
        
        response = requests.post(llm_api_url, json=payload, headers=headers, timeout=timeout)
        
        print(f"📥 [LLM] Отримано відповідь: статус {response.status_code}")
        sys.stdout.flush()
        
        if response.status_code == 200:
            try:
                response_data = response.json()
            except ValueError as e:
                print(f"⚠️ [LLM] Помилка парсингу JSON відповіді: {e}")
                print(f"📝 [LLM] Сира відповідь: {response.text[:200]}")
                sys.stdout.flush()
                return None
            
            # Безпечне отримання контенту з перевірками
            choices = response_data.get("choices", [])
            if not choices:
                print(f"⚠️ [LLM] Відповідь не містить choices: {response_data}")
                sys.stdout.flush()
                return None
            
            message = choices[0].get("message", {})
            if not message:
                print(f"⚠️ [LLM] Choice не містить message: {choices[0]}")
                sys.stdout.flush()
                return None
            
            content = message.get("content", "").strip()
            
            print(f"📝 [LLM] Відповідь LLM: '{content}' (довжина: {len(content)})")
            sys.stdout.flush()
            
            # Перевірка на порожню відповідь
            if not content:
                print(f"⚠️ [LLM] LLM повернув порожню відповідь. Це може бути помилка LLM.")
                sys.stdout.flush()
                return None
            
            # Парсимо відповідь (очікуємо тільки номер спікера)
            try:
                # Видаляємо всі нецифрові символи
                speaker_str = ''.join(filter(str.isdigit, content))
                if speaker_str:
                    speaker = int(speaker_str[0])  # Беремо першу цифру
                    # Перевірка на валідність спікера (0 або 1)
                    if speaker not in [0, 1]:
                        print(f"⚠️ [LLM] LLM повернув невалідний номер спікера: {speaker} (очікується 0 або 1)")
                        sys.stdout.flush()
                        return None
                    print(f"✅ [LLM] Визначено спікера: {speaker} (було: {current_seg['speaker']})")
                    sys.stdout.flush()
                    return speaker
                else:
                    print(f"⚠️ [LLM] LLM повернув відповідь без цифр: '{content}'")
                    sys.stdout.flush()
                    return None
            except (ValueError, IndexError, TypeError) as e:
                print(f"⚠️ [LLM] Помилка парсингу відповіді LLM: {e}, контент: '{content}'")
                sys.stdout.flush()
                return None
        else:
            print(f"⚠️  LLM API повернув статус {response.status_code}: {response.text[:200]}")
            sys.stdout.flush()
            return None
            
    except requests.exceptions.Timeout:
        print(f"⚠️  LLM API timeout (більше {timeout}s)")
        sys.stdout.flush()
        return None
    except Exception as e:
        print(f"⚠️  LLM request error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None


def combine_diarization_and_transcription(diarization_segments, words, llm_mode='local'):
    """
    Об'єднує результати діаризації та транскрипції на рівні слів для точності.
    
    Args:
        diarization_segments: сегменти діаризації [{'speaker': int, 'start': float, 'end': float}]
        words: список слів з timestamps [{'word': str, 'start': float, 'end': float}]
        llm_mode: Режим LLM для виправлення призначень спікерів ('local', 'fast', 'smart', 'smart-2')
    
    Returns:
        combined: список об'єднаних сегментів з спікером та текстом
    """
    if not words:
        print("⚠️  No words provided for combination")
        return []
    
    if not diarization_segments:
        print("⚠️  No diarization segments provided, using default speaker 0")
        # Якщо немає діаризації, повертаємо транскрипцію з одним спікером
        combined = []
        current_start = words[0]['start']
        current_words = []
        
        for word in words:
            if not word['word'].strip():
                continue
            current_words.append(word['word'])
            # Створюємо сегменти по проміжках >1 сек
            if len(combined) == 0 or (word['start'] - combined[-1]['end'] > 1.0):
                if current_words:
                    combined.append({
                        'speaker': 0,
                        'start': round(current_start, 2),
                        'end': round(word['end'], 2),
                        'text': ' '.join(current_words).strip()
                    })
                    current_words = []
                    current_start = word['start']
        
        if current_words:
            combined.append({
                'speaker': 0,
                'start': round(current_start, 2),
                'end': round(words[-1]['end'], 2),
                'text': ' '.join(current_words).strip()
            })
        return combined
    
    print(f"🔗 Combining {len(words)} words with {len(diarization_segments)} diarization segments")
    
    # Сортуємо сегменти діаризації за часом для швидшого пошуку
    sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
    
    # ТРІЗ: Визначаємо основного спікера на основі комбінації діаризації та транскрипції
    # Це важливо, бо якщо визначати після об'єднання, може бути неправильним, якщо слова вже неправильно призначені
    speaker_durations = {}
    for diar_seg in sorted_diar_segments:
        speaker = diar_seg['speaker']
        duration = diar_seg['end'] - diar_seg['start']
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
    
    # Перевіряємо, чи є чіткий переважаючий спікер в діаризації
    if speaker_durations:
        max_duration = max(speaker_durations.values())
        total_duration = sum(speaker_durations.values())
        max_ratio = max_duration / total_duration if total_duration > 0 else 0
        
        # Якщо є чіткий переважаючий спікер (>60%), використовуємо діаризацію
        if max_ratio > 0.6:
            main_speaker_from_diarization = max(speaker_durations.items(), key=lambda x: x[1])[0]
            print(f"👑 Main speaker from diarization: {main_speaker_from_diarization} (durations: {speaker_durations}, ratio: {max_ratio:.2%})")
        else:
            # Якщо немає чіткого переважаючого спікера, використовуємо діаризацію як fallback
            # (пізніше оновимо на основі транскрипції)
            main_speaker_from_diarization = max(speaker_durations.items(), key=lambda x: x[1])[0]
            print(f"👑 Main speaker from diarization (fallback, ratio: {max_ratio:.2%}): {main_speaker_from_diarization} (durations: {speaker_durations})")
    else:
        main_speaker_from_diarization = 0
        print(f"👑 Main speaker from diarization: {main_speaker_from_diarization} (no diarization segments)")
    
    # Для кожного слова знаходимо найкраще перекриття з сегментами діаризації
    word_speakers = []
    for word_idx, word in enumerate(words):
        word_start = word['start']
        word_end = word['end']
        word_center = (word_start + word_end) / 2.0
        word_text = word['word']
        
        if not word_text.strip():
            continue
        
        word_duration = word_end - word_start
        
        # КРИТИЧНО: Шукаємо всі сегменти, які перетинаються з словом (не тільки повністю містять)
        # Це важливо, бо слова можуть частково перетинатися з сегментами діаризації
        overlapping_segments = []
        for diar_seg in sorted_diar_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            segment_duration = diar_end - diar_start
            
            # Перевіряємо перекриття (не тільки повне вміщення)
            overlap_start = max(word_start, diar_start)
            overlap_end = min(word_end, diar_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                overlap_ratio = overlap / word_duration if word_duration > 0 else 0
                diar_center = (diar_start + diar_end) / 2.0
                center_distance = abs(word_center - diar_center)
                
                # Перевіряємо, чи слово повністю в межах (найвищий пріоритет)
                fully_contained = (word_start >= diar_start and word_end <= diar_end)
                
                # ТРІЗ: Виявлення проблемних великих сегментів
                # Сегменти >10 секунд вважаються "підозрілими" - вони можуть "поглинати" слова
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
        
        # Якщо є сегменти, що перетинаються, вибираємо найкращий
        if overlapping_segments:
            # ТРІЗ РІШЕННЯ: Віддаємо перевагу найближчому сегменту з достатнім перекриттям
            # Це вирішує проблему "поглинання" слів великими сегментами
            
            # КРИТИЧНО: Перевіряємо попереднє слово - якщо воно вже призначене іншому спікеру
            # і поточне слово дуже близьке (<0.5s), вони мають належати одному спікеру
            if len(word_speakers) > 0:
                prev_word = word_speakers[-1]
                gap_to_prev = word_start - prev_word['end']
                
                # Якщо попереднє слово дуже близьке (<0.5s), перевіряємо чи є сегмент того ж спікера
                if gap_to_prev < 0.5:
                    prev_speaker = prev_word['speaker']
                    current_speakers = set(s['speaker'] for s in overlapping_segments)
                    
                    # Якщо попереднє слово належить спікеру, який не в поточних overlapping_segments,
                    # але є сегмент того спікера в overlapping_segments, використовуємо його
                    if prev_speaker not in current_speakers:
                        # Шукаємо сегмент того спікера в overlapping_segments
                        matching_prev_speaker_segs = [
                            s for s in overlapping_segments 
                            if s['speaker'] == prev_speaker
                        ]
                        if matching_prev_speaker_segs:
                            # Використовуємо сегмент того спікера
                            best_seg = min(matching_prev_speaker_segs, key=lambda x: x['center_distance'])
                            speaker_id = best_seg['speaker']
                            word_speakers.append({
                                'word': word_text,
                                'start': word_start,
                                'end': word_end,
                                'speaker': speaker_id,
                                'triz_corrected': True
                            })
                            continue
                    else:
                        # Якщо попереднє слово належить спікеру, який є в overlapping_segments,
                        # але поточний найближчий сегмент від іншого спікера і підозрілий,
                        # віддаємо перевагу сегменту попереднього спікера
                        matching_prev_speaker_segs = [
                            s for s in overlapping_segments 
                            if s['speaker'] == prev_speaker
                        ]
                        if matching_prev_speaker_segs:
                            closest_current = min(overlapping_segments, key=lambda x: x['center_distance'])
                            closest_prev = min(matching_prev_speaker_segs, key=lambda x: x['center_distance'])
                            
                            # Якщо найближчий поточний сегмент підозрілий і від іншого спікера,
                            # а сегмент попереднього спікера не підозрілий, використовуємо його
                            if (closest_current['is_suspicious_large'] and 
                                closest_current['speaker'] != prev_speaker and
                                not closest_prev['is_suspicious_large']):
                                speaker_id = prev_speaker
                                word_speakers.append({
                                    'word': word_text,
                                    'start': word_start,
                                    'end': word_end,
                                    'speaker': speaker_id,
                                    'triz_corrected': True
                                })
                                continue
                        else:
                            # Якщо немає сегмента попереднього спікера в overlapping_segments,
                            # але gap дуже малий (<0.3s), просто використовуємо того ж спікера
                            # (це означає, що слова йдуть одразу один за одним і належать одному спікеру)
                            if gap_to_prev < 0.3:
                                speaker_id = prev_speaker
                                word_speakers.append({
                                    'word': word_text,
                                    'start': word_start,
                                    'end': word_end,
                                    'speaker': speaker_id,
                                    'triz_corrected': True
                                })
                                continue
            
            # КРИТИЧНО: Перевіряємо контекст - великі паузи між репліками
            # Якщо є велика пауза (>3s) перед або після слова, і є сегмент іншого спікера поблизу,
            # це може бути окрема репліка від іншого спікера
            word_index = [w['word'] for w in words].index(word_text) if word_text in [w['word'] for w in words] else -1
            
            if word_index >= 0:
                # Перевіряємо паузу перед словом
                gap_before = 0
                if word_index > 0:
                    prev_word = words[word_index - 1]
                    gap_before = word_start - prev_word['end']
                
                # Перевіряємо паузу після слова
                gap_after = 0
                if word_index < len(words) - 1:
                    next_word = words[word_index + 1]
                    gap_after = next_word['start'] - word_end
                
                # Якщо є велика пауза (>3s) перед або після, перевіряємо сегменти іншого спікера
                if gap_before > 3.0 or gap_after > 3.0:
                    current_speakers = set(s['speaker'] for s in overlapping_segments)
                    nearby_other_speaker_segments = []
                    
                    for diar_seg in sorted_diar_segments:
                        if diar_seg['speaker'] not in current_speakers:
                            diar_start = diar_seg['start']
                            diar_end = diar_seg['end']
                            
                            # Перевіряємо чи сегмент близький до слова (в межах паузи)
                            if gap_before > 3.0:
                                # Сегмент перед словом
                                distance_before = word_start - diar_end
                                if 0 <= distance_before < gap_before + 1.0:  # В межах паузи + 1s
                                    nearby_other_speaker_segments.append({
                                        'segment': diar_seg,
                                        'speaker': diar_seg['speaker'],
                                        'distance': distance_before,
                                        'type': 'before'
                                    })
                            
                            if gap_after > 3.0:
                                # Сегмент після слова
                                distance_after = diar_start - word_end
                                if 0 <= distance_after < gap_after + 1.0:  # В межах паузи + 1s
                                    nearby_other_speaker_segments.append({
                                        'segment': diar_seg,
                                        'speaker': diar_seg['speaker'],
                                        'distance': distance_after,
                                        'type': 'after'
                                    })
                    
                    # Якщо є сегменти іншого спікера в межах паузи, використовуємо їх
                    if nearby_other_speaker_segments:
                        closest_other = min(nearby_other_speaker_segments, key=lambda x: x['distance'])
                        closest_current = min(overlapping_segments, key=lambda x: x['center_distance'])
                        
                        # Якщо сегмент іншого спікера дуже близький (<2s) і поточний сегмент підозрілий,
                        # використовуємо сегмент іншого спікера
                        if (closest_other['distance'] < 2.0 and 
                            (closest_current['is_suspicious_large'] or closest_other['distance'] < 1.0)):
                            speaker_id = closest_other['speaker']
                            word_speakers.append({
                                'word': word_text,
                                'start': word_start,
                                'end': word_end,
                                'speaker': speaker_id,
                                'triz_corrected': True
                            })
                            continue
            
            # КРИТИЧНО: Якщо всі overlapping сегменти підозрілі (великі), перевіряємо чи є сегменти іншого спікера поблизу
            # (навіть якщо вони не перетинаються безпосередньо)
            all_suspicious = all(s['is_suspicious_large'] for s in overlapping_segments)
            if all_suspicious:
                # Шукаємо сегменти іншого спікера поблизу (навіть якщо не перетинаються)
                current_speakers = set(s['speaker'] for s in overlapping_segments)
                nearby_other_speaker_segments = []
                
                for diar_seg in sorted_diar_segments:
                    if diar_seg['speaker'] not in current_speakers:
                        diar_start = diar_seg['start']
                        diar_end = diar_seg['end']
                        diar_center = (diar_start + diar_end) / 2.0
                        
                        # Відстань від центру слова до центру сегмента
                        center_distance = abs(word_center - diar_center)
                        
                        # Відстань від кінця сегмента до початку слова (якщо сегмент перед словом)
                        distance_before = word_start - diar_end if word_start > diar_end else float('inf')
                        
                        # Відстань від кінця слова до початку сегмента (якщо сегмент після слова)
                        distance_after = diar_start - word_end if diar_start > word_end else float('inf')
                        
                        # Перевіряємо чи сегмент дуже близький (<2s)
                        if center_distance < 2.0 or distance_before < 2.0 or distance_after < 2.0:
                            nearby_other_speaker_segments.append({
                                'segment': diar_seg,
                                'speaker': diar_seg['speaker'],
                                'center_distance': center_distance,
                                'distance_before': distance_before,
                                'distance_after': distance_after,
                                'min_distance': min(center_distance, distance_before, distance_after)
                            })
                
                # Якщо є близькі сегменти іншого спікера, використовуємо найближчий
                if nearby_other_speaker_segments:
                    closest_other = min(nearby_other_speaker_segments, key=lambda x: x['min_distance'])
                    closest_suspicious = min(overlapping_segments, key=lambda x: x['center_distance'])
                    
                    # Якщо сегмент іншого спікера дуже близький (<1.5s), використовуємо його
                    # Особливо якщо підозрілий сегмент має велику center_distance
                    if (closest_other['min_distance'] < 1.5 and 
                        closest_suspicious['center_distance'] > 1.0):
                        speaker_id = closest_other['speaker']
                        word_speakers.append({
                            'word': word_text,
                            'start': word_start,
                            'end': word_end,
                            'speaker': speaker_id,
                            'triz_corrected': True
                        })
                        continue
            
            # КРИТИЧНО: Перевіряємо контекст - якщо наступні слова належать іншому спікеру,
            # і поточне слово дуже близько до них (<0.5s), це може бути частина репліки того спікера
            # Але якщо наступні слова належать поточному спікеру, поточне слово теж має належати йому
            word_index = words.index(word)
            if word_index < len(words) - 1:
                next_word = words[word_index + 1]
                next_word_start = next_word['start']
                gap_to_next = next_word_start - word_end
                
                # Якщо наступне слово дуже близько (<0.5s), перевіряємо його спікера
                # (але ми ще не знаємо його спікера, тому використовуємо діаризацію)
                if gap_to_next < 0.5:
                    # Знаходимо найближчий сегмент діаризації для наступного слова
                    next_word_center = (next_word_start + next_word['end']) / 2.0
                    closest_next_seg = None
                    min_next_distance = float('inf')
                    for diar_seg in sorted_diar_segments:
                        if next_word_start < diar_seg['end'] and next_word['end'] > diar_seg['start']:
                            diar_center = (diar_seg['start'] + diar_seg['end']) / 2.0
                            distance = abs(next_word_center - diar_center)
                            if distance < min_next_distance:
                                min_next_distance = distance
                                closest_next_seg = diar_seg
                    
                    # Якщо наступне слово належить іншому спікеру (не тому, що в overlapping_segments),
                    # і поточне слово перетинається з сегментом того ж спікера, використовуємо його
                    if closest_next_seg:
                        next_speaker = closest_next_seg['speaker']
                        current_speakers = set(s['speaker'] for s in overlapping_segments)
                        
                        # Якщо наступне слово належить спікеру, який не в поточних overlapping_segments,
                        # але є сегмент того спікера в overlapping_segments, використовуємо його
                        if next_speaker not in current_speakers:
                            # Шукаємо сегмент того спікера в overlapping_segments
                            matching_next_speaker_segs = [
                                s for s in overlapping_segments 
                                if s['speaker'] == next_speaker
                            ]
                            if matching_next_speaker_segs:
                                # Використовуємо сегмент того спікера
                                best_seg = min(matching_next_speaker_segs, key=lambda x: x['center_distance'])
                                speaker_id = best_seg['speaker']
                                word_speakers.append({
                                    'word': word_text,
                                    'start': word_start,
                                    'end': word_end,
                                    'speaker': speaker_id,
                                    'triz_corrected': True
                                })
                                continue
            
            # КРИТИЧНО: Для слів на початку файлу (<3.0s) перевіряємо сусідні сегменти інших спікерів
            # Якщо слово на початку і є сегмент іншого спікера, який починається дуже близько (<0.5s від слова),
            # це може бути частина репліки того спікера, навіть якщо поточний сегмент має overlap
            # ТРІЗ: Збільшено поріг з <1.0s до <3.0s для кращого виявлення фраз після коротких реплік
            if word_start < 3.0:
                # Шукаємо сегменти інших спікерів, які починаються дуже близько до слова
                nearby_other_speaker_segments = []
                for diar_seg in sorted_diar_segments:
                    # Перевіряємо сегменти, які починаються в межах 0.5s від кінця слова
                    distance_to_start = diar_seg['start'] - word_end
                    if distance_to_start >= 0 and distance_to_start < 0.5:
                        # Це сегмент, який починається після слова, але дуже близько
                        # Перевіряємо, чи він від іншого спікера (не від того, що вже в overlapping)
                        current_speakers = set(s['speaker'] for s in overlapping_segments)
                        if diar_seg['speaker'] not in current_speakers:
                            # Це сегмент іншого спікера
                            nearby_other_speaker_segments.append({
                                'segment': diar_seg,
                                'speaker': diar_seg['speaker'],
                                'distance_to_start': distance_to_start,
                                'segment_start': diar_seg['start'],
                                'segment_end': diar_seg['end']
                            })
                
                # Якщо є близькі сегменти інших спікерів, перевіряємо чи не є слово частиною їх репліки
                if nearby_other_speaker_segments:
                    # Знаходимо найближчий сегмент іншого спікера
                    closest_other = min(nearby_other_speaker_segments, key=lambda x: x['distance_to_start'])
                    
                    # Перевіряємо поточний найближчий сегмент
                    closest_current_seg = min(overlapping_segments, key=lambda x: x['center_distance'])
                    
                    # ТРІЗ: Якщо поточний сегмент починається з самого початку файлу (start=0.0),
                    # це може бути помилка діаризації - вона часто створює сегменти з 0.0
                    # У такому випадку віддаємо перевагу сегменту іншого спікера, який починається близько
                    current_seg_starts_at_zero = closest_current_seg['segment']['start'] < 0.1
                    
                    if closest_other['distance_to_start'] < 0.5:
                        # ТРІЗ: Перевіряємо тип фрази (питання/інструкція)
                        # Якщо слово є частиною питання/інструкції, воно має належати основному спікеру
                        # Очищаємо пунктуацію перед перевіркою
                        word_text_clean = clean_punctuation(word_text).lower()
                        is_question_or_instruction = any([
                            word_text_clean.startswith('hey'),
                            word_text_clean.startswith('did'),
                            word_text_clean.startswith('can'),
                            word_text_clean.startswith('try'),
                            word_text_clean.startswith('what'),
                            word_text_clean.startswith('how'),
                            word_text_clean.startswith('why'),
                            word_text_clean.startswith('when'),
                            word_text_clean.startswith('where'),
                        ])
                        
                        # Перевіряємо тривалість попереднього сегмента іншого спікера
                        prev_seg_duration = closest_other['segment_end'] - closest_other['segment_start']
                        is_prev_short = prev_seg_duration < 1.0
                        
                        # Якщо попередній сегмент короткий і поточне слово є питанням/інструкцією,
                        # воно має належати основному спікеру
                        if is_prev_short and is_question_or_instruction:
                            speaker_id = main_speaker_from_diarization
                            word_speakers.append({
                                'word': word_text,
                                'start': word_start,
                                'end': word_end,
                                'speaker': speaker_id,
                                'triz_corrected': True  # Позначка: виправлено ТРІЗ логікою для слів на початку (питання/інструкція)
                            })
                            continue
                        
                        # Сегмент іншого спікера дуже близько - це ймовірно частина його репліки
                        # Особливо якщо поточний сегмент починається з 0.0 (помилка діаризації)
                        # або підозрілий (великий)
                        if (current_seg_starts_at_zero or 
                            closest_current_seg['is_suspicious_large'] or 
                            closest_current_seg['center_distance'] > 0.2):
                            # Використовуємо сегмент іншого спікера
                            speaker_id = closest_other['speaker']
                            word_speakers.append({
                                'word': word_text,
                                'start': word_start,
                                'end': word_end,
                                'speaker': speaker_id,
                                'triz_corrected': True  # Позначка: виправлено ТРІЗ логікою для слів на початку
                            })
                            continue
            
            # КРИТИЧНО: Перевіряємо контекст - якщо наступні слова належать іншому спікеру,
            # і поточне слово дуже близько до них (<0.5s), це може бути частина репліки того спікера
            word_index = [w['word'] for w in words].index(word_text) if word_text in [w['word'] for w in words] else -1
            if word_index >= 0 and word_index < len(words) - 1:
                next_word = words[word_index + 1]
                next_word_start = next_word['start']
                gap_to_next = next_word_start - word_end
                
                # Якщо наступне слово дуже близько (<0.5s), перевіряємо його спікера через діаризацію
                if gap_to_next < 0.5:
                    # Знаходимо найближчий сегмент діаризації для наступного слова
                    next_word_center = (next_word_start + next_word['end']) / 2.0
                    closest_next_seg = None
                    min_next_distance = float('inf')
                    for diar_seg in sorted_diar_segments:
                        if next_word_start < diar_seg['end'] and next_word['end'] > diar_seg['start']:
                            diar_center = (diar_seg['start'] + diar_seg['end']) / 2.0
                            distance = abs(next_word_center - diar_center)
                            if distance < min_next_distance:
                                min_next_distance = distance
                                closest_next_seg = diar_seg
                    
                    # Якщо наступне слово належить іншому спікеру, використовуємо сегмент того спікера
                    if closest_next_seg:
                        next_speaker = closest_next_seg['speaker']
                        # Знаходимо найближчий сегмент для поточного слова
                        closest_current_seg = min(overlapping_segments, key=lambda x: x['center_distance'])
                        current_speaker = closest_current_seg['speaker']
                        
                        # Якщо наступне слово належить іншому спікеру, і є сегмент того спікера в overlapping_segments,
                        # використовуємо його (це означає, що поточне слово - частина репліки того спікера)
                        if next_speaker != current_speaker:
                            # Шукаємо сегмент того спікера в overlapping_segments
                            matching_next_speaker_segs = [
                                s for s in overlapping_segments 
                                if s['speaker'] == next_speaker
                            ]
                            if matching_next_speaker_segs:
                                # Використовуємо сегмент того спікера
                                best_seg = min(matching_next_speaker_segs, key=lambda x: x['center_distance'])
                                speaker_id = best_seg['speaker']
                                word_speakers.append({
                                    'word': word_text,
                                    'start': word_start,
                                    'end': word_end,
                                    'speaker': speaker_id,
                                    'triz_corrected': True
                                })
                                continue
            
            # Пріоритет 1: Найближчі сегменти (center_distance <0.5s) з overlap_ratio >5%
            # Це найточніші сегменти, які точно належать слову
            close_segments = [
                s for s in overlapping_segments 
                if s['center_distance'] < 0.5 and s['overlap_ratio'] > 0.05
            ]
            
            if close_segments:
                # Якщо є близькі сегменти, використовуємо найближчий
                # Віддаємо перевагу меншим сегментам (не підозрілим великим)
                non_suspicious_close = [s for s in close_segments if not s['is_suspicious_large']]
                if non_suspicious_close:
                    best_seg = min(non_suspicious_close, key=lambda x: x['center_distance'])
                else:
                    # Якщо всі близькі підозрілі, використовуємо найближчий
                    best_seg = min(close_segments, key=lambda x: x['center_distance'])
            else:
                # Пріоритет 2: Якщо немає близьких, вибираємо з урахуванням підозрілих великих сегментів
                # Віддаємо перевагу меншим сегментам, навіть якщо overlap менший
                non_suspicious = [s for s in overlapping_segments if not s['is_suspicious_large']]
                
                if non_suspicious:
                    # Якщо є непідозрілі сегменти, використовуємо найближчий з них
                    # Але тільки якщо він має overlap_ratio >5%
                    valid_non_suspicious = [s for s in non_suspicious if s['overlap_ratio'] > 0.05]
                    if valid_non_suspicious:
                        best_seg = min(valid_non_suspicious, key=lambda x: x['center_distance'])
                    else:
                        # Якщо немає валідних непідозрілих, використовуємо найближчий непідозрілий
                        best_seg = min(non_suspicious, key=lambda x: x['center_distance'])
                else:
                    # Якщо всі підозрілі, перевіряємо чи є сегменти іншого спікера, які ближчі
                    # Це важливо для випадків, коли великий сегмент "поглинає" слова від іншого спікера
                    suspicious_segments = [s for s in overlapping_segments if s['is_suspicious_large']]
                    closest_suspicious = min(suspicious_segments, key=lambda x: x['center_distance'])
                    
                    # Шукаємо сегменти іншого спікера (не підозрілі), які можуть бути ближчі
                    other_speaker_segments = [
                        s for s in overlapping_segments 
                        if s['speaker'] != closest_suspicious['speaker'] and not s['is_suspicious_large']
                    ]
                    
                    if other_speaker_segments:
                        # Якщо є сегменти іншого спікера, перевіряємо чи вони ближчі
                        closest_other = min(other_speaker_segments, key=lambda x: x['center_distance'])
                        
                        # Якщо сегмент іншого спікера ближчий або має кращу комбінацію center_distance + overlap,
                        # використовуємо його (особливо якщо різниця в center_distance не дуже велика)
                        distance_diff = closest_suspicious['center_distance'] - closest_other['center_distance']
                        
                        # Якщо сегмент іншого спікера ближчий на >0.5s або має overlap_ratio >10%,
                        # використовуємо його замість підозрілого великого
                        if (distance_diff > 0.5 or 
                            (closest_other['overlap_ratio'] > 0.10 and distance_diff > 0.2)):
                            best_seg = closest_other
                        else:
                            # Якщо різниця невелика, використовуємо найближчий підозрілий
                            # Але тільки якщо він має overlap_ratio >5% або center_distance <1.0s
                            valid_suspicious = [
                                s for s in suspicious_segments 
                                if s['overlap_ratio'] > 0.05 or s['center_distance'] < 1.0
                            ]
                            if valid_suspicious:
                                best_seg = min(valid_suspicious, key=lambda x: x['center_distance'])
                            else:
                                best_seg = closest_suspicious
                    else:
                        # Якщо немає сегментів іншого спікера, використовуємо найближчий підозрілий
                        # Але тільки якщо він має overlap_ratio >5% або center_distance <1.0s
                        valid_suspicious = [
                            s for s in suspicious_segments 
                            if s['overlap_ratio'] > 0.05 or s['center_distance'] < 1.0
                        ]
                        if valid_suspicious:
                            best_seg = min(valid_suspicious, key=lambda x: x['center_distance'])
                        else:
                            # Останній fallback: найближчий сегмент
                            best_seg = min(overlapping_segments, key=lambda x: x['center_distance'])
            
            speaker_id = best_seg['speaker']
            word_speakers.append({
                'word': word_text,
                'start': word_start,
                'end': word_end,
                'speaker': speaker_id,
                'triz_corrected': False  # Позначка для слів, виправлених ТРІЗ логікою
            })
            continue
        
        # Якщо немає сегментів, що перетинаються, шукаємо найближчий за часом
        best_speaker = None
        best_overlap_ratio = 0
        best_center_distance = float('inf')
        
        for diar_seg in sorted_diar_segments:
            diar_start = diar_seg['start']
            diar_end = diar_seg['end']
            diar_center = (diar_start + diar_end) / 2.0
            
            # Відстань між центрами
            center_distance = abs(word_center - diar_center)
            
            # Віддаємо перевагу найближчому сегменту
            if center_distance < best_center_distance:
                best_center_distance = center_distance
                best_speaker = diar_seg['speaker']
        
        # Якщо немає перекриття, використовуємо найближчий сегмент за часом
        # Але тільки якщо він дуже близький (<1 сек)
        if best_speaker is not None and best_center_distance < 1.0:
            speaker_id = best_speaker
        elif len(word_speakers) > 0:
            # Fallback: використовуємо спікера попереднього слова, якщо воно дуже близьке (<0.3 сек)
            last_word = word_speakers[-1]
            if (word_start - last_word['end']) < 0.3:
                speaker_id = last_word['speaker']
            else:
                # Якщо попереднє слово далеко, використовуємо найближчий сегмент, навіть якщо далеко
                speaker_id = best_speaker if best_speaker is not None else 0
        else:
            speaker_id = best_speaker if best_speaker is not None else 0
        
        word_speakers.append({
            'word': word_text,
            'start': word_start,
            'end': word_end,
            'speaker': speaker_id,
            'triz_corrected': False  # Позначка для звичайних слів
        })
    
    # ТРІЗ: Виправлення на рівні слів для фраз після коротких реплік
    # НЕЗАЛЕЖНО від діаризації та overlapping_segments
    # Якщо попереднє слово належить іншому спікеру (не поточному), і gap < 3.0s,
    # і попереднє слово коротке, і поточне слово є частиною питання/інструкції,
    # то поточне слово має належати іншому спікеру (не попередньому)
    print(f"🔧 ТРІЗ: Applying word-level corrections for phrases after short replies (independent of diarization)...")
    for iteration in range(3):
        changes_made = False
        for i in range(1, len(word_speakers) - 1):
            current_word = word_speakers[i]
            prev_word = word_speakers[i - 1]
            
            current_speaker = current_word['speaker']
            prev_speaker = prev_word['speaker']
            
            gap_to_prev = current_word['start'] - prev_word['end']
            
            # Перевіряємо, чи попереднє слово належить короткому сегменту діаризації
            prev_word_duration = prev_word['end'] - prev_word['start']
            prev_seg_duration = None
            for diar_seg in sorted_diar_segments:
                if (prev_word['start'] >= diar_seg['start'] and 
                    prev_word['end'] <= diar_seg['end']):
                    prev_seg_duration = diar_seg['end'] - diar_seg['start']
                    break
            
            is_prev_short = prev_word_duration < 0.5 or (prev_seg_duration and prev_seg_duration < 1.0)
            
            # Перевіряємо, чи поточне слово є частиною питання/інструкції
            # ТРІЗ: Очищаємо пунктуацію перед перевіркою типу фрази
            current_text_clean = clean_punctuation(current_word['word']).lower()
            is_question_start = any([
                current_text_clean.startswith('hey'),
                current_text_clean.startswith('did'),
                current_text_clean.startswith('can'),
                current_text_clean.startswith('try'),
                current_text_clean.startswith('what'),
                current_text_clean.startswith('how'),
                current_text_clean.startswith('why'),
                current_text_clean.startswith('when'),
                current_text_clean.startswith('where'),
                current_text_clean.startswith('you'),
            ])
            
            # ТРІЗ: Якщо попереднє слово належить іншому спікеру (не поточному), і gap < 3.0s,
            # і попереднє слово коротке, і поточне слово є частиною питання/інструкції,
            # то поточне слово має належати іншому спікеру (не попередньому),
            # НЕЗАЛЕЖНО від діаризації та overlapping_segments
            if (prev_speaker != current_speaker and
                gap_to_prev < 3.0 and
                is_prev_short and
                is_question_start):
                # Визначаємо іншого спікера як того, хто не є попереднім спікером
                all_speakers = set(w['speaker'] for w in word_speakers)
                other_speakers = all_speakers - {prev_speaker}
                
                if other_speakers:
                    # Використовуємо спікера з найбільшою кількістю слів (якщо є кілька)
                    speaker_word_counts = {}
                    for w in word_speakers:
                        speaker = w['speaker']
                        if speaker in other_speakers:
                            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + 1
                    
                    if speaker_word_counts:
                        target_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0]
                    else:
                        # Fallback: використовуємо першого іншого спікера
                        target_speaker = list(other_speakers)[0]
                else:
                    # Якщо немає інших спікерів, використовуємо 1 - prev_speaker
                    target_speaker = 1 - prev_speaker if prev_speaker == 0 else 0
                
                # Виправляємо призначення
                if current_speaker != target_speaker:
                    word_speakers[i]['speaker'] = target_speaker
                    word_speakers[i]['triz_corrected'] = True
                    changes_made = True
                    print(f"🔧 ТРІЗ (word-level, independent): Виправлено слово '{current_word['word']}' "
                          f"({current_word['start']:.2f}-{current_word['end']:.2f}s): "
                          f"Спікер {current_speaker} → {target_speaker} "
                          f"(після короткої репліки спікера {prev_speaker}, питання/інструкція, незалежно від діаризації)")
        
        if not changes_made:
            break
    
    # Діагностика: перевіряємо розподіл спікерів
    speakers_found = set(w['speaker'] for w in word_speakers)
    print(f"📊 Word-level speakers (before filtering): {len(speakers_found)} unique speakers found: {sorted(speakers_found)}")
    
    # ТРІЗ ПІДХІД: Діаризація - джерело правди про спікерів
    # Замість majority voting, використовуємо прямий мапінг на основі діаризації
    # Якщо слово не має достатнього перекриття з діаризацією, шукаємо найближчий сегмент діаризації
    
    filtered_word_speakers = []
    
    for i, word_info in enumerate(word_speakers):
        word_start = word_info['start']
        word_end = word_info['end']
        word_center = (word_start + word_end) / 2.0
        word_speaker = word_info['speaker']
        
        # Перевіряємо перекриття з діаризацією для цього спікера
        best_overlap = 0
        best_overlap_ratio = 0
        for diar_seg in sorted_diar_segments:
            if diar_seg['speaker'] == word_speaker:
                overlap_start = max(word_start, diar_seg['start'])
                overlap_end = min(word_end, diar_seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                word_duration = word_end - word_start
                overlap_ratio = overlap / word_duration if word_duration > 0 else 0
                if overlap_ratio > best_overlap_ratio:
                    best_overlap_ratio = overlap_ratio
                    best_overlap = overlap
        
        word_duration = word_end - word_start
        
        # КРИТИЧНО: Не змінюємо спікера, якщо він був виправлений ТРІЗ логікою для слів на початку файлу
        # Це запобігає перезапису правильного призначення спікера
        if word_info.get('triz_corrected', False):
            # Слово вже виправлено ТРІЗ логікою - не змінюємо спікера
            filtered_word_speakers.append({
                'word': word_info['word'],
                'start': word_start,
                'end': word_end,
                'speaker': word_speaker
            })
            continue
        
        # ТРІЗ: Якщо перекриття з поточним спікером занадто мале, шукаємо найближчий сегмент діаризації
        # замість використання majority voting (який може зміщувати до домінуючого спікера)
        if best_overlap_ratio < 0.3:  # Знижено поріг для більш агресивної перевірки
            # Знаходимо найближчий сегмент діаризації (за центром слова)
            closest_seg = None
            min_distance = float('inf')
            
            # Спочатку шукаємо сегменти, які перетинаються з словом (навіть частково)
            overlapping_segs = []
            for diar_seg in sorted_diar_segments:
                if word_start < diar_seg['end'] and word_end > diar_seg['start']:
                    diar_center = (diar_seg['start'] + diar_seg['end']) / 2.0
                    distance = abs(word_center - diar_center)
                    overlapping_segs.append((diar_seg, distance))
            
            # Якщо є сегменти, що перетинаються, використовуємо найближчий
            if overlapping_segs:
                overlapping_segs.sort(key=lambda x: x[1])
                closest_seg, min_distance = overlapping_segs[0]
            else:
                # Якщо немає перетинаючихся, шукаємо найближчий за часом
                for diar_seg in sorted_diar_segments:
                    diar_center = (diar_seg['start'] + diar_seg['end']) / 2.0
                    distance = abs(word_center - diar_center)
                    
                    if distance < min_distance and distance < 0.5:
                        min_distance = distance
                        closest_seg = diar_seg
            
            # Якщо знайшли близький сегмент, використовуємо його спікера
            if closest_seg and min_distance < 0.5:
                new_speaker = closest_seg['speaker']
                # КРИТИЧНО: Не змінюємо спікера, якщо він вже правильний або якщо це змінить спікера на основного
                # Тільки якщо новий спікер відрізняється від поточного
                if new_speaker != word_speaker:
                    word_speaker = new_speaker
                    if i < 5:  # Логуємо тільки перші 5 для діагностики
                        print(f"🔧 Word '{word_info['word']}' at {word_center:.2f}s: low overlap ({best_overlap_ratio:.2f}), "
                              f"using closest diarization segment (speaker {word_speaker}, distance: {min_distance:.2f}s)")
        
        filtered_word_speakers.append({
            'word': word_info['word'],
            'start': word_start,
            'end': word_end,
            'speaker': word_speaker
        })
    
    # ТРІЗ РІШЕННЯ: Виправлення призначень на основі двостороннього контексту
    # Якщо попереднє і наступне слова належать одному спікеру, і gap < 2s,
    # поточне слово теж має належати тому спікеру
    # Це вирішує проблему з фразами між репліками одного спікера (gap 1-2 секунди)
    # Проходимо кілька разів для виправлення всіх випадків
    for iteration in range(3):  # До 3 ітерацій для виправлення всіх випадків
        changes_made = False
        for i in range(1, len(filtered_word_speakers) - 1):
            current_word = filtered_word_speakers[i]
            prev_word = filtered_word_speakers[i - 1]
            next_word = filtered_word_speakers[i + 1]
            
            current_speaker = current_word['speaker']
            prev_speaker = prev_word['speaker']
            next_speaker = next_word['speaker']
            
            gap_to_prev = current_word['start'] - prev_word['end']
            gap_to_next = next_word['start'] - current_word['end']
            
            # ПРІОРИТЕТ 1: Якщо попереднє і наступне слова належать одному спікеру (не поточному),
            # і gap дуже малі (<2s), поточне слово теж має належати тому спікеру
            # Це вирішує проблему з фразами між репліками одного спікера (наприклад, "No, it should be 200.")
            if (prev_speaker == next_speaker and 
                prev_speaker != current_speaker and
                gap_to_prev < 2.0 and 
                gap_to_next < 2.0):
                # Виправляємо призначення
                filtered_word_speakers[i]['speaker'] = prev_speaker
                filtered_word_speakers[i]['triz_corrected'] = True
                changes_made = True
            # ПРІОРИТЕТ 2: Перевіряємо тільки попереднє слово, якщо gap < 1s
            # Але тільки якщо наступне слово не належить іншому спікеру (або gap великий)
            # Це запобігає неправильному призначенню, коли наступне слово від іншого спікера
            elif (prev_speaker != current_speaker and
                  gap_to_prev < 1.0 and
                  (next_speaker == prev_speaker or gap_to_next > 2.0)):
                # Якщо попереднє слово належить іншому спікеру і gap дуже малий,
                # і наступне слово не належить іншому спікеру (або gap великий),
                # поточне слово теж має належати попередньому спікеру
                filtered_word_speakers[i]['speaker'] = prev_speaker
                filtered_word_speakers[i]['triz_corrected'] = True
                changes_made = True
            # ПРІОРИТЕТ 3: Перевіряємо тільки попереднє слово, якщо gap дуже малий (<0.3s)
            # Це для випадків, коли слова йдуть одразу один за одним
            elif (prev_speaker != current_speaker and
                  gap_to_prev < 0.3):
                # Якщо попереднє слово належить іншому спікеру і gap дуже малий,
                # поточне слово теж має належати тому спікеру
                filtered_word_speakers[i]['speaker'] = prev_speaker
                filtered_word_speakers[i]['triz_corrected'] = True
                changes_made = True
        
        # Якщо не було змін, виходимо
        if not changes_made:
            break
    
    # Діагностика після фільтрації
    speakers_found_after = set(w['speaker'] for w in filtered_word_speakers)
    print(f"📊 Word-level speakers (after filtering): {len(speakers_found_after)} unique speakers found: {sorted(speakers_found_after)}")
    
    # Додаткова діагностика: перевіряємо розподіл слів по спікерах
    speaker_word_counts = {}
    for w in filtered_word_speakers:
        speaker = w['speaker']
        speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + 1
    print(f"📊 Word distribution by speaker: {speaker_word_counts}")
    
    # Групуємо послідовні слова одного спікера в сегменти
    # Зберігаємо word-level інформацію для подальшого розділення сегментів
    combined = []
    if not filtered_word_speakers:
        return combined
    
    current_speaker = filtered_word_speakers[0]['speaker']
    current_start = filtered_word_speakers[0]['start']
    current_words = [filtered_word_speakers[0]['word']]
    current_word_infos = [filtered_word_speakers[0]]  # Зберігаємо повну інформацію про слова
    
    for i in range(1, len(filtered_word_speakers)):
        word_info = filtered_word_speakers[i]
        
        # Якщо спікер змінився або великий проміжок (>1 сек), створюємо новий сегмент
        if (word_info['speaker'] != current_speaker or 
            word_info['start'] - filtered_word_speakers[i-1]['end'] > 1.0):
            
            # Зберігаємо поточний сегмент з word-level інформацією
            combined.append({
                'speaker': current_speaker,
                'start': round(current_start, 2),
                'end': round(filtered_word_speakers[i-1]['end'], 2),
                'text': ' '.join(current_words).strip(),
                'words': current_word_infos.copy()  # Зберігаємо word-level інформацію
            })
            
            # Починаємо новий сегмент
            current_speaker = word_info['speaker']
            current_start = word_info['start']
            current_words = [word_info['word']]
            current_word_infos = [word_info]
        else:
            # Додаємо слово до поточного сегмента
            current_words.append(word_info['word'])
            current_word_infos.append(word_info)
    
    # Додаємо останній сегмент
    if current_words:
        combined.append({
            'speaker': current_speaker,
            'start': round(current_start, 2),
            'end': round(filtered_word_speakers[-1]['end'], 2),
            'text': ' '.join(current_words).strip(),
            'words': current_word_infos.copy()  # Зберігаємо word-level інформацію
        })
    
    # АЛГОРИТМ: Розділення сегментів, які містять слова від різних спікерів
    # Це вирішує проблему, коли в одному сегменті об'єднані фрази від різних спікерів
    # (наприклад, "What speed does it show? Uh, per second.")
    print(f"🔍 [Segment Splitting] Перевірка {len(combined)} сегментів на наявність слів від різних спікерів...")
    sys.stdout.flush()
    
    split_combined = []
    for seg_idx, seg in enumerate(combined):
        if 'words' not in seg or len(seg['words']) == 0:
            # Якщо немає word-level інформації, додаємо сегмент як є
            split_combined.append(seg)
            continue
        
        # Перевіряємо, чи є в сегменті слова від різних спікерів
        word_speakers = set(w['speaker'] for w in seg['words'])
        
        if len(word_speakers) == 1:
            # Всі слова від одного спікера - додаємо сегмент як є
            # Видаляємо 'words' для економії пам'яті
            seg_clean = {k: v for k, v in seg.items() if k != 'words'}
            split_combined.append(seg_clean)
        else:
            # Є слова від різних спікерів - розділяємо сегмент
            print(f"🔧 [Segment Splitting] Сегмент {seg_idx}: '{seg['text'][:50]}...' містить слова від {len(word_speakers)} спікерів: {sorted(word_speakers)}")
            sys.stdout.flush()
            
            # Групуємо слова за спікером
            current_split_speaker = seg['words'][0]['speaker']
            current_split_start = seg['words'][0]['start']
            current_split_words = [seg['words'][0]['word']]
            
            for word_idx in range(1, len(seg['words'])):
                word = seg['words'][word_idx]
                
                # Якщо спікер змінився, створюємо новий підсегмент
                if word['speaker'] != current_split_speaker:
                    # Зберігаємо попередній підсегмент
                    prev_word = seg['words'][word_idx - 1]
                    split_combined.append({
                        'speaker': current_split_speaker,
                        'start': round(current_split_start, 2),
                        'end': round(prev_word['end'], 2),
                        'text': ' '.join(current_split_words).strip()
                    })
                    
                    # Починаємо новий підсегмент
                    current_split_speaker = word['speaker']
                    current_split_start = word['start']
                    current_split_words = [word['word']]
                else:
                    # Додаємо слово до поточного підсегмента
                    current_split_words.append(word['word'])
            
            # Додаємо останній підсегмент
            if current_split_words:
                last_word = seg['words'][-1]
                split_combined.append({
                    'speaker': current_split_speaker,
                    'start': round(current_split_start, 2),
                    'end': round(last_word['end'], 2),
                    'text': ' '.join(current_split_words).strip()
                })
            
            print(f"   ✅ Розділено на {len([s for s in split_combined if s['start'] >= seg['start'] and s['end'] <= seg['end']])} підсегментів")
            sys.stdout.flush()
    
    # Замінюємо combined на split_combined
    combined = split_combined
    print(f"✅ [Segment Splitting] Після розділення: {len(combined)} сегментів")
    sys.stdout.flush()
    
    # АЛГОРИТМ 2: Використання LLM для виявлення та розділення сегментів з питанням + відповіддю
    # Перевіряємо кожен сегмент на наявність питання + відповіді
    print(f"🤖 [LLM Segment Analysis] Перевірка сегментів на питання + відповідь...")
    sys.stdout.flush()
    
    llm_split_combined = []
    for seg_idx, seg in enumerate(combined):
        seg_text = seg.get('text', '').strip()
        
        # Перевіряємо, чи сегмент містить питання (?) та можливу відповідь
        has_question = '?' in seg_text
        has_potential_answer = any(
            phrase in seg_text.lower() for phrase in [
                'uh', 'um', 'well', 'yes', 'no', 'yeah', 'sure', 'okay', 'ok',
                'i did', 'i do', 'i have', 'i will', 'i can', 'i am', "i'm",
                'per second', 'per minute', 'per hour'
            ]
        )
        
        # Якщо є питання і потенційна відповідь, відправляємо на LLM
        if has_question and has_potential_answer and len(seg_text.split()) > 5:
            print(f"🔍 [LLM Segment Analysis] Сегмент {seg_idx}: '{seg_text[:60]}...' - виявлено питання + можлива відповідь")
            sys.stdout.flush()
            
            # Перевіряємо, чи є явні маркери відповіді ("Uh", "per second", тощо)
            explicit_answer_markers = ['uh,', 'um,', 'well,', 'per second', 'per minute', 'per hour']
            has_explicit_answer = any(marker in seg_text.lower() for marker in explicit_answer_markers)
            
            # Якщо є явні маркери відповіді, спробуємо розділити автоматично
            if has_explicit_answer:
                # Знаходимо позицію питання (?) та маркера відповіді
                question_pos = seg_text.find('?')
                if question_pos > 0:
                    # Розділяємо на питання та відповідь
                    question_text = seg_text[:question_pos + 1].strip()
                    answer_text = seg_text[question_pos + 1:].strip()
                    
                    # Визначаємо основного спікера
                    speaker_word_counts = {}
                    for s in combined:
                        speaker = s['speaker']
                        word_count = len(s.get('text', '').split())
                        speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
                    main_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0] if speaker_word_counts else 0
                    other_speaker = 1 if main_speaker == 0 else 0
                    
                    # Розділяємо сегмент
                    total_duration = seg['end'] - seg['start']
                    question_ratio = len(question_text) / len(seg_text) if len(seg_text) > 0 else 0.6
                    
                    question_duration = total_duration * question_ratio
                    answer_duration = total_duration * (1 - question_ratio)
                    
                    print(f"🔧 [Auto Split] Автоматичне розділення сегмента {seg_idx} (явні маркери відповіді)")
                    print(f"   Питання: '{question_text}' → Спікер {main_speaker}")
                    print(f"   Відповідь: '{answer_text}' → Спікер {other_speaker}")
                    sys.stdout.flush()
                    
                    llm_split_combined.append({
                        'speaker': main_speaker,
                        'start': seg['start'],
                        'end': seg['start'] + question_duration,
                        'text': question_text
                    })
                    llm_split_combined.append({
                        'speaker': other_speaker,
                        'start': seg['start'] + question_duration,
                        'end': seg['end'],
                        'text': answer_text
                    })
                    continue
            
            # Якщо немає явних маркерів, відправляємо на LLM
            split_result = call_llm_for_segment_splitting(seg, all_segments_context=combined, mode=llm_mode)
            
            if split_result and split_result.get('should_split') and split_result.get('parts'):
                # Розділяємо сегмент на частини
                print(f"✅ [LLM Segment Analysis] Сегмент {seg_idx} розділено на {len(split_result['parts'])} частин")
                for part_idx, part in enumerate(split_result['parts']):
                    print(f"   Частина {part_idx + 1}: '{part['text'][:50]}...' → Спікер {part['speaker']}")
                sys.stdout.flush()
                
                llm_split_combined.extend(split_result['parts'])
            else:
                # Не потрібно розділяти - додаємо сегмент як є
                llm_split_combined.append(seg)
        else:
            # Не підходить під критерії - додаємо сегмент як є
            llm_split_combined.append(seg)
    
    # Замінюємо combined на llm_split_combined
    combined = llm_split_combined
    print(f"✅ [LLM Segment Analysis] Після LLM аналізу: {len(combined)} сегментів")
    sys.stdout.flush()
    
    # ТРІЗ: Додаткова післяобробка на рівні сегментів
    # Якщо сегмент знаходиться між сегментами одного спікера з малими gap,
    # він має належати тому ж спікеру
    # Проходимо кілька разів для виправлення всіх випадків
    for iteration in range(3):
        changes_made = False
        for i in range(1, len(combined) - 1):
            current_seg = combined[i]
            prev_seg = combined[i - 1]
            next_seg = combined[i + 1]
            
            current_speaker = current_seg['speaker']
            prev_speaker = prev_seg['speaker']
            next_speaker = next_seg['speaker']
            
            gap_to_prev = current_seg['start'] - prev_seg['end']
            gap_to_next = next_seg['start'] - current_seg['end']
            
            # ПРІОРИТЕТ 1: Якщо попередній і наступний сегменти належать одному спікеру (не поточному),
            # і gap дуже малі (<2s), поточний сегмент теж має належати тому спікеру
            # Це вирішує проблему з фразами між репліками одного спікера (наприклад, "No, it should be 200.")
            if (prev_speaker == next_speaker and 
                prev_speaker != current_speaker and
                gap_to_prev < 2.0 and 
                gap_to_next < 2.0):
                # Виправляємо призначення
                combined[i]['speaker'] = prev_speaker
                changes_made = True
            # ПРІОРИТЕТ 2: Перевіряємо тільки попередній сегмент, якщо gap < 1s
            # Але тільки якщо наступний сегмент не належить іншому спікеру (або gap великий)
            elif (prev_speaker != current_speaker and
                  gap_to_prev < 1.0 and
                  (next_speaker == prev_speaker or gap_to_next > 2.0)):
                # Якщо попередній сегмент належить іншому спікеру і gap дуже малий,
                # і наступний сегмент не належить іншому спікеру (або gap великий),
                # поточний сегмент теж має належати попередньому спікеру
                combined[i]['speaker'] = prev_speaker
                changes_made = True
        
        if not changes_made:
            break
    
    # ТРІЗ: Виявлення коротких реплік та виправлення призначень спікерів
    # Якщо попередній сегмент короткий (<1s) і належить іншому спікеру,
    # і поточний сегмент є питанням/інструкцією, він має належати основному спікеру
    # Це вирішує проблему з фразами після коротких реплік інших спікерів
    # (наприклад, "Hey, did you try to reset your modem?" після "dropping.")
    # ТРІЗ: Використовуємо main_speaker_from_diarization (визначений на основі діаризації),
    # а не main_speaker (визначений на основі об'єднаної транскрипції)
    
    if len(combined) > 1:
        print(f"🔍 [Segment Processing] Початок обробки {len(combined)} сегментів, llm_mode={llm_mode}")
        sys.stdout.flush()
        
        # Ітеративна післяобробка для виявлення коротких реплік
        for iteration in range(3):
            changes_made = False
            print(f"🔄 [Iteration {iteration + 1}] Обробка сегментів...")
            sys.stdout.flush()
            
            for i in range(1, len(combined) - 1):
                current_seg = combined[i]
                prev_seg = combined[i - 1]
                next_seg = combined[i + 1] if i + 1 < len(combined) else None
                
                current_speaker = current_seg['speaker']
                prev_speaker = prev_seg['speaker']
                next_speaker = next_seg['speaker'] if next_seg else None
                
                gap_to_prev = current_seg['start'] - prev_seg['end']
                gap_to_next = next_seg['start'] - current_seg['end'] if next_seg else float('inf')
                
                prev_duration = prev_seg['end'] - prev_seg['start']
                current_duration = current_seg['end'] - current_seg['start']
                current_text_raw = current_seg.get('text', '').strip()
                current_text_lower = current_text_raw.lower().strip()
                
                # Визначаємо основного спікера (той, хто має більше слів)
                all_speakers = set(seg['speaker'] for seg in combined)
                speaker_word_counts = {}
                for seg in combined:
                    speaker = seg['speaker']
                    word_count = len(seg.get('text', '').split())
                    speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
                main_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0] if speaker_word_counts else 0
                
                # АЛГОРИТМ 1: Виявлення коротких відповідей неголовного спікера між репліками основного
                # Сценарій: [Основний спікер] -> [Коротка відповідь] -> [Основний спікер]
                if (prev_seg and next_seg and 
                    prev_speaker == main_speaker and 
                    next_speaker == main_speaker):
                    # Перевіряємо, чи поточний сегмент - коротка відповідь
                    is_short_duration = current_duration < 2.0
                    word_count = len(current_text_raw.split())
                    is_short_phrase = word_count <= 3
                    
                    # Список типових коротких відповідей
                    short_replies = [
                        'i did', 'i do', 'i have', 'i will', 'i can', 'i am', "i'm",
                        'yes', 'yeah', 'yep', 'yup', 'sure', 'okay', 'ok', 'alright', 'right',
                        'no', 'nope', 'nah', 'not', "don't", "didn't", "won't", "can't",
                        'thanks', 'thank you', 'please', 'sorry', 'excuse me',
                        'uh huh', 'mm hmm', 'hmm', 'ah', 'oh', 'well', 'um', 'uh'
                    ]
                    
                    is_short_reply = any(
                        current_text_lower.startswith(reply) or 
                        current_text_lower == reply or
                        current_text_lower.startswith(reply + ' ')
                        for reply in short_replies
                    )
                    
                    # Перевіряємо gap між сегментами (невеликий gap = природна пауза в діалозі)
                    gap_ok = gap_to_prev < 3.0 and gap_to_next < 3.0
                    
                    # Діагностика для коротких сегментів між репліками основного спікера
                    if is_short_duration and is_short_phrase:
                        print(f"🔍 [Short Reply Check] Segment {i}: '{current_text_raw}' "
                              f"(duration={current_duration:.2f}s, words={word_count}, "
                              f"speaker={current_speaker}, main={main_speaker}, "
                              f"prev={prev_speaker}, next={next_speaker}, "
                              f"is_short_reply={is_short_reply}, gap_ok={gap_ok})")
                        sys.stdout.flush()
                    
                    if is_short_duration and is_short_phrase and (is_short_reply or word_count <= 2) and gap_ok:
                        # Знаходимо неголовного спікера
                        other_speakers = all_speakers - {main_speaker}
                        if other_speakers:
                            other_speaker = list(other_speakers)[0]
                            
                            # Виправляємо призначення спікера
                            combined[i]['speaker'] = other_speaker
                            changes_made = True
                            print(f"🔧 [Algorithm] ✅ Виявлено коротку відповідь між репліками основного спікера: "
                                  f"'{current_text_raw}' ({current_seg['start']:.2f}-{current_seg['end']:.2f}s, "
                                  f"{current_duration:.2f}s, {word_count} words) "
                                  f"→ Спікер {current_speaker} → {other_speaker} "
                                  f"(між репліками спікера {main_speaker})")
                            sys.stdout.flush()
                            continue  # Переходимо до наступного сегмента
                
                # АЛГОРИТМ 2: Виявлення заперечень типу "No, it should be..." після питання/репліки
                # Якщо попередній сегмент містить питання або репліку, а поточний починається з "No" (або подібних заперечень),
                # це з великою ймовірністю новий спікер, але потрібен LLM для аналізу контексту
                prev_text_lower = prev_seg.get('text', '').strip().lower()
                has_question_in_prev = '?' in prev_seg.get('text', '')
                has_statement_in_prev = any(
                    phrase in prev_text_lower for phrase in [
                        'it shows', 'it should', 'it is', 'it was', 'it can', 'it will',
                        'speed', 'shows', 'should be', 'per second', 'per minute', 'per hour',
                        'uh,', 'um,', 'well,'
                    ]
                )
                
                # Перевіряємо також попередній-попередній сегмент (i-2), якщо gap між ними невеликий
                # Це вирішує випадок: "What speed does it show?" (i-2) -> "Uh, per second." (i-1) -> "No, it should be 200." (i)
                has_question_in_prev_prev = False
                if i >= 2:
                    prev_prev_seg = combined[i - 2]
                    prev_prev_text = prev_prev_seg.get('text', '').strip()
                    has_question_in_prev_prev = '?' in prev_prev_text
                    gap_to_prev_prev = prev_seg['start'] - prev_prev_seg['end']
                    # Якщо між попереднім-попереднім і попереднім gap невеликий (< 2 секунди), враховуємо питання
                    if has_question_in_prev_prev and gap_to_prev_prev < 2.0:
                        has_question_in_prev = True  # Враховуємо питання з попереднього-попереднього сегмента
                
                # Перевіряємо, чи поточний сегмент починається з заперечення
                negation_starters = ['no,', 'no ', 'nope,', 'nope ', 'nah,', 'nah ', 'not,', 'not ']
                starts_with_negation = any(
                    current_text_lower.startswith(neg) for neg in negation_starters
                )
                
                # Якщо попередній сегмент містить питання/репліку, а поточний починається з заперечення,
                # і gap невеликий (< 3 секунди), направляємо на LLM
                if (has_question_in_prev or has_statement_in_prev) and starts_with_negation and gap_to_prev < 3.0:
                    print(f"🔍 [Negation Detection] Сегмент {i}: Виявлено заперечення після питання/репліки")
                    print(f"   Попередній: '{prev_seg.get('text', '')[:50]}...' (speaker={prev_speaker})")
                    if i >= 2:
                        prev_prev_seg = combined[i - 2]
                        print(f"   Попередній-попередній: '{prev_prev_seg.get('text', '')[:50]}...' (speaker={prev_prev_seg['speaker']})")
                    print(f"   Поточний: '{current_text_raw[:50]}...' (speaker={current_speaker})")
                    print(f"   Gap: {gap_to_prev:.2f}s")
                    print(f"   has_question_in_prev: {has_question_in_prev}, has_statement_in_prev: {has_statement_in_prev}, starts_with_negation: {starts_with_negation}")
                    sys.stdout.flush()
                    
                    # Направляємо на LLM для аналізу контексту
                    llm_speaker = call_llm_for_speaker_correction(
                        prev_seg,
                        current_seg,
                        gap_to_prev,
                        all_segments_context=combined,
                        mode=llm_mode
                    )
                    
                    if llm_speaker is not None:
                        # LLM успішно визначила спікера
                        if current_speaker != llm_speaker:
                            combined[i]['speaker'] = llm_speaker
                            changes_made = True
                            print(f"🤖 [LLM Negation] ✅ Виправлено сегмент '{current_text_raw[:50]}...' "
                                  f"({current_seg['start']:.2f}-{current_seg['end']:.2f}s): "
                                  f"Спікер {current_speaker} → {llm_speaker} "
                                  f"(заперечення після питання/репліки, LLM рішення)")
                            sys.stdout.flush()
                        else:
                            print(f"ℹ️ [LLM Negation] LLM підтвердив поточного спікера {current_speaker} для '{current_text_raw[:50]}...'")
                            sys.stdout.flush()
                    else:
                        # LLM недоступна або повернула некоректну відповідь - використовуємо алгоритмічний fallback
                        print(f"⚠️ [LLM Negation] LLM недоступна або повернула некоректну відповідь для '{current_text_raw[:50]}...'")
                        print(f"   Використовуємо алгоритмічний fallback на основі контексту...")
                        sys.stdout.flush()
                        
                        # АЛГОРИТМІЧНИЙ FALLBACK для заперечень
                        # Правило: Якщо попередній-попередній сегмент містить питання від основного спікера,
                        # а попередній сегмент містить відповідь від неосновного спікера,
                        # то заперечення "No, it should be..." зазвичай належить основному спікеру (виправлення)
                        
                        # Визначаємо основного спікера
                        all_speakers = set(seg['speaker'] for seg in combined)
                        speaker_word_counts = {}
                        for seg in combined:
                            speaker = seg['speaker']
                            word_count = len(seg.get('text', '').split())
                            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
                        main_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0] if speaker_word_counts else 0
                        other_speaker = 1 if main_speaker == 0 else 0
                        
                        # Перевіряємо контекст: питання (основний) → відповідь (неосновний) → заперечення
                        algorithmic_speaker = None
                        if i >= 2:
                            prev_prev_seg = combined[i - 2]
                            prev_prev_speaker = prev_prev_seg['speaker']
                            prev_prev_text = prev_prev_seg.get('text', '').strip()
                            has_question_in_prev_prev = '?' in prev_prev_text
                            
                            # Якщо попередній-попередній сегмент - питання від основного спікера,
                            # а попередній сегмент - відповідь від неосновного,
                            # то заперечення належить основному спікеру (він виправляє відповідь)
                            if (has_question_in_prev_prev and 
                                prev_prev_speaker == main_speaker and 
                                prev_speaker == other_speaker):
                                algorithmic_speaker = main_speaker
                                print(f"   📊 [Fallback] Контекст: питання (спікер {main_speaker}) → "
                                      f"відповідь (спікер {other_speaker}) → заперечення → спікер {main_speaker} (виправлення)")
                            # Якщо попередній-попередній сегмент - питання від неосновного спікера,
                            # а попередній сегмент - відповідь від основного,
                            # то заперечення може належати неосновному спікеру (він виправляє)
                            elif (has_question_in_prev_prev and 
                                  prev_prev_speaker == other_speaker and 
                                  prev_speaker == main_speaker):
                                algorithmic_speaker = other_speaker
                                print(f"   📊 [Fallback] Контекст: питання (спікер {other_speaker}) → "
                                      f"відповідь (спікер {main_speaker}) → заперечення → спікер {other_speaker} (виправлення)")
                            # Якщо обидва попередні сегменти від одного спікера, заперечення належить іншому
                            elif prev_prev_speaker == prev_speaker:
                                algorithmic_speaker = other_speaker if prev_speaker == main_speaker else main_speaker
                                print(f"   📊 [Fallback] Контекст: обидва попередні сегменти від спікера {prev_speaker} → "
                                      f"заперечення → спікер {algorithmic_speaker} (альтернативний спікер)")
                        
                        # Якщо алгоритмічне рішення знайдено і воно відрізняється від поточного
                        if algorithmic_speaker is not None and current_speaker != algorithmic_speaker:
                            combined[i]['speaker'] = algorithmic_speaker
                            changes_made = True
                            print(f"🔧 [Algorithmic Fallback] ✅ Виправлено сегмент '{current_text_raw[:50]}...' "
                                  f"({current_seg['start']:.2f}-{current_seg['end']:.2f}s): "
                                  f"Спікер {current_speaker} → {algorithmic_speaker} "
                                  f"(заперечення після питання/репліки, алгоритмічне рішення)")
                            sys.stdout.flush()
                        elif algorithmic_speaker is None:
                            print(f"⚠️ [Algorithmic Fallback] Не вдалося визначити спікера алгоритмічно для '{current_text_raw[:50]}...'")
                            sys.stdout.flush()
                        else:
                            print(f"ℹ️ [Algorithmic Fallback] Алгоритмічне рішення підтвердило поточного спікера {current_speaker}")
                            sys.stdout.flush()
                    
                    continue  # Переходимо до наступного сегмента
                
                # Діагностика для кожного сегмента
                if i <= 3 or prev_duration < 1.0:  # Логуємо перші 3 або короткі попередні сегменти
                    print(f"  📊 [Segment {i}] prev: speaker={prev_speaker}, duration={prev_duration:.2f}s, text='{prev_seg.get('text', '')[:30]}...'")
                    print(f"     current: speaker={current_speaker}, gap={gap_to_prev:.2f}s, text='{current_text_raw[:30]}...'")
                    sys.stdout.flush()
                
                # ТРІЗ: Очищаємо пунктуацію перед перевіркою типу фрази
                # Беремо перше слово з очищеною пунктуацією для перевірки
                first_word = current_text_raw.split()[0] if current_text_raw.split() else ''
                current_text_clean = clean_punctuation(first_word).lower() if first_word else ''
                current_text_lower = current_text_raw.lower()
                
                # Виявляємо, чи поточний сегмент є питанням/інструкцією
                is_question_or_instruction = any([
                    current_text_clean.startswith('hey'),
                    current_text_lower.startswith('hey '),
                    current_text_lower.startswith('hey,'),
                    current_text_lower.startswith('did you'),
                    current_text_lower.startswith('can you'),
                    current_text_lower.startswith('try to'),
                    current_text_lower.startswith('you should'),
                    current_text_lower.startswith('you can'),
                    current_text_lower.startswith('you need'),
                    '?' in current_text_raw,
                    current_text_clean.startswith('what'),
                    current_text_clean.startswith('how'),
                    current_text_clean.startswith('why'),
                    current_text_clean.startswith('when'),
                    current_text_clean.startswith('where'),
                ])
                
                # Діагностика умов
                condition_prev_short = prev_duration < 1.0
                condition_diff_speaker = prev_speaker != current_speaker
                condition_gap_ok = gap_to_prev < 3.0
                
                # ПРІОРИТЕТ 1: Спочатку алгоритм намагається вирішити
                # Якщо попередній сегмент короткий (<1s) і поточний сегмент є питанням/інструкцією, і gap < 3s
                # Перевіряємо навіть якщо спікери однакові (це може бути помилка діаризації)
                should_check = (condition_prev_short and 
                               is_question_or_instruction and
                               condition_gap_ok)
                
                if should_check:
                    print(f"  🔍 [Segment {i}] Перевірка умов: prev_short={condition_prev_short}, "
                          f"diff_speaker={condition_diff_speaker}, gap_ok={condition_gap_ok}, "
                          f"is_question={is_question_or_instruction}")
                    sys.stdout.flush()
                    
                    print(f"🔍 [Segment {i}] Умови виконані: prev_duration={prev_duration:.2f}s, "
                          f"prev_speaker={prev_speaker}, current_speaker={current_speaker}, "
                          f"gap={gap_to_prev:.2f}s, is_question={is_question_or_instruction}, "
                          f"text='{current_text_raw[:50]}...'")
                    sys.stdout.flush()
                    
                    # Спочатку алгоритм намагається визначити правильного спікера
                    # Знаходимо основного спікера (той, хто має більше слів)
                    all_speakers = set(seg['speaker'] for seg in combined)
                    speaker_word_counts = {}
                    for seg in combined:
                        speaker = seg['speaker']
                        word_count = len(seg.get('text', '').split())
                        speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + word_count
                    
                    main_speaker = max(speaker_word_counts.items(), key=lambda x: x[1])[0] if speaker_word_counts else 0
                    other_speakers = all_speakers - {prev_speaker}
                    
                    # Алгоритмічне рішення: питання/інструкція після короткої репліки має належати основному спікеру
                    algorithmic_speaker = main_speaker
                    
                    # Визначаємо впевненість на основі того, чи поточний спікер відрізняється від основного
                    # Якщо поточний спікер = основному, але попередній короткий і поточний - питання,
                    # це може бути помилка діаризації (особливо якщо спікери однакові)
                    if current_speaker == main_speaker and prev_speaker == current_speaker:
                        # Підозрілий випадок: коротка репліка і питання мають однакового спікера
                        # але питання має належати основному спікеру
                        confidence = 0.4  # Низька впевненість - потрібен LLM
                    elif current_speaker != main_speaker:
                        # Поточний спікер не основний - впевненість середня
                        confidence = 0.6
                    else:
                        # Поточний спікер = основному і вони різні від попереднього - висока впевненість
                        confidence = 0.9
                    
                    # Викликаємо LLM якщо:
                    # 1. Впевненість низька (< 0.7)
                    # 2. Або поточний спікер не відповідає алгоритмічному рішенню
                    # 3. Або спікери однакові (можлива помилка діаризації)
                    use_llm = (confidence < 0.7 or 
                              algorithmic_speaker != current_speaker or
                              (prev_speaker == current_speaker and condition_prev_short))
                    
                    if use_llm:
                        print(f"🔍 [LLM Check] Складний випадок виявлено: prev_speaker={prev_speaker}, current_speaker={current_speaker}, "
                              f"algorithmic_speaker={algorithmic_speaker}, confidence={confidence:.2f}, mode={llm_mode}")
                        sys.stdout.flush()
                        
                        # Викликаємо LLM для вирішення складного випадку
                        llm_speaker = call_llm_for_speaker_correction(
                            prev_seg, 
                            current_seg, 
                            gap_to_prev, 
                            all_segments_context=combined,
                            mode=llm_mode
                        )
                        
                        if llm_speaker is not None:
                            # LLM визначив спікера - використовуємо його
                            if current_speaker != llm_speaker:
                                combined[i]['speaker'] = llm_speaker
                                changes_made = True
                                print(f"🤖 LLM (segment-level): Виправлено сегмент '{current_seg.get('text', '')[:50]}...' "
                                      f"({current_seg['start']:.2f}-{current_seg['end']:.2f}s): "
                                      f"Спікер {current_speaker} → {llm_speaker} "
                                      f"(після короткої репліки спікера {prev_speaker}, питання/інструкція)")
                        else:
                            # LLM недоступний - використовуємо алгоритмічне рішення
                            print(f"⚠️ LLM недоступний, використовуємо алгоритмічне рішення: {algorithmic_speaker}")
                            if current_speaker != algorithmic_speaker:
                                combined[i]['speaker'] = algorithmic_speaker
                                changes_made = True
                                print(f"🔧 Algorithmic (segment-level): Виправлено сегмент '{current_seg.get('text', '')[:50]}...' "
                                      f"({current_seg['start']:.2f}-{current_seg['end']:.2f}s): "
                                      f"Спікер {current_speaker} → {algorithmic_speaker} "
                                      f"(після короткої репліки спікера {prev_speaker}, питання/інструкція, confidence={confidence:.2f})")
                    else:
                        # Алгоритм впевнений - використовуємо його рішення без LLM
                        if current_speaker != algorithmic_speaker:
                            combined[i]['speaker'] = algorithmic_speaker
                            changes_made = True
                            print(f"✅ Algorithmic (segment-level, high confidence): Виправлено сегмент '{current_seg.get('text', '')[:50]}...' "
                                  f"({current_seg['start']:.2f}-{current_seg['end']:.2f}s): "
                                  f"Спікер {current_speaker} → {algorithmic_speaker} "
                                  f"(після короткої репліки спікера {prev_speaker}, питання/інструкція, confidence={confidence:.2f})")
            
            if not changes_made:
                break
    
    # Діагностика: перевіряємо фінальний результат
    final_speakers = set(seg['speaker'] for seg in combined)
    print(f"✅ Combined result: {len(combined)} segments, {len(final_speakers)} unique speakers: {sorted(final_speakers)}")
    
    return combined


def process_audio_background(job_id, filepath, num_speakers, language, segment_duration, overlap, processing_mode='fast'):
    """Обробка аудіо в фоновому потоці"""
    try:
        with jobs_lock:
            jobs[job_id]['status'] = 'processing'
            include_transcription = jobs[job_id].get('include_transcription', True)
        
        print(f"🔄 [Job {job_id}] Starting background processing (mode: {processing_mode})...")
        import sys
        sys.stdout.flush()
        
        # Обчислюємо тривалість аудіо
        try:
            audio_duration = librosa.get_duration(path=filepath)
            print(f"⏱️  [Job {job_id}] Audio duration: {audio_duration:.2f} seconds")
        except Exception as e:
            print(f"⚠️  [Job {job_id}] Could not determine audio duration: {e}")
            audio_duration = 0
        
        # Визначаємо мову для Speechmatics
        lang_code = 'en'
        if language:
            lang_map = {
                'english': 'en', 'en': 'en',
                'ukrainian': 'uk', 'uk': 'uk',
                'arabic': 'ar', 'ar': 'ar',
                'russian': 'ru', 'ru': 'ru'
            }
            lang_code = lang_map.get(language.lower(), 'en')
        
        # Обробка залежно від режиму
        if processing_mode == 'smart':
            # Smart mode: Speechmatics (транскрипція + діаризація)
            print(f"🎯 [Job {job_id}] Using Smart mode: Speechmatics")
            sys.stdout.flush()
            
            if include_transcription:
                try:
                    transcription, transcription_segments, words = transcribe_with_speechmatics(filepath, language=lang_code)
                    
                    print(f"📊 [Job {job_id}] Speechmatics result:")
                    print(f"   - transcription type: {type(transcription)}, length: {len(transcription) if transcription else 0}")
                    print(f"   - transcription_segments: {len(transcription_segments) if transcription_segments else 0} segments")
                    print(f"   - words: {len(words) if words else 0} words")
                    sys.stdout.flush()
                    
                    if not transcription or not words:
                        print(f"⚠️  [Job {job_id}] Speechmatics transcription failed or empty")
                        sys.stdout.flush()
                        with jobs_lock:
                            jobs[job_id]['status'] = 'processing'
                            jobs[job_id]['error'] = 'Speechmatics transcription is still processing or failed. Please wait or retry.'
                        return
                    
                    # Speechmatics вже містить діаризацію в words (speaker labels)
                    # Створюємо сегменти з діаризацією зі слів
                    diarization_segments = []
                    current_speaker = None
                    current_start = None
                    current_end = None
                    current_text = []
                    
                    for word in words:
                        word_speaker = word.get('speaker', 0)
                        word_start = word.get('start', 0)
                        word_end = word.get('end', 0)
                        word_text = word.get('word', '')
                        
                        if current_speaker is None:
                            current_speaker = word_speaker
                            current_start = word_start
                            current_text = [word_text]
                        elif word_speaker == current_speaker:
                            current_text.append(word_text)
                        else:
                            # Зберігаємо попередній сегмент
                            if current_start is not None and current_end is not None:
                                diarization_segments.append({
                                    'speaker': current_speaker,
                                    'start': round(current_start, 2),
                                    'end': round(current_end, 2),
                                    'text': ' '.join(current_text)
                                })
                            # Починаємо новий сегмент
                            current_speaker = word_speaker
                            current_start = word_start
                            current_text = [word_text]
                        
                        current_end = word_end
                    
                    # Додаємо останній сегмент
                    if current_speaker is not None and current_start is not None and current_end is not None:
                        diarization_segments.append({
                            'speaker': current_speaker,
                            'start': round(current_start, 2),
                            'end': round(current_end, 2),
                            'text': ' '.join(current_text)
                        })
                    
                    # Формуємо результат
                    result = {
                        'success': True,
                        'duration': round(audio_duration, 2),
                        'diarization': {
                            'segments': diarization_segments,
                            'num_speakers': len(set(seg.get('speaker', 0) for seg in diarization_segments)) if diarization_segments else 0
                        },
                        'transcription': {
                            'full_text': transcription,
                            'segments': transcription_segments or []
                        },
                        'combined': {
                            'segments': diarization_segments,  # Вже містить текст
                            'num_speakers': len(set(seg.get('speaker', 0) for seg in diarization_segments)) if diarization_segments else 0,
                            'num_segments': len(diarization_segments)
                        }
                    }
                    
                except Exception as e:
                    print(f"❌ [Job {job_id}] Error during Speechmatics transcription: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    with jobs_lock:
                        jobs[job_id]['status'] = 'failed'
                        jobs[job_id]['error'] = f'Speechmatics error: {str(e)}. Please check your internet connection and API key, then retry.'
                        jobs[job_id]['code'] = 'SPEECHMATICS_ERROR'
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                    except:
                        pass
                    return
            else:
                # Без транскрипції - все одно потрібна діаризація
                result = {
                    'success': True,
                    'duration': round(audio_duration, 2),
                    'diarization': {
                        'segments': [],
                        'num_speakers': 0
                    }
                }
        else:
            # Fast mode: Whisper + PyAnnote (оригінальна логіка)
            print(f"⚡ [Job {job_id}] Using Fast mode: Whisper + PyAnnote")
            sys.stdout.flush()
            
            # Витягуємо ембеддинги
            print(f"🔄 [Job {job_id}] Extracting speaker embeddings...")
            sys.stdout.flush()
            embeddings, timestamps = extract_speaker_embeddings(
                filepath, 
                segment_duration=segment_duration, 
                overlap=overlap
            )
            
            if embeddings is None:
                with jobs_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = 'Failed to extract speaker embeddings. Audio may be corrupted or unsupported format.'
                    jobs[job_id]['code'] = 'EMBEDDING_EXTRACTION_FAILED'
                os.remove(filepath)
                return
            
            if len(embeddings) == 0:
                with jobs_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = f'Audio too short (duration: {audio_duration:.2f}s). Minimum recommended: 2 seconds.'
                    jobs[job_id]['code'] = 'AUDIO_TOO_SHORT'
                os.remove(filepath)
                return
            
            # Діаризація
            print(f"🔄 [Job {job_id}] Performing diarization...")
            sys.stdout.flush()
            diarization_segments = diarize_audio(embeddings, timestamps, num_speakers)
            
            if not diarization_segments:
                with jobs_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = 'Diarization failed. Could not identify speakers.'
                    jobs[job_id]['code'] = 'DIARIZATION_FAILED'
                os.remove(filepath)
                return
            
            # Формуємо базовий результат
            result = {
                'success': True,
                'duration': round(audio_duration, 2),
                'diarization': {
                    'segments': diarization_segments,
                    'num_speakers': len(set(seg.get('speaker', 0) for seg in diarization_segments)) if diarization_segments else 0
                }
            }
            
            # Транскрипція (якщо потрібна)
            if include_transcription:
                print(f"🔄 [Job {job_id}] Transcribing audio...")
                sys.stdout.flush()
                try:
                    transcription, transcription_segments, words = transcribe_audio(filepath, language=language, transcription_provider='whisper')
                    
                    print(f"📊 [Job {job_id}] Transcription result:")
                    print(f"   - transcription type: {type(transcription)}, length: {len(transcription) if transcription else 0}")
                    print(f"   - transcription_segments: {len(transcription_segments) if transcription_segments else 0} segments")
                    print(f"   - words: {len(words) if words else 0} words")
                    if transcription:
                        print(f"   - transcription preview (first 100 chars): {transcription[:100]}")
                    sys.stdout.flush()
                    
                    # Перевіряємо, чи транскрипція успішна
                    if not transcription or not words:
                        print(f"⚠️  [Job {job_id}] Transcription failed or empty - keeping status 'processing'")
                        print(f"   - transcription is None/empty: {transcription is None or not transcription}")
                        print(f"   - words is None/empty: {words is None or not words}")
                        sys.stdout.flush()
                        # Залишаємо статус 'processing', щоб Shortcut продовжував полінг
                        with jobs_lock:
                            jobs[job_id]['status'] = 'processing'
                            jobs[job_id]['error'] = 'Transcription is still processing or failed. Please wait or retry.'
                        return  # Не завершуємо обробку, залишаємо статус 'processing'
                    
                    # Транскрипція успішна - продовжуємо
                    result['transcription'] = {
                        'full_text': transcription,
                        'segments': transcription_segments or []
                    }
                    
                    # Об'єднання діаризації та транскрипції
                    print(f"🔄 [Job {job_id}] Combining results...")
                    print(f"   - Diarization segments: {len(diarization_segments)}")
                    print(f"   - Words for combination: {len(words)}")
                    sys.stdout.flush()
                    # Отримуємо llm_mode з jobs або використовуємо 'local' за замовчуванням
                    llm_mode = jobs[job_id].get('llm_mode', 'local')
                    # Нормалізуємо режим
                    if llm_mode == 'smart2':
                        llm_mode = 'smart-2'
                    combined_segments = combine_diarization_and_transcription(
                        diarization_segments, 
                        words,
                        llm_mode=llm_mode
                    )
                    
                    print(f"✅ [Job {job_id}] Combined result: {len(combined_segments) if combined_segments else 0} segments")
                    sys.stdout.flush()
                    
                    result['combined'] = {
                        'segments': combined_segments if combined_segments else [],
                        'num_speakers': len(set(seg.get('speaker', 0) for seg in combined_segments)) if combined_segments else 0,
                        'num_segments': len(combined_segments) if combined_segments else 0
                    }
                except Exception as e:
                    print(f"❌ [Job {job_id}] Error during transcription: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    # Залишаємо статус 'processing' при помилці, щоб можна було спробувати знову
                    with jobs_lock:
                        jobs[job_id]['status'] = 'processing'
                        jobs[job_id]['error'] = f'Transcription error: {str(e)}. Please retry.'
                    return  # Не завершуємо обробку
            else:
                result['transcription'] = None
                result['combined'] = None
        
        # Встановлюємо статус 'completed' тільки якщо все успішно завершено
        with jobs_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = result
        
        # Видаляємо файл
        try:
            os.remove(filepath)
        except:
            pass
        
        print(f"✅ [Job {job_id}] Processing complete!")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"❌ [Job {job_id}] Error: {e}")
        import traceback
        import sys
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass


def allowed_file(filename):
    """Перевіряє, чи дозволений формат файлу"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_audio_format_from_base64(base64_data):
    """
    Визначає формат аудіо файлу з base64 даних за сигнатурою (magic bytes).
    Повертає розширення файлу (без крапки) або None, якщо не вдалося визначити.
    """
    try:
        import base64
        # Очищаємо base64 рядок
        base64_clean = str(base64_data).strip()
        
        # Видаляємо data URI префікс (якщо є)
        if ',' in base64_clean:
            base64_clean = base64_clean.split(',', 1)[1]
        
        # Видаляємо всі пробіли, переноси рядків
        base64_clean = base64_clean.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
        
        # Конвертуємо base64url (URL-safe) в стандартний base64
        if '-' in base64_clean or '_' in base64_clean:
            base64_clean = base64_clean.replace('-', '+').replace('_', '/')
        
        # Видаляємо крапки (невалідні в base64)
        if '.' in base64_clean:
            base64_clean = base64_clean.replace('.', '')
        
        # Додаємо padding, якщо потрібно
        missing_padding = len(base64_clean) % 4
        if missing_padding:
            base64_clean += '=' * (4 - missing_padding)
        
        # Декодуємо перші 20 байт для перевірки сигнатури
        decoded = base64.b64decode(base64_clean[:100], validate=False)  # Перші ~75 символів base64 = ~50 байт
        
        # Перевіряємо сигнатури форматів
        if decoded.startswith(b'RIFF') and b'WAVE' in decoded[:12]:
            return 'wav'
        elif decoded.startswith(b'\xff\xfb') or decoded.startswith(b'\xff\xf3') or decoded.startswith(b'\xff\xf2'):
            return 'mp3'
        elif decoded.startswith(b'\xff\xf1') or decoded.startswith(b'\xff\xf9'):
            return 'aac'
        elif decoded.startswith(b'fLaC'):
            return 'flac'
        elif decoded.startswith(b'OggS'):
            return 'ogg'
        elif b'ftyp' in decoded[:20]:
            # M4A/M4V/MP4 мають ftyp на початку
            if b'm4a' in decoded[:30] or b'M4A' in decoded[:30]:
                return 'm4a'
            elif b'mp4' in decoded[:30] or b'MP4' in decoded[:30]:
                return 'm4a'  # MP4 аудіо також обробляємо як m4a
            # Перевіряємо більш детально для M4A
            if b'ftypM4A' in decoded[:30] or b'ftypmp4' in decoded[:30]:
                return 'm4a'
            # Якщо є ftyp, але не визначили точно, спробуємо m4a (найпоширеніший для iOS)
            return 'm4a'
        
        return None
    except Exception as e:
        print(f"⚠️ Error detecting format from base64: {e}")
        return None


@app.errorhandler(400)
def handle_bad_request(e):
    """Обробка помилок 400 Bad Request"""
    print(f"❌ 400 Bad Request error: {e}")
    print(f"   Request method: {request.method}")
    print(f"   Content-Type: {request.content_type}")
    print(f"   Headers: {dict(request.headers)}")
    import sys
    sys.stdout.flush()
    return jsonify({
        'success': False,
        'error': f'Bad Request: {str(e)}. Make sure to send file as multipart/form-data.',
        'code': 'BAD_REQUEST',
        'debug_info': {
            'content_type': request.content_type,
            'method': request.method
        }
    }), 400


@app.route('/api/health', methods=['GET'])
def health():
    """Перевірка стану сервера"""
    return jsonify({
        'status': 'ok',
        'speaker_model_loaded': speaker_model is not None,
        'whisper_model_loaded': whisper_model is not None
    })


@app.route('/enhance-main-speaker', methods=['GET'])
def enhance_main_speaker_page():
    """Сторінка для виділення основного спікера"""
    return send_from_directory('.', 'enhance-main-speaker.html')


@app.route('/api/diarize', methods=['POST', 'OPTIONS'])
def api_diarize():
    """
    Асинхронний API ендпоінт для діаризації та транскрипції.
    Приймає JSON з base64-encoded файлом (швидше, ніж multipart/form-data).
    Повертає job_id одразу, обробка виконується в фоні.
    Використовуйте GET /api/diarize/{job_id}/status для перевірки статусу.
    """
    import sys
    import base64
    print(f"🔵 [API] /api/diarize called - Method: {request.method}, Remote: {request.remote_addr}")
    sys.stdout.flush()
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS preflight request received from", request.remote_addr)
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        sys.stdout.flush()
        return response
    
    print(f"📥 POST /api/diarize request received from {request.remote_addr}")
    print(f"📋 Request headers: {dict(request.headers)}")
    print(f"📋 Request method: {request.method}")
    print(f"📋 Request content type: {request.content_type}")
    print(f"📋 Request content length: {request.content_length} bytes")
    sys.stdout.flush()
    
    # Генеруємо job_id ДО try блоку
    job_id = str(uuid.uuid4())
    filepath = None
    
    try:
        print(f"📋 Step 1: Checking Content-Type: {request.content_type}")
        sys.stdout.flush()
        
        # Перевіряємо, чи це JSON (base64) або multipart/form-data (legacy)
        is_json = request.is_json or (request.content_type and 'application/json' in request.content_type)
        
        if is_json:
            print(f"📦 Step 2: Parsing JSON request (fast, non-blocking)...")
            print(f"📋 Step 2.1: Content-Type: {request.content_type}")
            print(f"📋 Step 2.2: Content-Length: {request.content_length} bytes")
            sys.stdout.flush()
            
            # JSON парситься швидко, не блокує
            try:
                data = request.get_json()
                if not data:
                    raise ValueError("No JSON data received")
                
                print(f"📦 Step 2.3: JSON parsed successfully")
                print(f"📋 Step 2.4: JSON keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                
                # Логуємо інформацію про файл (без самого base64, бо він великий)
                if 'file' in data:
                    file_data = data['file']
                    if isinstance(file_data, str):
                        print(f"📋 Step 2.5: File base64 length: {len(file_data)} characters")
                        print(f"📋 Step 2.6: File base64 preview (first 100 chars): {file_data[:100]}")
                        print(f"📋 Step 2.7: File base64 preview (last 50 chars): {file_data[-50:]}")
                        print(f"📋 Step 2.8: File data type check: {type(file_data)}")
                        print(f"📋 Step 2.9: Contains spaces: {file_data.count(' ')}, Contains newlines: {file_data.count(chr(10))}")
                        # Перевіряємо на data URI префікс
                        if file_data.startswith('data:'):
                            print(f"⚠️ Step 2.10: WARNING - base64 starts with data URI prefix!")
                        if ',' in file_data:
                            print(f"⚠️ Step 2.11: WARNING - base64 contains comma (possible data URI)")
                        # Перевіряємо, чи це не просто текст/назва файлу
                        if len(file_data) < 50:
                            print(f"❌ Step 2.12: ERROR - Base64 string is too short! This might be a filename, not base64 data!")
                            print(f"   Full content: '{file_data}'")
                        elif ' ' in file_data and file_data.count(' ') > len(file_data) * 0.1:
                            print(f"❌ Step 2.13: ERROR - Base64 string contains too many spaces! This might be a filename, not base64 data!")
                    else:
                        print(f"⚠️ Step 2.5: File data is not a string: {type(file_data)}")
                        print(f"   Value: {file_data}")
                
                if 'filename' in data:
                    print(f"📋 Step 2.10: Filename: {data['filename']}")
                
                sys.stdout.flush()
                
                # Отримуємо параметри (всі мають значення за замовчуванням)
                num_speakers = data.get('num_speakers', 2)  # Завжди 2
                language = data.get('language', 'English')  # Завжди English
                if language and language.lower() == 'auto':
                    language = None
                segment_duration = float(data.get('segment_duration', 2.5))  # Завжди 2.5
                overlap = float(data.get('overlap', 0.4))  # Завжди 0.4
                include_transcription = data.get('include_transcription', True)  # Завжди True
                processing_mode = data.get('mode', 'fast')  # 'smart' або 'fast'
                
                # Отримуємо base64 файл
                file_base64 = data.get('file')
                filename = data.get('filename', 'audio.wav')
                
                if not file_base64:
                    return jsonify({
                        'success': False,
                        'error': 'No file data provided. Send file as base64 string in "file" field.',
                        'code': 'NO_FILE'
                    }), 400
                
                # Перевіряємо, чи це дійсно base64 (а не просто текст)
                file_base64_str = str(file_base64).strip()
                # Base64 зазвичай довший за 100 символів для аудіо файлів
                # І містить багато букв/цифр, а не просто текст
                if len(file_base64_str) < 50:
                    print(f"⚠️ [Job {job_id}] WARNING: Base64 string is very short ({len(file_base64_str)} chars). This might be a filename instead of base64 data!")
                    print(f"📋 [Job {job_id}] Received data: {file_base64_str}")
                    return jsonify({
                        'success': False,
                        'error': f'Invalid base64 data: string is too short ({len(file_base64_str)} chars). Expected base64-encoded audio file, but received: "{file_base64_str[:50]}...". Make sure you use "Encode Media" action with Base64 format in Shortcut.',
                        'code': 'INVALID_BASE64_TOO_SHORT'
                    }), 400
                
                # Перевіряємо, чи це не просто текст (якщо містить багато пробілів або не містить типових base64 символів)
                if ' ' in file_base64_str and file_base64_str.count(' ') > len(file_base64_str) * 0.1:
                    print(f"⚠️ [Job {job_id}] WARNING: Base64 string contains many spaces. This might be a filename instead of base64 data!")
                    print(f"📋 [Job {job_id}] Received data: {file_base64_str}")
                    return jsonify({
                        'success': False,
                        'error': f'Invalid base64 data: contains too many spaces. This looks like a filename, not base64-encoded audio. Received: "{file_base64_str[:100]}...". Make sure you use "Encode Media" action with Base64 format in Shortcut.',
                        'code': 'INVALID_BASE64_FILENAME'
                    }), 400
                
                # Автоматично визначаємо формат, якщо filename не має розширення або має недозволене
                original_filename = filename
                if '.' not in filename or not allowed_file(filename):
                    print(f"🔍 Detecting audio format from base64 data (filename: {filename})...")
                    sys.stdout.flush()
                    
                    detected_format = detect_audio_format_from_base64(file_base64)
                    
                    if detected_format:
                        # Видаляємо старе розширення, якщо воно є, і додаємо визначене
                        if '.' in filename:
                            filename = filename.rsplit('.', 1)[0] + '.' + detected_format
                        else:
                            filename = filename + '.' + detected_format
                        print(f"✅ Detected format: {detected_format} → filename: {filename}")
                    else:
                        # Якщо не вдалося визначити, використовуємо .m4a для iOS файлів
                        if 'Screen Recording' in filename or 'screen' in filename.lower():
                            detected_format = 'm4a'
                        else:
                            detected_format = 'm4a'  # Дефолт для iOS
                        
                        if '.' in filename:
                            filename = filename.rsplit('.', 1)[0] + '.' + detected_format
                        else:
                            filename = filename + '.' + detected_format
                        print(f"⚠️ Could not detect format, using default: {detected_format} → filename: {filename}")
                    sys.stdout.flush()
                
                print(f"📝 Step 3: Parameters extracted: num_speakers={num_speakers}, language={language}, segment_duration={segment_duration}, overlap={overlap}, filename={filename} (original: {original_filename})")
                sys.stdout.flush()
                
                # Перевірка формату файлу
                if not allowed_file(filename):
                    return jsonify({
                        'success': False,
                        'error': f'Invalid audio format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}. Original filename: {original_filename}, processed: {filename}',
                        'code': 'INVALID_FORMAT'
                    }), 400
                
                # Створюємо завдання ДО декодування файлу
                with jobs_lock:
                    jobs[job_id] = {
                        'status': 'pending',
                        'result': None,
                        'error': None,
                        'created_at': datetime.now(),
                        'include_transcription': include_transcription,
                        'processing_mode': processing_mode
                    }
                
                print(f"✅ [Job {job_id}] Job created, returning job_id IMMEDIATELY")
                sys.stdout.flush()
                
                # Повертаємо job_id ОДРАЗУ (ДО декодування base64!)
                response = jsonify({
                    'success': True,
                    'job_id': job_id,
                    'status': 'pending',
                    'message': 'Processing started. Use GET /api/diarize/{job_id}/status to check progress.'
                })
                response.headers.add('Access-Control-Allow-Origin', '*')
                
                # Декодуємо base64 та зберігаємо файл в фоні ПОСЛЯ відправки відповіді
                def decode_and_process():
                    try:
                        print(f"💾 [Job {job_id}] Background: Starting base64 decode...")
                        print(f"📋 [Job {job_id}] Background: Original base64 type: {type(file_base64)}")
                        print(f"📋 [Job {job_id}] Background: Original base64 length: {len(str(file_base64))} characters")
                        sys.stdout.flush()
                        
                        # Очищаємо base64 рядок
                        file_base64_clean = str(file_base64).strip()
                        print(f"📋 [Job {job_id}] Background: After strip() length: {len(file_base64_clean)}")
                        
                        # Видаляємо data URI префікс (якщо є), наприклад: "data:audio/m4a;base64,"
                        if ',' in file_base64_clean:
                            before_split = file_base64_clean
                            file_base64_clean = file_base64_clean.split(',', 1)[1]
                            print(f"📋 [Job {job_id}] Background: Found comma, removed data URI prefix. Before: {len(before_split)}, After: {len(file_base64_clean)}")
                            print(f"📋 [Job {job_id}] Background: Removed prefix: {before_split[:before_split.index(',')]}")
                        
                        # Видаляємо всі пробіли, переноси рядків та інші зайві символи
                        before_clean = file_base64_clean
                        file_base64_clean = file_base64_clean.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
                        if len(before_clean) != len(file_base64_clean):
                            print(f"📋 [Job {job_id}] Background: Removed whitespace. Before: {len(before_clean)}, After: {len(file_base64_clean)}")
                        
                        # Конвертуємо base64url (URL-safe) в стандартний base64
                        # base64url використовує: - замість +, _ замість /
                        if '-' in file_base64_clean or '_' in file_base64_clean:
                            before_url = file_base64_clean
                            file_base64_clean = file_base64_clean.replace('-', '+').replace('_', '/')
                            print(f"📋 [Job {job_id}] Background: Converted base64url to standard base64. Found - or _ characters.")
                            print(f"📋 [Job {job_id}] Background: Count of -: {before_url.count('-')}, Count of _: {before_url.count('_')}")
                        
                        # Видаляємо крапки та інші невалідні символи (тільки залишаємо валідні base64 символи)
                        import re
                        before_invalid_removal = file_base64_clean
                        # Залишаємо тільки валідні base64 символи: A-Z, a-z, 0-9, +, /, =
                        file_base64_clean = re.sub(r'[^A-Za-z0-9+/=]', '', file_base64_clean)
                        if len(before_invalid_removal) != len(file_base64_clean):
                            removed_count = len(before_invalid_removal) - len(file_base64_clean)
                            removed_chars = set(before_invalid_removal) - set(file_base64_clean)
                            print(f"⚠️ [Job {job_id}] Background: Removed {removed_count} invalid characters: {removed_chars}")
                        
                        # Перевіряємо, що залишилися тільки валідні base64 символи
                        if not re.match(r'^[A-Za-z0-9+/=]+$', file_base64_clean):
                            # Це не повинно статися після re.sub, але на всяк випадок
                            invalid_chars = set(file_base64_clean) - set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                            print(f"❌ [Job {job_id}] Background: Invalid base64 characters found: {invalid_chars}")
                            print(f"📋 [Job {job_id}] Background: First 200 chars: {file_base64_clean[:200]}")
                            raise ValueError(f"Invalid base64 characters found: {invalid_chars}. Base64 should only contain A-Z, a-z, 0-9, +, /, and =")
                        
                        print(f"📋 [Job {job_id}] Background: Base64 validation passed. Length: {len(file_base64_clean)}")
                        
                        # Додаємо padding, якщо потрібно (base64 має бути кратний 4)
                        # Але спочатку видаляємо існуючі padding символи, щоб правильно розрахувати
                        original_length = len(file_base64_clean)
                        padding_at_end = len(file_base64_clean.rstrip('='))
                        file_base64_clean = file_base64_clean.rstrip('=')
                        
                        missing_padding = len(file_base64_clean) % 4
                        if missing_padding:
                            padding_needed = 4 - missing_padding
                            file_base64_clean += '=' * padding_needed
                            print(f"📋 [Job {job_id}] Background: Added {padding_needed} padding characters (length was {len(file_base64_clean) - padding_needed}, now {len(file_base64_clean)})")
                        else:
                            print(f"📋 [Job {job_id}] Background: No padding needed (length {len(file_base64_clean)} is multiple of 4)")
                        
                        print(f"📋 [Job {job_id}] Background: Final base64 length: {len(file_base64_clean)}")
                        print(f"📋 [Job {job_id}] Background: First 100 chars: {file_base64_clean[:100]}")
                        print(f"📋 [Job {job_id}] Background: Last 50 chars: {file_base64_clean[-50:]}")
                        print(f"📋 [Job {job_id}] Background: Base64 characters breakdown:")
                        print(f"   - Letters (A-Z, a-z): {sum(1 for c in file_base64_clean if c.isalpha())}")
                        print(f"   - Digits (0-9): {sum(1 for c in file_base64_clean if c.isdigit())}")
                        print(f"   - Plus (+): {file_base64_clean.count('+')}")
                        print(f"   - Slash (/): {file_base64_clean.count('/')}")
                        print(f"   - Equals (=): {file_base64_clean.count('=')}")
                        sys.stdout.flush()
                        
                        # Додаткова перевірка: якщо це виглядає як текст, а не base64
                        if len(file_base64_clean) < 100 and file_base64_clean.count('+') + file_base64_clean.count('/') < 2:
                            print(f"❌ [Job {job_id}] Background: ERROR - This doesn't look like base64 data!")
                            print(f"   Length: {len(file_base64_clean)}, Contains: {file_base64_clean}")
                            raise ValueError(f"This doesn't look like base64-encoded audio data. Received text: '{file_base64_clean}'. Make sure you use 'Encode Media' action with Base64 format in Shortcut, not just the filename.")
                        
                        # Декодуємо base64
                        try:
                            print(f"💾 [Job {job_id}] Background: Attempting base64 decode...")
                            file_data = base64.b64decode(file_base64_clean, validate=True)
                            print(f"✅ [Job {job_id}] Background: Base64 decode successful! Decoded size: {len(file_data)} bytes ({len(file_data) / (1024*1024):.2f} MB)")
                        except Exception as decode_error:
                            print(f"❌ [Job {job_id}] Background: Base64 decode error: {decode_error}")
                            print(f"📋 [Job {job_id}] Background: Base64 length: {len(file_base64_clean)}")
                            print(f"📋 [Job {job_id}] Background: First 200 chars: {file_base64_clean[:200]}")
                            print(f"📋 [Job {job_id}] Background: Last 100 chars: {file_base64_clean[-100:]}")
                            raise ValueError(f"Invalid base64 data: {str(decode_error)}")
                        sys.stdout.flush()
                        file_size = len(file_data)
                        print(f"📊 [Job {job_id}] Background: Decoded file size: {file_size / (1024*1024):.2f} MB")
                        sys.stdout.flush()
                        
                        if file_size > MAX_FILE_SIZE:
                            with jobs_lock:
                                jobs[job_id]['status'] = 'failed'
                                jobs[job_id]['error'] = f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB'
                                jobs[job_id]['code'] = 'FILE_SIZE_EXCEEDED'
                            return
                        
                        # Зберігаємо файл
                        safe_filename = secure_filename(filename)
                        filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{safe_filename}")
                        
                        print(f"💾 [Job {job_id}] Background: Saving file to: {filepath}")
                        sys.stdout.flush()
                        
                        with open(filepath, 'wb') as f:
                            f.write(file_data)
                        
                        print(f"✅ [Job {job_id}] Background: File saved, starting processing...")
                        sys.stdout.flush()
                        
                        # Запускаємо обробку
                        process_audio_background(job_id, filepath, num_speakers, language, segment_duration, overlap, processing_mode)
                    except Exception as e:
                        print(f"❌ [Job {job_id}] Background: Error in decode_and_process: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                        with jobs_lock:
                            jobs[job_id]['status'] = 'failed'
                            jobs[job_id]['error'] = str(e)
                            jobs[job_id]['code'] = 'PROCESSING_ERROR'
                
                thread = threading.Thread(target=decode_and_process, daemon=True)
                thread.start()
                
                return response, 202  # 202 Accepted
                
            except Exception as json_error:
                print(f"❌ [Job {job_id}] Error parsing JSON: {json_error}")
                sys.stdout.flush()
                return jsonify({
                    'success': False,
                    'error': f'Invalid JSON format: {str(json_error)}',
                    'code': 'INVALID_JSON'
                }), 400
        
        else:
            # Legacy: multipart/form-data (для сумісності)
            print(f"📦 Step 2: Using legacy multipart/form-data mode...")
            sys.stdout.flush()
            
            print(f"📋 Step 2: Content-Length: {request.content_length}")
            sys.stdout.flush()
            
            print(f"📦 Step 3: Accessing request.files (this may take time for large files)...")
            sys.stdout.flush()
            try:
                has_file = 'file' in request.files
                print(f"📦 Step 3: 'file' in request.files: {has_file}")
            except Exception as parse_error:
                print(f"❌ Step 3: Error parsing request.files: {parse_error}")
                raise
            sys.stdout.flush()
            
            if not has_file:
                return jsonify({
                    'success': False,
                    'error': 'No file uploaded. Use JSON with base64 for faster processing.',
                    'code': 'NO_FILE'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected',
                    'code': 'NO_FILE'
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f'Invalid audio format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                    'code': 'INVALID_FORMAT'
                }), 400
            
            # Отримуємо параметри (всі мають значення за замовчуванням)
            num_speakers = request.form.get('num_speakers', type=int) or 2  # Завжди 2
            language = request.form.get('language', type=str) or 'English'  # Завжди English
            if language and language.lower() == 'auto':
                language = None
            segment_duration = float(request.form.get('segment_duration', 2.5))  # Завжди 2.5
            overlap = float(request.form.get('overlap', 0.4))  # Завжди 0.4
            include_transcription = request.form.get('include_transcription', 'true').lower() == 'true'  # Завжди True
            processing_mode = request.form.get('mode', 'fast')  # 'smart' або 'fast'
            
            # Створюємо завдання
            with jobs_lock:
                jobs[job_id] = {
                    'status': 'pending',
                    'result': None,
                    'error': None,
                    'created_at': datetime.now(),
                    'include_transcription': include_transcription,
                    'processing_mode': processing_mode
                }
            
            # Зберігаємо файл та обробляємо
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
            file.save(filepath)
            
            file_size = os.path.getsize(filepath)
            if file_size > MAX_FILE_SIZE:
                os.remove(filepath)
                return jsonify({
                    'success': False,
                    'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB',
                    'code': 'FILE_SIZE_EXCEEDED'
                }), 413
            
            # Запускаємо обробку в фоні
            thread = threading.Thread(
                target=process_audio_background,
                args=(job_id, filepath, num_speakers, language, segment_duration, overlap, processing_mode),
                daemon=True
            )
            thread.start()
            
            response = jsonify({
                'success': True,
                'job_id': job_id,
                'status': 'pending',
                'message': 'Processing started. Use GET /api/diarize/{job_id}/status to check progress.'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 202
    
    except Exception as e:
        print(f"❌ [Job {job_id}] Error creating job: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Видаляємо тимчасовий файл у разі помилки
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        # Видаляємо job зі словника
        with jobs_lock:
            if job_id in jobs:
                del jobs[job_id]
        
        # Детальна інформація про помилку
        error_msg = str(e)
        if "Bad Request" in error_msg or "400" in error_msg:
            error_msg = "Invalid request format. Ensure file is sent as multipart/form-data with 'file' field."
        
        response = jsonify({
            'success': False,
            'error': error_msg,
            'code': 'PROCESSING_ERROR',
            'debug_info': {
                'request_method': request.method,
                'content_type': request.content_type,
                'files_keys': list(request.files.keys()) if hasattr(request, 'files') else [],
                'form_keys': list(request.form.keys()) if hasattr(request, 'form') else []
            }
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500


@app.route('/api/diarize/<job_id>/status', methods=['GET', 'OPTIONS'])
def get_diarize_status(job_id):
    """Перевірка статусу завдання діаризації"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found',
                'code': 'JOB_NOT_FOUND'
            }), 404
        
        job = jobs[job_id]
        
        if job['status'] == 'completed':
            result = job['result']
            # Повертаємо тільки метадані combined (без segments, бо segments парсяться засобами Shortcut)
            combined = result.get('combined', {})
            combined_metadata = {
                'num_speakers': combined.get('num_speakers', 0),
                'num_segments': combined.get('num_segments', 0)
            }
            return jsonify({
                'success': True,
                'status': 'completed',
                'combined': combined_metadata
            }), 200
        elif job['status'] == 'failed':
            return jsonify({
                'success': False,
                'status': 'failed',
                'error': job.get('error', 'Unknown error'),
                'code': job.get('code', 'PROCESSING_ERROR')
            }), 200
        else:
            return jsonify({
                'success': True,
                'status': job['status'],
                'message': 'Processing in progress...'
            }), 200


def remove_filler_words(text):
    """
    Видаляє filler words (Uh., Um.) з тексту як окремі слова, не частини інших слів.
    
    Args:
        text: текст для очищення
    
    Returns:
        очищений текст без filler words
    """
    import re
    # Видаляємо "Uh." та "Um." як окремі слова (з word boundaries)
    # Також обробляємо варіанти з пробілами та пунктуацією
    # \b - word boundary, щоб не видаляти частини інших слів
    text = re.sub(r'\bUh\.\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUm\.\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUh\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUm\s+', '', text, flags=re.IGNORECASE)
    # Видаляємо подвійні пробіли, які могли залишитися
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def format_dialogue_from_segments(segments):
    """
    Форматує сегменти діалогу у читабельний текст з таймстемпами та спікерами.
    Формат кожної репліки в одному рядку:
        MM:SS Speaker X: [text]
    
    Між репліками - перенос рядка (\n) для зручного розбиття в Shortcut.
    
    Args:
        segments: список сегментів [{'speaker': int, 'start': float, 'end': float, 'text': str}]
    
    Returns:
        formatted_text: відформатований діалог, де кожна репліка в одному рядку
    """
    if not segments:
        return "Error: No dialogue segments found"
    
    formatted_replicas = []
    
    for seg in segments:
        # Конвертуємо час з секунд у MM:SS
        start_time = seg.get('start', 0)
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Отримуємо спікера та текст
        speaker = seg.get('speaker', 0)
        text = seg.get('text', '').strip()
        
        if not text:
            continue
        
        # Видаляємо filler words (Uh., Um.) перед форматуванням
        text = remove_filler_words(text)
        
        if not text:  # Якщо після видалення filler words текст став порожнім
            continue
        
        # Форматуємо одну репліку в одному рядку: MM:SS Speaker X: [text]
        replica = f"{time_str} Speaker {speaker}: {text}"
        formatted_replicas.append(replica)
    
    # Об'єднуємо всі репліки переносом рядка (для зручного розбиття в Shortcut)
    return "\n".join(formatted_replicas)


def format_single_speaker_files_markdown(all_speakers_segments, original_diarization_segments=None):
    """
    Форматує результати обробки одноголосих файлів у Markdown формат.
    Створює окремі ключі для кожного файлу та спікера.
    
    Args:
        all_speakers_segments: dict {speaker_id: [segments]} - сегменти для кожного спікера з одноголосих файлів
        original_diarization_segments: list - оригінальні segments з діаризації (містить репліки обох спікерів)
    
    Returns:
        dict: {
            'File1Speaker0': markdown_text для спікера 0 з файлу 1,
            'File1Speaker1': markdown_text для спікера 1 з файлу 1,
            'File2Speaker0': markdown_text для спікера 0 з файлу 2,
            'File2Speaker1': markdown_text для спікера 1 з файлу 2
        }
    """
    result = {}
    
    # ВИКОРИСТОВУЄМО all_speakers_segments (відфільтровані репліки з одноголосих файлів)
    # замість original_diarization_segments, щоб показати тільки репліки, які залишилися після фільтрації
    
    # Нормалізуємо нумерацію спікерів до 0, 1
    unique_speakers = sorted(set(all_speakers_segments.keys()))
    speaker_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_speakers)}
    
    # Створюємо мапінг: який спікер відповідає якому файлу
    # File1 = перший спікер (за порядком), File2 = другий спікер
    file_to_speaker = {}
    for idx, speaker_id in enumerate(unique_speakers, start=1):
        file_to_speaker[idx] = speaker_id  # File1 -> speaker_id, File2 -> speaker_id
    
    # Логуємо для діагностики
    print(f"🔍 [format_markdown] Input data:")
    print(f"   - all_speakers_segments keys: {list(all_speakers_segments.keys())}")
    for speaker_id, segments in all_speakers_segments.items():
        print(f"   - Speaker {speaker_id}: {len(segments)} segments")
        if segments:
            print(f"     First segment: speaker={segments[0].get('speaker')}, text={segments[0].get('text', '')[:50]}")
    print(f"   - speaker_mapping: {speaker_mapping}")
    print(f"   - file_to_speaker: {file_to_speaker}")
    sys.stdout.flush()
    
    # Завжди створюємо 4 ключі: File1Speaker0, File1Speaker1, File2Speaker0, File2Speaker1
    for file_idx in range(1, 3):  # File1 і File2
        print(f"🔍 [format_markdown] File{file_idx}: processing...")
        sys.stdout.flush()
        
        # Для кожного нормалізованого спікера (0 і 1) створюємо ключ
        for normalized_speaker_id in range(2):  # Завжди Speaker 0 і 1
            key = f'File{file_idx}Speaker{normalized_speaker_id}'
            markdown_lines = []
            
            # Додаємо заголовок
            markdown_lines.append(f"# Репліки спікера {normalized_speaker_id}")
            markdown_lines.append("")  # Порожній рядок після заголовка
            
            # ВИПРАВЛЕННЯ: Перевіряємо, чи є спікер з normalized_speaker_id в all_speakers_segments
            # Знаходимо оригінальний ID спікера, який відповідає normalized_speaker_id
            original_speaker_id = None
            for orig_id, norm_id in speaker_mapping.items():
                if norm_id == normalized_speaker_id:
                    original_speaker_id = orig_id
                    break
            
            # Якщо знайшли оригінальний ID спікера, показуємо його репліки
            if original_speaker_id is not None and original_speaker_id in all_speakers_segments:
                file_segments = all_speakers_segments[original_speaker_id]
                if file_segments:
                    for seg in file_segments:
                        # Форматуємо час
                        start_time = seg.get('start', 0)
                        minutes = int(start_time // 60)
                        seconds = int(start_time % 60)
                        time_str = f"{minutes:02d}:{seconds:02d}"
                        
                        # Додаємо репліку
                        text = seg.get('text', '').strip()
                        if text:
                            # Видаляємо filler words (Uh., Um.) перед додаванням
                            text = remove_filler_words(text)
                            if text:  # Додаємо тільки якщо після очищення текст не порожній
                                markdown_lines.append(f"{time_str} Speaker {normalized_speaker_id}: {text}")
                else:
                    markdown_lines.append("(немає реплік)")
            else:
                # Спікер не знайдено - показуємо "(немає реплік)"
                markdown_lines.append("(немає реплік)")
            
            result[key] = "\n".join(markdown_lines)
            print(f"🔍 [format_markdown] {key}: {len(result[key])} chars, original_speaker_id={original_speaker_id}")
            sys.stdout.flush()
    
    return result


@app.route('/api/diarize/<job_id>/formatted', methods=['GET', 'OPTIONS'])
def get_diarize_formatted(job_id):
    """Отримує відформатований діалог у читабельному форматі"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found',
                'code': 'JOB_NOT_FOUND'
            }), 404
        
        job = jobs[job_id]
        
        if job['status'] == 'completed':
            result = job.get('result', {})
            combined = result.get('combined', {})
            segments = combined.get('segments', [])
            
            if not segments:
                return jsonify({
                    'success': False,
                    'error': 'No dialogue segments found in result',
                    'code': 'NO_SEGMENTS'
                }), 200
            
            # Форматуємо діалог
            formatted_dialogue = format_dialogue_from_segments(segments)
            
            response = jsonify({
                'success': True,
                'status': 'completed',
                'formatted_dialogue': formatted_dialogue
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        elif job['status'] == 'failed':
            response = jsonify({
                'success': False,
                'status': 'failed',
                'error': job.get('error', 'Unknown error'),
                'code': job.get('code', 'PROCESSING_ERROR')
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        else:
            response = jsonify({
                'success': True,
                'status': job['status'],
                'message': 'Processing in progress...'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200


@app.route('/process', methods=['POST', 'OPTIONS'])
def process_audio():
    """
    Основний ендпоінт для iOS Shortcuts - асинхронна обробка.
    Повертає job_id одразу, обробка виконується в фоні.
    """
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS preflight request received from", request.remote_addr)
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    print(f"📥 POST /process request received from {request.remote_addr}")
    print(f"📋 Headers: {dict(request.headers)}")
    print(f"📦 Files in request: {list(request.files.keys())}")
    print(f"📝 Form data keys: {list(request.form.keys())}")
    
    # Примусово скидаємо буфер виводу
    import sys
    sys.stdout.flush()
    
    filepath = None
    job_id = str(uuid.uuid4())
    
    try:
        print(f"🔵 [Job {job_id}] Starting request processing...")
        sys.stdout.flush()
        
        # Валідація файлу
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded',
                'code': 'NO_FILE'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'code': 'NO_FILE'
            }), 400
        
        # Перевірка формату файлу
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid audio format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                'code': 'INVALID_FORMAT'
            }), 400
        
        # Зберігаємо файл тимчасово з унікальним ім'ям
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(filepath)
        
        # Перевірка розміру файлу
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB',
                'code': 'FILE_SIZE_EXCEEDED'
            }), 413
        
        # Отримуємо параметри
        num_speakers = request.form.get('num_speakers', type=int)
        language = request.form.get('language', type=str) or None
        if language and language.lower() == 'auto':
            language = None
        
        # Збільшуємо мінімальну довжину сегмента для стабільності
        segment_duration = float(request.form.get('segment_duration', 1.5))  # Як в demo2 для швидкості
        overlap = float(request.form.get('overlap', 0.5))
        
        print(f"📁 [Job {job_id}] File saved: {filename} ({file_size / (1024*1024):.2f} MB)")
        
        # Створюємо завдання
        with jobs_lock:
            jobs[job_id] = {
                'status': 'pending',
                'result': None,
                'error': None,
                'created_at': datetime.now()
            }
        
        # Запускаємо обробку в фоні
        thread = threading.Thread(
            target=process_audio_background,
            args=(job_id, filepath, num_speakers, language, segment_duration, overlap),
            daemon=True
        )
        thread.start()
        
        print(f"✅ [Job {job_id}] Job created, processing started in background")
        sys.stdout.flush()
        
        # Повертаємо job_id одразу
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'pending',
            'message': 'Processing started. Use GET /process/{job_id}/status to check progress.'
        }), 202  # 202 Accepted
    
    except Exception as e:
        print(f"❌ [Job {job_id}] Error creating job: {e}")
        import traceback
        import sys
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Видаляємо тимчасовий файл у разі помилки
        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        # Видаляємо job зі словника
        with jobs_lock:
            if job_id in jobs:
                del jobs[job_id]
        
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/process/<job_id>/status', methods=['GET', 'OPTIONS'])
def get_job_status(job_id):
    """Перевірка статусу завдання"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    with jobs_lock:
        if job_id not in jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found',
                'code': 'JOB_NOT_FOUND'
            }), 404
        
        job = jobs[job_id]
        response_data = {
            'success': True,
            'job_id': job_id,
            'status': job['status']
        }
        
        if job['status'] == 'completed':
            response_data['result'] = job['result']
        elif job['status'] == 'failed':
            response_data['error'] = job['error']
            response_data['code'] = 'PROCESSING_ERROR'
        
        return jsonify(response_data)


@app.route('/process/<job_id>/result', methods=['GET', 'OPTIONS'])
def get_job_result(job_id):
    """Отримання результату завдання (alias для status)"""
    return get_job_status(job_id)


def clean_base64_string(base64_data):
    """
    Очищає base64 рядок від data URI префіксів, пробілів, конвертує base64url в стандартний base64,
    видаляє невалідні символи та додає правильне padding.
    """
    import re
    import base64
    
    # Конвертуємо в рядок та очищаємо
    base64_clean = str(base64_data).strip()
    
    # Видаляємо data URI префікс (якщо є), наприклад: "data:audio/m4a;base64,"
    if ',' in base64_clean:
        base64_clean = base64_clean.split(',', 1)[1]
    
    # Видаляємо всі пробіли, переноси рядків та інші зайві символи
    base64_clean = base64_clean.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
    
    # Конвертуємо base64url (URL-safe) в стандартний base64
    if '-' in base64_clean or '_' in base64_clean:
        base64_clean = base64_clean.replace('-', '+').replace('_', '/')
    
    # Видаляємо крапки та інші невалідні символи (тільки залишаємо валідні base64 символи)
    base64_clean = re.sub(r'[^A-Za-z0-9+/=]', '', base64_clean)
    
    # Додаємо padding, якщо потрібно (base64 має бути кратний 4)
    base64_clean = base64_clean.rstrip('=')
    missing_padding = len(base64_clean) % 4
    if missing_padding:
        base64_clean += '=' * (4 - missing_padding)
    
    return base64_clean


def determine_main_speaker_from_segments(combined_segments, duration=None):
    """
    Визначає основного спікера на основі сегментів транскрипції.
    Використовує ту саму логіку, що і в enhance_main_speaker_audio:
    - Критерій 1: Кількість слів (найточніший показник)
    - Критерій 2: Тривалість (якщо різниця в словах <10%)
    
    Args:
        combined_segments: список сегментів з полями 'speaker', 'start', 'end', 'text'
        duration: загальна тривалість аудіо (опціонально, для логування)
    
    Returns:
        main_speaker: номер основного спікера
        speaker_stats: словник зі статистикою для кожного спікера
    """
    import sys
    
    # Підраховуємо тривалість для кожного спікера
    speaker_durations = {}
    speaker_word_counts = {}
    speaker_first_segment = {}  # Час першого сегмента для кожного спікера
    
    for seg in combined_segments:
        speaker = seg['speaker']
        duration_seg = seg['end'] - seg['start']
        word_count = len(seg.get('text', '').split())
        
        if speaker not in speaker_durations:
            speaker_durations[speaker] = 0
            speaker_word_counts[speaker] = 0
            speaker_first_segment[speaker] = seg['start']
        
        speaker_durations[speaker] += duration_seg
        speaker_word_counts[speaker] += word_count
        
        # Оновлюємо час першого сегмента, якщо знайшли раніший
        if seg['start'] < speaker_first_segment[speaker]:
            speaker_first_segment[speaker] = seg['start']
    
    # КРИТИЧНО: Визначаємо основного спікера
    # Критерій 1: Кількість слів (найточніший показник)
    # Критерій 2: Тривалість (якщо різниця в словах <10%)
    main_speaker_by_words = max(speaker_word_counts.items(), key=lambda x: x[1])[0]
    main_word_count = speaker_word_counts[main_speaker_by_words]
    
    main_speaker_by_duration = max(speaker_durations.items(), key=lambda x: x[1])[0]
    main_duration = speaker_durations[main_speaker_by_duration]
    
    # Перевіряємо, чи різниця в словах невелика (<10%)
    if len(speaker_word_counts) > 1:
        sorted_word_counts = sorted(speaker_word_counts.items(), key=lambda x: x[1], reverse=True)
        first_word_count = sorted_word_counts[0][1]
        second_word_count = sorted_word_counts[1][1] if len(sorted_word_counts) > 1 else 0
        total_words = sum(speaker_word_counts.values())
        word_diff_ratio = (first_word_count - second_word_count) / total_words if total_words > 0 else 1.0
        
        # Якщо різниця в словах <10%, використовуємо тривалість
        if word_diff_ratio < 0.10:
            print(f"⚠️  Word count difference is small ({word_diff_ratio*100:.1f}%), using duration-based selection")
            main_speaker = main_speaker_by_duration
        else:
            # Використовуємо кількість слів (найточніший показник)
            main_speaker = main_speaker_by_words
    else:
        main_speaker = main_speaker_by_words
    
    # Формуємо статистику
    speaker_stats = {}
    for speaker in sorted(speaker_durations.keys()):
        dur = speaker_durations[speaker]
        words = speaker_word_counts[speaker]
        speaker_stats[speaker] = {
            'duration': dur,
            'word_count': words,
            'first_segment_time': speaker_first_segment[speaker]
        }
    if duration:
        print(f"📊 Speaker statistics from combined transcription:")
        for speaker in sorted(speaker_durations.keys()):
            dur = speaker_durations[speaker]
            words = speaker_word_counts[speaker]
            print(f"   Speaker {speaker}: {dur:.2f}s ({dur/duration*100:.1f}%), {words} words{' 👑' if speaker == main_speaker else ''}")
        print(f"✅ Main speaker determined: {main_speaker} ({main_word_count} words, {main_duration:.2f}s, {main_duration/duration*100:.1f}%)")
        sys.stdout.flush()
    
    return main_speaker, speaker_stats


def format_speaker_dialogue(segments, main_speaker):
    """
    Форматує сегменти основного спікера як діалог з одним спікером.
    Формат: Таймстемп, спікер номер, репліка
    
    Args:
        segments: список сегментів з полями 'speaker', 'start', 'end', 'text'
        main_speaker: номер основного спікера
    
    Returns:
        formatted_lines: список відформатованих рядків
    """
    formatted_lines = []
    
    for seg in segments:
        if seg['speaker'] == main_speaker:
            # Форматуємо таймстемп як MM:SS - MM:SS
            start_min = int(seg['start'] // 60)
            start_sec = int(seg['start'] % 60)
            end_min = int(seg['end'] // 60)
            end_sec = int(seg['end'] % 60)
            
            timestamp = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
            speaker_num = seg['speaker']
            text = seg.get('text', '').strip()
            
            formatted_lines.append(f"{timestamp}, спікер {speaker_num}, {text}")
    
    return formatted_lines


def separate_speakers_with_speechbrain(audio_path, output_dir):
    """
    Розділяє спікерів за допомогою SpeechBrain SepformerSeparation.
    Використовує той самий підхід, що і в speechbrain_separation.py для якісної нарізки.
    
    Args:
        audio_path: шлях до оригінального аудіо файлу
        output_dir: директорія для збереження розділених файлів
    
    Returns:
        dict: {
            'success': bool,
            'speaker_files': {speaker_id: {'path': str, 'speaker_label': str}},
            'error': str (якщо помилка)
        }
    """
    import sys
    
    try:
        # Імпортуємо необхідні бібліотеки
        try:
            import pyannote_patch  # noqa: F401
            from speechbrain.inference.separation import SepformerSeparation as Separator
            import torch
            import torchaudio
        except ImportError as e:
            print(f"⚠️ SpeechBrain separation not available: {e}, falling back to simple extraction")
            sys.stdout.flush()
            return {'success': False, 'error': f'SpeechBrain separation not available: {e}'}
        
        # Визначаємо device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"🔀 [SpeechBrain] Using device: {device}")
        sys.stdout.flush()
        
        # Завантажуємо модель
        cache_dir = os.path.expanduser(
            os.getenv("SPEECHBRAIN_CACHE_DIR", "~/.cache/speechbrain/sepformer-wsj02mix")
        )
        
        print(f"📦 [SpeechBrain] Loading sepformer-wsj02mix model...")
        sys.stdout.flush()
        
        try:
            model = Separator.from_hparams(
                source="speechbrain/sepformer-wsj02mix",
                savedir=cache_dir,
                run_opts={"device": device},
            )
            print(f"✅ [SpeechBrain] Model loaded successfully")
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ [SpeechBrain] Failed to load model: {e}, falling back to simple extraction")
            sys.stdout.flush()
            return {'success': False, 'error': f'Failed to load model: {e}'}
        
        # Завантажуємо аудіо через librosa (підтримує більше форматів, включаючи m4a)
        try:
            # Використовуємо librosa для завантаження (підтримує m4a, mp3, тощо)
            audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            
            # Конвертуємо в torch tensor
            if len(audio_data.shape) == 1:
                # Mono audio - додаємо вимір каналу
                waveform = torch.from_numpy(audio_data).unsqueeze(0).float()
            else:
                # Multi-channel audio - shape [channels, samples]
                waveform = torch.from_numpy(audio_data).float()
            
            print(f"✅ [SpeechBrain] Loaded via librosa: shape={waveform.shape}, sr={sample_rate}")
            sys.stdout.flush()
        except Exception as load_error:
            print(f"❌ [SpeechBrain] Audio loading failed with librosa: {load_error}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return {'success': False, 'error': f'Audio loading failed: {load_error}'}
        
        # Конвертуємо в mono якщо потрібно
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample до 8kHz (SpeechBrain вимагає 8kHz)
        if sample_rate != 8000:
            print(f"🔄 [SpeechBrain] Resampling from {sample_rate}Hz to 8000Hz")
            sys.stdout.flush()
            resampler = torchaudio.transforms.Resample(sample_rate, 8000)
            waveform = resampler(waveform)
            sample_rate = 8000
        
        total_samples = waveform.shape[1]
        
        # Налаштування chunking (як в speechbrain_separation.py)
        max_chunk_seconds = float(os.getenv("SPEECHBRAIN_CHUNK_SECONDS", "30"))
        max_chunk_samples = int(max_chunk_seconds * sample_rate)
        max_chunk_samples = max(max_chunk_samples, sample_rate * 5)  # мінімум 5 секунд
        
        print(f"🔍 [SpeechBrain] Waveform shape: {waveform.shape}, chunk size: {max_chunk_samples} samples")
        sys.stdout.flush()
        
        # Функція для обробки chunk
        def separate_chunk(chunk_tensor: torch.Tensor):
            chunk_tensor = chunk_tensor.to(device)
            with torch.no_grad():
                result = model.separate_batch(chunk_tensor)
            return result.cpu()
        
        # Запускаємо separation з chunking для довгих файлів
        print(f"🔄 [SpeechBrain] Running speaker separation...")
        sys.stdout.flush()
        
        if total_samples > max_chunk_samples:
            print(f"📦 [SpeechBrain] Processing in chunks (total: {total_samples} samples)")
            sys.stdout.flush()
            chunk_outputs = []
            for start in range(0, total_samples, max_chunk_samples):
                end = min(start + max_chunk_samples, total_samples)
                print(f"   🔄 [SpeechBrain] Separating chunk {start}:{end} ({start/sample_rate:.1f}s - {end/sample_rate:.1f}s)")
                sys.stdout.flush()
                chunk = waveform[:, start:end]
                chunk_outputs.append(separate_chunk(chunk))
            est_sources = torch.cat(chunk_outputs, dim=1)
        else:
            waveform = waveform.to(device)
            est_sources = separate_chunk(waveform)
        
        # Обробляємо результат (як в speechbrain_separation.py)
        if est_sources.dim() == 3:
            est_sources = est_sources[0]  # [time, num_speakers]
        
        if est_sources.dim() == 2:
            if est_sources.shape[0] == model.hparams.num_spks:
                # shape [num_speakers, time]
                sources_tensor = est_sources
            elif est_sources.shape[1] == model.hparams.num_spks:
                sources_tensor = est_sources.transpose(0, 1)
            else:
                raise ValueError(f"Unexpected est_sources shape: {est_sources.shape}")
        else:
            raise ValueError(f"Unsupported est_sources dimension: {est_sources.dim()}")
        
        sources_tensor = sources_tensor.cpu()
        
        num_speakers = sources_tensor.shape[0]
        print(f"✅ [SpeechBrain] Found {num_speakers} speakers")
        sys.stdout.flush()
        
        # Застосовуємо сильне приглушення слабких сигналів
        print(f"🔇 [SpeechBrain] Applying noise gate to suppress weak signals...")
        sys.stdout.flush()
        
        def apply_noise_gate(audio_tensor, threshold=0.05, ratio=10.0, attack=0.01, release=0.1):
            """
            Застосовує noise gate для приглушення слабких сигналів.
            
            Args:
                audio_tensor: torch.Tensor [samples] або [channels, samples]
                threshold: Поріг енергії (0.0-1.0), нижче якого сигнал приглушується
                ratio: Коефіцієнт приглушення (1.0 = без змін, 10.0 = сильне приглушення)
                attack: Час атаки (в секундах)
                release: Час відпускання (в секундах)
            """
            # Конвертуємо в numpy якщо потрібно
            if isinstance(audio_tensor, torch.Tensor):
                audio_np = audio_tensor.numpy()
            else:
                audio_np = audio_tensor
            
            # Обчислюємо енергію сигналу (RMS)
            if len(audio_np.shape) == 1:
                # Mono
                energy = np.abs(audio_np)
            else:
                # Multi-channel - беремо середнє
                energy = np.abs(audio_np).mean(axis=0)
            
            # Обчислюємо RMS в склянному вікні (для плавності)
            window_size = int(sample_rate * 0.05)  # 50ms вікно
            if window_size < 1:
                window_size = 1
            
            # Обчислюємо RMS
            rms = np.sqrt(np.convolve(energy ** 2, np.ones(window_size) / window_size, mode='same'))
            rms_normalized = rms / (np.max(rms) + 1e-8)  # Нормалізуємо до 0-1
            
            # Створюємо gate mask
            gate_mask = np.ones_like(rms_normalized)
            
            # Застосовуємо threshold
            below_threshold = rms_normalized < threshold
            gate_mask[below_threshold] = 1.0 / ratio  # Сильне приглушення слабких сигналів
            
            # Плавні переходи (attack/release)
            attack_samples = int(sample_rate * attack)
            release_samples = int(sample_rate * release)
            
            # Застосовуємо плавні переходи
            smoothed_mask = np.copy(gate_mask)
            for i in range(1, len(gate_mask)):
                if gate_mask[i] > gate_mask[i-1]:
                    # Attack - швидко відкриваємо
                    start = max(0, i - attack_samples)
                    smoothed_mask[start:i] = np.linspace(gate_mask[i-1], gate_mask[i], i - start)
                elif gate_mask[i] < gate_mask[i-1]:
                    # Release - повільно закриваємо
                    end = min(len(gate_mask), i + release_samples)
                    smoothed_mask[i:end] = np.linspace(gate_mask[i], gate_mask[i-1], end - i)
            
            gate_mask = smoothed_mask
            
            # Застосовуємо mask до аудіо
            if len(audio_np.shape) == 1:
                gated_audio = audio_np * gate_mask
            else:
                gated_audio = audio_np * gate_mask[np.newaxis, :]
            
            return gated_audio
        
        # Застосовуємо noise gate до кожного спікера
        gated_sources = []
        for idx in range(num_speakers):
            source_tensor = sources_tensor[idx]
            source_np = source_tensor.squeeze().numpy()
            
            # Застосовуємо noise gate з сильним приглушенням
            gated_audio = apply_noise_gate(
                source_np, 
                threshold=0.15,  # Поріг 15% від максимуму (менше переривань основного спікера)
                ratio=20.0,  # Сильне приглушення (20:1)
                attack=0.01,  # Швидка атака
                release=0.1  # Повільне відпускання
            )
            
            gated_sources.append(gated_audio)
        
        print(f"✅ [SpeechBrain] Noise gate applied (threshold=0.15, ratio=20:1)")
        sys.stdout.flush()
        
        # Створюємо output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Зберігаємо файли для кожного спікера (ПОВНІСТЮ розділені файли з приглушенням слабких сигналів)
        speaker_files = {}
        for idx in range(num_speakers):
            speaker_id = idx
            speaker_name = f"SPEAKER_{idx:02d}"
            output_path = os.path.join(output_dir, f"speaker_{speaker_id}.wav")
            
            gated_audio = gated_sources[idx]
            sf.write(output_path, gated_audio, sample_rate)
            
            speaker_files[speaker_id] = {
                'path': output_path,
                'speaker_label': speaker_name
            }
            
            duration = len(gated_audio) / sample_rate
            print(f"✅ [SpeechBrain] Saved speaker {speaker_id} ({speaker_name}): {duration:.2f}s (FULL SEPARATED TRACK with noise gate)")
            sys.stdout.flush()
        
        return {
            'success': True,
            'speaker_files': speaker_files,
            'speaker_map': {f"SPEAKER_{i:02d}": i for i in range(num_speakers)}  # Мапінг між labels та числовими ID
        }
        
    except Exception as e:
        print(f"❌ [SpeechBrain] Error in separation: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return {'success': False, 'error': str(e)}


def extract_single_speaker_audio(audio_path, speaker_segments, output_dir):
    """
    Витягує сегменти одного спікера з аудіо файлу та зберігає як окремий файл.
    FALLBACK метод - використовується, якщо PyAnnote separation не доступний.
    
    Args:
        audio_path: шлях до оригінального аудіо файлу
        speaker_segments: список сегментів для цього спікера [{'start': float, 'end': float}]
        output_dir: директорія для збереження витягнутого файлу
    
    Returns:
        output_path: шлях до збереженого одноголосого файлу
    """
    try:
        # Завантажуємо аудіо
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        # Збираємо всі сегменти спікера в один масив
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
        
        # Об'єднуємо всі сегменти в один аудіо файл
        combined_audio = np.concatenate(speaker_audio_segments)
        
        # Зберігаємо як WAV файл
        speaker_id = speaker_segments[0].get('speaker', 0)
        output_path = os.path.join(output_dir, f"speaker_{speaker_id}.wav")
        sf.write(output_path, combined_audio, sr)
        
        print(f"✅ Extracted speaker {speaker_id} audio: {len(combined_audio)/sr:.2f}s → {output_path}")
        return output_path
    
    except Exception as e:
        print(f"❌ Error extracting speaker audio: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_single_speaker_files_sync(audio_path, diarization_segments):
    """
    СИНХРОННА обробка одноголосих файлів (повертає результат одразу):
    1. Розрізає аудіо на одноголосі файли по спікерах
    2. Транскрибує кожен одноголосий файл
    3. Визначає головного спікера (той, хто більше говорив, не обривками)
    4. Видаляє другорядного спікера з результатів
    """
    import sys
    
    print(f"🔀 Step 1: Splitting audio into single-speaker files...")
    sys.stdout.flush()
    
    # Групуємо сегменти по спікерах
    speakers_segments = {}
    for seg in diarization_segments:
        speaker = seg.get('speaker', 0)
        if speaker not in speakers_segments:
            speakers_segments[speaker] = []
        speakers_segments[speaker].append(seg)
    
    print(f"📊 Found {len(speakers_segments)} speakers")
    sys.stdout.flush()
    
    # Створюємо тимчасову директорію для одноголосих файлів
    temp_job_id = str(uuid.uuid4())
    temp_dir = os.path.join(UPLOAD_FOLDER, f"single_speakers_{temp_job_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Витягуємо одноголосі файли
    speaker_files = {}
    for speaker, segments in speakers_segments.items():
        # Сортуємо сегменти за часом
        segments_sorted = sorted(segments, key=lambda x: x['start'])
        
        output_path = extract_single_speaker_audio(audio_path, segments_sorted, temp_dir)
        if output_path:
            speaker_files[speaker] = {
                'path': output_path,
                'segments': segments_sorted
            }
    
    print(f"✅ Step 1 completed: {len(speaker_files)} single-speaker files created")
    sys.stdout.flush()
    
    if not speaker_files:
        raise Exception('No single-speaker files could be extracted')
    
    # Крок 2: Транскрибуємо кожен одноголосий файл
    print(f"📝 Step 2: Transcribing single-speaker files...")
    sys.stdout.flush()
    
    speaker_transcriptions = {}
    for speaker, file_info in speaker_files.items():
        print(f"🎤 Transcribing speaker {speaker}...")
        sys.stdout.flush()
        
        transcription, transcription_segments, words = transcribe_audio(file_info['path'], transcription_provider='whisper')
        
        if transcription:
            # Обчислюємо загальну тривалість та кількість сегментів для визначення головного спікера
            total_duration = sum(seg['end'] - seg['start'] for seg in file_info['segments'])
            num_segments = len(file_info['segments'])
            
            speaker_transcriptions[speaker] = {
                'transcription': transcription,
                'segments': transcription_segments,
                'words': words,
                'total_duration': total_duration,
                'num_segments': num_segments,
                'file_path': file_info['path']
            }
            print(f"✅ Speaker {speaker} transcribed: {len(transcription)} chars, {total_duration:.2f}s duration")
        else:
            print(f"⚠️ Speaker {speaker} transcription failed or empty")
    
    if not speaker_transcriptions:
        raise Exception('No transcriptions could be generated')
    
    # Крок 3: Визначаємо головного спікера на основі транскрипції
    print(f"👤 Step 3: Determining main speaker from transcriptions...")
    sys.stdout.flush()
    
    # Формуємо combined_segments для всіх спікерів з транскрипцій
    all_combined_segments = []
    for speaker, info in speaker_transcriptions.items():
        transcription_segments = info['segments']
        diarization_segments = info.get('diarization_segments', [])
        
        # Об'єднуємо таймстемпи з діаризації та текст з транскрипції
        if diarization_segments:
            for i, diar_seg in enumerate(diarization_segments):
                transcript_text = ""
                if i < len(transcription_segments):
                    transcript_text = transcription_segments[i].get('text', '')
                elif transcription_segments:
                    transcript_text = info['transcription']
                
                all_combined_segments.append({
                    'speaker': speaker,
                    'start': diar_seg['start'],
                    'end': diar_seg['end'],
                    'text': transcript_text
                })
        else:
            for seg in transcription_segments:
                all_combined_segments.append({
                    'speaker': speaker,
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'text': seg.get('text', '')
                })
    
    # Обчислюємо загальну тривалість для логування
    total_duration = max(seg['end'] for seg in all_combined_segments) if all_combined_segments else 0
    
    # Використовуємо ту саму логіку, що і в enhance_main_speaker_audio
    main_speaker, speaker_stats = determine_main_speaker_from_segments(all_combined_segments, duration=total_duration)
    
    # Крок 4: Формуємо результат (тільки головний спікер)
    main_speaker_info = speaker_transcriptions[main_speaker]
    
    # Формуємо segments з транскрипції + таймстемпи з оригінальної діаризації
    combined_segments = []
    diarization_segments = main_speaker_info.get('diarization_segments', [])
    transcription_segments = main_speaker_info['segments']
    
    if diarization_segments:
        # Об'єднуємо: беремо текст з транскрипції, таймстемпи з діаризації
        for i, diar_seg in enumerate(diarization_segments):
            transcript_text = ""
            if i < len(transcription_segments):
                transcript_text = transcription_segments[i].get('text', '')
            elif transcription_segments:
                transcript_text = main_speaker_info['transcription']
            
            combined_segments.append({
                'speaker': main_speaker,
                'start': diar_seg['start'],
                'end': diar_seg['end'],
                'text': transcript_text
            })
    else:
        for seg in transcription_segments:
            combined_segments.append({
                'speaker': main_speaker,
                'start': seg.get('start', 0),
                'end': seg.get('end', 0),
                'text': seg.get('text', '')
            })
    
    # Форматуємо діалог основного спікера
    dialogue_lines = format_speaker_dialogue(combined_segments, main_speaker)
    
    result = {
        'success': True,
        'files': [{
            'speaker': main_speaker,
            'transcript': main_speaker_info['transcription'],
            'segments': combined_segments,  # Сегменти з таймстемпами відносно оригінального файлу
            'timestamps': [{'start': seg['start'], 'end': seg['end']} for seg in combined_segments],
            'total_duration': main_speaker_info['total_duration'],
            'num_segments': main_speaker_info['num_segments'],
            'dialogue': dialogue_lines  # Відформатований діалог: Таймстемп, спікер номер, репліка
        }],
        'main_speaker': main_speaker,
        'secondary_speakers_removed': [s for s in speaker_transcriptions.keys() if s != main_speaker]
    }
    
    print(f"✅ Processing completed successfully!")
    sys.stdout.flush()
    
    # Очищаємо тимчасові файли
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠️ Could not clean up temp files: {e}")
    
    return result


def process_single_speaker_files_background(job_id, audio_path, diarization_segments):
    """
    Фонова обробка одноголосих файлів:
    1. Розрізає аудіо на одноголосі файли по спікерах
    2. Транскрибує кожен одноголосий файл
    3. Визначає головного спікера (той, хто більше говорив, не обривками)
    4. Повертає всі сегменти з правильними таймстемпами для подальшої обробки в шорткатах
    
    Args:
        job_id: ID завдання
        audio_path: шлях до аудіо файлу
        diarization_segments: сегменти діаризації
    """
    import sys
    
    # Зберігаємо оригінальні segments ДО будь-якої обробки (глибока копія)
    import copy
    original_diarization_segments = copy.deepcopy(diarization_segments) if diarization_segments else []
    
    try:
        with jobs_lock:
            jobs[job_id]['status'] = 'processing'
        
        print(f"🔀 [Job {job_id}] Step 1: Splitting audio into single-speaker files using SpeechBrain separation...")
        sys.stdout.flush()
        
        # Створюємо тимчасову директорію для одноголосих файлів
        temp_dir = os.path.join(UPLOAD_FOLDER, f"single_speakers_{job_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Використовуємо SpeechBrain separation для якісної нарізки (як в speechbrain_separation.py)
        separation_result = separate_speakers_with_speechbrain(audio_path, temp_dir)
        
        speaker_files = {}
        
        if separation_result.get('success'):
            # SpeechBrain separation успішна - файли ПОВНІСТЮ розділені, не нарізані!
            speechbrain_speaker_files = separation_result['speaker_files']
            
            # SpeechBrain separation створює повні треки для кожного спікера
            # Не потрібно використовувати оригінальні segments - файли вже містять тільки одного спікера
            for speaker_id, file_info in speechbrain_speaker_files.items():
                # Для розділених файлів segments не потрібні - файл вже містить тільки одного спікера
                # Але зберігаємо порожній список segments для сумісності з кодом нижче
                speaker_files[speaker_id] = {
                    'path': file_info['path'],
                    'segments': [],  # Розділені файли не потребують segments - вони вже повні треки
                    'speaker_label': file_info.get('speaker_label', f'SPEAKER_{speaker_id:02d}'),
                    'is_separated': True  # Позначка, що це розділений файл, а не нарізаний
                }
            
            print(f"✅ [Job {job_id}] Step 1 completed: {len(speaker_files)} FULLY SEPARATED single-speaker files created using SpeechBrain")
            sys.stdout.flush()
        else:
            # Fallback до простого вирізання сегментів
            print(f"⚠️ [Job {job_id}] SpeechBrain separation failed: {separation_result.get('error', 'Unknown error')}")
            print(f"🔄 [Job {job_id}] Falling back to simple segment extraction...")
            sys.stdout.flush()
            
            # Групуємо сегменти по спікерах
            speakers_segments = {}
            for seg in diarization_segments:
                speaker = seg.get('speaker', 0)
                if speaker not in speakers_segments:
                    speakers_segments[speaker] = []
                speakers_segments[speaker].append(seg)
            
            print(f"📊 [Job {job_id}] Found {len(speakers_segments)} speakers")
            sys.stdout.flush()
            
            # Витягуємо одноголосі файли простим методом
            for speaker, segments in speakers_segments.items():
                # Сортуємо сегменти за часом
                segments_sorted = sorted(segments, key=lambda x: x['start'])
                
                output_path = extract_single_speaker_audio(audio_path, segments_sorted, temp_dir)
                if output_path:
                    speaker_files[speaker] = {
                        'path': output_path,
                        'segments': segments_sorted
                    }
            
            print(f"✅ [Job {job_id}] Step 1 completed: {len(speaker_files)} single-speaker files created using simple extraction")
            sys.stdout.flush()
        
        if not speaker_files:
            with jobs_lock:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = 'No single-speaker files could be extracted'
                jobs[job_id]['code'] = 'NO_FILES_EXTRACTED'
            return
        
        # Крок 2: Транскрибуємо кожен одноголосий файл
        print(f"📝 [Job {job_id}] Step 2: Transcribing single-speaker files...")
        sys.stdout.flush()
        
        speaker_transcriptions = {}
        for speaker, file_info in speaker_files.items():
            print(f"🎤 [Job {job_id}] Transcribing speaker {speaker}...")
            sys.stdout.flush()
            
            transcription, transcription_segments, words = transcribe_audio(file_info['path'], transcription_provider='whisper')
            
            if transcription:
                # Для розділених файлів використовуємо тривалість з транскрипції
                # (бо segments порожні для розділених файлів)
                if file_info.get('is_separated'):
                    # Розділений файл - використовуємо тривалість з транскрипції
                    total_duration = max(seg.get('end', 0) for seg in transcription_segments) if transcription_segments else 0
                    num_segments = len(transcription_segments)
                else:
                    # Нарізаний файл - використовуємо segments з діаризації
                    total_duration = sum(seg['end'] - seg['start'] for seg in file_info['segments'])
                    num_segments = len(file_info['segments'])
                
                speaker_transcriptions[speaker] = {
                    'transcription': transcription,
                    'segments': transcription_segments,  # Сегменти транскрипції одноголосого файлу
                    'words': words,
                    'total_duration': total_duration,
                    'num_segments': num_segments,
                    'file_path': file_info['path'],
                    'diarization_segments': file_info.get('segments', []),  # Може бути порожнім для розділених файлів
                    'is_separated': file_info.get('is_separated', False)  # Позначка, що це розділений файл
                }
                print(f"✅ [Job {job_id}] Speaker {speaker} transcribed: {len(transcription)} chars, {total_duration:.2f}s duration")
            else:
                print(f"⚠️ [Job {job_id}] Speaker {speaker} transcription failed or empty")
        
        if not speaker_transcriptions:
            with jobs_lock:
                jobs[job_id]['status'] = 'failed'
                jobs[job_id]['error'] = 'No transcriptions could be generated'
                jobs[job_id]['code'] = 'NO_TRANSCRIPTIONS'
            return
        
        # Крок 4: Визначаємо головного спікера на основі транскрипції
        print(f"👤 [Job {job_id}] Step 4: Determining main speaker from transcriptions...")
        sys.stdout.flush()
        
        # Формуємо combined_segments для всіх спікерів з транскрипцій
        all_combined_segments = []
        for speaker, info in speaker_transcriptions.items():
            transcription_segments = info['segments']
            diarization_segments = info.get('diarization_segments', [])
            
            # Об'єднуємо таймстемпи з діаризації та текст з транскрипції
            if diarization_segments:
                for i, diar_seg in enumerate(diarization_segments):
                    transcript_text = ""
                    if i < len(transcription_segments):
                        transcript_text = transcription_segments[i].get('text', '')
                    elif transcription_segments:
                        transcript_text = info['transcription']
                    
                    all_combined_segments.append({
                        'speaker': speaker,
                        'start': diar_seg['start'],
                        'end': diar_seg['end'],
                        'text': transcript_text
                    })
            else:
                for seg in transcription_segments:
                    all_combined_segments.append({
                        'speaker': speaker,
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': seg.get('text', '')
                    })
        
        # Обчислюємо загальну тривалість для логування
        total_duration = max(seg['end'] for seg in all_combined_segments) if all_combined_segments else 0
        
        # Використовуємо ту саму логіку, що і в enhance_main_speaker_audio
        main_speaker, speaker_stats = determine_main_speaker_from_segments(all_combined_segments, duration=total_duration)
        print(f"✅ [Job {job_id}] Main speaker determined: {main_speaker}")
        sys.stdout.flush()
        
        # Крок 4: Формуємо результат (тільки головний спікер)
        main_speaker_info = speaker_transcriptions[main_speaker]
        
        # Формуємо segments з транскрипції + таймстемпи з оригінальної діаризації
        # Кожен сегмент транскрипції має таймстемпи відносно одноголосого файлу
        # Потрібно перетворити їх на таймстемпи відносно оригінального файлу
        combined_segments = []
        diarization_segments = main_speaker_info.get('diarization_segments', [])
        transcription_segments = main_speaker_info['segments']
        
        # Якщо є сегменти діаризації, використовуємо їх таймстемпи
        # Якщо немає, використовуємо таймстемпи з транскрипції
        if diarization_segments:
            # Об'єднуємо: беремо текст з транскрипції, таймстемпи з діаризації
            for i, diar_seg in enumerate(diarization_segments):
                # Знаходимо відповідний текст з транскрипції (якщо є)
                transcript_text = ""
                if i < len(transcription_segments):
                    transcript_text = transcription_segments[i].get('text', '')
                elif transcription_segments:
                    # Якщо сегментів транскрипції менше, беремо весь текст
                    transcript_text = main_speaker_info['transcription']
                
                combined_segments.append({
                    'speaker': main_speaker,
                    'start': diar_seg['start'],
                    'end': diar_seg['end'],
                    'text': transcript_text
                })
        else:
            # Якщо немає сегментів діаризації, використовуємо сегменти транскрипції
            for seg in transcription_segments:
                combined_segments.append({
                    'speaker': main_speaker,
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'text': seg.get('text', '')
                })
        
        # Формуємо результати для ВСІХ спікерів (не тільки головного)
        files_result = []
        all_speakers_segments = {}  # Зберігаємо сегменти для всіх спікерів
        
        for speaker, info in speaker_transcriptions.items():
            # Формуємо segments для цього спікера
            # ТРІЗ РІШЕННЯ: Зіставлення на основі накопиченої тривалості
            # В одноголосому файлі сегменти йдуть без пауз, а в оригінальному - з паузами
            # Тому потрібно обчислити позиції сегментів діаризації в одноголосому файлі
            combined_segments = []
            diarization_segments = info.get('diarization_segments', [])
            transcription_segments = info['segments']
            
            if diarization_segments and transcription_segments:
                # КРИТИЧНО: Використовуємо сегменти діаризації як основу, а текст беремо з транскрипції
                # Це забезпечує правильні таймстемпи та правильну кількість сегментів
                
                # Крок 1: Обчислюємо позиції сегментів діаризації в одноголосому файлі
                diar_positions = []
                accumulated_duration = 0
                
                # Сортуємо сегменти діаризації за часом
                sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
                
                for diar_seg in sorted_diar_segments:
                    diar_duration = diar_seg['end'] - diar_seg['start']
                    diar_positions.append({
                        'position_in_single_file': accumulated_duration,  # Позиція в одноголосому файлі
                        'original_start': diar_seg['start'],  # Оригінальний таймстемп
                        'original_end': diar_seg['end'],  # Оригінальний таймстемп
                        'duration': diar_duration,
                        'index': len(diar_positions)  # Індекс для зіставлення
                    })
                    accumulated_duration += diar_duration
                
                print(f"🔍 [Job {job_id}] Speaker {speaker}: Matching diarization segments with transcription")
                print(f"   - Diarization segments: {len(sorted_diar_segments)}")
                print(f"   - Transcription segments: {len(transcription_segments)}")
                print(f"   - Total duration in single file: {accumulated_duration:.2f}s")
                sys.stdout.flush()
                
                # Крок 2: Для кожного сегмента діаризації знаходимо відповідний текст з транскрипції
                for diar_pos in diar_positions:
                    # Знаходимо сегмент транскрипції, який перетинається з цим сегментом діаризації
                    best_transcript = None
                    best_overlap = 0
                    
                    diar_start_in_single = diar_pos['position_in_single_file']
                    diar_end_in_single = diar_pos['position_in_single_file'] + diar_pos['duration']
                    
                    for trans_seg in transcription_segments:
                        trans_start = trans_seg.get('start', 0)  # Відносно одноголосого файлу
                        trans_end = trans_seg.get('end', 0)
                        text = trans_seg.get('text', '').strip()
                        
                        if not text:
                            continue
                        
                        # Обчислюємо перекриття
                        overlap_start = max(trans_start, diar_start_in_single)
                        overlap_end = min(trans_end, diar_end_in_single)
                        overlap = max(0, overlap_end - overlap_start)
                        
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_transcript = text
                    
                    # Якщо знайшли відповідний текст, використовуємо його
                    # Якщо ні, використовуємо весь текст транскрипції (fallback)
                    text_to_use = best_transcript if best_transcript else info.get('transcription', '')
                    
                    if text_to_use:
                        combined_segments.append({
                            'speaker': speaker,  # В одноголосому файлі є тільки цей спікер
                            'start': round(diar_pos['original_start'], 2),
                            'end': round(diar_pos['original_end'], 2),
                            'text': text_to_use
                        })
                
                print(f"✅ [Job {job_id}] Speaker {speaker}: Matched {len(combined_segments)} segments")
                if len(combined_segments) > 0:
                    print(f"   - First segment: {combined_segments[0].get('start', 0):.2f}s - {combined_segments[0].get('end', 0):.2f}s, text: {combined_segments[0].get('text', '')[:50]}")
                    if len(combined_segments) > 1:
                        print(f"   - Last segment: {combined_segments[-1].get('start', 0):.2f}s - {combined_segments[-1].get('end', 0):.2f}s, text: {combined_segments[-1].get('text', '')[:50]}")
                sys.stdout.flush()
                
            elif transcription_segments:
                # Якщо немає діаризації, використовуємо транскрипцію як є
                for seg in transcription_segments:
                    combined_segments.append({
                        'speaker': speaker,
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': seg.get('text', '').strip()
                    })
            else:
                # Якщо немає транскрипції, використовуємо діаризацію (fallback)
                for diar_seg in diarization_segments:
                    combined_segments.append({
                        'speaker': speaker,
                        'start': diar_seg['start'],
                        'end': diar_seg['end'],
                        'text': info.get('transcription', '')
                    })
            
            all_speakers_segments[speaker] = combined_segments
            
            # Форматуємо діалог для основного спікера
            dialogue_lines = format_speaker_dialogue(combined_segments, main_speaker) if speaker == main_speaker else []
            
            files_result.append({
                'speaker': speaker,
                'transcript': info['transcription'],
                'segments': combined_segments,
                'timestamps': [{'start': seg['start'], 'end': seg['end']} for seg in combined_segments],
                'total_duration': info['total_duration'],
                'num_segments': info['num_segments'],
                'dialogue': dialogue_lines if speaker == main_speaker else []  # Відформатований діалог тільки для основного спікера
            })
        
        # Крок 6: Зберігаємо одноголосі файли та створюємо посилання для завантаження
        print(f"🎵 [Job {job_id}] Step 6: Preparing single-speaker audio files for download...")
        sys.stdout.flush()
        
        # Створюємо директорію для збереження одноголосих файлів (не видаляємо temp_dir одразу)
        audio_files_urls = {}
        for speaker, file_info in speaker_files.items():
            file_path = file_info.get('path')
            if file_path and os.path.exists(file_path):
                try:
                    # Копіюємо файл в постійну директорію для завантаження
                    # Використовуємо job_id та speaker_id для унікальності
                    download_dir = os.path.join(UPLOAD_FOLDER, 'single_speaker_audio')
                    os.makedirs(download_dir, exist_ok=True)
                    
                    # Створюємо унікальне ім'я файлу
                    file_extension = os.path.splitext(file_path)[1] or '.wav'
                    download_filename = f"{job_id}_speaker_{speaker}{file_extension}"
                    download_path = os.path.join(download_dir, download_filename)
                    
                    # Копіюємо файл
                    import shutil
                    shutil.copy2(file_path, download_path)
                    
                    file_size = os.path.getsize(download_path)
                    
                    # Створюємо URL для завантаження
                    audio_files_urls[speaker] = {
                        'url': f'/api/single-speaker-audio/{job_id}/{speaker}',
                        'filename': f"speaker_{speaker}{file_extension}",
                        'size_bytes': file_size
                    }
                    
                    print(f"✅ [Job {job_id}] Prepared speaker {speaker} audio: {file_size} bytes → {download_path}")
                except Exception as e:
                    print(f"⚠️ [Job {job_id}] Failed to prepare speaker {speaker} audio: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
        
        print(f"🎵 [Job {job_id}] Prepared {len(audio_files_urls)} audio files for download")
        sys.stdout.flush()
        
        result = {
            'success': True,
            'files': files_result,  # Всі спікери, не тільки головний
            'main_speaker': main_speaker,  # Головний спікер між одноголосими файлами
            'all_speakers_segments': all_speakers_segments,  # Сегменти для всіх спікерів (для Markdown форматування)
            'original_diarization_segments': original_diarization_segments,  # Оригінальні segments з діаризації (для показу всіх реплік) - збережені ДО обробки
            'audio_files': audio_files_urls  # Посилання на одноголосі аудіо файли для завантаження
        }
        
        # Оновлюємо статус
        with jobs_lock:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['result'] = result
        
        print(f"✅ [Job {job_id}] Processing completed successfully!")
        sys.stdout.flush()
        
        # Очищаємо тимчасові файли
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"⚠️ [Job {job_id}] Could not clean up temp files: {e}")
    
    except Exception as e:
        print(f"❌ [Job {job_id}] Error in background processing: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['code'] = 'PROCESSING_ERROR'


@app.route('/api/process-single-speaker-files', methods=['POST', 'OPTIONS'])
def api_process_single_speaker_files():
    """
    АСИНХРОННИЙ API ендпоінт для розрізання аудіо на одноголосі файли та їх обробки.
    Приймає JSON з base64-encoded файлом та job_id діаризації.
    Повертає job_id одразу, обробка виконується в фоні.
    """
    import sys
    import base64
    
    print(f"🔵 [API] /api/process-single-speaker-files called - Method: {request.method}, Remote: {request.remote_addr}")
    sys.stdout.flush()
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS preflight request received from", request.remote_addr)
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        sys.stdout.flush()
        return response
    
    print(f"📥 POST /api/process-single-speaker-files request received from {request.remote_addr}")
    print(f"📋 Request headers: {dict(request.headers)}")
    print(f"📋 Request content type: {request.content_type}")
    sys.stdout.flush()
    
    # Генеруємо job_id ДО try блоку
    job_id = str(uuid.uuid4())
    filepath = None
    
    try:
        # Перевіряємо, чи це JSON
        is_json = request.is_json or (request.content_type and 'application/json' in request.content_type)
        
        if not is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        # Парсимо JSON
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data received',
                'code': 'NO_DATA'
            }), 400
        
        # Детальне логування JSON
        print(f"📦 [Job {job_id}] JSON parsed successfully")
        print(f"📋 [Job {job_id}] JSON keys: {list(data.keys())}")
        print(f"📋 [Job {job_id}] Full JSON structure:")
        print(f"   - file: {'present' if 'file' in data else 'MISSING'}, type: {type(data.get('file'))}, length: {len(str(data.get('file', ''))) if data.get('file') else 0}")
        print(f"   - filename: {data.get('filename', 'MISSING')}")
        print(f"   - diarization_job_id: {data.get('diarization_job_id', 'MISSING')}")
        
        # Логуємо перші 100 символів base64 (якщо є)
        if 'file' in data and data['file']:
            file_preview = str(data['file'])[:100]
            print(f"   - file preview (first 100 chars): {file_preview}...")
        
        # Логуємо весь JSON (обмежено, щоб не засмічувати логи)
        import json
        json_str = json.dumps(data, indent=2, default=str)
        if len(json_str) > 1000:
            print(f"   - JSON (first 1000 chars): {json_str[:1000]}...")
        else:
            print(f"   - Full JSON: {json_str}")
        sys.stdout.flush()
        
        # Отримуємо base64 файл
        file_base64 = data.get('file')
        filename = data.get('filename', 'audio.wav')
        diarization_job_id = data.get('diarization_job_id')
        
        if not file_base64:
            return jsonify({
                'success': False,
                'error': 'No file data provided. Send file as base64 string in "file" field.',
                'code': 'NO_FILE'
            }), 400
        
        if not diarization_job_id:
            return jsonify({
                'success': False,
                'error': 'No diarization_job_id provided. Send job_id from diarization in "diarization_job_id" field.',
                'code': 'NO_DIARIZATION_JOB_ID'
            }), 400
        
        # Витягуємо segments з результату діаризації
        with jobs_lock:
            if diarization_job_id not in jobs:
                return jsonify({
                    'success': False,
                    'error': f'Diarization job {diarization_job_id} not found. Make sure diarization is completed first.',
                    'code': 'DIARIZATION_JOB_NOT_FOUND'
                }), 404
            
            diarization_job = jobs[diarization_job_id]
            if diarization_job['status'] != 'completed':
                return jsonify({
                    'success': False,
                    'error': f'Diarization job {diarization_job_id} is not completed yet. Status: {diarization_job["status"]}',
                    'code': 'DIARIZATION_NOT_COMPLETED'
                }), 400
            
            diarization_result = diarization_job.get('result', {})
            
            # Беремо segments з combined, бо вони містять текст
            # combined.segments містять об'єднані дані з діаризації (speaker) та транскрипції (text)
            combined = diarization_result.get('combined', {})
            segments = combined.get('segments', [])
            
            # Логуємо кількість спікерів в combined
            if segments:
                unique_speakers = set(seg.get('speaker', 0) for seg in segments)
                print(f"📊 [Job {job_id}] Combined segments: {len(segments)} segments, {len(unique_speakers)} speakers: {sorted(unique_speakers)}")
                sys.stdout.flush()
            
            # Якщо немає в combined, спробуємо взяти з diarization (fallback, але без тексту)
            if not segments:
                diarization = diarization_result.get('diarization', {})
                segments = diarization.get('segments', [])
                if segments:
                    unique_speakers = set(seg.get('speaker', 0) for seg in segments)
                    print(f"⚠️ [Job {job_id}] Using diarization segments (fallback, no text): {len(segments)} segments, {len(unique_speakers)} speakers: {sorted(unique_speakers)}")
                    sys.stdout.flush()
        
        if not segments:
            return jsonify({
                'success': False,
                'error': 'No segments found in diarization result. Make sure diarization completed successfully.',
                'code': 'NO_SEGMENTS'
            }), 400
        
        print(f"📊 Diarization result: {len(segments)} segments from job {diarization_job_id}")
        sys.stdout.flush()
        
        # Автоматично визначаємо формат, якщо потрібно
        if '.' not in filename or not allowed_file(filename):
            detected_format = detect_audio_format_from_base64(file_base64)
            if detected_format:
                if '.' in filename:
                    filename = filename.rsplit('.', 1)[0] + '.' + detected_format
                else:
                    filename = filename + '.' + detected_format
            else:
                filename = filename + '.m4a' if '.' not in filename else filename
        
        # Створюємо завдання ДО декодування файлу
        with jobs_lock:
            jobs[job_id] = {
                'status': 'pending',
                'result': None,
                'error': None,
                'created_at': datetime.now()
            }
            print(f"✅ [Job {job_id}] Job created and stored in jobs dictionary")
            print(f"📊 Total jobs after creation: {len(jobs)}")
            print(f"📋 Job {job_id} exists in jobs: {job_id in jobs}")
            sys.stdout.flush()
        
        print(f"✅ [Job {job_id}] Job created, returning job_id IMMEDIATELY")
        sys.stdout.flush()
        
        # Повертаємо job_id ОДРАЗУ
        response = jsonify({
            'success': True,
            'job_id': job_id,
            'status': 'pending',
            'message': 'Processing started. Use GET /api/process-single-speaker-files/{job_id}/status to check progress.'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        # Декодуємо base64 та обробляємо в фоні
        def decode_and_process():
            try:
                print(f"💾 [Job {job_id}] Background: Starting base64 decode...")
                sys.stdout.flush()
                
                # Очищаємо base64
                file_base64_clean = clean_base64_string(file_base64)
                
                # Декодуємо base64
                file_data = base64.b64decode(file_base64_clean, validate=True)
                file_size = len(file_data)
                print(f"✅ [Job {job_id}] Background: Base64 decode successful! Decoded size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
                sys.stdout.flush()
                
                if file_size > MAX_FILE_SIZE:
                    with jobs_lock:
                        jobs[job_id]['status'] = 'failed'
                        jobs[job_id]['error'] = f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB'
                        jobs[job_id]['code'] = 'FILE_SIZE_EXCEEDED'
                    return
                
                # Зберігаємо файл тимчасово
                filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
                with open(filepath, 'wb') as f:
                    f.write(file_data)
                print(f"💾 [Job {job_id}] Background: File saved: {filepath}")
                sys.stdout.flush()
                
                # Логуємо segments перед передачею в process_single_speaker_files_background
                print(f"📊 [Job {job_id}] Passing segments to process_single_speaker_files_background:")
                print(f"   - Total segments: {len(segments)}")
                if segments:
                    unique_speakers = set(seg.get('speaker', 0) for seg in segments)
                    print(f"   - Unique speakers: {sorted(unique_speakers)}")
                    # Показуємо приклад segments для кожного спікера
                    for speaker_id in sorted(unique_speakers):
                        speaker_segments = [seg for seg in segments if seg.get('speaker', 0) == speaker_id]
                        print(f"   - Speaker {speaker_id}: {len(speaker_segments)} segments")
                        if speaker_segments:
                            first_seg = speaker_segments[0]
                            print(f"     Example: start={first_seg.get('start')}, text={first_seg.get('text', '')[:50]}")
                sys.stdout.flush()
                
                # Обробляємо одноголосі файли (LLM діаризація буде в шорткатах)
                process_single_speaker_files_background(job_id, filepath, segments)
                
            except Exception as e:
                print(f"❌ [Job {job_id}] Background: Error: {e}")
                import traceback
                traceback.print_exc()
                with jobs_lock:
                    jobs[job_id]['status'] = 'failed'
                    jobs[job_id]['error'] = str(e)
                    jobs[job_id]['code'] = 'PROCESSING_ERROR'
        
        # Запускаємо обробку в фоні
        thread = threading.Thread(target=decode_and_process, daemon=True)
        thread.start()
        
        return response, 202  # 202 Accepted
    
    except Exception as e:
        print(f"❌ [Job {job_id}] Error creating job: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        # Видаляємо job зі словника
        with jobs_lock:
            if job_id in jobs:
                del jobs[job_id]
        
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/api/process-single-speaker-files/<job_id>/status', methods=['GET', 'OPTIONS'])
def get_process_single_speaker_files_status(job_id):
    """Отримує статус обробки одноголосих файлів"""
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    print(f"🔵 [API] GET /api/process-single-speaker-files/{job_id}/status called from {request.remote_addr}")
    print(f"📋 Requested job_id: {job_id}")
    sys.stdout.flush()
    
    with jobs_lock:
        print(f"📊 Total jobs in memory: {len(jobs)}")
        print(f"📋 Available job_ids: {list(jobs.keys())[:5]}...")  # Показуємо перші 5
        sys.stdout.flush()
        
        if job_id not in jobs:
            print(f"❌ Job {job_id} not found in jobs dictionary")
            sys.stdout.flush()
            return jsonify({
                'success': False,
                'error': f'Job not found: {job_id}',
                'code': 'JOB_NOT_FOUND',
                'available_jobs_count': len(jobs)
            }), 404
        
        job = jobs[job_id]
        
        if job['status'] == 'completed':
            result = job['result']
            
            # Формуємо Markdown форматування
            markdown_data = {}
            if 'all_speakers_segments' in result:
                original_segments = result.get('original_diarization_segments', [])
                
                # Логуємо для діагностики
                print(f"📊 [Status {job_id}] Formatting markdown:")
                print(f"   - original_diarization_segments: {len(original_segments)} segments")
                if original_segments:
                    unique_speakers = set(seg.get('speaker', 0) for seg in original_segments)
                    print(f"   - Unique speakers in original: {sorted(unique_speakers)}")
                    # Показуємо перші 3 segments для діагностики
                    for i, seg in enumerate(original_segments[:3]):
                        print(f"   - Segment {i}: speaker={seg.get('speaker')}, start={seg.get('start')}, text={seg.get('text', '')[:50]}")
                print(f"   - all_speakers_segments keys: {list(result['all_speakers_segments'].keys())}")
                sys.stdout.flush()
                
                markdown_data = format_single_speaker_files_markdown(
                    result['all_speakers_segments'],
                    original_segments
                )
                
                # Додаємо відформатований діалог основного спікера
                main_speaker = result.get('main_speaker')
                if main_speaker is not None and 'files' in result:
                    # Знаходимо файл основного спікера
                    for file_info in result['files']:
                        if file_info.get('speaker') == main_speaker and 'dialogue' in file_info:
                            dialogue_lines = file_info['dialogue']
                            if dialogue_lines:
                                # Додаємо ключ з діалогом основного спікера
                                markdown_data['MainSpeakerDialogue'] = "\n".join(dialogue_lines)
                                print(f"📊 [Status {job_id}] Added MainSpeakerDialogue: {len(dialogue_lines)} lines")
                                sys.stdout.flush()
                
                # Логуємо результат форматування
                print(f"📊 [Status {job_id}] Markdown formatting result:")
                print(f"   - Markdown keys: {list(markdown_data.keys())}")
                for key in ['File1Speaker0', 'File1Speaker1', 'File2Speaker0', 'File2Speaker1', 'MainSpeakerDialogue']:
                    if key in markdown_data:
                        content = markdown_data[key]
                        content_preview = content[:100] if content else "(empty)"
                        print(f"   - {key}: {len(content)} chars, preview: {content_preview}")
                    else:
                        print(f"   - {key}: MISSING")
                sys.stdout.flush()
            
            # Формуємо списки реплік основного спікера для кожного файлу
            mainspeakerfile1 = []
            mainspeakerfile2 = []
            
            if 'files' in result:
                # Визначаємо, який спікер відповідає якому файлу
                unique_speakers = sorted(set(f['speaker'] for f in result['files']))
                file_to_speaker = {}
                for idx, speaker_id in enumerate(unique_speakers, start=1):
                    file_to_speaker[idx] = speaker_id  # File1 -> перший спікер, File2 -> другий спікер
                
                print(f"📊 [Status {job_id}] Processing main speaker files:")
                print(f"   - file_to_speaker: {file_to_speaker}")
                sys.stdout.flush()
                
                # Для File1: в одноголосому файлі є тільки один спікер, тому всі сегменти належать цьому спікеру
                file1_speaker_id = file_to_speaker.get(1)
                if file1_speaker_id is not None:
                    # Знаходимо інформацію про File1
                    file1_info = next((f for f in result['files'] if f.get('speaker') == file1_speaker_id), None)
                    if file1_info and 'segments' in file1_info:
                        file1_segments = file1_info['segments']
                        if file1_segments:
                            # В одноголосому файлі всі сегменти належать одному спікеру (file1_speaker_id)
                            # Визначаємо основного спікера на основі сегментів (для консистентності з enhance-main-speaker)
                            file1_duration = max(seg.get('end', 0) for seg in file1_segments) if file1_segments else 0
                            file1_main_speaker, _ = determine_main_speaker_from_segments(file1_segments, duration=file1_duration)
                            
                            print(f"   - File1: speaker_id={file1_speaker_id}, main_speaker={file1_main_speaker}, segments={len(file1_segments)}")
                            
                            # В одноголосому файлі всі сегменти належать одному спікеру, тому використовуємо всі сегменти
                            # Але фільтруємо тільки ті, що належать основному спікеру (для консистентності)
                            for seg in file1_segments:
                                # В одноголосому файлі всі сегменти мають speaker == file1_speaker_id
                                # Але для консистентності перевіряємо, чи це основний спікер
                                if seg.get('speaker') == file1_main_speaker:
                                    start_time = seg.get('start', 0)
                                    minutes = int(start_time // 60)
                                    seconds = int(start_time % 60)
                                    time_str = f"{minutes:02d}:{seconds:02d}"
                                    text = seg.get('text', '').strip()
                                    if text:
                                        mainspeakerfile1.append(f"{time_str} Speaker {file1_main_speaker}: {text}")
                
                # Для File2: в одноголосому файлі є тільки один спікер, тому всі сегменти належать цьому спікеру
                file2_speaker_id = file_to_speaker.get(2)
                if file2_speaker_id is not None:
                    # Знаходимо інформацію про File2
                    file2_info = next((f for f in result['files'] if f.get('speaker') == file2_speaker_id), None)
                    if file2_info and 'segments' in file2_info:
                        file2_segments = file2_info['segments']
                        if file2_segments:
                            # В одноголосому файлі всі сегменти належать одному спікеру (file2_speaker_id)
                            # Визначаємо основного спікера на основі сегментів (для консистентності з enhance-main-speaker)
                            file2_duration = max(seg.get('end', 0) for seg in file2_segments) if file2_segments else 0
                            file2_main_speaker, _ = determine_main_speaker_from_segments(file2_segments, duration=file2_duration)
                            
                            print(f"   - File2: speaker_id={file2_speaker_id}, main_speaker={file2_main_speaker}, segments={len(file2_segments)}")
                            
                            # В одноголосому файлі всі сегменти належать одному спікеру, тому використовуємо всі сегменти
                            # Але фільтруємо тільки ті, що належать основному спікеру (для консистентності)
                            for seg in file2_segments:
                                # В одноголосому файлі всі сегменти мають speaker == file2_speaker_id
                                # Але для консистентності перевіряємо, чи це основний спікер
                                if seg.get('speaker') == file2_main_speaker:
                                    start_time = seg.get('start', 0)
                                    minutes = int(start_time // 60)
                                    seconds = int(start_time % 60)
                                    time_str = f"{minutes:02d}:{seconds:02d}"
                                    text = seg.get('text', '').strip()
                                    if text:
                                        mainspeakerfile2.append(f"{time_str} Speaker {file2_main_speaker}: {text}")
                
                print(f"📊 [Status {job_id}] Main speaker files result:")
                print(f"   - mainspeakerfile1: {len(mainspeakerfile1)} replicas")
                print(f"   - mainspeakerfile2: {len(mainspeakerfile2)} replicas")
                sys.stdout.flush()
            
            # Отримуємо audio_files з результату (якщо є)
            audio_files = result.get('audio_files', {})
            
            # Додаємо повні URL для audio_files
            # Визначаємо базовий URL з запиту
            base_url = request.host_url.rstrip('/')
            audio_files_with_urls = {}
            for speaker, file_info in audio_files.items():
                if isinstance(file_info, dict) and 'url' in file_info:
                    # Створюємо повний URL (відносний шлях вже є в file_info['url'])
                    relative_url = file_info['url']
                    # Якщо URL вже повний (починається з http), не додаємо base_url
                    if relative_url.startswith('http://') or relative_url.startswith('https://'):
                        full_url = relative_url
                    else:
                        # Додаємо базовий URL до відносного шляху
                        full_url = f"{base_url}{relative_url}"
                    
                    audio_files_with_urls[speaker] = {
                        **file_info,
                        'url': full_url
                    }
                else:
                    audio_files_with_urls[speaker] = file_info
            
            # Повертаємо JSON з полями
            response_data = {
                'mainspeakerfile1': mainspeakerfile1,
                'mainspeakerfile2': mainspeakerfile2,
                'audio_files': audio_files_with_urls  # Посилання на одноголосі аудіо файли (повні URL)
            }
            
            # Логуємо повну відповідь для діагностики
            import json
            response_json = json.dumps(response_data, indent=2, ensure_ascii=False)
            print(f"📤 [Status {job_id}] Full response JSON (first 500 chars):")
            print(response_json[:500])
            print(f"📤 [Status {job_id}] Full response JSON length: {len(response_json)} chars")
            sys.stdout.flush()
            
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        elif job['status'] == 'failed':
            response = jsonify({
                'success': False,
                'status': 'failed',
                'error': job.get('error', 'Unknown error'),
                'code': job.get('code', 'PROCESSING_ERROR')
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
        else:
            response = jsonify({
                'success': True,
                'status': job['status'],
                'message': 'Processing in progress...'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200


@app.route('/api/separate-audio', methods=['POST', 'OPTIONS'])
def api_separate_audio():
    """
    Новий ендпоїнт для розділення аудіо на окремі голоси.
    Приймає аудіо файл, розбиває його на два голоси за допомогою SpeechBrain separation,
    і повертає два аудіо треки в JSON з полями file1, file2.
    
    Returns:
        JSON з полями:
        - file1: base64 encoded аудіо або URL до файлу
        - file2: base64 encoded аудіо або URL до файлу
        - success: bool
    """
    import sys
    import base64
    import uuid
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        # Перевіряємо, чи є файл
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided',
                'code': 'NO_FILE'
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        print(f"🎵 [Separate Audio] Received file: {audio_file.filename}")
        sys.stdout.flush()
        
        # Зберігаємо тимчасовий файл
        job_id = str(uuid.uuid4())
        file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
        temp_filename = f"separate_{job_id}{file_extension}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        audio_file.save(temp_path)
        
        print(f"💾 [Separate Audio] Saved to: {temp_path}")
        sys.stdout.flush()
        
        # Створюємо тимчасову директорію для розділених файлів
        output_dir = os.path.join(UPLOAD_FOLDER, f"separated_{job_id}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Виконуємо розділення за допомогою SpeechBrain
        print(f"🔀 [Separate Audio] Starting SpeechBrain separation...")
        sys.stdout.flush()
        
        separation_result = separate_speakers_with_speechbrain(temp_path, output_dir)
        
        if not separation_result.get('success'):
            # Видаляємо тимчасовий файл
            try:
                os.remove(temp_path)
            except:
                pass
            return jsonify({
                'success': False,
                'error': separation_result.get('error', 'Separation failed'),
                'code': 'SEPARATION_FAILED'
            }), 500
        
        speaker_files = separation_result['speaker_files']
        
        # Перевіряємо, чи є принаймні два спікери
        if len(speaker_files) < 2:
            # Видаляємо тимчасові файли
            try:
                os.remove(temp_path)
                import shutil
                shutil.rmtree(output_dir)
            except:
                pass
            return jsonify({
                'success': False,
                'error': f'Found only {len(speaker_files)} speaker(s), need at least 2',
                'code': 'INSUFFICIENT_SPEAKERS'
            }), 400
        
        # Беремо перші два спікери
        speaker_ids = sorted(speaker_files.keys())[:2]
        speaker_0_file = speaker_files[speaker_ids[0]]['path']
        speaker_1_file = speaker_files[speaker_ids[1]]['path']
        
        print(f"✅ [Separate Audio] Separation completed: speaker {speaker_ids[0]} and {speaker_ids[1]}")
        sys.stdout.flush()
        
        # Видаляємо тимчасовий оригінальний файл
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Створюємо URL-и для завантаження файлів
        base_url = request.host_url.rstrip('/')
        file1_url = f"{base_url}/api/separate-audio-file/{job_id}/0"
        file2_url = f"{base_url}/api/separate-audio-file/{job_id}/1"
        
        # Переміщуємо файли в постійну директорію для завантаження
        download_dir = os.path.join(UPLOAD_FOLDER, 'separated_audio')
        os.makedirs(download_dir, exist_ok=True)
        
        file1_download_path = os.path.join(download_dir, f"{job_id}_speaker_0.wav")
        file2_download_path = os.path.join(download_dir, f"{job_id}_speaker_1.wav")
        
        import shutil
        shutil.copy2(speaker_0_file, file1_download_path)
        shutil.copy2(speaker_1_file, file2_download_path)
        
        # Видаляємо тимчасову директорію з розділеними файлами
        try:
            shutil.rmtree(output_dir)
        except:
            pass
        
        # Повертаємо результат з URL-ами
        response_data = {
            'success': True,
            'file1': file1_url,
            'file2': file2_url
        }
        
        print(f"📤 [Separate Audio] Returning separated audio files")
        sys.stdout.flush()
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200
        
    except Exception as e:
        print(f"❌ [Separate Audio] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/api/separate-audio-file/<job_id>/<int:speaker_id>', methods=['GET', 'OPTIONS'])
def get_separate_audio_file(job_id, speaker_id):
    """
    Ендпоїнт для завантаження розділеного аудіо файлу.
    Після завантаження файл автоматично видаляється.
    
    Args:
        job_id: ID завдання розділення
        speaker_id: ID спікера (0 або 1)
    """
    import sys
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    try:
        # Знаходимо файл
        download_dir = os.path.join(UPLOAD_FOLDER, 'separated_audio')
        download_filename = f"{job_id}_speaker_{speaker_id}.wav"
        download_path = os.path.join(download_dir, download_filename)
        
        # Перевіряємо, чи файл існує
        if not os.path.exists(download_path):
            print(f"❌ [Separate Audio Download] File not found: {download_path}")
            sys.stdout.flush()
            return jsonify({
                'success': False,
                'error': f'Audio file for speaker {speaker_id} not found',
                'code': 'FILE_NOT_FOUND'
            }), 404
        
        print(f"📥 [Separate Audio Download] Serving file: {download_path} for job {job_id}, speaker {speaker_id}")
        sys.stdout.flush()
        
        # Відправляємо файл
        response = send_file(
            download_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f"speaker_{speaker_id}.wav"
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        # Видаляємо файл після відправки (в фоні, щоб не блокувати відповідь)
        def delete_file_after_delay():
            import time
            time.sleep(2)  # Затримка, щоб файл точно відправився
            try:
                if os.path.exists(download_path):
                    os.remove(download_path)
                    print(f"🗑️ [Separate Audio Download] Deleted file: {download_path}")
                    sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ [Separate Audio Download] Failed to delete file {download_path}: {e}")
                sys.stdout.flush()
        
        # Запускаємо видалення в окремому потоці
        import threading
        delete_thread = threading.Thread(target=delete_file_after_delay)
        delete_thread.daemon = True
        delete_thread.start()
        
        return response
        
    except Exception as e:
        print(f"❌ [Separate Audio Download] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'DOWNLOAD_ERROR'
        }), 500


@app.route('/api/diarize-and-transcribe', methods=['POST', 'OPTIONS'])
def api_diarize_and_transcribe():
    """
    Ендпоїнт для діаризації та транскрипції аудіо файлу.
    Приймає аудіо файл, виконує діаризацію та транскрипцію,
    повертає транскрипт з розділенням по спікерам.
    
    Returns:
        JSON з полями:
        - success: bool
        - transcript: список рядків у форматі "Таймстемп - Спікер номер - Репліка"
    """
    import sys
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        # Перевіряємо, чи є файл
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided',
                'code': 'NO_FILE'
            }), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        # Отримуємо параметри (опціонально)
        processing_mode = request.form.get('mode', 'fast')  # 'smart' або 'fast'
        transcription_provider = request.form.get('transcription_provider', 'whisper')
        num_speakers = request.form.get('num_speakers', None)
        if num_speakers:
            try:
                num_speakers = int(num_speakers)
            except:
                num_speakers = None
        
        print(f"🎵 [Diarize & Transcribe] Received file: {audio_file.filename}, mode: {processing_mode}, provider: {transcription_provider}")
        sys.stdout.flush()
        
        # Зберігаємо тимчасовий файл
        job_id = str(uuid.uuid4())
        file_extension = os.path.splitext(audio_file.filename)[1] or '.wav'
        temp_filename = f"diarize_{job_id}{file_extension}"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        audio_file.save(temp_path)
        
        print(f"💾 [Diarize & Transcribe] Saved to: {temp_path}")
        sys.stdout.flush()
        
        try:
            # Крок 1: Завантажуємо аудіо
            print(f"📂 [Diarize & Transcribe] Step 1: Loading audio...")
            sys.stdout.flush()
            audio, sr = librosa.load(temp_path, sr=16000, mono=True)
            duration = librosa.get_duration(y=audio, sr=sr)
            print(f"⏱️  [Diarize & Transcribe] Audio duration: {duration:.2f} seconds")
            sys.stdout.flush()
            
            # Обробка залежно від режиму
            if processing_mode == 'smart':
                # Smart mode: Speechmatics (транскрипція + діаризація)
                print(f"🎯 [Diarize & Transcribe] Using Smart mode: Speechmatics")
                sys.stdout.flush()
                
                transcription_text, transcription_segments, words = transcribe_with_speechmatics(temp_path, language='en')
                
                # Speechmatics вже містить діаризацію в words
                # Створюємо сегменти з діаризацією зі слів
                diarization_segments = []
                current_speaker = None
                current_start = None
                current_end = None
                current_text = []
                
                for word in words:
                    word_speaker = word.get('speaker', 0)
                    word_start = word.get('start', 0)
                    word_end = word.get('end', 0)
                    word_text = word.get('word', '')
                    
                    if current_speaker is None:
                        current_speaker = word_speaker
                        current_start = word_start
                        current_text = [word_text]
                    elif word_speaker == current_speaker:
                        current_text.append(word_text)
                    else:
                        # Зберігаємо попередній сегмент
                        if current_start is not None and current_end is not None:
                            diarization_segments.append({
                                'speaker': current_speaker,
                                'start': round(current_start, 2),
                                'end': round(current_end, 2),
                                'text': ' '.join(current_text)
                            })
                        # Починаємо новий сегмент
                        current_speaker = word_speaker
                        current_start = word_start
                        current_text = [word_text]
                    
                    current_end = word_end
                
                # Додаємо останній сегмент
                if current_speaker is not None and current_start is not None and current_end is not None:
                    diarization_segments.append({
                        'speaker': current_speaker,
                        'start': round(current_start, 2),
                        'end': round(current_end, 2),
                        'text': ' '.join(current_text)
                    })
                
                print(f"✅ [Diarize & Transcribe] Speechmatics: Found {len(diarization_segments)} segments")
                sys.stdout.flush()
                
                # Для Smart режиму combined_segments вже містить текст
                combined_segments = diarization_segments
            else:
                # Fast mode: Whisper + PyAnnote
                print(f"⚡ [Diarize & Transcribe] Using Fast mode: Whisper + PyAnnote")
                sys.stdout.flush()
                
                # Крок 2: Виконуємо діаризацію
                print(f"🔍 [Diarize & Transcribe] Step 2: Performing speaker diarization...")
                sys.stdout.flush()
                
                # Використовуємо SpeechBrain для діаризації
                embeddings, timestamps = extract_speaker_embeddings(
                    temp_path,
                    segment_duration=1.5,
                    overlap=0.5
                )
                
                if embeddings is None or len(embeddings) == 0:
                    raise ValueError("Failed to extract speaker embeddings")
                
                # Виконуємо діаризацію
                diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=num_speakers)
                
                if not diarization_segments:
                    raise ValueError("Diarization failed - no segments found")
                
                print(f"✅ [Diarize & Transcribe] Found {len(diarization_segments)} diarization segments")
                sys.stdout.flush()
                
                # Крок 3: Транскрибуємо оригінальне аудіо (жорстко задаємо англійську мову)
                print(f"📝 [Diarize & Transcribe] Step 3: Transcribing audio (language: en)...")
                sys.stdout.flush()
                
                transcription_text, transcription_segments, words = transcribe_audio(
                    temp_path,  # Використовуємо оригінальне аудіо без noise gate
                    language='en',  # Жорстко задаємо англійську мову
                    transcription_provider=transcription_provider
                )
                
                if not words:
                    raise ValueError("Transcription failed - no words found")
                
                print(f"✅ [Diarize & Transcribe] Transcribed {len(words)} words")
                sys.stdout.flush()
                
                # Крок 4: Об'єднуємо діаризацію з транскрипцією (тільки для Fast режиму)
                print(f"🔗 [Diarize & Transcribe] Step 4: Combining diarization with transcription...")
                sys.stdout.flush()
                
                # Використовуємо простий спосіб об'єднання (без LLM для швидкості)
                # ВАЖЛИВО: Відстежуємо використані слова, щоб уникнути дублікатів
                used_word_indices = set()
                combined_segments = []
                
                # Сортуємо сегменти діаризації за часом початку
                sorted_diar_segments = sorted(diarization_segments, key=lambda x: x['start'])
                
                for diar_seg in sorted_diar_segments:
                    # Знаходимо слова, які потрапляють в цей сегмент і ще не використані
                    segment_words = []
                for word_idx, word in enumerate(words):
                    # Пропускаємо вже використані слова
                    if word_idx in used_word_indices:
                        continue
                    
                    word_start = word.get('start', 0)
                    word_end = word.get('end', 0)
                    word_center = (word_start + word_end) / 2.0
                    
                    # Перевіряємо, чи слово потрапляє в сегмент (перевіряємо центр слова)
                    # Використовуємо м'яку умову: якщо центр слова в межах сегменту або слово частково перетинається
                    if (word_center >= diar_seg['start'] and word_center <= diar_seg['end']) or \
                       (word_start < diar_seg['end'] and word_end > diar_seg['start']):
                        segment_words.append((word_idx, word.get('word', '')))
                
                    # Якщо знайшли слова для цього сегменту, додаємо їх
                    if segment_words:
                        text = ' '.join([w[1] for w in segment_words]).strip()
                        if text:
                            # Позначаємо слова як використані
                            for word_idx, _ in segment_words:
                                used_word_indices.add(word_idx)
                            
                            combined_segments.append({
                                'speaker': diar_seg['speaker'],
                                'start': diar_seg['start'],
                                'end': diar_seg['end'],
                                'text': text
                            })
                
                # Додаємо слова, які не потрапили в жоден сегмент діаризації
                # (може статися, якщо діаризація не покриває весь час транскрипції)
                unused_words = []
                for word_idx, word in enumerate(words):
                    if word_idx not in used_word_indices:
                        word_text = word.get('word', '').strip()
                        if word_text:
                            unused_words.append((word_idx, word))
                
                if unused_words:
                    print(f"⚠️  [Diarize & Transcribe] Found {len(unused_words)} words not assigned to any segment, adding them...")
                    sys.stdout.flush()
                    
                    # Групуємо невикористані слова за часом (сегменти по 1 секунді)
                    unused_words_sorted = sorted(unused_words, key=lambda x: x[1].get('start', 0))
                    current_group = []
                    current_start = None
                    
                    for word_idx, word in unused_words_sorted:
                        word_start = word.get('start', 0)
                        
                        if current_start is None:
                            current_start = word_start
                            current_group = [(word_idx, word)]
                        elif word_start - current_start < 1.0:  # Групуємо слова в межах 1 секунди
                            current_group.append((word_idx, word))
                        else:
                            # Зберігаємо поточну групу
                            if current_group:
                                text = ' '.join([w[1].get('word', '') for w in current_group]).strip()
                                if text:
                                    # Визначаємо спікера на основі найближчого сегменту діаризації
                                    speaker = 0
                                    if sorted_diar_segments:
                                        # Знаходимо найближчий сегмент
                                        min_dist = float('inf')
                                        for seg in sorted_diar_segments:
                                            seg_center = (seg['start'] + seg['end']) / 2.0
                                            dist = abs(current_start - seg_center)
                                            if dist < min_dist:
                                                min_dist = dist
                                                speaker = seg['speaker']
                                    
                                    combined_segments.append({
                                        'speaker': speaker,
                                        'start': round(current_start, 2),
                                        'end': round(current_group[-1][1].get('end', current_start), 2),
                                        'text': text
                                    })
                            
                            # Починаємо нову групу
                            current_start = word_start
                            current_group = [(word_idx, word)]
                    
                    # Додаємо останню групу
                    if current_group:
                        text = ' '.join([w[1].get('word', '') for w in current_group]).strip()
                        if text:
                            speaker = 0
                            if sorted_diar_segments:
                                min_dist = float('inf')
                                for seg in sorted_diar_segments:
                                    seg_center = (seg['start'] + seg['end']) / 2.0
                                    dist = abs(current_start - seg_center)
                                    if dist < min_dist:
                                        min_dist = dist
                                        speaker = seg['speaker']
                            
                            combined_segments.append({
                                'speaker': speaker,
                                'start': round(current_start, 2),
                                'end': round(current_group[-1][1].get('end', current_start), 2),
                                'text': text
                            })
                
                # Додаткова перевірка: видаляємо дублікати на основі тексту та часу
                # Але тільки для ідентичних текстів з дуже близьким часом (<1 сек)
                unique_segments = []
                seen_exact = set()
                for seg in combined_segments:
                    text_key = seg['text'].strip().lower()
                    time_key = int(seg['start'])
                    exact_key = (text_key, time_key)
                    
                    # Перевіряємо тільки на точні дублікати (ідентичний текст + той самий час)
                    if exact_key not in seen_exact:
                        unique_segments.append(seg)
                        seen_exact.add(exact_key)
                
                combined_segments = unique_segments
                
                print(f"✅ [Diarize & Transcribe] Combined {len(combined_segments)} segments (after deduplication)")
                sys.stdout.flush()
            
            # Крок 5: Форматуємо результат (для обох режимів)
            print(f"📋 [Diarize & Transcribe] Step 5: Formatting transcript...")
            sys.stdout.flush()
            
            transcript_lines = []
            for seg in combined_segments:
                start_time = seg['start']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                speaker_num = seg['speaker']
                text = seg['text']
                
                transcript_lines.append(f"{timestamp} - Спікер {speaker_num} - {text}")
            
            # Видаляємо тимчасовий файл
            try:
                os.remove(temp_path)
            except:
                pass
            
            # Повертаємо результат
            response_data = {
                'success': True,
                'transcript': transcript_lines
            }
            
            print(f"📤 [Diarize & Transcribe] Returning transcript with {len(transcript_lines)} lines")
            sys.stdout.flush()
            
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 200
            
        except Exception as processing_error:
            # Видаляємо тимчасовий файл при помилці
            try:
                os.remove(temp_path)
            except:
                pass
            raise processing_error
        
    except Exception as e:
        print(f"❌ [Diarize & Transcribe] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'PROCESSING_ERROR'
        }), 500


@app.route('/api/single-speaker-audio/<job_id>/<int:speaker_id>', methods=['GET', 'OPTIONS'])
def get_single_speaker_audio(job_id, speaker_id):
    """
    Ендпоінт для завантаження одноголосого аудіо файлу.
    Після завантаження файл автоматично видаляється.
    
    Args:
        job_id: ID завдання обробки одноголосих файлів
        speaker_id: ID спікера (0, 1, тощо)
    """
    import sys
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, OPTIONS')
        return response
    
    try:
        # Знаходимо файл
        download_dir = os.path.join(UPLOAD_FOLDER, 'single_speaker_audio')
        file_extension = '.wav'  # За замовчуванням
        download_filename = f"{job_id}_speaker_{speaker_id}{file_extension}"
        download_path = os.path.join(download_dir, download_filename)
        
        # Перевіряємо, чи файл існує
        if not os.path.exists(download_path):
            # Спробуємо знайти файл з іншим розширенням
            found = False
            for ext in ['.wav', '.m4a', '.mp3', '.flac']:
                alt_path = os.path.join(download_dir, f"{job_id}_speaker_{speaker_id}{ext}")
                if os.path.exists(alt_path):
                    download_path = alt_path
                    download_filename = f"{job_id}_speaker_{speaker_id}{ext}"
                    found = True
                    break
            
            if not found:
                print(f"❌ [Audio Download] File not found: {download_path}")
                sys.stdout.flush()
                return jsonify({
                    'success': False,
                    'error': f'Audio file for speaker {speaker_id} not found',
                    'code': 'FILE_NOT_FOUND'
                }), 404
        
        print(f"📥 [Audio Download] Serving file: {download_path} for job {job_id}, speaker {speaker_id}")
        sys.stdout.flush()
        
        # Визначаємо MIME type на основі розширення
        mime_types = {
            '.wav': 'audio/wav',
            '.m4a': 'audio/m4a',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac'
        }
        file_ext = os.path.splitext(download_path)[1].lower()
        mime_type = mime_types.get(file_ext, 'audio/wav')
        
        # Відправляємо файл
        response = send_file(
            download_path,
            mimetype=mime_type,
            as_attachment=True,
            download_name=f"speaker_{speaker_id}{file_ext}"
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        
        # Видаляємо файл після відправки (в фоні, щоб не блокувати відповідь)
        def delete_file_after_delay():
            import time
            time.sleep(2)  # Затримка, щоб файл точно відправився
            try:
                if os.path.exists(download_path):
                    os.remove(download_path)
                    print(f"🗑️ [Audio Download] Deleted file: {download_path}")
                    sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ [Audio Download] Failed to delete file {download_path}: {e}")
                sys.stdout.flush()
        
        # Запускаємо видалення в окремому потоці
        import threading
        delete_thread = threading.Thread(target=delete_file_after_delay)
        delete_thread.daemon = True
        delete_thread.start()
        
        return response
        
    except Exception as e:
        print(f"❌ [Audio Download] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'DOWNLOAD_ERROR'
        }), 500


def enhance_main_speaker_audio(audio_path, suppression_factor=0.1, num_speakers=None, llm_mode='local', transcription_provider='whisper'):
    """
    Виділяє основного спікера в аудіо, приглушуючи інших спікерів.
    
    Args:
        audio_path: шлях до вхідного аудіофайлу
        suppression_factor: коефіцієнт приглушення (0.0 = повне видалення, 1.0 = без змін)
        num_speakers: кількість спікерів (None = автоматичне визначення)
        llm_mode: Режим LLM для виправлення призначень спікерів ('local', 'fast', 'smart', 'smart-2')
        suppression_factor: коефіцієнт приглушення для неосновних спікерів (0.0-1.0, де 0.0 = повне видалення, 1.0 = без змін)
        num_speakers: кількість спікерів (None для автоматичного визначення)
    
    Returns:
        output_path: шлях до обробленого аудіофайлу
        main_speaker: ID основного спікера
        segments_info: інформація про сегменти діаризації
    """
    import sys
    import shutil
    
    print(f"🎯 Starting main speaker enhancement for: {audio_path}")
    sys.stdout.flush()
    
    try:
        # Крок 1: Завантажуємо аудіо
        print(f"📂 Step 1: Loading audio...")
        sys.stdout.flush()
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)
        print(f"⏱️  Audio duration: {duration:.2f} seconds, sample rate: {sr} Hz")
        sys.stdout.flush()
        
        # Крок 2: Виконуємо діаризацію
        print(f"🔍 Step 2: Performing speaker diarization...")
        sys.stdout.flush()
        
        # Спробуємо використати pyannote для більш точної діаризації (якщо доступна)
        diarization_segments = None
        use_pyannote = os.getenv('HUGGINGFACE_TOKEN') is not None
        
        if use_pyannote:
            try:
                print(f"🎯 Attempting to use PyAnnote for more accurate diarization...")
                sys.stdout.flush()
                
                # Використовуємо той самий підхід, що і в pyannote_separation.py
                try:
                    import pyannote_patch  # noqa: F401
                    from pyannote.audio import Pipeline
                    import torch
                    import torchaudio
                except ImportError:
                    raise ImportError("pyannote.audio not available")
                
                hf_token = os.getenv('HUGGINGFACE_TOKEN')
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                print(f"📦 Loading PyAnnote speaker-diarization-3.1 pipeline...")
                sys.stdout.flush()
                
                try:
                    # Використовуємо той самий підхід, що і в pyannote_separation.py
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
                    
                    if pipeline is None:
                        raise ValueError("Pipeline is None after loading")
                    pipeline.to(device)
                except Exception as load_error:
                    print(f"⚠️  Failed to load PyAnnote pipeline: {load_error}")
                    # Викидаємо помилку, щоб використати SpeechBrain як fallback
                    raise
                
                print(f"✅ PyAnnote pipeline loaded, running diarization on: {audio_path}")
                sys.stdout.flush()
                
                # Завантажуємо аудіо так само, як в pyannote_separation.py
                # soundfile вже імпортовано на початку файлу
                try:
                    data, sample_rate = sf.read(audio_path, dtype='float32')
                    if len(data.shape) == 1:
                        waveform = torch.from_numpy(data).unsqueeze(0).float()
                    else:
                        waveform = torch.from_numpy(data).transpose(0, 1).float()
                except Exception as load_error:
                    # Fallback до torchaudio якщо soundfile не працює
                    print(f"⚠️  soundfile failed: {load_error}, trying torchaudio...")
                    sys.stdout.flush()
                    waveform, sample_rate = torchaudio.load(audio_path)
                
                # Конвертуємо в mono якщо потрібно
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample до 16kHz якщо потрібно
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000
                
                # Запускаємо діаризацію
                diarization = pipeline({
                    "waveform": waveform,
                    "sample_rate": sample_rate
                })
                
                # Конвертуємо результат pyannote в наш формат
                diarization_segments = []
                speaker_map = {}  # Мапінг pyannote labels до числових ID
                next_speaker_id = 0
                
                # Спочатку збираємо всі унікальні спікерів
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker not in speaker_map:
                        speaker_map[speaker] = next_speaker_id
                        next_speaker_id += 1
                
                # Тепер створюємо сегменти
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
                sys.stdout.flush()
                
            except Exception as e:
                print(f"⚠️  PyAnnote diarization failed: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                diarization_segments = None
        
        # Якщо pyannote не спрацювала, використовуємо SpeechBrain
        if diarization_segments is None:
            print(f"📊 Using SpeechBrain diarization...")
            sys.stdout.flush()
            
            # Витягуємо ембеддинги
            embeddings, timestamps = extract_speaker_embeddings(
                audio_path,
                segment_duration=1.5,
                overlap=0.5
            )
            
            if embeddings is None or len(embeddings) == 0:
                raise ValueError("Failed to extract speaker embeddings")
            
            # Виконуємо діаризацію
            diarization_segments = diarize_audio(embeddings, timestamps, num_speakers=num_speakers)
            
            if not diarization_segments:
                raise ValueError("Diarization failed - no segments found")
        
        print(f"✅ Found {len(diarization_segments)} diarization segments")
        sys.stdout.flush()
        
        # Крок 3: Транскрибуємо аудіо та об'єднуємо з діаризацією
        # КРИТИЧНО: Спочатку об'єднуємо, потім визначаємо основного спікера на основі об'єднаної транскрипції
        print(f"📝 Step 3: Transcribing and combining with diarization...")
        sys.stdout.flush()
        
        transcription_text, transcription_segments, words = transcribe_audio(audio_path, language=None, transcription_provider=transcription_provider)
        
        # Об'єднуємо діаризацію з транскрипцією
        combined_segments = combine_diarization_and_transcription(diarization_segments, words, llm_mode=llm_mode)
        
        print(f"✅ Combined {len(combined_segments)} segments from transcription and diarization")
        sys.stdout.flush()
        
        # Крок 4: Визначаємо основного спікера на основі ОБ'ЄДНАНОЇ транскрипції
        print(f"👤 Step 4: Determining main speaker from combined transcription...")
        sys.stdout.flush()
        
        # Використовуємо уніфіковану функцію для визначення основного спікера
        main_speaker, speaker_stats = determine_main_speaker_from_segments(combined_segments, duration=duration)
        
        # Отримуємо статистику для подальшого використання
        speaker_durations = {spk: stats['duration'] for spk, stats in speaker_stats.items()}
        main_duration = speaker_stats[main_speaker]['duration'] if main_speaker in speaker_stats else 0
        
        # Крок 5: Створюємо маску для аудіо на основі ОБ'ЄДНАНОЇ транскрипції
        print(f"🎚️  Step 5: Creating audio mask (suppression factor: {suppression_factor})...")
        sys.stdout.flush()
        
        num_samples = len(audio)
        
        # Діагностика: перевіряємо наявність combined_segments
        if not combined_segments:
            print(f"⚠️  WARNING: combined_segments is empty! Cannot create mask.")
            sys.stdout.flush()
            # Якщо немає сегментів, створюємо маску без змін (1.0 для всього)
            mask = np.ones(num_samples, dtype=np.float32)
            enhanced_audio = audio * mask
            print(f"⚠️  No mask applied - using original audio")
            sys.stdout.flush()
        else:
            print(f"📊 Using {len(combined_segments)} segments for mask creation, main_speaker={main_speaker}")
            sys.stdout.flush()
            
            # Якщо suppression_factor = 0, повністю видаляємо звук інших спікерів через маску
            if suppression_factor == 0.0:
                print(f"🔇 Suppression factor is 0.0 - completely removing other speakers using COMBINED transcription timestamps...")
                sys.stdout.flush()
                
                # Створюємо маску: 1.0 для основного спікера, 0.0 для інших
                mask = np.zeros(num_samples, dtype=np.float32)
                
                # Застосовуємо маску для кожного сегмента з ОБ'ЄДНАНОЇ транскрипції
                main_speaker_segments_count = 0
                other_speaker_segments_count = 0
                
                for seg_idx, seg in enumerate(combined_segments):
                    speaker = seg['speaker']
                    start_time = seg['start']
                    end_time = seg['end']
                    text = seg.get('text', '')[:50]
                    
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    # Обмежуємо межі масиву
                    start_sample = max(0, min(start_sample, num_samples))
                    end_sample = max(0, min(end_sample, num_samples))
                    
                    if start_sample < end_sample:
                        if speaker == main_speaker:
                            # Основний спікер - залишаємо звук (1.0)
                            mask[start_sample:end_sample] = 1.0
                            main_speaker_segments_count += 1
                        else:
                            # Інші спікери - повністю видаляємо звук (0.0)
                            mask[start_sample:end_sample] = 0.0
                            other_speaker_segments_count += 1
                            # Детальне логування для неосновних спікерів
                            print(f"   🔇 [Mask] Segment {seg_idx}: Removing speaker {speaker} "
                                  f"({start_time:.2f}-{end_time:.2f}s, {end_time-start_time:.2f}s): '{text}...'")
                            sys.stdout.flush()
                
                print(f"📊 Mask created: {main_speaker_segments_count} segments of main speaker (kept), {other_speaker_segments_count} segments of other speakers (removed)")
                sys.stdout.flush()
                
                # Застосовуємо маску до аудіо
                # Перевіряємо розміри: якщо аудіо 2D (stereo), маска має бути 2D теж
                print(f"🔍 [Mask Debug] audio shape: {audio.shape}, mask shape: {mask.shape}")
                sys.stdout.flush()
                
                if len(audio.shape) == 2:
                    # Stereo аудіо - маска має бути 2D
                    mask_2d = mask[:, np.newaxis]  # Додаємо вимір для каналів
                    enhanced_audio = audio * mask_2d
                    print(f"🔍 [Mask Debug] Applied 2D mask to stereo audio")
                else:
                    # Mono аудіо - маска 1D
                    enhanced_audio = audio * mask
                    print(f"🔍 [Mask Debug] Applied 1D mask to mono audio")
                sys.stdout.flush()
                
                # Перевіряємо, чи маска дійсно застосувалася
                max_audio_before = np.max(np.abs(audio))
                max_audio_after = np.max(np.abs(enhanced_audio))
                print(f"🔍 [Mask Debug] Max audio before mask: {max_audio_before:.6f}, after mask: {max_audio_after:.6f}")
                sys.stdout.flush()
                
                # Обчислюємо статистику
                main_speaker_duration_samples = np.sum(mask > 0)
                main_speaker_duration = main_speaker_duration_samples / sr
                print(f"✅ Applied mask: main speaker audio kept ({main_speaker_duration:.2f}s), other speakers completely removed")
                sys.stdout.flush()
            else:
                # Створюємо масив масок (1.0 для основного спікера, suppression_factor для інших)
                # ВАЖЛИВО: Ініціалізуємо як suppression_factor, щоб проміжки між сегментами теж були приглушені
                mask = np.full(num_samples, suppression_factor, dtype=np.float32)
                
                # Застосовуємо маску для кожного сегмента з ОБ'ЄДНАНОЇ транскрипції
                main_speaker_segments_count = 0
                other_speaker_segments_count = 0
                
                for seg_idx, seg in enumerate(combined_segments):
                    speaker = seg['speaker']
                    start_time = seg['start']
                    end_time = seg['end']
                    text = seg.get('text', '')[:50]
                    
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    # Обмежуємо межі масиву
                    start_sample = max(0, min(start_sample, num_samples))
                    end_sample = max(0, min(end_sample, num_samples))
                    
                    if start_sample < end_sample:
                        if speaker == main_speaker:
                            # Основний спікер - залишаємо без змін (1.0)
                            mask[start_sample:end_sample] = 1.0
                            main_speaker_segments_count += 1
                        else:
                            # Неосновний спікер - приглушуємо
                            mask[start_sample:end_sample] = suppression_factor
                            other_speaker_segments_count += 1
                            # Детальне логування для неосновних спікерів
                            print(f"   🔇 [Mask] Segment {seg_idx}: Suppressing speaker {speaker} "
                                  f"({start_time:.2f}-{end_time:.2f}s, {end_time-start_time:.2f}s): '{text}...'")
                            sys.stdout.flush()
                
                print(f"📊 Mask created: {main_speaker_segments_count} segments of main speaker (kept at 1.0), {other_speaker_segments_count} segments of other speakers (suppressed to {suppression_factor})")
                
                # Перевіряємо, чи маска дійсно застосувалася до сегментів неосновного спікера
                for seg in combined_segments:
                    if seg['speaker'] != main_speaker:
                        start_sample = int(seg['start'] * sr)
                        end_sample = int(seg['end'] * sr)
                        start_sample = max(0, min(start_sample, num_samples))
                        end_sample = max(0, min(end_sample, num_samples))
                        if start_sample < end_sample:
                            mask_values = mask[start_sample:end_sample]
                            avg_mask_value = np.mean(mask_values)
                            if abs(avg_mask_value - suppression_factor) > 0.01:
                                print(f"   ⚠️ [Mask Check] Segment '{seg.get('text', '')[:50]}...' "
                                      f"({seg['start']:.2f}-{seg['end']:.2f}s): "
                                      f"expected mask={suppression_factor}, actual={avg_mask_value:.3f}")
                            else:
                                print(f"   ✅ [Mask Check] Segment '{seg.get('text', '')[:50]}...' "
                                      f"({seg['start']:.2f}-{seg['end']:.2f}s): "
                                      f"mask correctly applied={avg_mask_value:.3f}")
                sys.stdout.flush()
                
                # Застосовуємо маску до аудіо
                # Перевіряємо розміри: якщо аудіо 2D (stereo), маска має бути 2D теж
                print(f"🔍 [Mask Debug] audio shape: {audio.shape}, mask shape: {mask.shape}")
                sys.stdout.flush()
                
                if len(audio.shape) == 2:
                    # Stereo аудіо - маска має бути 2D
                    mask_2d = mask[:, np.newaxis]  # Додаємо вимір для каналів
                    enhanced_audio = audio * mask_2d
                    print(f"🔍 [Mask Debug] Applied 2D mask to stereo audio")
                else:
                    # Mono аудіо - маска 1D
                    enhanced_audio = audio * mask
                    print(f"🔍 [Mask Debug] Applied 1D mask to mono audio")
                sys.stdout.flush()
                
                # Перевіряємо, чи маска дійсно застосувалася
                max_audio_before = np.max(np.abs(audio))
                max_audio_after = np.max(np.abs(enhanced_audio))
                print(f"🔍 [Mask Debug] Max audio before mask: {max_audio_before:.6f}, after mask: {max_audio_after:.6f}")
                
                # Перевіряємо конкретні сегменти спікера 1 - чи вони дійсно приглушені
                print(f"🔍 [Audio Level Check] Checking audio levels for non-main speaker segments...")
                for seg in combined_segments:
                    if seg['speaker'] != main_speaker:
                        start_sample = int(seg['start'] * sr)
                        end_sample = int(seg['end'] * sr)
                        start_sample = max(0, min(start_sample, num_samples))
                        end_sample = max(0, min(end_sample, num_samples))
                        if start_sample < end_sample:
                            # Порівнюємо аудіо до та після маски для цього сегмента
                            audio_before_seg = np.max(np.abs(audio[start_sample:end_sample]))
                            audio_after_seg = np.max(np.abs(enhanced_audio[start_sample:end_sample]))
                            expected_after = audio_before_seg * suppression_factor
                            ratio = audio_after_seg / audio_before_seg if audio_before_seg > 0 else 0
                            print(f"   🔊 [Audio Check] Segment '{seg.get('text', '')[:40]}...' "
                                  f"({seg['start']:.2f}-{seg['end']:.2f}s): "
                                  f"before={audio_before_seg:.6f}, after={audio_after_seg:.6f}, "
                                  f"ratio={ratio:.3f} (expected ~{suppression_factor:.3f})")
                            if abs(ratio - suppression_factor) > 0.05:
                                print(f"      ⚠️ WARNING: Audio not properly suppressed! Ratio should be ~{suppression_factor:.3f}")
                sys.stdout.flush()
                
                # Обчислюємо статистику
                main_speaker_duration_samples = np.sum(mask == 1.0)
                main_speaker_duration = main_speaker_duration_samples / sr
                suppressed_duration_samples = np.sum((mask > 0) & (mask < 1.0))
                suppressed_duration = suppressed_duration_samples / sr
                print(f"✅ Applied mask: main speaker audio kept ({main_speaker_duration:.2f}s), other speakers suppressed ({suppressed_duration:.2f}s at {suppression_factor*100:.0f}% volume)")
                sys.stdout.flush()
        
        # Крок 6: Зберігаємо оброблений файл
        print(f"💾 Step 6: Saving enhanced audio...")
        sys.stdout.flush()
        
        # Створюємо вихідний файл
        output_dir = os.path.join(UPLOAD_FOLDER, 'enhanced')
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_main_speaker.wav")
        
        # Перевіряємо, чи enhanced_audio визначено
        if 'enhanced_audio' not in locals():
            print(f"❌ ERROR: enhanced_audio is not defined! Using original audio.")
            sys.stdout.flush()
            enhanced_audio = audio
        
        # Перевіряємо розміри перед збереженням
        print(f"🔍 [Save Debug] enhanced_audio shape: {enhanced_audio.shape}, dtype: {enhanced_audio.dtype}")
        print(f"🔍 [Save Debug] Max value in enhanced_audio: {np.max(np.abs(enhanced_audio)):.6f}")
        
        # Перевіряємо, чи enhanced_audio дійсно відрізняється від оригінального аудіо
        audio_diff = np.max(np.abs(audio - enhanced_audio))
        print(f"🔍 [Save Debug] Max difference between original and enhanced audio: {audio_diff:.6f}")
        if audio_diff < 0.001:
            print(f"⚠️ [Save Debug] WARNING: Enhanced audio is almost identical to original! Mask might not be applied correctly.")
        else:
            print(f"✅ [Save Debug] Enhanced audio differs from original (difference: {audio_diff:.6f})")
        sys.stdout.flush()
        
        # Зберігаємо оброблений аудіо
        sf.write(output_path, enhanced_audio, sr)
        
        # Перевіряємо, чи файл дійсно збережено
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Enhanced audio saved to: {output_path} (size: {file_size} bytes)")
            
            # ДОДАТКОВА ПЕРЕВІРКА: Читаємо збережений файл і порівнюємо з enhanced_audio
            try:
                loaded_audio, loaded_sr = sf.read(output_path)
                print(f"🔍 [File Verify] Loaded file: shape={loaded_audio.shape}, sr={loaded_sr}, max={np.max(np.abs(loaded_audio)):.6f}")
                
                # Порівнюємо з enhanced_audio
                if loaded_audio.shape == enhanced_audio.shape:
                    diff = np.max(np.abs(loaded_audio - enhanced_audio))
                    print(f"🔍 [File Verify] Difference between saved and enhanced_audio: {diff:.6f}")
                    if diff > 0.001:
                        print(f"⚠️ [File Verify] WARNING: Saved file differs from enhanced_audio!")
                    else:
                        print(f"✅ [File Verify] Saved file matches enhanced_audio")
                else:
                    print(f"⚠️ [File Verify] WARNING: Shape mismatch! saved={loaded_audio.shape}, enhanced={enhanced_audio.shape}")
            except Exception as e:
                print(f"⚠️ [File Verify] Could not verify saved file: {e}")
        else:
            print(f"❌ ERROR: File was not saved! Path: {output_path}")
        sys.stdout.flush()
        
        # Крок 7: Використовуємо вже об'єднану транскрипцію для відображення
        print(f"📝 Step 7: Using combined transcription for display...")
        sys.stdout.flush()
        
        # combined_segments вже створені на кроці 3, не потрібно повторно транскрибувати
        
        # Створюємо дані для візуалізації маски
        mask_data = []
        for seg in sorted(diarization_segments, key=lambda x: x['start']):
            mask_data.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'is_main_speaker': seg['speaker'] == main_speaker,
                'suppression_applied': suppression_factor if seg['speaker'] != main_speaker else 1.0
            })
        
        segments_info = {
            'total_segments': len(diarization_segments),
            'main_speaker': main_speaker,
            'main_speaker_duration': main_duration,
            'main_speaker_percentage': main_duration / duration * 100,
            'all_speakers': {speaker: dur for speaker, dur in speaker_durations.items()},
            'transcription': transcription_text,
            'transcription_segments': combined_segments,
            'mask_data': mask_data,
            'audio_duration': duration
        }
        
        return output_path, main_speaker, segments_info
        
    except Exception as e:
        print(f"❌ Error in enhance_main_speaker_audio: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise


@app.route('/api/enhance-main-speaker', methods=['POST', 'OPTIONS'])
def api_enhance_main_speaker():
    """
    Ендпоїнт для виділення основного спікера в аудіо.
    Приймає аудіофайл, виконує діаризацію, визначає основного спікера
    і повертає оброблений аудіофайл з приглушеними неосновними спікерами.
    
    Параметри (multipart/form-data):
    - file: аудіофайл (обов'язково)
    - num_speakers: кількість спікерів (опціонально, за замовчуванням автоматично)
    - suppression_factor: коефіцієнт приглушення 0.0-1.0 (опціонально, за замовчуванням 0.1)
    
    Returns:
    - Оброблений аудіофайл (WAV) або JSON з помилкою
    """
    import sys
    
    print(f"🔵 [API] /api/enhance-main-speaker called - Method: {request.method}, Remote: {request.remote_addr}")
    sys.stdout.flush()
    
    # Обробка OPTIONS для preflight запитів (CORS)
    if request.method == 'OPTIONS':
        print("✅ OPTIONS preflight request received from", request.remote_addr)
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        sys.stdout.flush()
        return response
    
    print(f"📥 POST /api/enhance-main-speaker request received from {request.remote_addr}")
    sys.stdout.flush()
    
    filepath = None
    
    try:
        # Перевіряємо наявність файлу
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided. Send file in "file" field.',
                'code': 'NO_FILE'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected.',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid audio format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}',
                'code': 'INVALID_FORMAT'
            }), 400
        
        # Отримуємо параметри
        num_speakers = request.form.get('num_speakers')
        if num_speakers:
            try:
                num_speakers = int(num_speakers)
            except ValueError:
                num_speakers = None
        else:
            num_speakers = None
        
        suppression_factor = request.form.get('suppression_factor', '0.1')
        try:
            suppression_factor = float(suppression_factor)
            # Обмежуємо значення від 0.0 до 1.0
            suppression_factor = max(0.0, min(1.0, suppression_factor))
        except ValueError:
            suppression_factor = 0.1
        
        # Отримуємо режим LLM (fast, smart, smart-2, local)
        llm_mode = request.form.get('llm_mode', 'local')
        # Нормалізуємо режим (smart-2 -> smart-2, smart2 -> smart-2)
        if llm_mode == 'smart2':
            llm_mode = 'smart-2'
        # Валідація режиму
        valid_modes = ['local', 'fast', 'smart', 'smart-2', 'test', 'test2']
        if llm_mode not in valid_modes:
            llm_mode = 'local'
        
        # Отримуємо провайдера транскрипції (whisper, azure, speechmatics)
        transcription_provider = request.form.get('transcription_provider', 'whisper')
        # Валідація провайдера
        valid_providers = ['whisper', 'azure', 'speechmatics']
        if transcription_provider not in valid_providers:
            transcription_provider = 'whisper'
        
        print(f"📋 Parameters: num_speakers={num_speakers}, suppression_factor={suppression_factor}, llm_mode={llm_mode}, transcription_provider={transcription_provider}")
        sys.stdout.flush()
        
        # Зберігаємо завантажений файл
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        print(f"💾 File saved to: {filepath}")
        sys.stdout.flush()
        
        # Обробляємо аудіо
        output_path, main_speaker, segments_info = enhance_main_speaker_audio(
            filepath,
            suppression_factor=suppression_factor,
            num_speakers=num_speakers,
            llm_mode=llm_mode,
            transcription_provider=transcription_provider
        )
        
        # Перевіряємо, чи потрібно повернути JSON з метаданими
        return_json = request.form.get('return_json', 'false').lower() == 'true'
        
        if return_json:
            # Повертаємо JSON з метаданими та URL файлу
            import base64
            
            # Читаємо файл та кодуємо в base64 для передачі
            print(f"📂 [File Return] Reading file for client: {output_path}")
            if not os.path.exists(output_path):
                print(f"❌ [File Return] ERROR: Output file does not exist: {output_path}")
                raise FileNotFoundError(f"Output file not found: {output_path}")
            
            file_size = os.path.getsize(output_path)
            print(f"📂 [File Return] File exists, size: {file_size} bytes")
            
            with open(output_path, 'rb') as f:
                audio_data = f.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            print(f"📂 [File Return] File read successfully, base64 length: {len(audio_base64)} chars")
            sys.stdout.flush()
            
            response_data = {
                'success': True,
                'audio_file_base64': audio_base64,
                'audio_filename': f"enhanced_main_speaker_{os.path.basename(filepath)}",
                'main_speaker': main_speaker,
                'segments_info': segments_info
            }
            
            response = jsonify(response_data)
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        else:
            # Повертаємо файл (legacy режим)
            print(f"📤 Sending enhanced audio file: {output_path}")
            sys.stdout.flush()
            
            response = send_file(
                output_path,
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f"enhanced_main_speaker_{os.path.basename(filepath)}"
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('X-Main-Speaker', str(main_speaker))
            response.headers.add('X-Segments-Count', str(segments_info['total_segments']))
            response.headers.add('X-Main-Speaker-Duration', f"{segments_info['main_speaker_duration']:.2f}")
            response.headers.add('X-Main-Speaker-Percentage', f"{segments_info['main_speaker_percentage']:.1f}")
            
            return response
        
    except ValueError as e:
        error_msg = str(e)
        print(f"❌ ValueError: {error_msg}")
        sys.stdout.flush()
        response = jsonify({
            'success': False,
            'error': error_msg,
            'code': 'PROCESSING_ERROR'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in /api/enhance-main-speaker: {error_msg}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        response = jsonify({
            'success': False,
            'error': f'Processing failed: {error_msg}',
            'code': 'INTERNAL_ERROR'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
        
    finally:
        # Очищаємо тимчасові файли (залишаємо оброблений файл для завантаження)
        try:
            if filepath and os.path.exists(filepath):
                # Видаляємо оригінальний завантажений файл через деякий час
                # (не одразу, щоб не видалити під час відправки)
                pass  # Можна додати очищення пізніше
        except Exception as e:
            print(f"⚠️ Could not clean up temp files: {e}")


if __name__ == '__main__':
    port = int(os.environ.get('IOS_SHORTCUTS_PORT', 5005))
    print(f"🚀 Starting Flask server for iOS Shortcuts on port {port}")
    print(f"📂 Upload folder: {UPLOAD_FOLDER}")
    print(f"🌐 Server will be accessible at: http://0.0.0.0:{port}")
    print(f"📱 Use your Mac's IP address for iOS Shortcuts")
    app.run(host='0.0.0.0', port=port, debug=False)

