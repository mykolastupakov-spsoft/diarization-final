# Аналіз проблеми з об'єднанням сегментів (Overlap Merge)

## Контекст

Система діаризації аудіо працює в два етапи:
1. **Primary Diarization** - аналіз оригінального аудіо з перекриваючими голосами (Speechmatics/Azure STT)
2. **Voice Track Separation** - розділення аудіо на окремі доріжки для кожного спікера (SpeechBrain/PyAnnote)
3. **Voice Track Transcription** - транскрипція кожної окремої доріжки (Speechmatics/Azure STT)
4. **Merge** - об'єднання результатів з primary та voice tracks в один коректний JSON

**Проблема**: Після об'єднання залишаються дублікати - одна й та сама репліка з'являється кілька разів з різними таймстемпами або з незначними варіаціями тексту.

---

## Архітектура поточної реалізації

### 1. Джерела даних

#### Primary Diarization (`primarySegments`)
- Результат транскрипції оригінального аудіо з перекриваючими голосами
- Може містити неповні або неточні транскрипції через перекриття
- Приклад:
```json
{
  "speaker": "SPEAKER_00",
  "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health...",
  "start": 0.69,
  "end": 5.24
}
```

#### Voice Tracks (`voiceTrackSegments`)
- Результат транскрипції окремих доріжок (кожна доріжка = один спікер)
- Містить повні та точні транскрипції, оскільки аудіо вже розділено
- Проблема: Azure STT може виявити кілька спікерів навіть в одній доріжці (через залишковий шум)
- Приклад:
```json
{
  "speaker": "SPEAKER_00",  // Примусово встановлено з track.speaker для Azure
  "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm your appointment with our doctor. Are you available for a short conversation right now?",
  "start": 0.66,
  "end": 8.31,
  "source": "voice-track"
}
```

### 2. Процес об'єднання (merge)

#### Крок 1: `collectVoiceTrackSegments()`
**Файл**: `overlap_merge_utils.js:45-220`

**Для Azure STT**:
- Примусово встановлює `speaker = track.speaker` для всіх сегментів
- Ігнорує внутрішню діаризацію Azure (яка може виявити кілька спікерів в одній доріжці)

**Для Speechmatics**:
- Групує сегменти за виявленим спікером
- Визначає головного спікера (найбільше сегментів/тривалість)
- Мапить інших виявлених спікерів на правильні мітки через primary segments
- Фільтрує залишковий аудіо (короткі сегменти без overlap з primary)

**Проблема**: Навіть після нормалізації, один voice track може дати кілька сегментів з однаковим спікером, але різними таймстемпами.

#### Крок 2: Дедuплікація voice tracks
**Файл**: `overlap_merge_utils.js:246-301`

**Логіка**:
```javascript
// Для кожного voice track сегменту:
// 1. Перевіряє overlap з вже доданими сегментами (того ж спікера)
// 2. Якщо overlap > 80% або (text similarity + overlap > 0.1s) → дублікат
// 3. Залишає сегмент з довшим текстом
```

**Проблеми**:
- Поріг 80% overlap може пропустити дублікати з overlap 70-79%
- Text similarity (`includes()`) занадто простий - не ловить варіації типу "Hi, I'm Jessica" vs "Hi I'm Jessica"
- Не враховує випадки, коли один сегмент містить інший, але з іншими словами на початку/кінці

#### Крок 3: Додавання primary segments
**Файл**: `overlap_merge_utils.js:411-448`

**Логіка**:
```javascript
// Для кожного primary сегменту:
// 1. Валідує короткі сегменти (< 0.8s) через validateShortSegment()
// 2. Перевіряє overlap з voice tracks (того ж спікера)
// 3. Додає тільки якщо НЕ перекривається з voice track
```

**Проблеми**:
- Перевірка overlap: `overlapDuration > Math.max(0.2, pDuration * 0.5)` або `textSimilarity && overlapDuration > 0.1`
- Якщо voice track сегмент коротший за primary, але перекривається на 40% - primary все одно додається
- Text similarity знову використовує простий `includes()`, який не ловить варіації

#### Крок 4: Об'єднання послідовних сегментів
**Файл**: `overlap_merge_utils.js:456-549`

**Логіка**:
- Об'єднує сегменти одного спікера, якщо gap < 2s або small overlap < 0.5s
- Перевіряє, чи немає інших спікерів між сегментами

**Проблеми**:
- Може об'єднати сегменти, які насправді є дублікатами (різні транскрипції одного фрагменту)
- Не перевіряє text similarity перед об'єднанням

#### Крок 5: Хронологічна валідація
**Файл**: `overlap_merge_utils.js:553-607`

**Логіка**:
- Видаляє сегменти, які перекриваються з попередніми (того ж спікера)
- Об'єднує continuation segments (починається до кінця попереднього, але продовжується далі)

**Проблеми**:
- Поріг overlap 0.1s може пропустити дублікати з overlap 0.05-0.09s
- Не перевіряє text similarity - може видалити різні репліки, які випадково перекриваються

#### Крок 6: Фінальна дедуплікація
**Файл**: `overlap_merge_utils.js:622-670`

**Логіка**:
```javascript
// Для кожного сегменту:
// 1. Перевіряє overlap з вже доданими (того ж спікера)
// 2. Якщо coverage > 60% або (similarText && overlap > 0.2s) → замінює або пропускає
// 3. Використовує priority score (voice-track > primary)
```

**Проблеми**:
- Поріг 60% coverage може пропустити дублікати з 50-59% overlap
- `areTextsSimilar()` використовує простий `includes()` з порогом 0.72, що не ловить варіації
- Priority score може зберегти гірший сегмент (наприклад, коротший primary замість довшого voice track)

---

## Конкретні проблемні кейси

### Кейс 1: Дублікат через різні таймстемпи

**Primary segment**:
```json
{
  "speaker": "SPEAKER_00",
  "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm your appointment with our doctor.",
  "start": 0.69,
  "end": 5.24
}
```

**Voice track segment**:
```json
{
  "speaker": "SPEAKER_00",
  "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm your appointment with our doctor. Are you available for a short conversation right now?",
  "start": 0.66,
  "end": 8.31
}
```

**Проблема**: 
- Voice track містить primary + додатковий текст
- Overlap: 5.24 - 0.69 = 4.55s (100% primary, ~55% voice track)
- Поточна логіка: voice track не перекривається з primary на 50%+ (4.55/8.31 = 54.8% < порогу для voice track)
- Результат: ОБА сегменти залишаються

### Кейс 2: Дублікат через варіації тексту

**Voice track segment 1**:
```json
{
  "speaker": "SPEAKER_01",
  "text": "Yeah, Jessica, I do. Available. What do you have for me?",
  "start": 9.33,
  "end": 13.17
}
```

**Voice track segment 2** (з іншого voice track або primary):
```json
{
  "speaker": "SPEAKER_01",
  "text": "Yeah Jessica I do available What do you have for me",
  "start": 9.39,
  "end": 13.03
}
```

**Проблема**:
- Тексти майже ідентичні (різниця тільки в пунктуації/пробілах)
- Overlap: ~3.7s (майже 100% обох сегментів)
- `areTextsSimilar()` використовує `includes()`, який може не спрацювати через різницю в пунктуації
- Результат: ОБА сегменти залишаються

### Кейс 3: Дублікат через частковий overlap

**Primary segment**:
```json
{
  "speaker": "SPEAKER_00",
  "text": "Great, to ensure I have the correct information for you, could you please confirm your date of birth for me please?",
  "start": 15.94,
  "end": 22.92
}
```

**Voice track segment**:
```json
{
  "speaker": "SPEAKER_00",
  "text": "Great, to ensure I have the correct information for you, could you please confirm your date of birth for me please?",
  "start": 15.93,
  "end": 23.13
}
```

**Проблема**:
- Тексти ідентичні
- Overlap: ~6.98s (100% primary, ~99% voice track)
- Але перевірка overlap в кроці 3: `overlapDuration > Math.max(0.2, pDuration * 0.5)` = `6.98 > 3.5` ✅
- Однак, якщо voice track вже додано в кроці 2, primary не повинен додаватися
- Можлива проблема: voice track не був правильно дедуплікований в кроці 2

### Кейс 4: Дублікат через об'єднання послідовних сегментів

**Сегмент 1** (voice track):
```json
{
  "speaker": "SPEAKER_01",
  "text": "I want to reschedule this one.",
  "start": 101.31,
  "end": 103.87
}
```

**Сегмент 2** (primary):
```json
{
  "speaker": "SPEAKER_01",
  "text": "I want to reschedule this one.",
  "start": 101.04,
  "end": 103.52
}
```

**Проблема**:
- Обидва сегменти мають однаковий текст
- Overlap: ~2.4s (майже 100%)
- В кроці 4 (об'єднання послідовних): gap = 101.31 - 103.52 = -2.21s (overlap!)
- Але smallOverlap = true (2.21s < 0.5s? НІ, це більше 0.5s)
- Може не об'єднатися, якщо перевірка overlap не спрацює правильно
- Результат: ОБА сегменти залишаються

---

## Кореневі причини проблем

### 1. Недостатня дедуплікація на ранніх етапах
- Крок 2 (дедуплікація voice tracks) використовує занадто високі пороги (80% overlap)
- Не враховує випадки, коли один сегмент містить інший (substring matching)

### 2. Слабка text similarity функція
- `areTextsSimilar()` використовує простий `includes()` з порогом 0.72
- Не нормалізує текст (пунктуація, пробіли, регістр)
- Не використовує більш складні метрики (Levenshtein, Jaccard, token-based similarity)

### 3. Конфліктуюча логіка між кроками
- Крок 3 додає primary segments, якщо overlap < 50%
- Крок 6 видаляє сегменти, якщо overlap > 60%
- Між 50-60% overlap - "сіра зона", де дублікати можуть залишитися

### 4. Не враховується контекст
- Не перевіряє, чи сегмент з voice track вже покриває primary segment повністю
- Не враховує, що voice track сегменти зазвичай точніші за primary

### 5. Пороги overlap занадто жорсткі/м'які
- 80% для voice tracks - занадто високо (пропускає 70-79%)
- 60% для фінальної дедуплікації - занадто високо (пропускає 50-59%)
- 0.1s для хронологічної валідації - занадто мало (пропускає 0.05-0.09s)

---

## Що саме не працює

### 1. Дублікати залишаються в фінальному результаті
**Симптом**: В експортованому діалозі одна й та сама репліка з'являється 2-3 рази з незначними варіаціями:
```
Speaker 1: Hi, I'm Jessica. I'm calling on behalf of Future Health...
Speaker 1: Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm your appointment...
Speaker 1: Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm your appointment with our doctor.
```

### 2. Різні транскрипції одного фрагменту
**Симптом**: Один фрагмент аудіо має кілька варіантів транскрипції:
- "Yeah, Jessica, I do. Available." (з комами)
- "Yeah Jessica I do available" (без ком)
- "Yeah Jessica I do Available" (з великої літери)

### 3. Часткові дублікати
**Симптом**: Сегмент містить інший сегмент, але з додатковим текстом:
- Короткий: "I want to reschedule this one."
- Довгий: "Uh, I want to reschedule this one." (з "Uh," на початку)

### 4. Дублікати через різні таймстемпи
**Симптом**: Одна репліка з різними start/end:
- [9.33-13.17] "Yeah, Jessica, I do. Available."
- [9.39-13.03] "Yeah, Jessica, I do. Available."

---

## Потрібні зміни

### 1. Покращена text similarity
- Нормалізація тексту (lowercase, remove punctuation, normalize whitespace)
- Використання token-based similarity (Jaccard, cosine similarity)
- Поріг similarity: 0.85-0.9 замість 0.72

### 2. Більш агресивна дедуплікація
- Знизити поріг overlap для voice tracks: 80% → 60-70%
- Знизити поріг для фінальної дедуплікації: 60% → 50%
- Додати перевірку substring matching (один текст містить інший)

### 3. Уніфікована логіка
- Створити одну функцію `isDuplicate(seg1, seg2)` з чіткими правилами
- Використовувати її на всіх етапах (кроки 2, 3, 6)

### 4. Пріоритет voice tracks
- Якщо voice track сегмент перекривається з primary на >30%, завжди використовувати voice track
- Primary segments додавати тільки якщо вони НЕ перекриваються з voice tracks

### 5. Контекстна дедуплікація
- Перевіряти не тільки pairwise overlap, але й чи сегмент повністю міститься в іншому
- Якщо сегмент A містить сегмент B (time + text), завжди видаляти B

---

## Приклад очікуваного результату

**До виправлення** (3 дублікати):
```json
[
  {"speaker": "SPEAKER_00", "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health...", "start": 0.69, "end": 5.24},
  {"speaker": "SPEAKER_00", "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm...", "start": 0.66, "end": 8.31},
  {"speaker": "SPEAKER_00", "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm your appointment with our doctor.", "start": 0.66, "end": 5.24}
]
```

**Після виправлення** (1 сегмент):
```json
[
  {"speaker": "SPEAKER_00", "text": "Hi, I'm Jessica. I'm calling on behalf of Future Health to reconfirm your appointment with our doctor. Are you available for a short conversation right now?", "start": 0.66, "end": 8.31, "source": "voice-track"}
]
```

---

## Технічні деталі реалізації

### Файли, які потрібно змінити:
1. `overlap_merge_utils.js` - основна логіка merge
2. `app.js` - використовує `overlap_merge_utils.js` через wrapper
3. `server.js` - має свою версію `collectVoiceTrackSegments()` (потрібна синхронізація)

### Ключові функції:
- `areTextsSimilar(text1, text2, threshold)` - потрібно покращити
- `mergeVoiceTrackSegments()` - потрібно зробити більш агресивну дедуплікацію
- `collectVoiceTrackSegments()` - вже правильно обробляє Azure, але може покращитися

### Тестування:
- Використовувати `test_overlap_fixes.js` для автоматичного тестування
- Перевіряти на реальному аудіо `OverlappingCallCenterWithoutBackground.MP3`
- Перевіряти, що кількість дублікатів зменшилася

---

## Висновок

Проблема полягає в тому, що поточна логіка дедуплікації використовує занадто високі пороги overlap та слабку text similarity функцію. В результаті дублікати проходять через всі етапи фільтрації і залишаються в фінальному результаті.

**Рішення**: Покращити text similarity, знизити пороги overlap, додати перевірку substring matching, і зробити дедуплікацію більш агресивною на всіх етапах.

