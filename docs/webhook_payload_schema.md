# Webhook Payload Schema для Text Analysis

## Огляд

Цей документ описує структуру JSON payload, який надсилається на webhook `http://localhost:5678/webhook/text-analysis` після завершення аналізу ролей (Step 4) та перед початком корекції overlap (Step 5).

## Загальна структура

```json
{
  "general": { ... },    // Дані з Step 1: Primary Diarization
  "speaker1": { ... },   // Дані з Step 3: Voice Track 1 Transcription
  "speaker2": { ... },   // Дані з Step 3: Voice Track 2 Transcription
  "markdown": "..."      // Markdown таблиця з усіх сегментів
}
```

---

## 1. Секція `general` (Step 1: Primary Diarization)

Містить дані з первинного аналізу аудіо з перекриваючими голосами.

### Структура:

```json
{
  "general": {
    "originalText": "string",           // Об'єднаний текст з усіх сегментів primary diarization
    "speechmatics": { ... },            // Повний об'єкт результатів Speechmatics
    "recording": {
      "name": "string | null",          // Назва запису
      "duration": "number | null",      // Тривалість в секундах
      "speakerCount": "string | null",  // Кількість спікерів (наприклад, "2")
      "language": "string | null"       // Код мови (наприклад, "en-US")
    },
    "segments": [ ... ],                // Масив сегментів з primary diarization
    "segmentsCount": 0                  // Кількість сегментів
  }
}
```

### Деталі `general.segments`:

Кожен сегмент має структуру:

```json
{
  "speaker": "SPEAKER_00",     // Ідентифікатор спікера
  "text": "Hello, how can I help you?",  // Текст репліки
  "start": 0.64,               // Час початку в секундах
  "end": 2.15,                 // Час закінчення в секундах
  "words": [ ... ],            // Масив слів (опціонально)
  "confidence": 0.95,          // Впевненість (опціонально)
  "role": "operator",          // Роль спікера (operator/client) - опціонально
  "overlap": false             // Чи є перекриття з іншим спікером
}
```

### Приклад:

```json
{
  "general": {
    "originalText": "Hello, how can I help you? Hi, I need help with my account.",
    "speechmatics": {
      "success": true,
      "serviceName": "Speechmatics",
      "engine": "speechmatics",
      "segments": [
        {
          "speaker": "SPEAKER_00",
          "text": "Hello, how can I help you?",
          "start": 0.64,
          "end": 2.15,
          "words": [],
          "role": "operator",
          "overlap": false
        },
        {
          "speaker": "SPEAKER_01",
          "text": "Hi, I need help with my account.",
          "start": 2.30,
          "end": 5.45,
          "words": [],
          "role": "client",
          "overlap": false
        }
      ]
    },
    "recording": {
      "name": "Call Recording",
      "duration": 120.5,
      "speakerCount": "2",
      "language": "en-US"
    },
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "text": "Hello, how can I help you?",
        "start": 0.64,
        "end": 2.15,
        "words": [],
        "role": "operator",
        "overlap": false
      },
      {
        "speaker": "SPEAKER_01",
        "text": "Hi, I need help with my account.",
        "start": 2.30,
        "end": 5.45,
        "words": [],
        "role": "client",
        "overlap": false
      }
    ],
    "segmentsCount": 2
  }
}
```

---

## 2. Секція `speaker1` (Step 3: Voice Track 1 Transcription)

Містить дані транскрипції першої окремої аудіодоріжки (voice track 1).

### Структура:

```json
{
  "speaker1": {
    "speaker": "SPEAKER_00",            // Ідентифікатор спікера з voice track
    "role": "operator",                 // Роль: "operator" | "client" | null
    "roleConfidence": 0.95,             // Впевненість у ролі (0-1)
    "roleSummary": "string | null",     // Короткий опис ролі
    "speechmatics": { ... },            // Повний об'єкт результатів Speechmatics для цього треку
    "recording": {
      "name": "string | null",          // Назва запису
      "duration": "number | null",      // Тривалість в секундах
      "language": "string | null"       // Код мови
    },
    "segments": [ ... ],                // Масив сегментів з транскрипції voice track 1
    "segmentsCount": 0,                 // Кількість сегментів
    "transcriptText": "string | null"   // Об'єднаний текст транскрипції
  }
}
```

### Деталі `speaker1.segments`:

Кожен сегмент має структуру:

```json
{
  "speaker": "SPEAKER_00",              // Ідентифікатор спікера (може відрізнятися від speaker1.speaker!)
  "text": "Hello, how can I help you?", // Текст репліки
  "start": 0.64,                        // Час початку в секундах
  "end": 2.15,                          // Час закінчення в секундах
  "words": [ ... ],                     // Масив слів (опціонально)
  "confidence": 0.95                    // Впевненість (опціонально)
}
```

**ВАЖЛИВО**: У `speaker1.segments` поле `speaker` може містити різні значення (наприклад, `SPEAKER_00`, `SPEAKER_01`), оскільки навіть в окремій доріжці можуть бути виявлені різні спікери через залишковий шум. Для визначення правильного спікера використовуйте `speaker1.speaker` та `speaker1.role`.

### Приклад:

```json
{
  "speaker1": {
    "speaker": "SPEAKER_00",
    "role": "operator",
    "roleConfidence": 0.95,
    "roleSummary": "Call center operator offering assistance",
    "speechmatics": {
      "success": true,
      "serviceName": "Speechmatics",
      "engine": "speechmatics",
      "segments": [
        {
          "speaker": "SPEAKER_00",
          "text": "Hello, how can I help you?",
          "start": 0.64,
          "end": 2.15,
          "words": []
        },
        {
          "speaker": "SPEAKER_00",
          "text": "Sure, I can help you with that.",
          "start": 5.20,
          "end": 7.80,
          "words": []
        }
      ]
    },
    "recording": {
      "name": "Voice Track 1",
      "duration": 120.5,
      "language": "en-US"
    },
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "text": "Hello, how can I help you?",
        "start": 0.64,
        "end": 2.15,
        "words": []
      },
      {
        "speaker": "SPEAKER_00",
        "text": "Sure, I can help you with that.",
        "start": 5.20,
        "end": 7.80,
        "words": []
      }
    ],
    "segmentsCount": 2,
    "transcriptText": "Hello, how can I help you? Sure, I can help you with that."
  }
}
```

---

## 3. Секція `speaker2` (Step 3: Voice Track 2 Transcription)

Містить дані транскрипції другої окремої аудіодоріжки (voice track 2).

### Структура:

Аналогічна до `speaker1`, але для другого спікера:

```json
{
  "speaker2": {
    "speaker": "SPEAKER_01",            // Ідентифікатор спікера з voice track
    "role": "client",                   // Роль: "operator" | "client" | null
    "roleConfidence": 0.92,             // Впевненість у ролі (0-1)
    "roleSummary": "string | null",     // Короткий опис ролі
    "speechmatics": { ... },            // Повний об'єкт результатів Speechmatics для цього треку
    "recording": {
      "name": "string | null",          // Назва запису
      "duration": "number | null",      // Тривалість в секундах
      "language": "string | null"       // Код мови
    },
    "segments": [ ... ],                // Масив сегментів з транскрипції voice track 2
    "segmentsCount": 0,                 // Кількість сегментів
    "transcriptText": "string | null"   // Об'єднаний текст транскрипції
  }
}
```

### Приклад:

```json
{
  "speaker2": {
    "speaker": "SPEAKER_01",
    "role": "client",
    "roleConfidence": 0.92,
    "roleSummary": "Customer requesting assistance",
    "speechmatics": {
      "success": true,
      "serviceName": "Speechmatics",
      "engine": "speechmatics",
      "segments": [
        {
          "speaker": "SPEAKER_01",
          "text": "Hi, I need help with my account.",
          "start": 2.30,
          "end": 5.45,
          "words": []
        }
      ]
    },
    "recording": {
      "name": "Voice Track 2",
      "duration": 120.5,
      "language": "en-US"
    },
    "segments": [
      {
        "speaker": "SPEAKER_01",
        "text": "Hi, I need help with my account.",
        "start": 2.30,
        "end": 5.45,
        "words": []
      }
    ],
    "segmentsCount": 1,
    "transcriptText": "Hi, I need help with my account."
  }
}
```

---

## 4. Секція `markdown` (Markdown Table)

Містить markdown таблицю, згенеровану з сегментів обох voice tracks, відсортовану за часом початку.

### Формат:

```markdown
| Segment ID | Speaker | Text | Start Time | End Time |
|------------|---------|------|------------|----------|
| 1 | Agent | Hello, how can I help you? | 0.64 | 2.15 |
| 2 | Client | Hi, I need help with my account. | 2.30 | 5.45 |
| 3 | Agent | Sure, I can help you with that. | 5.20 | 7.80 |
```

### Правила генерації:

1. **Сегменти об'єднуються** з `speaker1.segments` та `speaker2.segments`
2. **Сортуються** за `start` (час початку) в порядку зростання
3. **Мітки спікерів** визначаються на основі ролей:
   - `role === "operator" || role === "agent"` → `"Agent"`
   - `role === "client" || role === "customer"` → `"Client"`
   - Інакше → `"Speaker 1"` або `"Speaker 2"`
4. **Segment ID** - послідовний номер від 1
5. **Час** форматується з 2 знаками після коми (наприклад, `0.64`, `2.15`)

### Приклад:

```markdown
| Segment ID | Speaker | Text | Start Time | End Time |
|------------|---------|------|------------|----------|
| 1 | Agent | Hello, how can I help you? | 0.64 | 2.15 |
| 2 | Client | Hi, I need help with my account. | 2.30 | 5.45 |
| 3 | Agent | Sure, I can help you with that. | 5.20 | 7.80 |
```

---

## Як знайти репліки для порівняння з Markdown

### Алгоритм для AI:

1. **Парсити Markdown таблицю**:
   - Розбити markdown на рядки
   - Витягти для кожного рядка: `Segment ID`, `Speaker`, `Text`, `Start Time`, `End Time`

2. **Знайти відповідні репліки в JSON**:

   **Для кожної репліки з Markdown:**
   
   a. **Визначити джерело** на основі `Speaker`:
      - Якщо `Speaker === "Agent"` → шукати в `speaker1.segments` (якщо `speaker1.role === "operator"`)
      - Якщо `Speaker === "Client"` → шукати в `speaker2.segments` (якщо `speaker2.role === "client"`)
      - Або навпаки, залежно від ролей
   
   b. **Знайти сегмент** за часом:
      - Порівняти `Start Time` та `End Time` з markdown
      - Знайти сегмент в `speaker1.segments` або `speaker2.segments`, де:
        - `start` ≈ `Start Time` (допустима різниця ±0.5 секунди)
        - `end` ≈ `End Time` (допустима різниця ±0.5 секунди)
   
   c. **Порівняти текст**:
      - Порівняти `Text` з markdown з `text` з JSON сегменту
      - Враховувати можливі незначні відмінності (великі/малі літери, пунктуація)

3. **Перевірити primary diarization** (опціонально):
   - Якщо потрібно порівняти з оригінальним аналізом, використовувати `general.segments`
   - Знайти сегмент за часом та текстом

### Приклад коду для пошуку репліки:

```javascript
// Для репліки з markdown:
// Segment ID: 1
// Speaker: Agent
// Text: "Hello, how can I help you?"
// Start Time: 0.64
// End Time: 2.15

// 1. Визначити джерело
const isAgent = markdownSpeaker === "Agent";
const sourceTrack = isAgent && speaker1.role === "operator" ? speaker1 : speaker2;

// 2. Знайти сегмент за часом
const matchingSegment = sourceTrack.segments.find(seg => {
  const timeDiff = Math.abs(parseFloat(seg.start) - parseFloat(markdownStartTime));
  return timeDiff < 0.5; // Допустима різниця 0.5 секунди
});

// 3. Порівняти текст
if (matchingSegment) {
  const textMatches = matchingSegment.text.trim().toLowerCase() === 
                      markdownText.trim().toLowerCase();
  // Виконати порівняння
}
```

---

## Повний приклад payload

```json
{
  "general": {
    "originalText": "Hello, how can I help you? Hi, I need help with my account. Sure, I can help you with that.",
    "speechmatics": {
      "success": true,
      "serviceName": "Speechmatics",
      "engine": "speechmatics",
      "segments": [
        {
          "speaker": "SPEAKER_00",
          "text": "Hello, how can I help you?",
          "start": 0.64,
          "end": 2.15,
          "words": [],
          "role": "operator",
          "overlap": false
        },
        {
          "speaker": "SPEAKER_01",
          "text": "Hi, I need help with my account.",
          "start": 2.30,
          "end": 5.45,
          "words": [],
          "role": "client",
          "overlap": false
        },
        {
          "speaker": "SPEAKER_00",
          "text": "Sure, I can help you with that.",
          "start": 5.20,
          "end": 7.80,
          "words": [],
          "role": "operator",
          "overlap": true
        }
      ]
    },
    "recording": {
      "name": "Call Recording",
      "duration": 120.5,
      "speakerCount": "2",
      "language": "en-US"
    },
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "text": "Hello, how can I help you?",
        "start": 0.64,
        "end": 2.15,
        "words": [],
        "role": "operator",
        "overlap": false
      },
      {
        "speaker": "SPEAKER_01",
        "text": "Hi, I need help with my account.",
        "start": 2.30,
        "end": 5.45,
        "words": [],
        "role": "client",
        "overlap": false
      },
      {
        "speaker": "SPEAKER_00",
        "text": "Sure, I can help you with that.",
        "start": 5.20,
        "end": 7.80,
        "words": [],
        "role": "operator",
        "overlap": true
      }
    ],
    "segmentsCount": 3
  },
  "speaker1": {
    "speaker": "SPEAKER_00",
    "role": "operator",
    "roleConfidence": 0.95,
    "roleSummary": "Call center operator offering assistance",
    "speechmatics": {
      "success": true,
      "serviceName": "Speechmatics",
      "engine": "speechmatics",
      "segments": [
        {
          "speaker": "SPEAKER_00",
          "text": "Hello, how can I help you?",
          "start": 0.64,
          "end": 2.15,
          "words": []
        },
        {
          "speaker": "SPEAKER_00",
          "text": "Sure, I can help you with that.",
          "start": 5.20,
          "end": 7.80,
          "words": []
        }
      ]
    },
    "recording": {
      "name": "Voice Track 1",
      "duration": 120.5,
      "language": "en-US"
    },
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "text": "Hello, how can I help you?",
        "start": 0.64,
        "end": 2.15,
        "words": []
      },
      {
        "speaker": "SPEAKER_00",
        "text": "Sure, I can help you with that.",
        "start": 5.20,
        "end": 7.80,
        "words": []
      }
    ],
    "segmentsCount": 2,
    "transcriptText": "Hello, how can I help you? Sure, I can help you with that."
  },
  "speaker2": {
    "speaker": "SPEAKER_01",
    "role": "client",
    "roleConfidence": 0.92,
    "roleSummary": "Customer requesting assistance",
    "speechmatics": {
      "success": true,
      "serviceName": "Speechmatics",
      "engine": "speechmatics",
      "segments": [
        {
          "speaker": "SPEAKER_01",
          "text": "Hi, I need help with my account.",
          "start": 2.30,
          "end": 5.45,
          "words": []
        }
      ]
    },
    "recording": {
      "name": "Voice Track 2",
      "duration": 120.5,
      "language": "en-US"
    },
    "segments": [
      {
        "speaker": "SPEAKER_01",
        "text": "Hi, I need help with my account.",
        "start": 2.30,
        "end": 5.45,
        "words": []
      }
    ],
    "segmentsCount": 1,
    "transcriptText": "Hi, I need help with my account."
  },
  "markdown": "| Segment ID | Speaker | Text | Start Time | End Time |\n|------------|---------|------|------------|----------|\n| 1 | Agent | Hello, how can I help you? | 0.64 | 2.15 |\n| 2 | Client | Hi, I need help with my account. | 2.30 | 5.45 |\n| 3 | Agent | Sure, I can help you with that. | 5.20 | 7.80 |\n"
}
```

---

## Важливі примітки

1. **Ролі спікерів**: Використовуйте `speaker1.role` та `speaker2.role` для визначення, хто є Agent, а хто Client. Не покладайтеся тільки на мітки в markdown.

2. **Час**: Час в markdown та JSON може мати незначні відмінності через округлення. Використовуйте допустиму різницю ±0.5 секунди при порівнянні.

3. **Текст**: Текст може відрізнятися через:
   - Різне форматування пунктуації
   - Великі/малі літери
   - Додаткові пробіли
   - Нормалізацію при генерації markdown

4. **Сегменти в voice tracks**: Поле `speaker` в `speaker1.segments` та `speaker2.segments` може містити різні значення через залишковий шум. Завжди використовуйте `speaker1.speaker` та `speaker2.speaker` як основні ідентифікатори.

5. **Primary diarization**: `general.segments` містить сегменти з оригінального аналізу з перекриваючими голосами. Це може бути корисним для порівняння, але може містити менш точні транскрипції через overlap.

---

## Посилання на джерела даних

- **Step 1 (Primary Diarization)**: `general.segments` - сегменти з первинного аналізу
- **Step 3 (Voice Track 1)**: `speaker1.segments` - сегменти з транскрипції першої доріжки
- **Step 3 (Voice Track 2)**: `speaker2.segments` - сегменти з транскрипції другої доріжки
- **Markdown Table**: `markdown` - об'єднана та відсортована таблиця з обох voice tracks

