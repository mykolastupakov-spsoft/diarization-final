# Логіка формування даних сегментів для LLM

## Дата: 2025-11-28

## Опис

Створено логіку для формування даних сегментів у потрібному форматі для передачі на LLM для виправлення змішування спікерів.

## Функції

### 1. `prepareSegmentsForLLM(segments)`

**Призначення**: Формує масив сегментів у потрібному форматі для LLM.

**Вхідні дані**:
- `segments` - масив об'єктів сегментів з результату diarization

**Вихідні дані**:
```json
[
  {
    "segment_id": 1,
    "text": "Hello",
    "start": 0.00,
    "end": 1.20,
    "speaker_id": "SPEAKER_00"
  },
  {
    "segment_id": 2,
    "text": "How are you?",
    "start": 0.80,
    "end": 1.90,
    "speaker_id": "SPEAKER_01"
  }
]
```

**Логіка**:
1. Сортує сегменти за часом початку (start)
2. Присвоює унікальний `segment_id` (1-based index)
3. Округлює `start` та `end` до 2 знаків після коми
4. Використовує `speaker` як `speaker_id`
5. Очищає текст від зайвих пробілів

**Розташування**: `server.js`, перед функцією `correctPrimaryDiarizationWithTracks`

### 2. `sendSegmentsToLLMForFixes(segmentsForLLM, options)`

**Призначення**: Підготовка та передача даних на LLM для виправлення змішування спікерів.

**Вхідні дані**:
- `segmentsForLLM` - масив відформатованих сегментів
- `options` - об'єкт з параметрами:
  - `mode` - режим LLM ('fast', 'smart', 'smart-2')
  - `language` - код мови
  - `requestId` - ID запиту для логування
  - `sendUpdate` - функція для відправки SSE оновлень

**Вихідні дані**:
- Promise з масивом виправлених сегментів від LLM

**Поточний стан**: 
- ✅ Функція повністю реалізована
- ✅ Викликає LLM API (OpenRouter або Local LLM)
- ✅ Підтримує режими: 'fast', 'smart', 'smart-2', 'local'
- ✅ Обробляє відповідь та валідує результати
- ✅ Логує детальну інформацію про процес

**Реалізація**:
- Використовує існуючу логіку вибору моделі (`FAST_MODEL_ID`, `SMART_MODEL_ID`, `SMART_2_MODEL_ID`, `LOCAL_LLM_MODEL`)
- Використовує існуючі функції для headers (`getOpenRouterHeaders`, `getLocalLLMHeaders`)
- Створює детальний prompt для виправлення змішування спікерів
- Парсить JSON відповідь з підтримкою markdown code blocks
- Валідує структуру виправлених сегментів
- Логує кількість змін спікерів

**Розташування**: `server.js`, перед функцією `correctPrimaryDiarizationWithTracks`

## Інтеграція

### Місце виклику

Функція `prepareSegmentsForLLM` викликається в `correctPrimaryDiarizationWithTracks` після програмного об'єднання:

```javascript
// Step 1: Programmatic merge
let correctedResult = mergeTranscriptsProgrammatically(primaryDiarization, voiceTracks);

// Prepare segments data for LLM processing
const programmaticSegments = correctedResult?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
const segmentsForLLM = prepareSegmentsForLLM(programmaticSegments);

// Step 2: Send segments to LLM for speaker mixing fixes
let llmCorrectedSegments = null;
try {
  llmCorrectedSegments = await sendSegmentsToLLMForFixes(segmentsForLLM, {
    mode: mode || 'smart',
    language: language,
    requestId: requestId,
    sendUpdate: sendUpdate
  });
  
  // Apply LLM corrections to result
  if (llmCorrectedSegments && llmCorrectedSegments.length > 0) {
    // Update segments in correctedResult with LLM corrections
    // ...
  }
} catch (llmError) {
  // Continue with programmatic result if LLM fails
}
```

**Інтеграція результатів**:
- Виправлені сегменти від LLM конвертуються назад у формат системи
- Оновлюються `speaker` поля в `correctedResult`
- Додається метадані про LLM виправлення (`llmFixed: true`, `llmCorrectionsCount`)
- Оновлюється `serviceName` на "Overlap Corrected (Programmatic + LLM Fixed)"

### Логування

Додано детальне логування:
- Кількість підготовлених сегментів
- Приклад перших 3 сегментів
- Список унікальних спікерів
- Діапазон часу (start - end)
- Перші 5 та останні 5 сегментів у JSON форматі

## Приклад використання

### Вхідні дані (з diarization)

```javascript
const segments = [
  {
    speaker: "SPEAKER_00",
    text: "Hello, how can I help you?",
    start: 0.64,
    end: 2.15,
    words: [...]
  },
  {
    speaker: "SPEAKER_01",
    text: "Hi, I need help",
    start: 2.30,
    end: 3.45,
    words: [...]
  }
];
```

### Вихідні дані (для LLM)

```json
[
  {
    "segment_id": 1,
    "text": "Hello, how can I help you?",
    "start": 0.64,
    "end": 2.15,
    "speaker_id": "SPEAKER_00"
  },
  {
    "segment_id": 2,
    "text": "Hi, I need help",
    "start": 2.30,
    "end": 3.45,
    "speaker_id": "SPEAKER_01"
  }
]
```

## Наступні кроки

1. ✅ Створено функцію `prepareSegmentsForLLM` - формує дані у потрібному форматі
2. ✅ Створено функцію `sendSegmentsToLLMForFixes` - структура для передачі на LLM
3. ✅ Реалізовано виклик LLM API в `sendSegmentsToLLMForFixes`
4. ✅ Створено prompt для LLM з інструкціями для виправлення змішування спікерів
5. ✅ Оброблено відповідь від LLM та конвертацію назад у формат сегментів
6. ✅ Інтегровано виправлені сегменти в результат

## Технічні деталі

### Формат сегментів

- `segment_id`: унікальний номер (1-based, послідовний)
- `text`: текст фрази (trimmed)
- `start`: час початку в секундах (округлено до 2 знаків)
- `end`: час закінчення в секундах (округлено до 2 знаків)
- `speaker_id`: ідентифікатор спікера (SPEAKER_00, SPEAKER_01, ...)

### Сортування

Сегменти сортуються за:
1. `start` (час початку) - основний критерій
2. `end` (час закінчення) - якщо start однаковий

### Валідація

- Перевірка на порожній масив
- Обробка відсутніх значень (start, end, speaker, text)
- Округлення до 2 знаків після коми для start/end

### LLM Prompt

**System Prompt**:
```
You are an expert in dialogue transcription and speaker diarization.
Your task is to review the list of segments below, identify any mismatched 
speaker assignments caused by overlapping speech or diarization errors, 
and return a corrected list of segments.

CRITICAL RULES:
1. Each segment must keep its original start/end times (do not modify timestamps)
2. Do not alter the text content unless absolutely necessary
3. If a segment's speaker_id is wrong, change it to the correct one based on:
   - Contextual flow of conversation
   - Logical speaker alternation
   - Semantic content (questions vs answers, greetings vs responses)
4. Maintain chronological order
5. Return ONLY valid JSON array with the same structure as input
```

**User Prompt**:
```
Review the following segments and correct any speaker assignment errors.

Segments (in chronological order):
[список сегментів з segment_id, text, start, end, speaker_id]

TASK:
1. Identify segments where speaker_id is incorrect
2. Correct the speaker_id while keeping all other fields unchanged
3. Return the corrected JSON array in the exact same format
```

### Підтримувані моделі

- **OpenRouter**: 
  - `fast` → `FAST_MODEL_ID` (gpt-oss-120b)
  - `smart` → `SMART_MODEL_ID` (gpt-5.1)
  - `smart-2` → `SMART_2_MODEL_ID` (google/gemini-3-pro-preview)
- **Local LLM**: 
  - `local` → `LOCAL_LLM_MODEL` (openai/gpt-oss-20b)

### Обробка помилок

- Якщо LLM запит не вдався → продовжуємо з програмним результатом
- Якщо парсинг JSON не вдався → викидаємо помилку з деталями
- Якщо валідація не пройшла → викидаємо помилку з деталями
- Логування всіх помилок з `requestId` для відстеження

## Логування

Після підготовки даних в консолі сервера з'являться логи:

```
📋 Prepared 27 segments for LLM processing
📋 Segments data structure: {
  totalSegments: 27,
  sampleSegments: [...],
  speakers: ['SPEAKER_00', 'SPEAKER_01'],
  timeRange: { start: 0.64, end: 355.23 }
}
📋 First 5 segments: [...]
```

Це дозволить перевірити правильність форматування перед передачею на LLM.

