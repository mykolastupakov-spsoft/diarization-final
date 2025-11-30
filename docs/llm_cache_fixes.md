# Виправлення кешування LLM відповідей

## Проблема

Кешування LLM відповідей не працювало для Step 5 (overlap correction) та інших місць, де викликається LLM.

## Виявлені проблеми

### 1. Step 5 (Overlap Correction) - `generateOverlapCorrectionResult`
**Проблема**: Функція не приймала `filename` параметр, тому кешування не працювало.

**Виправлення**:
- Додано `filename` параметр в `generateOverlapCorrectionResult`
- Передається `filename` в `handleDiarizationRequest` для кешування
- Додано передачу `filename` при виклику `generateOverlapCorrectionResult` в `/api/diarize-overlap`
- Додано передачу `filename` при виклику `generateOverlapCorrectionResult` в `correctPrimaryDiarizationWithTracks`

### 2. Step 3 (Voice Track Transcription)
**Статус**: Step 3 використовує `runPythonDiarization`, який має своє кешування для транскрипції (не LLM). LLM не використовується на цьому етапі.

### 3. Step 4 (Role Analysis) - `analyzeVoiceRole`
**Статус**: Має своє окреме кешування на основі hash транскрипту. Працює правильно.

### 4. Overlap Fixes - `sendSegmentsToLLMForFixes`
**Статус**: Вже має кешування та передачу `filename`. Працює правильно.

## Виправлення

### 1. Додано `filename` параметр в `generateOverlapCorrectionResult`

```javascript
async function generateOverlapCorrectionResult({
  primaryDiarization,
  voiceTracks,
  transcript,
  existingLLMResult,
  mode = 'smart',
  requestId = null,
  filename = null  // ← Додано
}) {
  // ...
  structured = await handleDiarizationRequest({
    transcript,
    mode,
    promptVariant: 'voice-tracks',
    filename: filename // ← Передається для кешування
  });
}
```

### 2. Передача `filename` при виклику `generateOverlapCorrectionResult`

В `/api/diarize-overlap`:
```javascript
correctionResult = await generateOverlapCorrectionResult({
  primaryDiarization,
  voiceTracks,
  transcript: combinedTranscript,
  existingLLMResult: voiceTrackLLMResult,
  mode: mode || 'smart',
  requestId,
  filename: uploadedFile?.originalname || (url ? path.parse(new URL(url).pathname.split('/').pop() || 'audio').name + '.wav' : null)  // ← Додано
});
```

В `correctPrimaryDiarizationWithTracks`:
```javascript
const llmRefined = await generateOverlapCorrectionResult({
  primaryDiarization: correctedResult,
  voiceTracks,
  transcript,
  existingLLMResult: null,
  mode: mode || 'smart',
  requestId,
  filename: filename // ← Додано
});
```

### 3. Додано діагностичне логування

Додано детальне логування для:
- Перевірки кешу (cache hit/miss)
- Проблем з filename (якщо відсутній)
- Проблем з побудовою cache key
- Збереження в кеш

Логи виводяться з префіксами:
- `🔍 Checking LLM cache` - перевірка кешу
- `✅ Using cached LLM response` - використання кешу
- `📝 LLM cache miss` - кеш не знайдено
- `💾 Saving LLM response to cache` - збереження в кеш
- `⚠️ LLM cache check skipped` - filename відсутній

## Структура кешування

### Кеш ключ
Формується на основі:
- `filename` - базова назва файлу (без розширення)
- `promptHash` - hash промпту (перші 16 символів SHA256)
- `model` - модель LLM
- `mode` - режим (fast, smart, smart-2, local, test, test2)
- `promptVariant` - варіант промпту (default, voice-tracks, overlap-fixes)

Формат: `${filenameBase}_${promptHash}_${modelSafe}_${modeSafe}_${variantSafe}`

### Кеш файли
Зберігаються в: `cache/llm_responses/${cacheKey}.json`

Структура файлу:
```json
{
  "llmOutput": "...",
  "model": "...",
  "mode": "...",
  "promptVariant": "...",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

### TTL (Time To Live)
- 30 днів (2,592,000 секунд)
- Автоматично видаляються при читанні, якщо застарілі

## Тестування

Для перевірки роботи кешування:

1. Запустити overlap diarization з файлом
2. Перевірити логи сервера:
   - `🔍 Checking LLM cache` - має з'являтися при перевірці
   - `📝 LLM cache miss` - при першому запиті
   - `💾 Saving LLM response to cache` - при збереженні
   - `✅ Using cached LLM response` - при другому запиті
3. Перевірити директорію `cache/llm_responses/` - мають з'являтися файли `.json`
4. Запустити той самий файл повторно - має використовуватися кеш

## Місця, де працює кешування

1. ✅ **Step 5 (Overlap Correction)** - `generateOverlapCorrectionResult` → `handleDiarizationRequest`
2. ✅ **Overlap Fixes** - `sendSegmentsToLLMForFixes`
3. ✅ **Combined Diarization** - `/api/diarize-combined`
4. ✅ **Text Mode Diarization** - `handleDiarizationRequest`
5. ✅ **Step 4 (Role Analysis)** - `analyzeVoiceRole` (окреме кешування)

## Примітки

- Кешування можна вимкнути через змінну середовища: `LLM_CACHE_ENABLED=false`
- Якщо `filename` відсутній, кешування пропускається (з попередженням в логах)
- Кеш ключ залежить від всього промпту, тому зміни в промпті призведуть до нового кешу

