# Налаштування AI нод для Blue, Green, Red

## Огляд

Всі три ноди (Blue, Green, Red) використовують AI для аналізу транскрипцій. Це простіше і інтуїтивніше, ніж Function ноди з JavaScript кодом.

## Переваги AI підходу

- ✅ Просто отримує дані з Webhook
- ✅ Не потрібно писати складну логіку порівняння
- ✅ AI розуміє контекст і може знаходити складні патерни
- ✅ Легко налаштувати через промпт
- ✅ Однаковий підхід для всіх трьох нод

## Структура Workflow

```
Webhook → Blue AI → Parse → Green AI → Parse → Red AI → Parse → Set → Webhook Response
```

Або паралельно (якщо підтримується):

```
Webhook → [Blue AI, Green AI, Red AI] → [Parse, Parse, Parse] → Set → Webhook Response
```

## Налаштування AI нод

### Загальні налаштування для всіх AI нод

**Model**: `gpt-4o-mini` (швидкий і дешевий) або `gpt-4o` (точніший)
**Temperature**: `0` (детерміновані результати)
**Max Tokens**: `4000` (достатньо для великої кількості результатів)
**Input**: `{{ $json.body }}` (або `{{ $json }}` якщо дані безпосередньо в json)

---

## 1. Blue AI Node (Repeated Phrases)

**Назва**: `Blue AI` або `Blue`

**Промпт**: Скопіюйте з `docs/n8n_ai_blue_repeated_phrases_prompt.txt`

Або вставте напряму:
```
You are analyzing audio transcription diarization results to find repeated phrases.

Your task is to identify phrases that appear in BOTH:
1. The **general** transcript (primary diarization with all speakers)
2. The individual speaker tracks (**speaker1** and/or **speaker2**)

These are phrases that were correctly transcribed in both the general diarization and the individual speaker tracks.

**IMPORTANT OUTPUT FORMAT:**
Return ONLY a valid JSON array. Each item must have exactly these fields:
- `text`: The text of the repeated phrase (string)
- `start`: Start time in seconds (number)
- `end`: End time in seconds (number)

Example output:
[
  {"text": "Are you available for a short conversation", "start": 5.2, "end": 7.8},
  {"text": "Thank you", "start": 12.5, "end": 13.1}
]

If no repeated phrases are found, return an empty array: []

**RULES:**
- Only include phrases that exist in BOTH general transcript AND at least one speaker track
- Use the timing from the speaker track (speaker1 or speaker2), not from general
- Ignore very short phrases (less than 3 words)
- Sort results by start time
- Return ONLY the JSON array, no explanations, no markdown, no code blocks

**Input data:**
{{ $json.body }}
```

---

## 2. Green AI Node (Overlaps)

**Назва**: `Green AI` або `Green`

**Промпт**: Скопіюйте з `docs/n8n_ai_green_overlaps_prompt.txt`

Або вставте напряму:
```
You are analyzing audio transcription diarization results to find overlapping speech segments.

Your task is to identify moments when **speaker1** and **speaker2** spoke simultaneously (overlapping timestamps).

Compare the timestamps from both speaker tracks to find segments where:
- speaker1's segment overlaps in time with speaker2's segment
- The overlap duration is at least 0.1 seconds

**IMPORTANT OUTPUT FORMAT:**
Return ONLY a valid JSON array. Each item must have exactly these fields:
- `text`: Combined text from both speakers, separated by " | " (string)
- `start`: Start time of the overlap in seconds (number)
- `end`: End time of the overlap in seconds (number)

Example output:
[
  {"text": "Hello there | Yes I'm here", "start": 5.2, "end": 7.8},
  {"text": "Thank you | You're welcome", "start": 12.5, "end": 13.5}
]

If no overlaps are found, return an empty array: []

**RULES:**
- Only include segments where timestamps actually overlap (start1 < end2 AND start2 < end1)
- Minimum overlap duration: 0.1 seconds
- Format combined text as: "speaker1_text | speaker2_text"
- Use the overlap time range (max(start1, start2) to min(end1, end2))
- Sort results by start time
- Return ONLY the JSON array, no explanations, no markdown, no code blocks

**Input data:**
{{ $json.body }}
```

---

## 3. Red AI Node (Discrepancies)

**Назва**: `Red AI` або `Red`

**Промпт**: Скопіюйте з `docs/n8n_ai_red_discrepancies_prompt.txt`

Або вставте напряму:
```
You are analyzing audio transcription diarization results to find discrepancies and transcription errors.

Your task is to compare three transcriptions:
1. **general** - The primary diarization transcript (contains all speakers)
2. **speaker1** - Individual track for speaker 1
3. **speaker2** - Individual track for speaker 2

Find and return:
1. **Missing phrases**: Segments that exist in speaker1 or speaker2 tracks but are missing from the general transcript
2. **Transcription errors**: Segments where the text differs significantly between speaker tracks and general transcript (similarity < 95%)

**IMPORTANT OUTPUT FORMAT:**
Return ONLY a valid JSON array. Each item must have exactly these fields:
- `text`: The text of the discrepancy (string)
- `start`: Start time in seconds (number)
- `end`: End time in seconds (number)

Example output:
[
  {"text": "Missing phrase from speaker1", "start": 5.2, "end": 7.8},
  {"text": "Different transcription", "start": 12.5, "end": 15.3}
]

If no discrepancies are found, return an empty array: []

**RULES:**
- Only return segments where text similarity is less than 95% OR the segment is completely missing from general
- Ignore very short segments (less than 3 characters after normalization)
- Sort results by start time
- Return ONLY the JSON array, no explanations, no markdown, no code blocks

**Input data:**
{{ $json.body }}
```

---

## 4. Parse Node (для кожної AI ноди)

Після кожної AI ноди додайте **Code/Function ноду** для парсингу JSON.

**Назва**: `Parse Blue`, `Parse Green`, `Parse Red`

**Код**: Скопіюйте з `docs/n8n_function_parse_ai_response.js`

Або вставте напряму:
```javascript
/**
 * n8n Function Node: Parse AI Response
 * Парсить відповідь від AI ноди і конвертує в формат для Set ноди
 */

const aiResponse = $input.item.json;

// Отримуємо текст відповіді
let responseText = '';

if (aiResponse.choices && aiResponse.choices[0] && aiResponse.choices[0].message) {
  responseText = aiResponse.choices[0].message.content;
} else if (aiResponse.content && Array.isArray(aiResponse.content)) {
  responseText = aiResponse.content.map(c => c.text).join('');
} else if (aiResponse.content && typeof aiResponse.content === 'string') {
  responseText = aiResponse.content;
} else if (typeof aiResponse === 'string') {
  responseText = aiResponse;
} else if (Array.isArray(aiResponse)) {
  if (aiResponse.length > 0 && aiResponse[0].text !== undefined) {
    return aiResponse.map(item => ({
      json: {
        text: item.text || '',
        start: parseFloat(item.start) || 0,
        end: parseFloat(item.end) || 0
      }
    }));
  }
  responseText = JSON.stringify(aiResponse);
} else {
  responseText = JSON.stringify(aiResponse);
}

// Очищаємо від markdown
responseText = responseText
  .replace(/```json\n?/g, '')
  .replace(/```\n?/g, '')
  .replace(/^```/g, '')
  .replace(/```$/g, '')
  .trim();

if (!responseText || responseText.length === 0) {
  return [];
}

// Парсимо JSON
let results = [];
try {
  results = JSON.parse(responseText);
  if (!Array.isArray(results)) {
    if (results.discrepancies && Array.isArray(results.discrepancies)) {
      results = results.discrepancies;
    } else if (results.results && Array.isArray(results.results)) {
      results = results.results;
    } else {
      results = [results];
    }
  }
} catch (e) {
  return [{
    json: {
      error: 'Failed to parse AI response',
      text: `Error: ${e.message}`,
      start: 0,
      end: 0
    }
  }];
}

// Конвертуємо в потрібний формат
return results
  .filter(item => item && (item.text || item.text === ''))
  .map(item => ({
    json: {
      text: String(item.text || ''),
      start: parseFloat(item.start) || 0,
      end: parseFloat(item.end) || parseFloat(item.start) || 0
    }
  }));
```

---

## 5. Set Node (об'єднання результатів)

Після всіх Parse нод додайте **Set ноду** для об'єднання результатів.

**Назва**: `Set`

**Налаштування**:
- **Keep Only Set Fields**: `false`
- **Fields to Set**:
  - `Blue`: `{{ $('Parse Blue').all() }}`
  - `Green`: `{{ $('Parse Green').all() }}`
  - `Red`: `{{ $('Parse Red').all() }}`

Або використовуйте Expression:
```javascript
{
  "Blue": $('Parse Blue').all().map(item => item.json),
  "Green": $('Parse Green').all().map(item => item.json),
  "Red": $('Parse Red').all().map(item => item.json)
}
```

---

## 6. Webhook Response Node

**Назва**: `Webhook Response`

**Налаштування**:
- **Response Code**: `200`
- **Response Data**: `{{ $json }}`

---

## Перевірка роботи

1. Запустіть workflow з тестовими даними
2. Перевірте вихід кожної AI ноди - має бути JSON масив
3. Перевірте вихід Parse нод - має бути масив об'єктів з `text`, `start`, `end`
4. Перевірте вихід Set ноди - має бути об'єкт з ключами `Blue`, `Green`, `Red`

---

## Оптимізація витрат

- Використовуйте `gpt-4o-mini` замість `gpt-4o` для економії
- Встановіть `temperature: 0` для детермінованих результатів
- Обмежте `max_tokens` до мінімуму (4000 достатньо)
- Можна запускати всі три AI ноди паралельно (якщо підтримується)

---

## Troubleshooting

### Проблема: AI нода не повертає жодних результатів

**Крок 1: Додайте Debug ноду**
1. Додайте Function ноду після AI ноди
2. Скопіюйте код з `docs/n8n_debug_ai_node.js`
3. Запустіть workflow і перевірте вихід Debug ноди

**Крок 2: Перевірте вхідні дані**
1. Додайте Debug ноду ПЕРЕД AI нодою
2. Перевірте, чи дані передаються правильно
3. Переконайтеся, що `{{ $json.body }}` містить `general`, `speaker1`, `speaker2`

**Крок 3: Перевірте налаштування AI ноди**
- **Model**: Переконайтеся, що модель підтримується (наприклад, `gpt-4o-mini`)
- **API Key**: Перевірте, чи API ключ введений правильно
- **Temperature**: Має бути `0` для детермінованих результатів
- **Max Tokens**: Має бути достатньо (4000+)

**Крок 4: Перевірте формат відповіді**
- OpenAI повертає: `{choices: [{message: {content: "..."}}]}`
- Anthropic повертає: `{content: [...]}` або `{content: "..."}`
- Якщо формат інший, оновіть Parse ноду

**Крок 5: Перевірте промпт**
- Переконайтеся, що промпт чітко вказує повертати ТІЛЬКИ JSON масив
- Додайте в кінець промпту: "Return ONLY a valid JSON array. Do NOT wrap it in any object."

### Інші проблеми

**Проблема**: AI повертає текст замість JSON
- **Рішення**: Додайте в промпт "Return ONLY valid JSON, no explanations"

**Проблема**: AI повертає `{"output": "[]"}`
- **Рішення**: Parse нода автоматично обробляє це. Перевірте, чи Parse нода оновлена

**Проблема**: AI повертає markdown code blocks
- **Рішення**: Parse нода автоматично очищає markdown

**Проблема**: AI не знаходить результати
- **Рішення**: 
  1. Перевірте, чи правильно передаються дані через `{{ $json.body }}`
  2. Перевірте, чи дані містять сегменти в `general.speechmatics.segments`
  3. Спробуйте зменшити поріг схожості в промпті (з 80% до 70%)

**Проблема**: Parse нода не працює
- **Рішення**: 
  1. Використайте Debug ноду, щоб побачити структуру відповіді
  2. Оновіть Parse ноду відповідно до формату
  3. Перевірте формат відповіді AI (OpenAI vs Anthropic мають різні формати)

---

## Файли з промптами

- `docs/n8n_ai_blue_repeated_phrases_prompt.txt` - промпт для Blue
- `docs/n8n_ai_green_overlaps_prompt.txt` - промпт для Green
- `docs/n8n_ai_red_discrepancies_prompt.txt` - промпт для Red
- `docs/n8n_function_parse_ai_response.js` - код для Parse ноди

