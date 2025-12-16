# Інструкція по оновленню iOS Shortcut

## Що змінилося

API тепер повертає додаткове поле `MainSpeakerDialogue` в об'єкті `markdown`, яке містить відформатований діалог основного спікера у форматі:
```
MM:SS - MM:SS, спікер номер, репліка
```

## Оновлення Shortcut

### Витягування реплік спікера 1 та спікера 2 через LLM

Оскільки бекенд вже визначає основного спікера, вам потрібно просто витягнути репліки обох спікерів окремо з поля `MainSpeakerDialogue` та інших полів markdown.

**Крок 1: Після отримання відповіді від `/api/process-single-speaker-files/{job_id}/status`**

Додайте блок **"Ask AI"** (Cloud Apple Intelligence) з наступним промптом:

```
# Extract Speaker Utterances from Processed Files

## Context
You are a JSON parser that extracts speaker utterances from the processed single-speaker files JSON response. Your task is to extract utterances for Speaker 1 and Speaker 2 separately.

## Input Data
**JSON response from /api/process-single-speaker-files/{job_id}/status:**
{process_status_response}

## Task
1. Parse the JSON response to find the `markdown` object
2. Extract utterances from File1 (File1Speaker0 and File1Speaker1)
3. Return a JSON object with two fields: `speaker_1` and `speaker_2`

## Response Format
Return a JSON object with the following structure:

{
  "speaker_1": "[content from File1Speaker0]",
  "speaker_2": "[content from File1Speaker1]"
}

Where:
- `speaker_1` contains the text content from `File1Speaker0` key
- `speaker_2` contains the text content from `File1Speaker1` key

## Instructions
1. Parse the JSON response to find the `markdown` object (it is nested: `response.markdown` or `response["markdown"]`)
2. Inside the `markdown` object, find the keys `File1Speaker0` and `File1Speaker1`
3. Extract the text content from both keys:
   - Remove markdown headers (lines starting with "# Репліки спікера X")
   - Remove empty lines after headers
   - Preserve all lines with timestamps and text (format: "MM:SS Speaker X: [text]")
   - If the key contains only "(немає реплік)", use an empty string ""
4. Return ONLY the JSON object with `speaker_1` and `speaker_2` fields
5. Do not include any explanations, comments, or additional text
6. If a key is missing from the `markdown` object, use an empty string "" for that field
7. IMPORTANT: The JSON structure is: `{"markdown": {"File1Speaker0": "...", "File1Speaker1": "...", ...}}`. Make sure you navigate to `markdown` first, then access the keys.
8. Do NOT determine which speaker is main - just extract both speakers' utterances as-is

## Example
If the input JSON markdown object contains:
File1Speaker0: "# Репліки спікера 0\n\n00:06 Speaker 0: Hello\n00:12 Speaker 0: How can I help?"
File1Speaker1: "# Репліки спікера 1\n\n00:08 Speaker 1: Hi\n00:15 Speaker 1: I need help"

The output should be:
{
  "speaker_1": "00:06 Speaker 0: Hello\n00:12 Speaker 0: How can I help?",
  "speaker_2": "00:08 Speaker 1: Hi\n00:15 Speaker 1: I need help"
}

**Note:** Remove markdown headers (lines starting with "#") from the extracted content, but preserve timestamps and utterance text. Do not try to determine which speaker is main - just extract both speakers' utterances.
```

**Крок 2: Обробка результату**

Після отримання JSON від AI:

1. **"Get Dictionary Value"**
   - Dictionary: результат від AI
   - Key: `speaker_1`
   - Result: `speaker_1_utterances`

2. **"Get Dictionary Value"**
   - Dictionary: результат від AI
   - Key: `speaker_2`
   - Result: `speaker_2_utterances`

3. **"Set Variable"** для кожного спікера:
   - `speaker_1_text` = `speaker_1_utterances`
   - `speaker_2_text` = `speaker_2_utterances`

**Приклад структури дій:**

```
40. Get Contents of URL
    - URL: http://YOUR_IP:5005/api/process-single-speaker-files/{job_id}/status
    - Method: GET
    - Result: process_status_response

41. Ask AI (Cloud Apple Intelligence)
    - Input: [промпт вище з {process_status_response}]
    - Result: speakers_json

42. Get Dictionary Value
    - Dictionary: speakers_json
    - Key: speaker_1
    - Result: speaker_1_utterances

43. Get Dictionary Value
    - Dictionary: speakers_json
    - Key: speaker_2
    - Result: speaker_2_utterances

44. Set Variable: speaker_1_text = (result from action 42)
45. Set Variable: speaker_2_text = (result from action 43)
```

**Важливо:**
- Не потрібно визначати, який спікер основний - це вже зроблено на бекенді
- Просто витягуйте репліки обох спікерів окремо
- Використовуйте `File1Speaker0` та `File1Speaker1` з об'єкта `markdown`

## Структура відповіді API

Після оновлення, відповідь від `/api/process-single-speaker-files/{job_id}/status` містить:

```json
{
  "markdown": {
    "File1Speaker0": "...",
    "File1Speaker1": "...",
    "File2Speaker0": "...",
    "File2Speaker1": "...",
    "MainSpeakerDialogue": "00:00 - 00:05, спікер 0, Hello world\n00:10 - 00:15, спікер 0, How are you?"
  },
  "main_speaker": 0
}
```

## Формат `MainSpeakerDialogue`

Кожен рядок має формат:
```
MM:SS - MM:SS, спікер номер, репліка
```

**Приклад:**
```
00:00 - 00:05, спікер 0, Hello, how can I help you?
00:10 - 00:15, спікер 0, I need assistance with my account.
00:20 - 00:25, спікер 0, Sure, let me check that for you.
```

## Визначення основного спікера

Основний спікер визначається за тією ж логікою, що використовується в `enhance_main_speaker_audio`:
- **Критерій 1**: Кількість слів (найточніший показник)
- **Критерій 2**: Тривалість (якщо різниця в словах <10%)

Це забезпечує узгодженість між різними частинами системи.

## Важливі примітки

1. **Визначення основного спікера**: Бекенд вже визначає основного спікера за логікою (кількість слів + тривалість), тому в шорткаті не потрібно це робити
2. **Витягування реплік**: LLM використовується тільки для витягування та форматування реплік з JSON, а не для визначення основного спікера
3. **Структура даних**: Репліки обох спікерів доступні в `markdown.File1Speaker0` та `markdown.File1Speaker1`

## Перевірка оновлення

Після оновлення шорткату:
1. Запустіть обробку аудіо файлу
2. Перевірте, чи в відповіді є поле `MainSpeakerDialogue`
3. Перевірте формат діалогу - кожен рядок повинен мати формат `MM:SS - MM:SS, спікер номер, репліка`
4. Переконайтеся, що змінна `main_speaker_dialogue_text` містить правильний текст

## Приклад використання результату

Після витягування реплік обох спікерів, ви можете:

1. **Відобразити текст**:
   - Дія "Show Notification" або "Show Alert"
   - Input: `speaker_1_text` або `speaker_2_text`

2. **Зберегти в файл**:
   - Дія "Save File"
   - Input: `speaker_1_text` або `speaker_2_text`
   - File Name: `speaker_1_utterances.txt` або `speaker_2_utterances.txt`

3. **Скопіювати в буфер обміну**:
   - Дія "Copy to Clipboard"
   - Input: `speaker_1_text` або `speaker_2_text`

4. **Відправити повідомлення**:
   - Дія "Send Message"
   - Message: `speaker_1_text` або `speaker_2_text`

5. **Об'єднати репліки обох спікерів**:
   - Дія "Text"
   - Input: `Speaker 1:\n{speaker_1_text}\n\nSpeaker 2:\n{speaker_2_text}`

