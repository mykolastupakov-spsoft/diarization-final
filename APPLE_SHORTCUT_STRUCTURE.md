# Структура Apple Shortcut для Whisper + SpeechBrain Діаризації

## Загальна концепція

Shortcut виконує роль фронтенду, бекенд обробляє аудіо через Flask API (`app_ios_shortcuts.py`). Обробка JSON відбувається через Cloud Apple Intelligence (Ask AI дія в Shortcuts).

---

## 📚 СЛОВНИК ЗМІННИХ

**Єдиний словник всіх змінних, які використовуються в Shortcut:**

### Вхідні дані
- `input_audio` - вхідний аудіо файл
- `base64_audio` - аудіо файл, закодований в base64
- `audio_filename` - ім'я аудіо файлу

### Налаштування сервера
- `server_url` - URL сервера (наприклад: "http://100.67.135.103:5005")

### Діаризація (БЛОК 3)
- `diarize_response` - JSON відповідь від POST /api/diarize
- `job_id` - ID завдання діаризації
- `status_url` - URL для перевірки статусу діаризації
- `status_response` - JSON відповідь від GET /api/diarize/{job_id}/status
- `parsed_status` - розпарсений статус (формат: "status|data" або "status|error")
- `job_status` - статус завдання ("completed", "failed", "pending", "processing")
- `diarize_combined` - словник з combined даними (якщо статус "completed")
- `formatted_dialogue_text` - відформатований діалог з сервера (GET /api/diarize/{job_id}/formatted)

### Заміна спікерів на ролі (БЛОК 4.1)
- `dialogue_lines_array` - масив рядків діалогу (розбитий по \n)
- `dialogue_with_roles` - діалог з заміненими спікерами на ролі (Agent/Client)

### Обробка одноголосих файлів (БЛОК 5)
- `process_response` - JSON відповідь від POST /api/process-single-speaker-files
- `process_job_id` - ID завдання обробки одноголосих файлів
- `process_status_url` - URL для перевірки статусу обробки одноголосих файлів
- `process_status_response` - JSON відповідь від GET /api/process-single-speaker-files/{job_id}/status
- `process_status_dict` - словник зі статусом обробки
- `process_job_status` - статус завдання обробки ("completed", "failed", "pending", "processing")
- `single_speaker_results` - JSON з результатами обробки одноголосих файлів (містить markdown з File1Speaker0, File1Speaker1, File2Speaker0, File2Speaker1)

### Визначення головного спікера (БЛОК 5.1)
- `main_speaker_utterances` - **PLAIN TEXT** з репліками головного спікера у форматі `MM:SS [utterance text]` (кожна репліка на окремому рядку)

### Витягування реплік з JSON (БЛОК 5.2.1, 5.2.2)
- `file1_speakers_json` - **JSON** з двома полями `first_speaker` та `second_speaker` для File1
- `file2_speakers_json` - **JSON** з двома полями `first_speaker` та `second_speaker` для File2

### Об'єднання файлів у список (БЛОК 5.3)
- `files_list` - список з двох JSON об'єктів [file1_speakers_json, file2_speakers_json]

### Ідентифікація головного спікера (БЛОК 5.4)
- `current_file_json` - поточний JSON з репліками спікерів в циклі
- `main_speaker_utterances` - **PLAIN TEXT** з репліками головного спікера для поточного файлу (всередині циклу)
- `all_main_speakers_list` - список з репліками головних спікерів для всіх файлів (проміжна змінна)
- **`file1_main_speaker_utterances`** - **PLAIN TEXT** з репліками головного спікера для **File1** (фінальна змінна)
- **`file2_main_speaker_utterances`** - **PLAIN TEXT** з репліками головного спікера для **File2** (фінальна змінна)

### Визначення ролі головного спікера (БЛОК 6)
- `files_with_roles_list` - список об'єктів з ролями та репліками (ініціалізується як [])
- `current_main_speaker_utterances` - поточні репліки головного спікера в циклі (PLAIN TEXT)
- `speaker_role_json` - JSON з полями `role` та `reasoning`
- `speaker_role_dict` - словник з speaker_role_json
- `current_speaker_role` - роль поточного спікера ("operator" або "client")
- `files_with_roles` - фінальний список з ролями та репліками для всіх файлів

### Складання діалогу (БЛОК 7)
- `dialogue_from_single_files` - JSON зі складеним діалогом з одноголосих транскрайбів

---

## 🎯 ПОВНИЙ FLOW (згідно зі схемою обробки)

### БЛОК 1: Вхідні дані та підготовка
```
1. Get Shortcut Input / Record Audio / Get File
   → input_audio

2. Encode Media
   - Input: input_audio
   - Format: Base64
   → base64_string

3. Set Variable: base64_audio = base64_string

4. Get Name
   - Input: input_audio
   → filename

5. Set Variable: audio_filename = filename
```

### БЛОК 2: Налаштування сервера
```
6. Set Variable: server_url = "http://100.67.135.103:5005"
```
**⚠️ ВСІ ПАРАМЕТРИ ВСТАНОВЛЕНІ АВТОМАТИЧНО:**
- `num_speakers` = 2 (завжди)
- `language` = "English" (завжди)
- `include_transcription` = true (завжди)
- `segment_duration` = 2.5 (завжди)
- `overlap` = 0.4 (завжди)
```

### БЛОК 3: Крок 1 - Діаризація багатоголосого файлу + транскрайб (АСИНХРОННО)

**⚠️ ВАЖЛИВО: Це АСИНХРОННИЙ запит! Сервер повертає job_id одразу (за 1-2 секунди), обробка виконується в фоні.**

9. Get Contents of URL
   - Endpoint: POST /api/diarize
   - URL: http://100.67.135.103:5005/api/diarize
   - Method: POST
   - Headers:
     * Content-Type: application/json
   - Request Body: JSON
     {
       "file": base64_audio,
       "filename": audio_filename
     }
   - ⚠️ ВСІ ІНШІ ПАРАМЕТРИ ВСТАНОВЛЕНІ АВТОМАТИЧНО:
     * num_speakers = 2 (завжди)
     * language = "English" (завжди)
     * include_transcription = true (завжди)
     * segment_duration = 2.5 (завжди)
     * overlap = 0.4 (завжди)
   - Опис: Створює асинхронне завдання, повертає job_id ОДРАЗУ (не чекає на обробку!)
   - Response (швидко, за <1 секунди):
     {
       "success": true,
       "job_id": "uuid-here",
       "status": "pending",
       "message": "Processing started..."
     }

8. Ask AI (Cloud Apple Intelligence)
   - Input:
     "You are a JSON parser. Your task is to extract the job_id from a JSON response.
     
     Here is the JSON response:
     {результат з дії 7}
     
     TASK:
     1. Parse the JSON to check the structure
     2. Check the 'success' field:
        - If success = false, return: \"Error: [error message from 'error' field]\"
        - If success = true, continue to step 3
     3. Extract the 'job_id' field value
     4. Return ONLY the job_id value (without quotes, without any additional text, comments, or explanations)
     
     EXAMPLES:
     
     Input JSON:
     {
       \"success\": true,
       \"job_id\": \"abc-123-def-456\",
       \"status\": \"pending\",
       \"message\": \"Processing started...\"
     }
     Output: abc-123-def-456
     
     Input JSON:
     {
       \"success\": false,
       \"error\": \"File too large\",
       \"code\": \"FILE_SIZE_EXCEEDED\"
     }
     Output: Error: File too large
     
     IMPORTANT:
     - Return ONLY the job_id value (if success = true)
     - Return ONLY \"Error: [error message]\" (if success = false)
     - Do NOT include quotes around the job_id
     - Do NOT include any additional text, explanations, or formatting
     - Do NOT return JSON, return only the plain value"

9. Set Variable: job_id = (результат з дії 8)
   - Якщо job_id починається з "Error:", покажи помилку і зупини виконання

10. Set Variable: status_url = "http://100.67.135.103:5005/api/diarize/{job_id}/status"
    - Заміни {job_id} на значення змінної job_id

11. Set Variable: max_attempts = 120 (максимум 120 спроб = 10 хвилин)

12. Repeat (max_attempts разів):
    12.1. Wait 5 seconds
    12.2. Get Contents of URL: status_url
         - Method: GET
         - Request Timeout: 5 seconds
    12.3. Set Variable: status_response = (результат з дії 12.2)
    
    12.4. Ask AI (Cloud Apple Intelligence)
        - Input:
          "You are a JSON parser. Extract status information from the API response.

          Here is the JSON response:
          {status_response}

          TASK:
          1. Check the 'status' field value (pending, processing, completed, or failed)
          2. If status = 'completed':
             - Return: \"completed\"
             - Do NOT extract combined or segments (they will be parsed by Shortcut actions)
          3. If status = 'failed':
             - Extract the 'error' field
             - Return: \"failed|{error_message}\"
          4. If status = 'pending' or 'processing':
             - Return: \"pending\" or \"processing\"

          IMPORTANT:
          - Return only the status and error message (if failed), separated by pipe (|)
          - If completed, return ONLY \"completed\" without any additional data
          - Do NOT include combined, segments, or any other data
          - Do NOT include any additional text, explanations, or formatting"

    12.5. Set Variable: parsed_status = (результат з дії 12.4)
    
    12.6. Split Text
        - Input: parsed_status
        - Split By: Custom
        - Custom Separator: "|"
        → status_parts
    
    12.7. Get Item from List: status_parts[0] → job_status
    
    12.8. If job_status == "completed":
        12.8.1. Get Item from List: status_parts[1] → diarize_combined_json
        12.8.2. Get Dictionary from Input: diarize_combined_json → diarize_combined
        12.8.3. Exit Repeat
    12.9. If job_status == "failed":
        12.9.1. Get Item from List: status_parts[1] → error_message
        12.9.2. Show Result: "Помилка: {error_message}"
        12.9.3. Exit Repeat
    12.10. If job_status == "pending" або "processing":
        - Продовжуємо цикл (чекаємо далі)

13. If job_status != "completed":
    - Show Result: "Таймаут: обробка зайняла більше 10 хвилин"
    - Exit

14. Set Variable: diarize_combined = diarize_combined
```

### БЛОК 4: Отримання відформатованого діалогу з сервера

**⚠️ ВАЖЛИВО: Сервер сам форматує діалог, просто витягуємо текст з JSON**

```
15. Set Variable: formatted_url = "http://100.67.135.103:5005/api/diarize/{job_id}/formatted"
    - Replace {job_id} with the value of the job_id variable

16. Get Contents of URL: formatted_url
    - Method: GET
    - Request Timeout: 5 seconds

17. Set Variable: formatted_response = (result from action 16)

18. Get Dictionary from Input: formatted_response
    → formatted_dict

19. Get Dictionary Value
    - Dictionary: formatted_dict
    - Key: "formatted_dialogue"
    → formatted_dialogue

20. If formatted_dialogue is empty or not found:
    20.1. Get Dictionary Value
         - Dictionary: formatted_dict
         - Key: "error"
         → error_message
    20.2. Show Result: "Error: {error_message}"
    20.3. Exit

21. Set Variable: formatted_dialogue_text = formatted_dialogue
    - Примітка: Зберігаємо відформатований діалог для подальшої обробки
```

### БЛОК 4.1: Крок 1.1 - Заміна спікерів на ролі через Cloud Apple Intelligence

**⚠️ ВАЖЛИВО: Це обов'язковий етап перед розрізанням на одноголосі файли**

**Опис:** Розбиваємо діалог на репліки, для кожної визначаємо роль (Агент/Клієнт) через LLM в контексті всього діалогу.

```
22. Split Text
    - Input: formatted_dialogue_text
    - Split By: Custom
    - Custom Separator: "\n" (одинарний перенос рядка - кожна репліка в одному рядку)
    → dialogue_lines_array

23. Set Variable: dialogue_with_roles = "" 
    - Опис: Ініціалізуємо порожній рядок для накопичення оновленого діалогу з ролями (Agent/Client замість Speaker 0/1)

24. Repeat (for each item in dialogue_lines_array):
    24.1. Set Variable: current_line = (current item from dialogue_lines_array)
    
    24.2. If current_line is empty or contains only whitespace:
        24.2.1. Continue to next iteration (skip empty lines)
    
    24.3. Ask AI (Cloud Apple Intelligence)
        - Input:
          "You are an expert in analyzing call center dialogues.

          CONTEXT:
          You are analyzing a dialogue from a call center. The dialogue below is provided as REFERENCE ONLY.
          IMPORTANT: This dialogue may contain transcription errors and missing phrases due to speaker overlaps.
          Use it only as context to understand the conversation flow, but focus on the SPECIFIC REPLICA you need to analyze.

          FULL DIALOGUE (for context only):
          {formatted_dialogue_text}

          SPECIFIC REPLICA TO ANALYZE:
          {current_line}

          TASK:
          1. Parse the replica line in format: \"MM:SS Speaker X: [text]\"
          2. Extract the timestamp (MM:SS format)
          3. Extract the speaker label (Speaker 0 or Speaker 1)
          4. Extract the text content
          5. Analyze the text content in the context of the full dialogue to determine the role:
             - Agent (Агент): call center employee, provides services, asks questions, offers help, greets professionally, uses formal language
             - Client (Клієнт): customer, receives services, asks questions, makes requests, explains problems, seeks assistance
          6. Replace 'Speaker 0' or 'Speaker 1' with 'Agent' or 'Client' based on your analysis
          7. Return ONLY the modified line in the same single-line format:
             
             Format: MM:SS [role]: [text]
             
             Example input:
             00:00 Speaker 0: Доброго дня, чим я можу вам допомогти.
             
             Example output:
             00:00 Agent: Доброго дня, чим я можу вам допомогти.
             
             Example input:
             00:05 Speaker 1: Доброго дня, я хотів би перевірити баланс.
             
             Example output:
             00:05 Client: Доброго дня, я хотів би перевірити баланс.

          IMPORTANT:
          - Keep the exact same single-line format: \"MM:SS [role]: [text]\"
          - Use 'Agent' for call center employees (not 'Агент')
          - Use 'Client' for customers (not 'Клієнт')
          - Consider the full dialogue context, but analyze based on the specific replica's content
          - Account for possible transcription errors or overlaps - focus on the meaning and intent
          - Return ONLY the modified line, nothing else
          - Do NOT add empty lines or extra formatting"

    24.4. Set Variable: modified_line = (result from action 24.3)
        - Опис: Зберігаємо результат обробки однієї репліки (з заміненою роллю)
    
    24.5. If dialogue_with_roles is empty:
        24.5.1. Set Variable: dialogue_with_roles = modified_line
            - Опис: Якщо це перша репліка, просто присвоюємо її значення
    24.6. Otherwise:
        24.6.1. Set Variable: dialogue_with_roles = dialogue_with_roles + "\n" + modified_line
            - Опис: Додаємо нову репліку до накопиченого діалогу через перенос рядка

25. Set Variable: diarized_with_roles = dialogue_with_roles
    - Опис: Зберігаємо фінальний оновлений діалог з визначеними ролями (Agent/Client замість Speaker 0/1)
    - Формат результату: кожна репліка в одному рядку, розділені переносами рядка (\n)
    - Приклад:
      00:00 Agent: Доброго дня, чим я можу вам допомогти.
      00:05 Client: Доброго дня, я хотів би перевірити баланс.
      00:12 Agent: Звичайно, зараз перевірю ваш баланс.
```

### БЛОК 5: Крок 2-3 - Розрізання та процесінг одноголосих файлів (на бекенді)

**⚠️ IMPORTANT: This is an ASYNCHRONOUS request! The server returns a job_id immediately (in <1-2 seconds), processing runs in the background.**

```
30. Get Contents of URL
    - Endpoint: POST /api/process-single-speaker-files
    - URL: http://100.67.135.103:5005/api/process-single-speaker-files
    - Method: POST
    - Headers:
      * Content-Type: application/json
    - Request Body: JSON
      {
        "file": base64_audio,
        "filename": audio_filename,
        "diarization_job_id": job_id
      }
    - Примітка: Бекенд сам витягне segments з результату діаризації за job_id
    - Опис: Розрізання на одноголосі файли + процесінг (транскрайб одноголосих файлів)
    - Response (fast, in <1 second):
      {
        "success": true,
        "job_id": "uuid-here",
        "status": "pending",
        "message": "Processing started..."
      }
    - Примітка: Бекенд виконує:
      - Розрізання на одноголосі файли по спікерах
      - Транскрайб кожного одноголосого файлу
      - Повертає транскрипції обох одноголосих файлів (без визначення головного спікера)
      - Визначення головного спікера виконується через LLM в Shortcut (див. БЛОК 5.1)

31. Set Variable: process_response = (result from action 30)

32. Ask AI (Cloud Apple Intelligence)
    - Input:
      "You are a JSON parser. Your task is to extract the job_id from a JSON response.

      Here is the JSON response:
      {process_response}

      TASK:
      1. Parse the JSON to check the structure
      2. Check the 'success' field:
         - If success = false, return: \"Error: [error message from 'error' field]\"
         - If success = true, continue to step 3
      3. Extract the 'job_id' field value
      4. Return ONLY the job_id value (without quotes, without any additional text, comments, or explanations)

      IMPORTANT:
      - Return ONLY the job_id value (if success = true)
      - Return ONLY \"Error: [error message]\" (if success = false)
      - Do NOT include quotes around the job_id
      - Do NOT include any additional text, explanations, or formatting
      - Do NOT return JSON, return only the plain value"

33. Set Variable: process_job_id = (result from action 32)
    - If process_job_id starts with "Error:", show error and stop execution

34. Set Variable: process_status_url = "http://100.67.135.103:5005/api/process-single-speaker-files/{process_job_id}/status"
    - Replace {process_job_id} with the value of the process_job_id variable

35. Repeat (max_attempts times):
    35.1. Wait 5 seconds
    35.2. Get Contents of URL: process_status_url
         - Method: GET
         - Request Timeout: 5 seconds
    35.3. Set Variable: process_status_response = (result from action 35.2)
    35.4. Get Dictionary from Input: process_status_response → process_status_dict
    35.5. Get Dictionary Value: process_status_dict["status"] → process_job_status
    35.6. If process_job_status == "completed":
        35.6.1. Get Dictionary Value: process_status_dict["result"] → single_speaker_results
        35.6.2. Exit Repeat
    35.7. If process_job_status == "failed":
        35.7.1. Get Dictionary Value: process_status_dict["error"] → error_message
        35.7.2. Show Result: "Error: {error_message}"
        35.7.3. Exit Repeat
    35.8. If process_job_status == "pending" or "processing":
        - Continue loop (wait further)

36. If process_job_status != "completed":
    - Show Result: "Timeout: processing took more than 10 minutes"
    - Exit

37. Set Variable: single_speaker_results = single_speaker_results
```

### БЛОК 5.1: Визначення головного спікера між одноголосими файлами (через LLM)

**⚠️ ВАЖЛИВО: Це обов'язковий етап перед визначенням ролі головного спікера**

**Опис:** Використовуємо Cloud Apple Intelligence для визначення, який з одноголосих файлів містить головного спікера (той, чиї репліки мають бути збережені для подальшого аналізу).

```
38. Ask AI (Cloud Apple Intelligence)
    - Input:
      "# Prompt for Identifying the Main Speaker

## Context

You are analyzing a diarized transcript of a phone conversation between an agent and a client. Your task is to determine which of the speakers is the main one in this file (the one whose utterances should be preserved for further analysis) and return ONLY the utterances of that main speaker.

## Input Data

**Main conversation transcript:**

{formatted_dialogue_text}

**Utterances to analyze (diarized file):**

{single_speaker_results}

## Criteria for Identifying the Main Speaker

1. **Duration of continuous speech:** Who has the longest uninterrupted segments (considering that short fragments may be noise)

2. **Frequency of completed utterances:** Who more often completes their thought rather than getting cut off

3. **Interruption ratio:** Who interrupts more often vs who gets interrupted (agents typically interrupt less frequently)

4. **Stability of presence:** Who is present throughout the entire dialogue vs episodic speakers

5. **Semantic completeness:** Whose utterances contain more meaningful information, even if they are shorter

6. **Contextual role:** Who provides information/services vs who requests them

7. **Minimum segment duration:** Filtering segments shorter than 1 second as noise

## Task

1. Analyze each speaker's utterances according to the provided criteria
2. Determine who is the main speaker in this diarized file
3. Extract and return ONLY the utterances of the main speaker

## Response Format

Return the main speaker's utterances as plain text. Each utterance should be on a separate line with its timestamp and text in the format:

MM:SS [utterance text]
MM:SS [utterance text]
...

Where:
- MM:SS is the timestamp in minutes:seconds format
- [utterance text] is the exact text of the utterance

## Additional Instructions

- Consider the context of the entire conversation for more accurate identification
- Return ONLY the utterances of the identified main speaker
- Preserve the exact text and timestamps from the input
- Do not include any explanations, analysis, or additional text
- If you cannot confidently identify the main speaker, return an empty response"

39. Set Variable: main_speaker_utterances = (результат з дії 38)
```

**Вхідні змінні:**
- `formatted_dialogue_text` - відформатований діалог з сервера
- `single_speaker_results` - JSON з результатами обробки одноголосих файлів

**Вихідні змінні:**
- `main_speaker_utterances` - **PLAIN TEXT** з репліками головного спікера у форматі `MM:SS [utterance text]` (кожна репліка на окремому рядку)

**Примітки:**
- Промпт аналізує обидва одноголосі файли в контексті повного діалогу
- Враховує кілька критеріїв: тривалість, завершеність реплік, частота переривань, стабільність присутності, семантична повнота, контекстуальна роль
- Повертає репліки головного спікера як plain text у форматі `MM:SS [utterance text]` (кожна репліка на окремому рядку)
- Результат вже містить відфільтровані репліки головного спікера, готові для подальшої обробки
- Якщо не вдається впевнено визначити головного спікера, повертається порожня відповідь

---

### БЛОК 5.2.1: Витягування реплік з File1

**⚠️ ВАЖЛИВО: Це перша спроба витягування реплік з File1**

**Опис:** Витягуємо репліки з File1 з JSON відповіді після процесингу одноголосих файлів.

```
40. Ask AI (Cloud Apple Intelligence)
    - Input:
      "# Extract Speaker Utterances from File1

## Context

You are a JSON parser that extracts speaker utterances from the processed single-speaker files JSON response. Your task is to extract utterances from File1 and return them in a structured JSON format.

## Input Data

**JSON response from /api/process-single-speaker-files/{job_id}/status:**

{process_status_response}

## Task

1. Parse the JSON response to find the `markdown` object
2. Extract utterances from File1 (File1Speaker0 and File1Speaker1)
3. Return a JSON object with two fields: `first_speaker` and `second_speaker`

## Response Format

Return a JSON object with the following structure:

{
  \"first_speaker\": \"[content from File1Speaker0]\",
  \"second_speaker\": \"[content from File1Speaker1]\"
}

Where:
- `first_speaker` contains the text content from `File1Speaker0` key
- `second_speaker` contains the text content from `File1Speaker1` key

## Instructions

1. Parse the JSON response to find the `markdown` object (it is nested: `response.markdown` or `response["markdown"]`)
2. Inside the `markdown` object, find the keys `File1Speaker0` and `File1Speaker1`
3. Extract the text content from both keys:
   - Remove markdown headers (lines starting with "# Репліки спікера X")
   - Remove empty lines after headers
   - Preserve all lines with timestamps and text (format: "MM:SS Speaker X: [text]")
   - If the key contains only "(немає реплік)", use an empty string ""
4. Return ONLY the JSON object with `first_speaker` and `second_speaker` fields
5. Do not include any explanations, comments, or additional text
6. If a key is missing from the `markdown` object, use an empty string "" for that field
7. IMPORTANT: The JSON structure is: `{"markdown": {"File1Speaker0": "...", "File1Speaker1": "...", ...}}`. Make sure you navigate to `markdown` first, then access the keys.

## Example

If the input JSON markdown object contains:
File1Speaker0: \"# Репліки спікера 0\n\n00:06 Speaker 0: Hello\n00:12 Speaker 0: How can I help?\"
File1Speaker1: \"# Репліки спікера 1\n\n00:08 Speaker 1: Hi\n00:15 Speaker 1: I need help\"

The output should be:
{
  \"first_speaker\": \"00:06 Speaker 0: Hello\n00:12 Speaker 0: How can I help?\",
  \"second_speaker\": \"00:08 Speaker 1: Hi\n00:15 Speaker 1: I need help\"
}

**Note:** Remove markdown headers (lines starting with \"#\") from the extracted content, but preserve timestamps and utterance text."

41. Set Variable: file1_speakers_json = (результат з дії 40)
```

**Вхідні змінні:**
- `process_status_response` - JSON відповідь від GET /api/process-single-speaker-files/{job_id}/status

**Вихідні змінні:**
- `file1_speakers_json` - **JSON** з двома полями `first_speaker` та `second_speaker` для File1

**Примітки:**
- Промпт витягує репліки з File1
- Повертає JSON з двома полями: `first_speaker` та `second_speaker`
- Markdown заголовки (як "# Репліки спікера X") видаляються, але таймстемпи та текст зберігаються

---

### БЛОК 5.2.2: Витягування реплік з File2

**⚠️ ВАЖЛИВО: Це друга спроба витягування реплік з File2**

**Опис:** Витягуємо репліки з File2 з JSON відповіді після процесингу одноголосих файлів.

```
42. Ask AI (Cloud Apple Intelligence)
    - Input:
      "# Extract Speaker Utterances from File2

## Context

You are a JSON parser that extracts speaker utterances from the processed single-speaker files JSON response. Your task is to extract utterances from File2 and return them in a structured JSON format.

## Input Data

**JSON response from /api/process-single-speaker-files/{job_id}/status:**

{process_status_response}

**Expected JSON structure:**
{
  "markdown": {
    "File1Speaker0": "# Репліки спікера 0\n\n00:06 Speaker 0: [text]\n...",
    "File1Speaker1": "# Репліки спікера 1\n\n00:08 Speaker 1: [text]\n...",
    "File2Speaker0": "# Репліки спікера 0\n\n...",
    "File2Speaker1": "# Репліки спікера 1\n\n..."
  }
}

## Task

1. Parse the JSON response to find the `markdown` object
2. Extract utterances from File2 (File2Speaker0 and File2Speaker1)
3. Return a JSON object with two fields: `first_speaker` and `second_speaker`

## Response Format

Return a JSON object with the following structure:

{
  \"first_speaker\": \"[content from File2Speaker0]\",
  \"second_speaker\": \"[content from File2Speaker1]\"
}

Where:
- `first_speaker` contains the text content from `File2Speaker0` key
- `second_speaker` contains the text content from `File2Speaker1` key

## Instructions

1. Parse the JSON response to find the `markdown` object (it is nested: `response.markdown` or `response["markdown"]`)
2. Inside the `markdown` object, find the keys `File2Speaker0` and `File2Speaker1`
3. Extract the text content from both keys:
   - Remove markdown headers (lines starting with "# Репліки спікера X")
   - Remove empty lines after headers
   - Preserve all lines with timestamps and text (format: "MM:SS Speaker X: [text]")
   - If the key contains only "(немає реплік)", use an empty string ""
4. Return ONLY the JSON object with `first_speaker` and `second_speaker` fields
5. Do not include any explanations, comments, or additional text
6. If a key is missing from the `markdown` object, use an empty string "" for that field
7. IMPORTANT: The JSON structure is: `{"markdown": {"File2Speaker0": "...", "File2Speaker1": "...", ...}}`. Make sure you navigate to `markdown` first, then access the keys.

## Example

If the input JSON markdown object contains:
File2Speaker0: \"# Репліки спікера 0\n\n00:06 Speaker 0: Hello\n00:12 Speaker 0: How can I help?\"
File2Speaker1: \"# Репліки спікера 1\n\n00:08 Speaker 1: Hi\n00:15 Speaker 1: I need help\"

The output should be:
{
  \"first_speaker\": \"00:06 Speaker 0: Hello\n00:12 Speaker 0: How can I help?\",
  \"second_speaker\": \"00:08 Speaker 1: Hi\n00:15 Speaker 1: I need help\"
}

**Note:** Remove markdown headers (lines starting with \"#\") from the extracted content, but preserve timestamps and utterance text."

43. Set Variable: file2_speakers_json = (результат з дії 42)
```

**Вхідні змінні:**
- `process_status_response` - JSON відповідь від GET /api/process-single-speaker-files/{job_id}/status

**Вихідні змінні:**
- `file2_speakers_json` - **JSON** з двома полями `first_speaker` та `second_speaker` для File2

**Примітки:**
- Промпт витягує репліки з File2
- Повертає JSON з двома полями: `first_speaker` та `second_speaker`
- Markdown заголовки (як "# Репліки спікера X") видаляються, але таймстемпи та текст зберігаються

---

### БЛОК 5.3: Об'єднання JSON файлів у список

**Опис:** Створюємо список з JSON обох файлів для подальшої обробки в циклі.

```
44. Set Variable: files_list = [file1_speakers_json, file2_speakers_json]
```

**Вхідні змінні:**
- `file1_speakers_json` - JSON з репліками спікерів File1
- `file2_speakers_json` - JSON з репліками спікерів File2

**Вихідні змінні:**
- `files_list` - список з двох JSON об'єктів

**Примітки:**
- Використовуємо дію "List" або "Set Variable" для створення списку
- Список містить два елементи: JSON для File1 та JSON для File2

---

### БЛОК 5.4: Ідентифікація головного спікера для кожного файлу (цикл)

**⚠️ ВАЖЛИВО: Це обов'язковий етап для визначення головного спікера в кожному файлі**

**Опис:** Проходимо через кожен файл у циклі та визначаємо головного спікера для кожного.

```
44. Set Variable: all_main_speakers_list = [] (порожній список)

45. Repeat with Each Item: files_list
    - Input: files_list
    - Current Item: current_file_json
    
    45.1. Ask AI (Cloud Apple Intelligence)
        - Input:
          "# Identify Agent or Client Utterances from Single-Speaker File

## Context

You are analyzing a diarized transcript from a single-speaker audio file that was extracted from a phone conversation between an agent and a client. This single-speaker file was created by separating the original multi-speaker audio, but due to imperfect voice isolation, it may contain residual utterances from both speakers mixed together. Your task is to determine whether this file primarily belongs to the AGENT or the CLIENT, and then return ONLY the utterances of the speaker to whom this file belongs, filtering out any residual utterances from the other speaker.

**IMPORTANT CONTEXT:** This is a diarized single-speaker audio file that was extracted from a multi-speaker conversation. During the extraction process, the audio was separated by speaker, but due to imperfect voice isolation, there may be residual utterances from the secondary speaker mixed in. The file should primarily contain utterances from either the agent OR the client - you need to determine which one. The utterances from the other speaker are noise/artifacts that need to be filtered out. Your goal is to identify whether this file belongs to the agent or the client, and return ONLY the utterances of the speaker to whom this file belongs, effectively filtering out any residual utterances from the other speaker that may have been incorrectly included during the voice separation process.

## Input Data

**Speaker utterances JSON:**

{current_file_json}

This JSON contains two fields:
- `first_speaker`: utterances from Speaker 0
- `second_speaker`: utterances from Speaker 1

## Criteria for Identifying File Ownership (Agent vs Client)

1. **Duration of continuous speech:** Who has the longest uninterrupted segments (considering that short fragments may be noise)

2. **Frequency of completed utterances:** Who more often completes their thought rather than getting cut off

3. **Interruption ratio:** Who interrupts more often vs who gets interrupted (agents typically interrupt less frequently)

4. **Stability of presence:** Who is present throughout the entire dialogue vs episodic speakers

5. **Semantic completeness:** Whose utterances contain more meaningful information, even if they are shorter

6. **Contextual role indicators:** 
   - Agent typically: provides information, offers solutions, asks clarifying questions, uses professional language
   - Client typically: describes problems, requests help, responds to agent's questions, uses more casual language

7. **Minimum segment duration:** Filtering segments shorter than 1 second as noise

## Task

1. Analyze each speaker's utterances according to the provided criteria
2. Determine whether this file belongs to the AGENT or the CLIENT
3. Identify which speaker (first_speaker or second_speaker) corresponds to the file owner (agent or client)
4. Extract and return ONLY the utterances of the speaker to whom this file belongs

## Response Format

Return the utterances of the speaker to whom this file belongs (agent or client) as plain text. Each utterance should be on a separate line with its timestamp and text in the format:

MM:SS [utterance text]
MM:SS [utterance text]
...

Where:
- MM:SS is the timestamp in minutes:seconds format
- [utterance text] is the exact text of the utterance

## Additional Instructions

- Consider the context of the entire conversation for more accurate identification
- Focus on determining whether this file belongs to the agent or the client based on the content and speaking patterns
- Return ONLY the utterances of the speaker to whom this file belongs (agent or client)
- Filter out any residual utterances from the other speaker (these are noise/artifacts from imperfect voice separation)
- Preserve the exact text and timestamps from the input
- Do not include any explanations, analysis, or additional text
- If you cannot confidently determine file ownership (agent vs client), return an empty response"

    45.2. Set Variable: main_speaker_utterances = (результат з дії 45.1)
    
    45.3. Append to Variable: all_main_speakers_list
        - Add main_speaker_utterances to the list

46. Get Item from List: all_main_speakers_list
    - Index: 0 (перший елемент)
    → file1_main_speaker_utterances

47. Get Item from List: all_main_speakers_list
    - Index: 1 (другий елемент)
    → file2_main_speaker_utterances
```

**Вхідні змінні:**
- `files_list` - список з JSON об'єктів для File1 та File2
- `current_file_json` - поточний JSON з репліками спікерів (first_speaker, second_speaker)

**Вихідні змінні:**
- `main_speaker_utterances` - **PLAIN TEXT** з репліками головного спікера для поточного файлу (всередині циклу)
- `all_main_speakers_list` - список з репліками головних спікерів для всіх файлів (проміжна змінна)
- **`file1_main_speaker_utterances`** - **PLAIN TEXT** з репліками головного спікера для **File1** (фінальна змінна)
- **`file2_main_speaker_utterances`** - **PLAIN TEXT** з репліками головного спікера для **File2** (фінальна змінна)

**Примітки:**
- Цикл виконується двічі: один раз для File1, один раз для File2
- Для кожного файлу визначається головний спікер на основі критеріїв
- **Результат аналізу головного спікера для кожного файлу зберігається в окремі змінні:**
  - **`file1_main_speaker_utterances`** = результат для **File1** (репліки головного спікера File1)
  - **`file2_main_speaker_utterances`** = результат для **File2** (репліки головного спікера File2)
- Кожна змінна містить репліки головного спікера у форматі `MM:SS [utterance text]` (PLAIN TEXT)
- Список `all_main_speakers_list` використовується як проміжна змінна для зберігання результатів під час циклу

---

### БЛОК 6: Крок 4 - Визначення ролі головного спікера в одноголосих транскрайбах

**⚠️ ВАЖЛИВО: Цей блок виконується ПІСЛЯ визначення головного спікера (БЛОК 5.4)**

**Опис:** Визначаємо роль (operator/client) головного спікера для кожного файлу. Обробляємо File1 та File2 окремо.

```
48. Set Variable: files_with_roles_list = [] (порожній список)

49. Ask AI (Cloud Apple Intelligence) - для File1
    - Input:
      "You are an expert in analyzing call center dialogues.
      
      You receive utterances from a main speaker (secondary speaker already removed).
      
      Utterances:
      {file1_main_speaker_utterances}
      
      TASK:
      Determine the role of the main speaker (operator/client) based on the transcript content.
      
      DO NOT look at previous roles from diarization - analyze only the text of the utterances.
      
      Return JSON:
      {
        \"role\": \"operator\" or \"client\",
        \"reasoning\": \"Brief explanation of why this role was assigned\"
      }
      
      Return only JSON."

50. Set Variable: file1_speaker_role_json = (результат з дії 49)

51. Get Dictionary from Input: file1_speaker_role_json
    → file1_speaker_role_dict

52. Get Dictionary Value: file1_speaker_role_dict["role"]
    → file1_speaker_role

53. Ask AI (Cloud Apple Intelligence) - для File2
    - Input:
      "You are an expert in analyzing call center dialogues.
      
      You receive utterances from a main speaker (secondary speaker already removed).
      
      Utterances:
      {file2_main_speaker_utterances}
      
      TASK:
      Determine the role of the main speaker (operator/client) based on the transcript content.
      
      DO NOT look at previous roles from diarization - analyze only the text of the utterances.
      
      Return JSON:
      {
        \"role\": \"operator\" or \"client\",
        \"reasoning\": \"Brief explanation of why this role was assigned\"
      }
      
      Return only JSON."

54. Set Variable: file2_speaker_role_json = (результат з дії 53)

55. Get Dictionary from Input: file2_speaker_role_json
    → file2_speaker_role_dict

56. Get Dictionary Value: file2_speaker_role_dict["role"]
    → file2_speaker_role

57. Set Variable: files_with_roles_list = [
    {"role": file1_speaker_role, "utterances": file1_main_speaker_utterances},
    {"role": file2_speaker_role, "utterances": file2_main_speaker_utterances}
]
```
```

### БЛОК 7: Крок 5 - Складання діалогу з одноголосих транскрайбів
```
49. Ask AI (Cloud Apple Intelligence)
    - Input:
      "You are an expert in analyzing call center dialogues.
      
      You receive a list of single-speaker transcripts with their roles.
      
      List:
      {files_with_roles}
      
      TASK:
      Compose a dialogue by combining all single-speaker transcripts in chronological order.
      For each replica, preserve: timestamp, text, role.
      
      Return JSON:
      {
        \"dialogue\": [
          {\"timestamp\": 0.5, \"text\": \"...\", \"role\": \"operator\", \"source_file\": \"...\"},
          {\"timestamp\": 4.3, \"text\": \"...\", \"role\": \"client\", \"source_file\": \"...\"}
        ]
      }
      
      Return only JSON."
49. Set Variable: dialogue_from_single_files = (результат з дії 48)
```

### БЛОК 8: Крок 6 - Визначення ролі кожної репліки в діалозі (без підглядання)
```
23. Ask AI (Cloud Apple Intelligence)
    - Input:
      "You are an expert in analyzing call center dialogues.
      
      You receive a dialogue composed from single-speaker transcripts.
      
      JSON:
      {dialogue_from_single_files}
      
      CRITICALLY IMPORTANT:
      - DO NOT look at the 'role' field in JSON
      - Determine the role of EACH replica independently based on CONTENT
      - Analyze conversation context, but do not use previous roles
      
      TASK:
      For each replica in the dialogue, determine the role (operator/client) based on text content and context.
      Add a 'final_role' field to each replica.
      
      Return JSON:
      {
        \"dialogue\": [
          {\"timestamp\": 0.5, \"text\": \"...\", \"final_role\": \"operator\", \"source_file\": \"...\"},
          {\"timestamp\": 4.3, \"text\": \"...\", \"final_role\": \"client\", \"source_file\": \"...\"}
        ]
      }
      
      Return only JSON."
18. Set Variable: dialogue_with_final_roles = (результат з дії 17)
```

### БЛОК 9: Фінальне форматування та відображення
```
25. Ask AI (Cloud Apple Intelligence)
    - Input:
      "Format this dialogue into readable text:
      
      {dialogue_with_final_roles}
      
      Format:
      [MM:SS] role: replica text
      
      Return only the formatted text."
26. Set Variable: formatted_output = (результат з дії 25)
27. Show Result: formatted_output
28. Copy to Clipboard: formatted_output (optional)
```
```
8. Ask AI (Cloud Apple Intelligence)
    - Input: 
      "You are an expert in analyzing call center dialogues for the largest bank in Abu Dhabi. 
      You receive a raw JSON response from the audio diarization and transcription API.
      
      Here is the JSON response from the API:
      {diarize_response}
      
      ===== YOUR TASKS =====
      
      1. SUCCESS CHECK:
         - Check the 'success' field in JSON
         - If success = false, return a short message: \"Error: [text from error field]\"
         - If success = true, continue to step 2
      
      2. DATA EXTRACTION:
         - Find the segments array in one of these fields:
           * 'combined.segments' (priority)
           * 'segments' (if combined doesn't exist)
         - If segments not found, return: \"Error: segments not found in JSON\"
      
      3. SPEAKER ROLE DETERMINATION:
         For each segment, analyze the 'text' field and determine the role:
         
         OPERATOR (call center employee) - indicators:
         - Greets first (\"Hello\", \"Good morning\", \"السلام عليكم\")
         - Uses formal, professional language
         - Mentions bank name, department, services
         - Asks for customer information (account number, ID, name)
         - Uses phrases: \"How can I help you?\", \"May I have your...\", \"What can I do for you?\"
         - Provides information about services, policies, procedures
         - Confirms information or asks verification questions
         - Ends with phrases: \"Is there anything else?\", \"Thank you for calling\"
         
         CLIENT (customer) - indicators:
         - Responds to operator's greeting
         - Asks questions about their account, services, requests help
         - Provides personal information when asked
         - Expresses problems, complaints, or requests
         - Uses less formal language (may be more emotional)
         - Uses phrases: \"Can you help me with...\", \"I need to...\", \"I want to...\"
         - May express dissatisfaction or satisfaction
      
      4. CRITICALLY IMPORTANT:
         - DO NOT assume the first speaker is always operator
         - Analyze CONTENT and CONTEXT of conversation, not order
         - Who provides services = operator, who receives = client
         - If indicators conflict, trust the conversation content
      
      5. RESULT FORMATTING:
         Format the result as text:
         [MM:SS] role (Speaker N): replica text
         
         Where:
         - MM:SS - time from 'start' field, converted to minutes:seconds format
         - role - operator or client
         - N - speaker number (speaker + 1)
         - replica text - exact text from 'text' field (do not change, do not correct)
      
      EXAMPLE OUTPUT FORMAT:
      [00:05] operator (Speaker 1): Hello, how can I help you today?
      [00:12] client (Speaker 2): I need to check my account balance
      [00:18] operator (Speaker 1): May I have your account number please?
      [00:25] client (Speaker 2): Yes, it's 1234567890
      
      ===== OUTPUT TEXT REQUIREMENTS =====
      - Return ONLY the formatted text
      - WITHOUT headers, comments, explanations
      - WITHOUT additional text before or after
      - Preserve exact text from JSON (do not fix, do not improve)
      - Each replica on a separate line"
    - Опис: Cloud Apple Intelligence обробляє весь JSON, перевіряє success, витягує segments, визначає ролі на основі змісту, форматує результат
9. Set Variable: formatted_output = (результат з дії 8)
```

### БЛОК 5: Відображення результату
```
10. Show Result: formatted_output (вже оброблений Cloud Apple Intelligence - з перевіркою помилок, ролями та форматуванням)
11. Copy to Clipboard: formatted_output (optional)
```

---

## 🔄 СПРОЩЕНИЙ ВАРІАНТ (для прототипу без розрізання на одноголосі файли)

Якщо ендпоінт `/api/process-single-speaker-files` ще не реалізований, можна використати спрощений варіант:

### БЛОК 3-4 (спрощений): Діаризація + заміна ролей
```
7. Get Contents of URL: POST /api/diarize (як в основному flow)
8. Set Variable: diarize_response = (результат)

9. Ask AI (Cloud Apple Intelligence)
    - Input: 
      "You are an expert in analyzing call center dialogues.
      
      JSON from diarization:
      {diarize_response}
      
      TASK:
      1. Check success
      2. Extract segments
      3. Determine role (operator/client) for each segment
      4. Format as text:
         [MM:SS] role (Speaker N): text
      
      Return only the formatted text."
10. Set Variable: formatted_output = (результат)
11. Show Result: formatted_output
```

**Цей варіант пропускає кроки 2-6 (розрізання, одноголосі файли, подвійне визначення ролей).**

---

## 🔄 АЛЬТЕРНАТИВНИЙ ВАРІАНТ (якщо потрібен JSON замість тексту)

Якщо потрібно отримати JSON з ролями для подальшої обробки:

### БЛОК 4 (альтернатива): Запит до Cloud Apple Intelligence з поверненням JSON
```
8. Ask AI (Cloud Apple Intelligence)
    - Input: 
      "You are an expert in analyzing call center dialogues for a bank in Abu Dhabi. 
      You receive a JSON response from the diarization and transcription API.
      
      Here is the JSON response:
      {diarize_response}
      
      YOUR TASKS:
      1. Check the 'success' field - if false, return JSON: {\"error\": \"error description\"}
      2. If success = true, extract the segments array from 'combined.segments' field (or 'segments' if combined doesn't exist)
      3. For each segment, determine the speaker role (operator/client) based on text content:
         - operator: call center employee, greets first, provides services, asks questions about customer
         - client: customer, receives services, answers questions, may express problems
      4. IMPORTANT: DO NOT assume the first speaker is always operator - analyze CONTENT
      5. Add 'role' field to each segment in JSON
      6. Return the same JSON structure with added 'role' field for each segment
      
      EXAMPLE OUTPUT JSON:
      {
        \"success\": true,
        \"segments\": [
          {\"speaker\": 0, \"start\": 0.5, \"end\": 4.2, \"text\": \"Hello\", \"role\": \"operator\"},
          {\"speaker\": 1, \"start\": 4.3, \"end\": 8.1, \"text\": \"I need help\", \"role\": \"client\"}
        ]
      }
      
      Return only valid JSON without additional text, comments, or explanations."
9. Set Variable: result_with_roles = (результат з дії 8 - вже JSON з ролями)
    (далі можна обробити JSON як потрібно)
```

---

## 📝 Примітки

**Цей flow виконує повну схему обробки згідно з вимогами:**

1. **Діаризація багатоголосого** - через `/api/diarize`
2. **Заміна спікерів на ролі** - Cloud Apple Intelligence аналізує діаризований діалог
3. **Розрізання на одноголосі файли** - на бекенді (потрібен ендпоінт `/api/process-single-speaker-files`)
4. **Процесінг одноголосих файлів** - на бекенді (транскрайб + визначення головного спікера)
5. **Визначення ролі головного спікера** - Cloud Apple Intelligence аналізує одноголосі транскрайби
6. **Складання діалогу** - Cloud Apple Intelligence об'єднує одноголосі транскрайби
7. **Фінальне визначення ролей** - Cloud Apple Intelligence аналізує кожну репліку без підглядання в попередні ролі

**Важливо:** 
- Синхронний ендпоінт `/api/diarize` в `app_ios_shortcuts.py` ✅ (вже додано)
- Ендпоінт `/api/process-single-speaker-files` потрібно додати на бекенді (поки що може бути не реалізований)
- Для прототипу можна використати спрощений варіант (див. нижче)

**Переваги синхронного підходу:**
- Не потрібен polling (чекає завершення обробки)
- Простіша структура Shortcut
- Швидше для коротких файлів (Whisper працює швидше за SpeechBrain)

**LLM обробка через Cloud Apple Intelligence:**
- Використовує вбудовану інтеграцію Cloud Apple Intelligence в Apple Shortcuts
- Не потрібен API ключ - використовується Cloud Apple Intelligence через дію "Ask AI"
- **ВСЯ обробка JSON відбувається в Cloud Apple Intelligence:**
  - Перевірка success/error
  - Витягування segments з combined.segments або segments
  - Визначення ролей спікерів на основі змісту
  - Форматування результату
- Shortcut тільки передає сирий JSON від API та отримує готовий відформатований текст
- **НЕ потрібні дії:** Get Dictionary, Get Dictionary Value, Make JSON, парсинг, зіставлення

**Детальний промпт для Cloud Apple Intelligence:**
- Включає всі інструкції для обробки JSON
- Описує як визначати ролі (operator/client) на основі змісту
- Вказує формат вихідного тексту
- Обробляє помилки (якщо success = false)
- Підтримує обидва формати JSON (combined.segments та segments)

**Використання Cloud Apple Intelligence в Shortcuts:**
- Дія "Ask AI" (Cloud Apple Intelligence) - доступна в iOS 18+ / macOS Sequoia+
- Cloud Apple Intelligence автоматично обробляє JSON промпти та повертає готовий відформатований текст
- Не потрібна додаткова авторизація - працює через системну інтеграцію

**Переваги:**
- Мінімальна структура Shortcut (тільки 2 дії: Get Contents of URL → Ask AI → Show Result)
- Менше помилок (Cloud Apple Intelligence краще обробляє JSON ніж Shortcut)
- Гнучкість (можна змінити промпт для іншого формату виводу)
- Автоматична обробка помилок (Cloud Apple Intelligence перевіряє success)

---

## 📋 ДЕТАЛЬНИЙ ПСЕВДОКОД (повна версія)

### 📋 БЛОК 1: Вхідні дані та налаштування

#### Дія 1.1: Get Shortcut Input (опціонально)
```
Action: Get Shortcut Input
Parameters:
  - Input Type: Audio/File
  - Allow Multiple: No
Output Variable: input_audio
```

**АБО**

#### Дія 1.2: Record Audio
```
Action: Record Audio
Parameters:
  - Maximum Duration: 600 seconds (10 хвилин)
  - Preset: None
  - Stop Recording: Manually
Output Variable: input_audio
```

**АБО**

#### Дія 1.3: Get File
```
Action: Get File
Parameters:
  - Show Document Picker: Yes
  - File Path: (empty - user selects)
Output Variable: input_audio
```

---

### 📋 БЛОК 2: Налаштування параметрів (опціонально)

#### Дія 2.1: Ask for Input - Кількість спікерів
```
Action: Ask for Input
Parameters:
  - Question: "Кількість спікерів? (Залишити порожнім для авто-визначення)"
  - Input Type: Number
  - Allow Decimal Numbers: No
  - Default Answer: (empty)
Output Variable: num_speakers_input
```

#### Дія 2.2: If - Перевірка наявності num_speakers
```
Action: If
Parameters:
  - Condition: num_speakers_input is not empty
  - Then:
    - Set Variable: num_speakers = num_speakers_input
  - Otherwise:
    - Set Variable: num_speakers = (empty string)
```

#### Дія 2.3: Ask for Input - Мова (опціонально)
```
Action: Ask for Input
Parameters:
  - Question: "Мова аудіо? (uk/en/auto)"
  - Input Type: Text
  - Default Answer: "auto"
  - Allow Multiple Lines: No
Output Variable: language_input
```

#### Дія 2.4: Set Variable - language
```
Action: Set Variable
Parameters:
  - Variable Name: language
  - Value: language_input
```

---

### 📋 БЛОК 3: Конфігурація сервера

#### Дія 3.1: Set Variable - Server URL
```
Action: Set Variable
Parameters:
  - Variable Name: server_url
  - Value: "http://192.168.31.219:5005" (або Tailscale IP: "http://100.67.135.103:5005")
```

**АБО з Ask for Input:**

#### Дія 3.2: Ask for Input - Server IP
```
Action: Ask for Input
Parameters:
  - Question: "IP адреса сервера?"
  - Input Type: Text
  - Default Answer: "192.168.31.219"
Output Variable: server_ip
```

#### Дія 3.3: Text - Формування URL
```
Action: Text
Parameters:
  - Text: "http://{server_ip}:5005"
  - Variables: server_ip
Output Variable: server_url
```

---

### 📋 БЛОК 4: Перевірка стану сервера (опціонально)

#### Дія 4.1: Get Contents of URL - Health Check
```
Action: Get Contents of URL
Parameters:
  - Endpoint: GET /api/health
  - URL: "http://100.67.135.103:5005/api/health"
  - Method: GET
  - Опис: Перевіряє доступність сервера та стан завантаження моделей
  - Headers: (empty)
Output Variable: health_response
```

#### Дія 4.2: Get Dictionary from Input
```
Action: Get Dictionary from Input
Parameters:
  - Input: health_response
Output Variable: health_dict
```

#### Дія 4.3: Get Dictionary Value - status
```
Action: Get Dictionary Value
Parameters:
  - Dictionary: health_dict
  - Key: "status"
Output Variable: health_status
```

#### Дія 4.4: If - Перевірка статусу
```
Action: If
Parameters:
  - Condition: health_status != "ok"
  - Then:
    - Show Alert: "Сервер недоступний. Перевірте підключення."
    - Stop This Shortcut
```

---

### 📋 БЛОК 5: Відправка аудіо на обробку

#### Дія 5.1: Get Contents of URL - POST /process
```
Action: Get Contents of URL
Parameters:
  - Endpoint: POST /process
  - URL: "http://100.67.135.103:5005/process"
  - Method: POST
  - Опис: Створює асинхронне завдання на діаризацію + транскрайб багатоголосого файлу
  - Headers: (empty)
  - Request Body: Form
    - Field 1:
      - Name: "file"
      - Value: input_audio (File type)
      - Type: File
    - Field 2:
      - Name: "num_speakers"
      - Value: num_speakers (Text type)
      - Type: Text
    - Field 3:
      - Name: "language"
      - Value: language (Text type)
      - Type: Text
    - Field 4 (опціонально):
      - Name: "segment_duration"
      - Value: "2.0" (Text type)
      - Type: Text
    - Field 5 (опціонально):
      - Name: "overlap"
      - Value: "0.5" (Text type)
      - Type: Text
Output Variable: process_response
```

#### Дія 5.2: Get Dictionary from Input
```
Action: Get Dictionary from Input
Parameters:
  - Input: process_response
Output Variable: process_dict
```

#### Дія 5.3: Get Dictionary Value - success
```
Action: Get Dictionary Value
Parameters:
  - Dictionary: process_dict
  - Key: "success"
Output Variable: process_success
```

#### Дія 5.4: If - Перевірка успішності
```
Action: If
Parameters:
  - Condition: process_success != true
  - Then:
    - Get Dictionary Value → error (from process_dict)
    - Show Alert: "Помилка: {error}"
    - Stop This Shortcut
```

#### Дія 5.5: Get Dictionary Value - job_id
```
Action: Get Dictionary Value
Parameters:
  - Dictionary: process_dict
  - Key: "job_id"
Output Variable: job_id
```

---

### 📋 БЛОК 6: Polling статусу обробки

#### Дія 6.1: Set Variable - max_attempts
```
Action: Set Variable
Parameters:
  - Variable Name: max_attempts
  - Value: 60 (60 спроб × 5 сек = 5 хвилин максимум)
```

#### Дія 6.2: Set Variable - attempt_count
```
Action: Set Variable
Parameters:
  - Variable Name: attempt_count
  - Value: 0
```

#### Дія 6.3: Repeat - Polling loop
```
Action: Repeat
Parameters:
  - Repeat: max_attempts times
  - Inside Repeat:
    
    **6.3.1: Set Variable - attempt_count**
    Action: Set Variable
    Parameters:
      - Variable Name: attempt_count
      - Value: attempt_count + 1
    
    **6.3.2: Wait**
    Action: Wait
    Parameters:
      - Wait: 5 seconds
    
    **6.3.3: Text - Формування URL статусу**
    Action: Text
    Parameters:
      - Text: "http://100.67.135.103:5005/process/{job_id}/status"
      - Variables: job_id
    Output Variable: status_url
    
    **6.3.4: Get Contents of URL - GET статусу**
    Action: Get Contents of URL
    Parameters:
      - Endpoint: GET /process/{job_id}/status
      - URL: status_url
      - Method: GET
      - Опис: Перевіряє статус обробки завдання (pending/processing/completed/failed)
      - Headers: (empty)
    Output Variable: status_response
    
    **6.3.5: Get Dictionary from Input**
    Action: Get Dictionary from Input
    Parameters:
      - Input: status_response
    Output Variable: status_dict
    
    **6.3.6: Get Dictionary Value - status**
    Action: Get Dictionary Value
    Parameters:
      - Dictionary: status_dict
      - Key: "status"
    Output Variable: job_status
    
    **6.3.7: If - Перевірка статусу**
    Action: If
    Parameters:
      - Condition: job_status == "completed"
      - Then:
        - Exit Repeat
      - Otherwise:
        - If: job_status == "failed"
          - Then:
            - Get Dictionary Value → error (from status_dict)
            - Show Alert: "Помилка обробки: {error}"
            - Stop This Shortcut
          - Otherwise:
            - Continue (status == "pending" або "processing")
```

---

### 📋 БЛОК 7: Отримання результату

#### Дія 7.1: Get Dictionary Value - result
```
Action: Get Dictionary Value
Parameters:
  - Dictionary: status_dict (останній результат з polling)
  - Key: "result"
Output Variable: result_dict
```

#### Дія 7.2: Get Dictionary Value - segments
```
Action: Get Dictionary Value
Parameters:
  - Dictionary: result_dict
  - Key: "segments"
Output Variable: segments_array
```

#### Дія 7.3: Get Dictionary Value - full_text (опціонально)
```
Action: Get Dictionary Value
Parameters:
  - Dictionary: result_dict
  - Key: "full_text"
Output Variable: full_text
```

#### Дія 7.4: Get Dictionary Value - duration (опціонально)
```
Action: Get Dictionary Value
Parameters:
  - Dictionary: result_dict
  - Key: "duration"
Output Variable: duration
```

---

### 📋 БЛОК 8: Обробка та форматування результатів (локально)

#### Дія 8.1: Set Variable - formatted_output
```
Action: Set Variable
Parameters:
  - Variable Name: formatted_output
  - Value: (empty string)
```

#### Дія 8.2: Text - Заголовок
```
Action: Text
Parameters:
  - Text: "=== Діаризація та транскрипція ===\n\nТривалість: {duration} сек\n\n"
  - Variables: duration
Output Variable: header_text
```

#### Дія 8.3: Set Variable - formatted_output
```
Action: Set Variable
Parameters:
  - Variable Name: formatted_output
  - Value: header_text
```

#### Дія 8.4: Repeat with Each - Обробка сегментів
```
Action: Repeat with Each
Parameters:
  - Input: segments_array
  - Inside Repeat:
    
    **8.4.1: Get Dictionary Value - speaker**
    Action: Get Dictionary Value
    Parameters:
      - Dictionary: Repeat Item
      - Key: "speaker"
    Output Variable: segment_speaker
    
    **8.4.2: Get Dictionary Value - start**
    Action: Get Dictionary Value
    Parameters:
      - Dictionary: Repeat Item
      - Key: "start"
    Output Variable: segment_start
    
    **8.4.3: Get Dictionary Value - end**
    Action: Get Dictionary Value
    Parameters:
      - Dictionary: Repeat Item
      - Key: "end"
    Output Variable: segment_end
    
    **8.4.4: Get Dictionary Value - text**
    Action: Get Dictionary Value
    Parameters:
      - Dictionary: Repeat Item
      - Key: "text"
    Output Variable: segment_text
    
    **8.4.5: Calculate - Форматування часу (start)**
    Action: Calculate
    Parameters:
      - Operation: segment_start / 60 (для хвилин)
    Output Variable: start_minutes
    
    Action: Calculate
    Parameters:
      - Operation: segment_start mod 60 (для секунд)
    Output Variable: start_seconds
    
    **8.4.6: Text - Форматування часу start**
    Action: Text
    Parameters:
      - Text: "{start_minutes:02.0f}:{start_seconds:02.0f}"
      - Variables: start_minutes, start_seconds
    Output Variable: formatted_start
    
    **8.4.7: Text - Форматування сегмента**
    Action: Text
    Parameters:
      - Text: "[{formatted_start}] Спікер {segment_speaker + 1}: {segment_text}\n"
      - Variables: formatted_start, segment_speaker, segment_text
    Output Variable: segment_line
    
    **8.4.8: Append to Variable**
    Action: Append to Variable
    Parameters:
      - Variable Name: formatted_output
      - Value: segment_line
```

---

### 📋 БЛОК 9: Відображення результатів

#### Дія 9.1: Show Result
```
Action: Show Result
Parameters:
  - Input: formatted_output
```

#### Дія 9.2: Copy to Clipboard (опціонально)
```
Action: Copy to Clipboard
Parameters:
  - Input: formatted_output
```

#### Дія 9.3: Show Notification (опціонально)
```
Action: Show Notification
Parameters:
  - Title: "Діаризація завершена"
  - Body: "Оброблено {duration} секунд аудіо"
  - Variables: duration
  - Sound: Default
```

---

### 📋 БЛОК 10: Експорт результатів (опціонально)

#### Дія 10.1: Save File (опціонально)
```
Action: Save File
Parameters:
  - File: formatted_output
  - File Name: "diarization_{current_date}.txt"
  - Ask Where to Save: Yes
```

#### Дія 10.2: Share (опціонально)
```
Action: Share
Parameters:
  - Input: formatted_output
  - Share Sheet: Yes
```

---

## Альтернативна структура (асинхронна з polling)

Якщо потрібна асинхронна обробка (для дуже довгих файлів або якщо не хочете чекати), можна використати асинхронний endpoint `/process`:

### БЛОК 3 (альтернатива): Асинхронна обробка

```
5. Get Contents of URL
   - Endpoint: POST /process
   - URL: http://100.67.135.103:5005/process
   - Method: POST
   - Опис: Створює асинхронне завдання, повертає job_id
   - Body: Form
     * file = input_audio
     * num_speakers = num_speakers
     * language = language
   → process_response

6. Get Dictionary from Input: process_response → process_dict
7. Get Dictionary Value: process_dict["job_id"] → job_id

8. Repeat (60 разів):
   8.1. Wait 5 seconds
   8.2. Get Contents of URL: http://100.67.135.103:5005/process/{job_id}/status
   8.3. Get Dictionary from Input → status_dict
   8.4. Get Dictionary Value: status_dict["status"] → job_status
   8.5. If job_status == "completed": Exit Repeat
   8.6. If job_status == "failed": Show Alert + Stop

9. Get Dictionary Value: status_dict["result"] → result_dict
10. Get Dictionary Value: result_dict["segments"] → segments_array
```

**Примітка:** Асинхронний підхід доступний в `app_ios_shortcuts.py`. Рекомендується використовувати синхронний `/api/diarize` для більшості випадків.

---

## Параметри конфігурації

### Змінні, які можна налаштувати:

1. **server_url** - IP адреса MacBook
   - Локальна мережа: `http://192.168.31.219:5005`
   - Tailscale: `http://100.67.135.103:5005`

2. **max_attempts** - Максимальна кількість спроб polling (за замовчуванням: 60)
3. **wait_interval** - Інтервал між спробами в секундах (за замовчуванням: 5)
4. **segment_duration** - Довжина сегмента для SpeechBrain (за замовчуванням: 2.0)
5. **overlap** - Перекриття між сегментами (за замовчуванням: 0.5)

---

## 📡 API Endpoints

### GET /api/health
**Повна адреса:** `http://100.67.135.103:5005/api/health`  
**Опис:** Перевірка доступності сервера та стану завантаження моделей  
**Метод:** GET  
**Параметри:** немає  
**Відповідь:**
```json
{
  "status": "ok",
  "speaker_model_loaded": true,
  "whisper_model_loaded": true
}
```

### POST /api/diarize (синхронний, рекомендовано)
**Повна адреса:** `http://100.67.135.103:5005/api/diarize`  
**Опис:** Синхронна діаризація + транскрайб багатоголосого файлу (чекає завершення обробки, повертає результат одразу)  
**Метод:** POST  
**Параметри (multipart/form-data):**
- `file` (обов'язково) - аудіофайл
- `num_speakers` (опціонально) - кількість спікерів
- `language` (опціонально) - код мови (uk, en, auto)
- `include_transcription` (опціонально) - "true" або "false" (default: "true")
- `segment_duration` (опціонально) - довжина сегмента в секундах (default: 1.5)
- `overlap` (опціонально) - перекриття між сегментами 0-1 (default: 0.5)

**Примітка:** Цей ендпоінт доступний в `app_demo2.py`. Якщо використовуєте `app_ios_shortcuts.py`, потрібно або додати такий ендпоінт, або запустити `app_demo2.py` на тому ж порту.

### POST /process (асинхронний, альтернатива)
**Повна адреса:** `http://100.67.135.103:5005/process`  
**Опис:** Створює асинхронне завдання на діаризацію + транскрайб багатоголосого файлу  
**Метод:** POST  
**Параметри (multipart/form-data):**
- `file` (обов'язково) - аудіофайл
- `num_speakers` (опціонально) - кількість спікерів
- `language` (опціонально) - код мови (uk, en, auto)
- `segment_duration` (опціонально) - довжина сегмента в секундах (default: 2.0)
- `overlap` (опціонально) - перекриття між сегментами 0-1 (default: 0.5)

### GET /process/{job_id}/status
**Повна адреса:** `http://100.67.135.103:5005/process/{job_id}/status`  
**Опис:** Перевіряє статус обробки завдання (використовується тільки з асинхронним `/process`)  
**Метод:** GET  
**Параметри:** `job_id` в URL (замініть `{job_id}` на реальний ID)  
**Статуси:** `pending`, `processing`, `completed`, `failed`

### GET /process/{job_id}/result
**Повна адреса:** `http://100.67.135.103:5005/process/{job_id}/result`  
**Опис:** Alias для `/process/{job_id}/status` (той самий ендпоінт)  
**Метод:** GET  
**Параметри:** `job_id` в URL (замініть `{job_id}` на реальний ID)

---

## Формат результату API

### POST /api/diarize (200 OK, синхронний):
```json
{
  "success": true,
  "diarization": {
    "segments": [
      {
        "speaker": 0,
        "start": 0.5,
        "end": 5.2
      }
    ],
    "num_speakers": 2
  },
  "transcription": {
    "full_text": "Повний текст транскрипції...",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Текст сегменту"
      }
    ]
  },
  "combined": {
    "segments": [
      {
        "speaker": 0,
        "start": 0.5,
        "end": 4.2,
        "text": "Привіт, як справи?"
      },
      {
        "speaker": 1,
        "start": 4.3,
        "end": 8.1,
        "text": "Все добре, дякую."
      }
    ],
    "num_speakers": 2,
    "num_segments": 2
  }
}
```

### POST /process (202 Accepted, асинхронний):
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending"
}
```

### GET /process/{job_id}/status (200 OK):
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "success": true,
    "duration": 125.5,
    "full_text": "Повний текст транскрипції...",
    "segments": [
      {
        "speaker": 0,
        "start": 0.5,
        "end": 4.2,
        "text": "Привіт, як справи?"
      },
      {
        "speaker": 1,
        "start": 4.3,
        "end": 8.1,
        "text": "Все добре, дякую."
      }
    ]
  }
}
```

---

## Примітки для реалізації

1. **Обробка помилок:** Додайте перевірки на кожному кроці для кращого UX
2. **Прогрес:** Можна додати Show Notification з прогресом (attempt_count / max_attempts)
3. **Кешування:** Можна зберігати job_id для подальшого отримання результатів
4. **Експорт:** Можна додати експорт у різні формати (JSON, VTT, SRT)
5. **Візуалізація:** Можна створити HTML для відображення результатів з таймлайном

---

## Наступні кроки

1. Створити Shortcut в Apple Shortcuts app згідно з цією структурою
2. Протестувати з різними аудіофайлами
3. Налаштувати параметри під свої потреби
4. Додати додаткові функції (експорт, візуалізація тощо)

