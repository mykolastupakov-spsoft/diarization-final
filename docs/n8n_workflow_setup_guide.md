# n8n Workflow Setup Guide: Text Analysis with Blue, Green, Red Nodes

## Огляд

Цей гайд описує налаштування n8n workflow для аналізу результатів діаризації від Speechmatics. Workflow приймає три JSON-файли (general, speaker1, speaker2) і повертає категорізовані фрагменти тексту для візуалізації на фронтенді.

## Структура Workflow

```
Webhook → Blue Node → Green Node → Red Node → Set Node → Webhook Response
```

### 1. Webhook Node

**Налаштування:**
- **HTTP Method**: POST
- **Path**: `text-analysis` (або будь-який інший шлях)
- **Response Mode**: `responseNode` (використовуємо Webhook Response node)

**Очікуваний формат payload:**
```json
{
  "general": {
    "speechmatics": {
      "segments": [
        {
          "speaker": "SPEAKER_00",
          "text": "...",
          "start": 0.6,
          "end": 8.24
        }
      ]
    }
  },
  "speaker1": {
    "speaker": "SPEAKER_00",
    "role": "operator",
    "speechmatics": {
      "segments": [...]
    }
  },
  "speaker2": {
    "speaker": "SPEAKER_01",
    "role": "client",
    "speechmatics": {
      "segments": [...]
    }
  }
}
```

### 2. Blue Node (Repeated Phrases)

**Тип**: Code / Function Node

**Призначення**: Знаходить фрази, які присутні і в general транскрайбі, і в окремих доріжках (speaker1/speaker2).

**Код**: Скопіюйте вміст файлу `docs/n8n_function_blue_repeated_phrases.js`

**Вихідний формат:**
```json
{
  "speaker": "SPEAKER_00",
  "role": "operator",
  "roleLabel": "Agent",
  "text": "Hi I'm Jessica...",
  "start": 0.64,
  "end": 8.2,
  "source": "speaker1",
  "confidence": 1.0,
  "type": "repeated",
  "metadata": {
    "totalRepeated": 45,
    "summary": {
      "agentRepeated": 20,
      "clientRepeated": 25
    }
  }
}
```

### 3. Green Node (Overlaps)

**Тип**: Code / Function Node

**Призначення**: Знаходить моменти одночасної мови (overlaps) шляхом порівняння таймстемпів з обох доріжок.

**Код**: Скопіюйте вміст файлу `docs/n8n_function_green_overlaps.js`

**Вихідний формат:**
```json
{
  "speaker1": {
    "speaker": "SPEAKER_00",
    "role": "operator",
    "roleLabel": "Agent",
    "text": "Great!",
    "start": 15.8,
    "end": 16.2,
    "source": "speaker1"
  },
  "speaker2": {
    "speaker": "SPEAKER_01",
    "role": "client",
    "roleLabel": "Client",
    "text": "Yes, please",
    "start": 15.9,
    "end": 16.5,
    "source": "speaker2"
  },
  "overlap": {
    "start": 15.9,
    "end": 16.2,
    "duration": 0.3
  },
  "type": "overlap",
  "metadata": {
    "totalOverlaps": 12,
    "summary": {
      "agentOverlaps": 5,
      "clientOverlaps": 7,
      "totalOverlapDuration": 3.5
    }
  }
}
```

### 4. Red Node (Discrepancies & Errors)

**Тип**: Code / Function Node

**Призначення**: Знаходить розбіжності та помилки транскрибації:
- Фрази, які є в окремих доріжках, але відсутні в general
- Фрази з різним текстом між general та окремими доріжками

**Код**: Скопіюйте вміст файлу `docs/n8n_function_red_discrepancies.js`

**Вихідний формат:**
```json
{
  "speaker": "SPEAKER_01",
  "role": "client",
  "roleLabel": "Client",
  "text": "April 29th 1981",
  "start": 22.5,
  "end": 24.0,
  "source": "speaker2",
  "confidence": 0.95,
  "errorType": "missing",
  "description": "Phrase present in speaker track but missing from primary transcript",
  "metadata": {
    "totalDiscrepancies": 8,
    "summary": {
      "missing": 5,
      "transcriptionErrors": 3,
      "agentDiscrepancies": 2,
      "clientDiscrepancies": 6
    }
  }
}
```

Або для помилок транскрибації:
```json
{
  "speaker": "SPEAKER_00",
  "role": "operator",
  "roleLabel": "Agent",
  "text": "I see you have an appointment",
  "primaryText": "I see you have appointment",
  "start": 25.0,
  "end": 28.5,
  "primaryStart": 25.1,
  "primaryEnd": 28.4,
  "source": "speaker1",
  "confidence": 0.92,
  "similarity": 0.85,
  "errorType": "transcription_error",
  "description": "Text differs between speaker track and primary transcript"
}
```

### 5. Set Node (Combine Results)

**Тип**: Set Node

**Призначення**: Об'єднує результати з трьох попередніх нод в один JSON з ключами `Blue`, `Green`, `Red`.

**Налаштування:**

**Mode**: `Combine` або `Manual`

**Values to Set:**

1. **Blue** (Repeated Phrases):
   - **Name**: `Blue`
   - **Value**: `{{ $json }}` (від Blue node)

2. **Green** (Overlaps):
   - **Name**: `Green`
   - **Value**: `{{ $json }}` (від Green node)

3. **Red** (Discrepancies):
   - **Name**: `Red`
   - **Value**: `{{ $json }}` (від Red node)

**Альтернативний підхід (якщо потрібен масив):**

Якщо кожна нода повертає масив елементів, можна використати Function node для об'єднання:

```javascript
// Combine results from Blue, Green, Red nodes
const blueData = $input.all().find(item => item.json.type === 'repeated') || [];
const greenData = $input.all().find(item => item.json.type === 'overlap') || [];
const redData = $input.all().find(item => item.json.errorType) || [];

return [{
  json: {
    Blue: blueData.map(item => item.json),
    Green: greenData.map(item => item.json),
    Red: redData.map(item => item.json)
  }
}];
```

### 6. Webhook Response Node

**Тип**: Webhook Response

**Призначення**: Повертає фінальний JSON клієнту.

**Налаштування:**
- **Response Code**: 200
- **Response Data**: `{{ $json }}` (від Set node)

**Очікуваний фінальний формат відповіді:**
```json
{
  "Blue": [
    {
      "speaker": "SPEAKER_00",
      "role": "operator",
      "roleLabel": "Agent",
      "text": "...",
      "start": 0.64,
      "end": 8.2,
      "type": "repeated"
    }
  ],
  "Green": [
    {
      "speaker1": {...},
      "speaker2": {...},
      "overlap": {...},
      "type": "overlap"
    }
  ],
  "Red": [
    {
      "speaker": "SPEAKER_01",
      "role": "client",
      "text": "...",
      "errorType": "missing",
      "description": "..."
    }
  ]
}
```

## Підключення нод

1. **Webhook** → **Blue Node** (паралельно)
2. **Webhook** → **Green Node** (паралельно)
3. **Webhook** → **Red Node** (паралельно)
4. **Blue Node** → **Set Node**
5. **Green Node** → **Set Node**
6. **Red Node** → **Set Node**
7. **Set Node** → **Webhook Response**

Або, якщо використовуєте Function node для об'єднання:

1. **Webhook** → **Blue Node**
2. **Blue Node** → **Green Node**
3. **Green Node** → **Red Node**
4. **Red Node** → **Combine Function Node**
5. **Combine Function Node** → **Webhook Response**

## Тестування

### Тестовий запит (curl):

```bash
curl -X POST http://localhost:5678/webhook/text-analysis \
  -H "Content-Type: application/json" \
  -d @docs/json_coloring_example.json
```

### Перевірка результатів:

1. **Blue (Repeated)**: Перевірте, що фрази присутні і в general, і в speaker tracks
2. **Green (Overlaps)**: Перевірте, що знайдені перекриваючі таймстемпи
3. **Red (Discrepancies)**: Перевірте, що знайдені відсутні або відрізняються фрази

## Обробка на фронтенді

Фронтенд отримує JSON з ключами `Blue`, `Green`, `Red` і може:

1. **Blue**: Відображати синім кольором - стандартна діаризація
2. **Green**: Відображати зеленим кольором - моменти накладок
3. **Red**: Відображати червоним кольором - помилки та відсутні фрагменти

## Налаштування порогів

У всіх трьох функціях можна налаштувати пороги:

- **Blue Node**: `similarityThreshold = 0.7` (в функції `existsInPrimary`)
- **Green Node**: `minOverlapSeconds = 0.1` (в функції `timeRangesOverlap`)
- **Red Node**: 
  - `similarityThreshold = 0.7` (для перевірки наявності)
  - `similarity < 0.9 && similarity >= 0.5` (для виявлення помилок транскрибації)

## Troubleshooting

### Проблема: Нода повертає помилку "Missing required fields"

**Рішення**: Перевірте, що webhook payload містить поля `general`, `speaker1`, `speaker2`.

### Проблема: Не знаходить overlaps

**Рішення**: Перевірте, що таймстемпи валідні (start < end) і зменште `minOverlapSeconds` в Green node.

### Проблема: Занадто багато/мало знайдених фраз

**Рішення**: Налаштуйте пороги схожості (`similarityThreshold`) в Blue та Red nodes.

## Додаткові ресурси

- `docs/n8n_function_blue_repeated_phrases.js` - код для Blue node
- `docs/n8n_function_green_overlaps.js` - код для Green node
- `docs/n8n_function_red_discrepancies.js` - код для Red node
- `docs/json_coloring_example.json` - приклад payload для тестування

