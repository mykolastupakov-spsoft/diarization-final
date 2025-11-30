# Інструкція по використанню n8n Function Node для знаходження відсутніх фраз

## Огляд

Ця Function нода аналізує webhook payload з транскрипціями та знаходить фрази, які присутні в окремих доріжках спікерів (speaker1/speaker2), але відсутні в первинному транскрайбі (general).

---

## Налаштування в n8n

### 1. Створення Workflow

1. Створіть новий workflow в n8n
2. Додайте **Webhook** ноду (HTTP Request Trigger)
3. Налаштуйте webhook:
   - Method: `POST`
   - Path: `/webhook/text-analysis`
   - Response Mode: `Last Node`

### 2. Додавання Function Node

1. Після Webhook ноди додайте **Function** ноду
2. Вставте код з файлу `n8n_function_find_missing_phrases.js` в поле коду
3. Збережіть зміни

### 3. Налаштування Function Node

Function нода автоматично отримає дані з попередньої ноди (Webhook). Код очікує структуру:

```json
{
  "general": { ... },
  "speaker1": { ... },
  "speaker2": { ... },
  "markdown": "..."
}
```

---

## Формат вхідних даних

Function нода очікує payload з такою структурою:

```json
{
  "general": {
    "segments": [
      {
        "speaker": "SPEAKER_00",
        "text": "Hi I'm Jessica...",
        "start": 0.6,
        "end": 8.24
      }
    ],
    "speechmatics": {
      "segments": [ ... ]
    }
  },
  "speaker1": {
    "speaker": "SPEAKER_00",
    "role": "operator",
    "segments": [ ... ],
    "speechmatics": {
      "segments": [ ... ]
    }
  },
  "speaker2": {
    "speaker": "SPEAKER_01",
    "role": "client",
    "segments": [ ... ],
    "speechmatics": {
      "segments": [ ... ]
    }
  }
}
```

---

## Формат вихідних даних

Function нода повертає масив об'єктів, де кожен об'єкт представляє одну відсутню фразу:

```json
[
  {
    "json": {
      "speaker": "SPEAKER_00",
      "role": "operator",
      "roleLabel": "Agent",
      "text": "Well wait wait wait for a second",
      "start": 35.16,
      "end": 36.96,
      "source": "speaker1",
      "confidence": null,
      "metadata": {
        "totalMissing": 5,
        "summary": {
          "agentMissing": 2,
          "clientMissing": 3,
          "unknownMissing": 0
        }
      }
    }
  },
  {
    "json": {
      "speaker": "SPEAKER_01",
      "role": "client",
      "roleLabel": "Client",
      "text": "Uh can you please uh make the correction",
      "start": 37.84,
      "end": 42.88,
      "source": "speaker2",
      "confidence": null,
      "metadata": {
        "totalMissing": 5,
        "summary": {
          "agentMissing": 2,
          "clientMissing": 3,
          "unknownMissing": 0
        }
      }
    }
  }
]
```

### Поля об'єкта відсутньої фрази:

- **speaker**: Ідентифікатор спікера (SPEAKER_00, SPEAKER_01)
- **role**: Роль спікера (operator, client, agent, customer)
- **roleLabel**: Мітка ролі для відображення (Agent, Client)
- **text**: Текст фрази
- **start**: Час початку в секундах
- **end**: Час закінчення в секундах
- **source**: Джерело (speaker1 або speaker2)
- **confidence**: Впевненість транскрипції (якщо доступна)
- **metadata**: Метадані з підсумком аналізу

---

## Алгоритм роботи

1. **Нормалізація текстів**: Всі тексти нормалізуються (нижній регістр, видалення пунктуації, нормалізація пробілів)

2. **Порівняння сегментів**: Для кожного сегменту з speaker1 та speaker2:
   - Перевіряється, чи існує подібний сегмент в primary транскрайбі
   - Використовується кілька методів порівняння:
     - Точна відповідність нормалізованих текстів
     - Перевірка на включення (один текст містить інший)
     - Обчислення схожості на основі спільних слів (Jaccard similarity)

3. **Фільтрація**: 
   - Ігноруються дуже короткі фрази (< 3 символи)
   - Уникаються дублікати
   - Поріг схожості: 0.7 (70%)

4. **Сортування**: Відсутні фрази сортуються за часом початку

---

## Приклад використання в Workflow

```
Webhook → Function (Find Missing Phrases) → IF Node → Set Node → Response
```

### Рекомендований workflow:

1. **Webhook** - отримує payload
2. **Function** - знаходить відсутні фрази
3. **IF Node** - перевіряє, чи є відсутні фрази (`{{ $json.metadata.totalMissing }} > 0`)
4. **Set Node** (якщо є відсутні) - форматує результат для відображення
5. **Response** - повертає результат

---

## Налаштування параметрів

Якщо потрібно змінити чутливість порівняння, можна змінити параметри в коді:

```javascript
// В функції existsInPrimary
function existsInPrimary(segment, primarySegments, similarityThreshold = 0.7) {
  // Змініть 0.7 на інше значення (0.0 - 1.0)
  // Нижче значення = більше відсутніх фраз (менш строгий)
  // Вище значення = менше відсутніх фраз (більш строгий)
}
```

---

## Обробка помилок

Якщо вхідні дані некоректні, Function нода поверне:

```json
[
  {
    "json": {
      "error": "Missing required fields: general, speaker1, or speaker2",
      "missingPhrases": []
    }
  }
]
```

Або:

```json
[
  {
    "json": {
      "error": "No segments found in primary diarization",
      "missingPhrases": []
    }
  }
]
```

---

## Тестування

Для тестування можна використати приклад payload з файлу `docs/json_coloring_example.json`:

1. Скопіюйте payload з файлу
2. Відправте його на webhook endpoint
3. Перевірте результат в Function ноді

---

## Оптимізація продуктивності

Для великих транскрипцій (> 100 сегментів):

1. Можна додати обмеження на мінімальну довжину тексту
2. Можна додати кешування нормалізованих текстів
3. Можна використовувати паралельну обробку (якщо n8n підтримує)

---

## Приклади результатів

### Приклад 1: Відсутня фраза агента

```json
{
  "speaker": "SPEAKER_00",
  "role": "operator",
  "roleLabel": "Agent",
  "text": "Well wait wait wait for a second",
  "start": 35.16,
  "end": 36.96,
  "source": "speaker1"
}
```

### Приклад 2: Відсутня фраза клієнта

```json
{
  "speaker": "SPEAKER_01",
  "role": "client",
  "roleLabel": "Client",
  "text": "Uh can you please uh make the correction uh in my date of birth",
  "start": 37.84,
  "end": 42.88,
  "source": "speaker2"
}
```

---

## Підтримка

Якщо виникають проблеми:

1. Перевірте формат вхідних даних
2. Перевірте, чи всі необхідні поля присутні
3. Перевірте логи в n8n для виявлення помилок
4. Зменште поріг схожості, якщо знаходиться занадто багато/мало відсутніх фраз

