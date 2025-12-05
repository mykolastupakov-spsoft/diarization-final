# Діагностичні дані для аналізу класифікації

## Запуск діаризації

Для автоматичного запуску діаризації та збереження результатів:

```bash
node run-diarization-debug.js
```

Скрипт:
1. Запускає діаризацію для файлу `Call centre example.MP3`
2. Обробляє SSE потік відповіді
3. Зберігає результати в папку `Debug/`:
   - `diarization_full_result.json` - повний результат
   - `general_segments.json` - сегменти з primary diarization
   - `speaker1_segments.json` - сегменти з voice track 1
   - `speaker2_segments.json` - сегменти з voice track 2
   - `diagnostic_examples.md` - діагностичний файл з прикладами

## Структура даних

### General (Primary Diarization)
- Джерело: Step 1 - первинна діаризація всього аудіо
- Містить обидва голоси одночасно
- Може пропускати фрази при overlap

### Speaker1 / Speaker2 (Voice Tracks)
- Джерело: Step 3 - окремі аудіо треки після розділення
- Кожен трек містить тільки один голос
- Більш точна транскрипція без overlap

### Markdown Table
- Джерело: Step 5 - LLM-коректована таблиця
- Містить усі сегменти з виправленими текстами та спікерами

### Text Analysis
- Blue: фрази присутні в general І (speaker1 АБО speaker2)
- Green: фрази присутні в (speaker1 АБО speaker2), але НЕ в general
- Red: фрази відсутні в усіх джерелах (галюцинації)




