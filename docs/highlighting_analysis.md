# Аналіз проблеми з виділенням тексту кольорами

## Поточна логіка

### 1. Структура даних
- **Blue**: Фрази, які є і в general, і в speaker1/speaker2 (звичайний діаризатор)
- **Green**: Overlaps - одночасна мова (наша унікальна технологія)
- **Red**: Розбіжності - те, що наш діаризатор "галюцинує"

### 2. Потік даних
1. Сервер генерує `textAnalysis` з полями `Blue`, `Green`, `Red` (кожне - масив об'єктів з `text`, `start`, `end`)
2. `textAnalysis` передається через SSE в `fullResult`
3. `attachOverlapResultToRecording` зберігає `textAnalysis` в `recording.textAnalysis`
4. `renderTextComparisonTableFromSegments` отримує `textAnalysis` та передає його в `highlightTextWithColors`
5. `highlightTextWithColors` шукає фрази в тексті сегментів та виділяє їх кольорами

## Виявлені проблеми

### Проблема 1: Неправильне обчислення позицій в оригінальному тексті

**Місце**: `app.js`, функція `findPhraseInText` (рядки 11447-11481)

**Проблема**:
```javascript
const index = normalizedText.indexOf(normalizedPhrase);
// ...
const beforeMatch = normalizeText(text.substring(0, index));
const originalIndex = beforeMatch.length;
return { start: originalIndex, end: originalIndex + normalizedPhrase.length };
```

**Чому не працює**:
- `index` - це позиція в **нормалізованому** тексті (без пунктуації, з нижнім регістром)
- `text.substring(0, index)` - бере перші `index` символів з **оригінального** тексту
- Після нормалізації `beforeMatch.length` може не відповідати реальній позиції в оригінальному тексті
- Наприклад: якщо оригінальний текст "Hello, world!" і нормалізований "hello world", то `index = 5` (позиція "world"), але `text.substring(0, 5) = "Hello"`, після нормалізації `"hello"` має довжину 5, але це не відповідає позиції "world" в оригінальному тексті

**Наслідки**: Фрази не знаходяться або виділяються не в тому місці

### Проблема 2: Пошук окремих слів повертає позиції в нормалізованому тексті

**Місце**: `app.js`, функція `findPhraseInText` (рядки 11454-11473)

**Проблема**:
```javascript
const wordIndex = normalizedText.indexOf(word, currentIndex);
// ...
return { start: foundWords[0].index, end: foundWords[foundWords.length - 1].index + foundWords[foundWords.length - 1].word.length };
```

**Чому не працює**:
- `wordIndex` - це позиція в нормалізованому тексті
- Повертаються позиції в нормалізованому тексті, а не в оригінальному
- Це призводить до неправильного виділення тексту

### Проблема 3: Нормалізація може змінити структуру тексту

**Місце**: `app.js`, функція `normalizeText` (рядки 11441-11444)

**Проблема**:
```javascript
const normalizeText = (str) => {
  if (!str || typeof str !== 'string') return '';
  return str.toLowerCase().replace(/[^\w\s]/g, ' ').replace(/\s+/g, ' ').trim();
};
```

**Чому може бути проблемою**:
- Видаляє всю пунктуацію: "Hello, world!" → "hello world"
- Замінює множинні пробіли на один: "Hello    world" → "Hello world"
- Може змінити довжину тексту, що ускладнює обчислення позицій

**Наслідки**: Якщо текст у `textAnalysis` має пунктуацію, а текст у сегментах - ні (або навпаки), фрази не знайдуться

### Проблема 4: Фільтрація за часом може бути занадто строгою

**Місце**: `app.js`, функція `highlightTextWithColors` (рядки 11489-11492, 11504-11507, 11523-11526)

**Проблема**:
```javascript
if (segmentStart !== null && segmentEnd !== null) {
  const itemStart = parseFloat(item.start) || 0;
  const itemEnd = parseFloat(item.end) || itemStart;
  if (!(itemStart < segmentEnd && itemEnd > segmentStart)) return;
}
```

**Чому може бути проблемою**:
- Використовується строге перекриття часу
- Якщо `item.start` і `item.end` не точно збігаються з `segmentStart` і `segmentEnd`, фраза пропускається
- Можливі невеликі розбіжності в часі через округлення або різні джерела даних

### Проблема 5: Відсутність CSS стилів для класів (не критично)

**Місце**: `lib/textHighlighting.js` (рядок 247)

**Проблема**:
- Використовуються класи `text-highlight-blue`, `text-highlight-green`, `text-highlight-red`
- В CSS файлах немає стилів для цих класів
- **Але**: В `app.js` використовуються inline стилі, тому це не критично

## Рекомендації для виправлення

### Виправлення 1: Правильний пошук позицій в оригінальному тексті

Замість нормалізації всього тексту та пошуку в нормалізованому, потрібно:
1. Шукати фразу в оригінальному тексті з урахуванням регістру та пунктуації
2. Використовувати fuzzy matching для знаходження приблизних збігів
3. Зберігати точні позиції в оригінальному тексті

### Виправлення 2: Покращена нормалізація

Використовувати більш розумну нормалізацію:
- Зберігати позиції символів під час нормалізації
- Використовувати мапу для відстеження співвідношення між нормалізованим та оригінальним текстом

### Виправлення 3: Більш гнучка фільтрація за часом

Додати tolerance для перевірки часу:
```javascript
const timeTolerance = 0.5; // секунди
if (!(itemStart < segmentEnd + timeTolerance && itemEnd > segmentStart - timeTolerance)) return;
```

### Виправлення 4: Додати діагностичне логування

Додати більше логів для діагностики:
- Логувати знайдені фрази та їх позиції
- Логувати порівняння нормалізованих текстів
- Логувати результати пошуку

## Приклад правильного пошуку позицій

```javascript
function findPhraseInTextOriginal(text, phrase) {
  // Спочатку спробуємо точний пошук (з урахуванням регістру)
  let index = text.indexOf(phrase);
  if (index !== -1) {
    return { start: index, end: index + phrase.length };
  }
  
  // Потім спробуємо case-insensitive пошук
  const lowerText = text.toLowerCase();
  const lowerPhrase = phrase.toLowerCase();
  index = lowerText.indexOf(lowerPhrase);
  if (index !== -1) {
    return { start: index, end: index + phrase.length };
  }
  
  // Нарешті, спробуємо пошук по словах з нормалізацією
  // (але зберігаючи позиції в оригінальному тексті)
  return findPhraseByWords(text, phrase);
}
```

