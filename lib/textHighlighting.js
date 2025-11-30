/**
 * Text Highlighting Utilities
 * 
 * Функції для виділення слів кольорами на основі результатів textAnalysis
 * - Blue: Фрази, які є і в general, і в speaker1/speaker2 (звичайний діаризатор)
 * - Green: Overlaps - одночасна мова (наша унікальна технологія)
 * - Red: Розбіжності - те, що наш діаризатор "галюцинує"
 */

/**
 * Нормалізує текст для порівняння
 */
function normalizeTextForMatching(text) {
  if (!text || typeof text !== 'string') return '';
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Знаходить позиції фрази в тексті
 */
function findPhrasePositions(text, phrase, tolerance = 0.8) {
  const positions = [];
  const normalizedText = normalizeTextForMatching(text);
  const normalizedPhrase = normalizeTextForMatching(phrase);
  
  if (!normalizedPhrase || normalizedPhrase.length < 3) return positions;
  
  // Точний збіг
  let index = normalizedText.indexOf(normalizedPhrase);
  if (index !== -1) {
    // Знаходимо відповідну позицію в оригінальному тексті
    const words = text.split(/(\s+)/);
    let charIndex = 0;
    let wordStart = 0;
    
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const normalizedWord = normalizeTextForMatching(word);
      
      if (charIndex <= index && index < charIndex + normalizedWord.length) {
        wordStart = text.indexOf(word, wordStart);
        positions.push({
          start: wordStart,
          end: wordStart + word.length,
          phrase: phrase
        });
        break;
      }
      
      charIndex += normalizedWord.length;
      if (i < words.length - 1) {
        charIndex += 1; // пробіл
      }
    }
  }
  
  // Якщо точного збігу немає, шукаємо по словах
  if (positions.length === 0) {
    const phraseWords = normalizedPhrase.split(/\s+/).filter(w => w.length > 2);
    const textWords = normalizedText.split(/\s+/);
    
    for (let i = 0; i <= textWords.length - phraseWords.length; i++) {
      const window = textWords.slice(i, i + phraseWords.length).join(' ');
      const similarity = computeSimilarity(window, normalizedPhrase);
      
      if (similarity >= tolerance) {
        // Знаходимо позиції слів в оригінальному тексті
        const words = text.split(/(\s+)/);
        let wordIndex = 0;
        let charIndex = 0;
        
        for (let j = 0; j < words.length && wordIndex < i + phraseWords.length; j++) {
          const word = words[j];
          if (word.trim()) {
            if (wordIndex >= i) {
              const wordStart = text.indexOf(word, charIndex);
              positions.push({
                start: wordStart,
                end: wordStart + word.length,
                phrase: phrase
              });
            }
            wordIndex++;
          }
          charIndex += word.length;
        }
        break;
      }
    }
  }
  
  return positions;
}

/**
 * Обчислює схожість двох текстів
 */
function computeSimilarity(text1, text2) {
  if (!text1 || !text2) return 0;
  if (text1 === text2) return 1;
  
  const words1 = new Set(text1.split(/\s+/).filter(w => w.length > 2));
  const words2 = new Set(text2.split(/\s+/).filter(w => w.length > 2));
  
  if (words1.size === 0 || words2.size === 0) return 0;
  
  const intersection = new Set([...words1].filter(x => words2.has(x)));
  const union = new Set([...words1, ...words2]);
  
  return intersection.size / union.size;
}

/**
 * Виділяє слова в тексті кольорами на основі textAnalysis
 */
function highlightTextWithColors(text, textAnalysis, segmentStart = null, segmentEnd = null) {
  if (!text || !textAnalysis) return text;
  
  const highlights = [];
  
  // Blue: повторювані фрази (звичайний діаризатор)
  if (textAnalysis.Blue && Array.isArray(textAnalysis.Blue)) {
    textAnalysis.Blue.forEach(item => {
      // Перевіряємо, чи сегмент відповідає часу
      if (segmentStart !== null && segmentEnd !== null) {
        const itemStart = parseFloat(item.start) || 0;
        const itemEnd = parseFloat(item.end) || itemStart;
        // Перевіряємо перекриття часу
        if (!(itemStart < segmentEnd && itemEnd > segmentStart)) {
          return; // Пропускаємо, якщо час не збігається
        }
      }
      
      const positions = findPhrasePositions(text, item.text);
      positions.forEach(pos => {
        highlights.push({
          start: pos.start,
          end: pos.end,
          color: 'blue',
          type: 'repeated'
        });
      });
    });
  }
  
  // Green: overlaps (наша унікальна технологія)
  if (textAnalysis.Green && Array.isArray(textAnalysis.Green)) {
    textAnalysis.Green.forEach(item => {
      if (segmentStart !== null && segmentEnd !== null) {
        const itemStart = parseFloat(item.start) || 0;
        const itemEnd = parseFloat(item.end) || itemStart;
        if (!(itemStart < segmentEnd && itemEnd > segmentStart)) {
          return;
        }
      }
      
      // Green містить текст у форматі "text1 | text2"
      const combinedText = item.text;
      const parts = combinedText.split('|').map(p => p.trim());
      
      parts.forEach(part => {
        const positions = findPhrasePositions(text, part);
        positions.forEach(pos => {
          highlights.push({
            start: pos.start,
            end: pos.end,
            color: 'green',
            type: 'overlap'
          });
        });
      });
    });
  }
  
  // Red: розбіжності (галюцинації)
  if (textAnalysis.Red && Array.isArray(textAnalysis.Red)) {
    textAnalysis.Red.forEach(item => {
      if (segmentStart !== null && segmentEnd !== null) {
        const itemStart = parseFloat(item.start) || 0;
        const itemEnd = parseFloat(item.end) || itemStart;
        if (!(itemStart < segmentEnd && itemEnd > segmentStart)) {
          return;
        }
      }
      
      const positions = findPhrasePositions(text, item.text);
      positions.forEach(pos => {
        highlights.push({
          start: pos.start,
          end: pos.end,
          color: 'red',
          type: 'discrepancy'
        });
      });
    });
  }
  
  // Сортуємо за позицією
  highlights.sort((a, b) => a.start - b.start);
  
  // Об'єднуємо перекриваючі виділення (пріоритет: Red > Green > Blue)
  const mergedHighlights = [];
  for (let i = 0; i < highlights.length; i++) {
    const current = highlights[i];
    let merged = { ...current };
    
    // Перевіряємо наступні виділення на перекриття
    for (let j = i + 1; j < highlights.length; j++) {
      const next = highlights[j];
      if (next.start < merged.end) {
        // Перекриття - об'єднуємо та визначаємо пріоритет
        merged.end = Math.max(merged.end, next.end);
        // Пріоритет: Red > Green > Blue
        const priority = { red: 3, green: 2, blue: 1 };
        if (priority[next.color] > priority[merged.color]) {
          merged.color = next.color;
          merged.type = next.type;
        }
        i = j; // Пропускаємо наступне виділення
      } else {
        break;
      }
    }
    
    mergedHighlights.push(merged);
  }
  
  // Застосовуємо виділення до тексту
  if (mergedHighlights.length === 0) return text;
  
  // Будуємо HTML з виділеннями
  let result = '';
  let lastIndex = 0;
  
  mergedHighlights.forEach(highlight => {
    // Додаємо текст до виділення
    if (highlight.start > lastIndex) {
      result += escapeHtml(text.substring(lastIndex, highlight.start));
    }
    
    // Додаємо виділений текст
    const highlightedText = text.substring(highlight.start, highlight.end);
    const colorClass = `text-highlight-${highlight.color}`;
    result += `<span class="${colorClass}" data-type="${highlight.type}">${escapeHtml(highlightedText)}</span>`;
    
    lastIndex = highlight.end;
  });
  
  // Додаємо залишок тексту
  if (lastIndex < text.length) {
    result += escapeHtml(text.substring(lastIndex));
  }
  
  return result;
}

/**
 * Екранує HTML
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

module.exports = {
  highlightTextWithColors,
  findPhrasePositions,
  normalizeTextForMatching
};

