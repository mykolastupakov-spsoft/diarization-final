/**
 * Text Analysis Functions: Blue, Green, Red
 * 
 * Аналізує результати діаризації та знаходить:
 * - Blue: Повторювані фрази (є і в general, і в speaker1/speaker2)
 * - Green: Overlaps (одночасна мова двох спікерів)
 * - Red: Розбіжності та помилки транскрибації
 */

// Функція для нормалізації тексту
function normalizeText(text) {
  if (!text || typeof text !== 'string') return '';
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Функція для обчислення схожості текстів
function computeTextSimilarity(text1, text2) {
  const normalized1 = normalizeText(text1);
  const normalized2 = normalizeText(text2);
  
  if (!normalized1 || !normalized2) return 0;
  if (normalized1 === normalized2) return 1;
  
  // Перевірка на включення
  if (normalized1.includes(normalized2) || normalized2.includes(normalized1)) {
    const shorter = normalized1.length < normalized2.length ? normalized1 : normalized2;
    const longer = normalized1.length >= normalized2.length ? normalized1 : normalized2;
    return shorter.length / longer.length;
  }
  
  // Jaccard similarity на основі спільних слів
  const words1 = new Set(normalized1.split(/\s+/).filter(w => w.length > 2));
  const words2 = new Set(normalized2.split(/\s+/).filter(w => w.length > 2));
  
  if (words1.size === 0 || words2.size === 0) return 0;
  
  const intersection = new Set([...words1].filter(x => words2.has(x)));
  const union = new Set([...words1, ...words2]);
  
  return intersection.size / union.size;
}

// Функція для перевірки, чи існує сегмент в primary транскрайбі
function existsInPrimary(segment, primarySegments, similarityThreshold = 0.7) {
  const segmentText = normalizeText(segment.text || '');
  if (!segmentText || segmentText.length < 3) return false;
  
  for (const primarySeg of primarySegments) {
    const primaryText = normalizeText(primarySeg.text || '');
    if (!primaryText) continue;
    
    if (segmentText === primaryText) return true;
    
    if (segmentText.includes(primaryText) || primaryText.includes(segmentText)) {
      const shorter = segmentText.length < primaryText.length ? segmentText : primaryText;
      const longer = segmentText.length >= primaryText.length ? segmentText : primaryText;
      if (shorter.length / longer.length >= 0.8) return true;
    }
    
    const similarity = computeTextSimilarity(segmentText, primaryText);
    if (similarity >= similarityThreshold) return true;
  }
  
  return false;
}

/**
 * BLUE: Знаходить повторювані фрази
 * Фрази, які є і в general (транскрайб всього аудіо), і в окремих доріжках (speaker1/speaker2)
 */
function findRepeatedPhrases(payload) {
  const repeatedPhrases = [];
  
  // Перевіряємо наявність необхідних даних
  if (!payload || !payload.general || !payload.speaker1 || !payload.speaker2) {
    return {
      error: 'Missing required fields: general, speaker1, or speaker2',
      repeatedPhrases: []
    };
  }
  
  // Отримуємо сегменти з primary транскрайбу
  const primarySegments = payload.general.segments || 
                         payload.general.speechmatics?.segments || 
                         [];
  
  if (primarySegments.length === 0) {
    return {
      error: 'No segments found in primary diarization',
      repeatedPhrases: []
    };
  }
  
  // Обробляємо speaker1
  const speaker1Segments = payload.speaker1.segments || 
                          payload.speaker1.speechmatics?.segments || 
                          [];
  
  // Обробляємо speaker2
  const speaker2Segments = payload.speaker2.segments || 
                          payload.speaker2.speechmatics?.segments || 
                          [];
  
  // Знаходимо повторювані фрази для speaker1
  const seenTexts = new Set();
  for (const segment of speaker1Segments) {
    if (!segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 3) continue;
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    // Якщо фраза присутня в primary транскрайбі, вона повторюється
    if (existsInPrimary(segment, primarySegments)) {
      repeatedPhrases.push({
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0
      });
    }
  }
  
  // Знаходимо повторювані фрази для speaker2
  for (const segment of speaker2Segments) {
    if (!segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 3) continue;
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    // Якщо фраза присутня в primary транскрайбі, вона повторюється
    if (existsInPrimary(segment, primarySegments)) {
      repeatedPhrases.push({
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0
      });
    }
  }
  
  // Сортуємо за часом початку
  repeatedPhrases.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return repeatedPhrases;
}

/**
 * GREEN: Знаходить overlaps (одночасна мова)
 * Порівнює таймстемпи з обох доріжок для виявлення одночасної мови
 */
function findOverlaps(payload) {
  const overlaps = [];
  
  // Перевіряємо наявність необхідних даних
  if (!payload || !payload.speaker1 || !payload.speaker2) {
    return {
      error: 'Missing required fields: speaker1 or speaker2',
      overlaps: []
    };
  }
  
  // Отримуємо сегменти з обох доріжок
  const speaker1Segments = payload.speaker1.segments || 
                          payload.speaker1.speechmatics?.segments || 
                          [];
  const speaker2Segments = payload.speaker2.segments || 
                          payload.speaker2.speechmatics?.segments || 
                          [];
  
  if (speaker1Segments.length === 0 || speaker2Segments.length === 0) {
    return [];
  }
  
  // Функція для перевірки перекриття часових інтервалів
  function timeRangesOverlap(start1, end1, start2, end2, minOverlapSeconds = 0.1) {
    const overlapStart = Math.max(start1, start2);
    const overlapEnd = Math.min(end1, end2);
    const overlapDuration = overlapEnd - overlapStart;
    return overlapDuration >= minOverlapSeconds;
  }
  
  // Знаходимо перекриваючі сегменти
  for (const seg1 of speaker1Segments) {
    if (!seg1.text || !seg1.text.trim()) continue;
    if (!seg1.start && seg1.start !== 0) continue;
    if (!seg1.end && seg1.end !== 0) continue;
    
    const start1 = parseFloat(seg1.start) || 0;
    const end1 = parseFloat(seg1.end) || start1;
    
    if (end1 <= start1) continue;
    
    for (const seg2 of speaker2Segments) {
      if (!seg2.text || !seg2.text.trim()) continue;
      if (!seg2.start && seg2.start !== 0) continue;
      if (!seg2.end && seg2.end !== 0) continue;
      
      const start2 = parseFloat(seg2.start) || 0;
      const end2 = parseFloat(seg2.end) || start2;
      
      if (end2 <= start2) continue;
      
      // Перевіряємо перекриття
      if (timeRangesOverlap(start1, end1, start2, end2)) {
        const overlapStart = Math.max(start1, start2);
        const overlapEnd = Math.min(end1, end2);
        
        overlaps.push({
          text: `${seg1.text.trim()} | ${seg2.text.trim()}`,
          start: overlapStart,
          end: overlapEnd
        });
      }
    }
  }
  
  // Сортуємо за часом початку перекриття
  overlaps.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return overlaps;
}

/**
 * RED: Знаходить розбіжності та помилки транскрибації
 * Фрази, які є в окремих доріжках, але відсутні або відрізняються в general
 */
function findDiscrepancies(payload) {
  const discrepancies = [];
  
  // Перевіряємо наявність необхідних даних
  if (!payload || !payload.general || !payload.speaker1 || !payload.speaker2) {
    return {
      error: 'Missing required fields: general, speaker1, or speaker2',
      discrepancies: []
    };
  }
  
  // Отримуємо сегменти з primary транскрайбу
  const primarySegments = payload.general.segments || 
                         payload.general.speechmatics?.segments || 
                         [];
  
  if (primarySegments.length === 0) {
    return {
      error: 'No segments found in primary diarization',
      discrepancies: []
    };
  }
  
  // Обробляємо speaker1
  const speaker1Segments = payload.speaker1.segments || 
                          payload.speaker1.speechmatics?.segments || 
                          [];
  
  // Обробляємо speaker2
  const speaker2Segments = payload.speaker2.segments || 
                          payload.speaker2.speechmatics?.segments || 
                          [];
  
  // Функція для перевірки, чи існує сегмент в primary (з поверненням схожості)
  function existsInPrimaryWithSimilarity(segment, primarySegments, similarityThreshold = 0.85) {
    const segmentText = normalizeText(segment.text || '');
    if (!segmentText || segmentText.length < 3) return { exists: true, match: null, similarity: 1.0 };
    
    let bestMatch = null;
    let bestSimilarity = 0;
    
    for (const primarySeg of primarySegments) {
      const primaryText = normalizeText(primarySeg.text || '');
      if (!primaryText) continue;
      
      // Точний збіг
      if (segmentText === primaryText) {
        return { exists: true, match: primarySeg, similarity: 1.0 };
      }
      
      // Перевірка на включення
      if (segmentText.includes(primaryText) || primaryText.includes(segmentText)) {
        const shorter = segmentText.length < primaryText.length ? segmentText : primaryText;
        const longer = segmentText.length >= primaryText.length ? segmentText : primaryText;
        const ratio = shorter.length / longer.length;
        if (ratio >= 0.9) {
          return { exists: true, match: primarySeg, similarity: ratio };
        }
      }
      
      // Обчислюємо схожість
      const similarity = computeTextSimilarity(segmentText, primaryText);
      if (similarity > bestSimilarity) {
        bestSimilarity = similarity;
        bestMatch = primarySeg;
      }
    }
    
    // Якщо знайшли дуже схожий сегмент, вважаємо що існує
    if (bestSimilarity >= similarityThreshold) {
      return { exists: true, match: bestMatch, similarity: bestSimilarity };
    }
    
    return { exists: false, match: bestMatch, similarity: bestSimilarity };
  }
  
  // Знаходимо розбіжності для speaker1
  const seenTexts = new Set();
  for (const segment of speaker1Segments) {
    if (!segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 3) continue;
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    const checkResult = existsInPrimaryWithSimilarity(segment, primarySegments);
    
    if (!checkResult.exists) {
      // Фраза відсутня в primary - це помилка
      discrepancies.push({
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0
      });
    } else if (checkResult.match && checkResult.similarity) {
      // Перевіряємо, чи текст відрізняється (помилка транскрибації)
      if (checkResult.similarity < 0.95) {
        discrepancies.push({
          text: segment.text.trim(),
          start: segment.start || 0,
          end: segment.end || segment.start || 0
        });
      }
    }
  }
  
  // Знаходимо розбіжності для speaker2
  for (const segment of speaker2Segments) {
    if (!segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 3) continue;
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    const checkResult = existsInPrimaryWithSimilarity(segment, primarySegments);
    
    if (!checkResult.exists) {
      // Фраза відсутня в primary - це помилка
      discrepancies.push({
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0
      });
    } else if (checkResult.match && checkResult.similarity) {
      // Перевіряємо, чи текст відрізняється (помилка транскрибації)
      if (checkResult.similarity < 0.95) {
        discrepancies.push({
          text: segment.text.trim(),
          start: segment.start || 0,
          end: segment.end || segment.start || 0
        });
      }
    }
  }
  
  // Сортуємо за часом початку
  discrepancies.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return discrepancies;
}

/**
 * Головна функція для аналізу тексту
 * Повертає об'єкт з Blue, Green, Red
 */
function analyzeText(payload) {
  try {
    const blue = findRepeatedPhrases(payload);
    const green = findOverlaps(payload);
    const red = findDiscrepancies(payload);
    
    const result = {
      Blue: Array.isArray(blue) ? blue : [],
      Green: Array.isArray(green) ? green : [],
      Red: Array.isArray(red) ? red : []
    };
    
    // Логування результатів на сервері
    console.log('📊 Text Analysis Results:', {
      Blue: { count: result.Blue.length, samples: result.Blue.slice(0, 3).map(i => i.text?.substring(0, 50)) },
      Green: { count: result.Green.length, samples: result.Green.slice(0, 3).map(i => i.text?.substring(0, 50)) },
      Red: { count: result.Red.length, samples: result.Red.slice(0, 3).map(i => i.text?.substring(0, 50)) }
    });
    
    return result;
  } catch (error) {
    console.error('❌ Text analysis error:', error);
    return {
      Blue: [],
      Green: [],
      Red: [],
      error: error.message
    };
  }
}

module.exports = {
  findRepeatedPhrases,
  findOverlaps,
  findDiscrepancies,
  analyzeText
};

