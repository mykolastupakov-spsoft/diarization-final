/**
 * Text Analysis Functions: Blue, Green, Red
 * 
 * Аналізує результати діаризації та знаходить:
 * - Blue: Повторювані фрази (є і в general, і в speaker1/speaker2)
 * - Green: Overlaps (одночасна мова двох спікерів)
 * - Red: Розбіжності та помилки транскрибації
 */

const fs = require('fs');
const path = require('path');
const SegmentAlignmentEngine = require('./alignment-engine');

// Створюємо глобальний екземпляр alignment engine з адаптивними порогами
const alignmentEngine = new SegmentAlignmentEngine({
  exactThreshold: 0.98,
  temporalThreshold: 0.70,
  semanticThreshold: 0.55,
  partialThreshold: 0.40,
  timeTolerance: {
    exact: 0.5,
    temporal: 2.0,
    semantic: 4.0
  },
  minPhraseLength: 1,
  maxPhraseLength: 500,
  enableShortPhraseOptimization: true,
  metrics: {
    levenshtein: 0.4,
    jaccard: 0.3,
    sequence: 0.2,
    lcs: 0.1
  }
});

// Функція для нормалізації тексту (залишаємо для сумісності, але використовуємо engine.normalizeText)
function normalizeText(text) {
  return alignmentEngine.normalizeText(text);
}

// Покращена функція для обчислення схожості текстів (використовує alignment engine)
function computeTextSimilarity(text1, text2) {
  return alignmentEngine.computeAdvancedSimilarity(text1, text2);
}

// Покращена функція для перевірки, чи існує сегмент в primary транскрайбі
// Використовує SegmentAlignmentEngine для багатошарового вирівнювання
// ВАЖЛИВО: Завжди повертає об'єкт { found, similarity, timeMatch } для консистентності
function existsInPrimary(segment, primarySegments, similarityThreshold = 0.7, timeTolerance = 2.0) {
  // Валідація вхідних даних
  if (!segment || !primarySegments || !Array.isArray(primarySegments) || primarySegments.length === 0) {
    return { found: false, similarity: 0, timeMatch: false };
  }
  
  // Валідація сегмента
  if (!alignmentEngine.isValidSegment(segment)) {
    return { found: false, similarity: 0, timeMatch: false };
  }
  
  // Визначаємо sourceType на основі timeTolerance
  // Більший tolerance означає, що це може бути markdown або overlap
  const sourceType = timeTolerance >= 4.0 ? 'markdown' : 'general';
  
  // Використовуємо alignment engine для знаходження найкращого збігу
  const alignmentResult = alignmentEngine.alignSegment(segment, primarySegments, sourceType);
  
  // Адаптуємо пороги залежно від довжини фрази
  const normalizedText = normalizeText(segment.text || '');
  const isShortPhrase = normalizedText.length < 3;
  const isVeryShort = normalizedText.length < 2;
  
  // Адаптивні пороги на основі довжини фрази
  let effectiveThreshold;
  if (isVeryShort) {
    effectiveThreshold = 0.45; // Для дуже коротких фраз (1-2 символи)
  } else if (isShortPhrase) {
    effectiveThreshold = 0.50; // Для коротких фраз (3-5 символів)
  } else if (normalizedText.length < 6) {
    effectiveThreshold = 0.55; // Для середніх фраз (6-15 символів)
  } else {
    effectiveThreshold = similarityThreshold; // Для довгих фраз
  }
  
  // Перевіряємо, чи знайдено збіг з достатньою впевненістю
  if (alignmentResult.found && alignmentResult.confidence >= effectiveThreshold) {
    return {
      found: true,
      similarity: alignmentResult.similarity || alignmentResult.confidence,
      timeMatch: alignmentResult.timeMatch || false
    };
  }
  
  return { found: false, similarity: alignmentResult.similarity || 0, timeMatch: false };
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
  
  // Отримуємо інформацію про спікерів
  const speaker1Label = payload.speaker1?.role === 'operator' || payload.speaker1?.role === 'agent' ? 'Agent' : 
                        payload.speaker1?.role === 'client' || payload.speaker1?.role === 'customer' ? 'Client' : 
                        payload.speaker1?.speaker || 'Speaker 1';
  const speaker2Label = payload.speaker2?.role === 'operator' || payload.speaker2?.role === 'agent' ? 'Agent' : 
                        payload.speaker2?.role === 'client' || payload.speaker2?.role === 'customer' ? 'Client' : 
                        payload.speaker2?.speaker || 'Speaker 2';
  
  // Знаходимо повторювані фрази для speaker1
  const seenTexts = new Set();
  for (const segment of speaker1Segments) {
    if (!segment || !segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 1) continue; // Знижено з 3 для обробки коротких фраз
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    // Валідація часу
    const start = parseFloat(segment.start);
    const end = parseFloat(segment.end);
    if (isNaN(start) || isNaN(end) || end < start) continue;
    
    // Якщо фраза присутня в primary транскрайбі, вона повторюється
    // Використовуємо alignment engine для багатошарового вирівнювання
    // Адаптивні пороги залежно від довжини фрази
    const threshold = normalizedText.length < 3 ? 0.50 : (normalizedText.length < 6 ? 0.65 : 0.70);
    const matchResult = existsInPrimary(segment, primarySegments, threshold, 2.0);
    if (matchResult && matchResult.found && matchResult.similarity >= threshold) {
      repeatedPhrases.push({
        text: segment.text.trim(),
        start: start,
        end: end,
        speaker: speaker1Label // Додаємо інформацію про спікера
      });
    }
  }
  
  // Знаходимо повторювані фрази для speaker2
  for (const segment of speaker2Segments) {
    if (!segment || !segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 1) continue; // Знижено з 3 для обробки коротких фраз
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    // Валідація часу
    const start = parseFloat(segment.start);
    const end = parseFloat(segment.end);
    if (isNaN(start) || isNaN(end) || end < start) continue;
    
    // Якщо фраза присутня в primary транскрайбі, вона повторюється
    // Використовуємо alignment engine для багатошарового вирівнювання
    // Адаптивні пороги залежно від довжини фрази
    const threshold = normalizedText.length < 3 ? 0.50 : (normalizedText.length < 6 ? 0.65 : 0.70);
    const matchResult = existsInPrimary(segment, primarySegments, threshold, 2.0);
    if (matchResult && matchResult.found && matchResult.similarity >= threshold) {
      repeatedPhrases.push({
        text: segment.text.trim(),
        start: start,
        end: end,
        speaker: speaker2Label // Додаємо інформацію про спікера
      });
    }
  }
  
  // Сортуємо за часом початку
  repeatedPhrases.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return repeatedPhrases;
}

/**
 * GREEN: Знаходить фрази, які є в voice tracks (speaker1/speaker2), але відсутні в primary (general)
 * Це фрази, які були виявлені на третьому етапі (voice tracks), але не були в initial діаризації
 */
function findOverlaps(payload) {
  const overlaps = [];
  
  // Перевіряємо наявність необхідних даних
  if (!payload || !payload.general || (!payload.speaker1 && !payload.speaker2)) {
    return {
      error: 'Missing required fields: general, speaker1, or speaker2',
      overlaps: []
    };
  }
  
  // Отримуємо сегменти з primary транскрайбу
  const primarySegments = payload.general.segments || 
                         payload.general.speechmatics?.segments || 
                         [];
  
  if (primarySegments.length === 0) {
    return [];
  }
  
  // Отримуємо сегменти з voice tracks
  const speaker1Segments = payload.speaker1?.segments || 
                          payload.speaker1?.speechmatics?.segments || 
                          [];
  const speaker2Segments = payload.speaker2?.segments || 
                          payload.speaker2?.speechmatics?.segments || 
                          [];
  
  // Об'єднуємо всі voice track сегменти
  const allVoiceTrackSegments = [...speaker1Segments, ...speaker2Segments];
  
  if (allVoiceTrackSegments.length === 0) {
    return [];
  }
  
  // Отримуємо інформацію про спікерів
  const speaker1Label = payload.speaker1?.role === 'operator' || payload.speaker1?.role === 'agent' ? 'Agent' : 
                        payload.speaker1?.role === 'client' || payload.speaker1?.role === 'customer' ? 'Client' : 
                        payload.speaker1?.speaker || 'Speaker 1';
  const speaker2Label = payload.speaker2?.role === 'operator' || payload.speaker2?.role === 'agent' ? 'Agent' : 
                        payload.speaker2?.role === 'client' || payload.speaker2?.role === 'customer' ? 'Client' : 
                        payload.speaker2?.speaker || 'Speaker 2';
  
  // Знаходимо фрази, які є в voice tracks, але відсутні в primary
  const seenTexts = new Set();
  
  // Обробляємо speaker1 сегменти
  for (const segment of speaker1Segments) {
    if (!segment || !segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 1) continue; // Знижено з 3 для обробки коротких фраз
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    // Валідація часу
    const start = parseFloat(segment.start);
    const end = parseFloat(segment.end);
    if (isNaN(start) || isNaN(end) || end < start) continue;
    
    // Перевіряємо, чи ця фраза відсутня в primary
    // Якщо фраза є в voice tracks, але не в primary - це Green
    // Використовуємо більш гнучку перевірку для Green (overlap діаризація)
    // Знижуємо поріг схожості та збільшуємо time tolerance для кращого виявлення overlap
    const threshold = normalizedText.length < 3 ? 0.4 : 0.55; // Знижено пороги для кращого виявлення
    const timeTolerance = 4.0; // Збільшено time tolerance для overlap
    const matchResult = existsInPrimary(segment, primarySegments, threshold, timeTolerance);
    if (!matchResult || !matchResult.found || matchResult.similarity < threshold) {
      overlaps.push({
        text: segment.text.trim(),
        start: start,
        end: end,
        speaker: speaker1Label // Додаємо інформацію про спікера
      });
    }
  }
  
  // Обробляємо speaker2 сегменти
  for (const segment of speaker2Segments) {
    if (!segment || !segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 1) continue; // Знижено з 3 для обробки коротких фраз
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    // Валідація часу
    const start = parseFloat(segment.start);
    const end = parseFloat(segment.end);
    if (isNaN(start) || isNaN(end) || end < start) continue;
      
    // Перевіряємо, чи ця фраза відсутня в primary
    // Якщо фраза є в voice tracks, але не в primary - це Green
    // Використовуємо більш гнучку перевірку для Green (overlap діаризація)
    // Знижуємо поріг схожості та збільшуємо time tolerance для кращого виявлення overlap
    const threshold = normalizedText.length < 3 ? 0.4 : 0.55; // Знижено пороги для кращого виявлення
    const timeTolerance = 4.0; // Збільшено time tolerance для overlap
    const matchResult = existsInPrimary(segment, primarySegments, threshold, timeTolerance);
    if (!matchResult || !matchResult.found || matchResult.similarity < threshold) {
        overlaps.push({
        text: segment.text.trim(),
        start: start,
        end: end,
        speaker: speaker2Label // Додаємо інформацію про спікера
        });
      }
    }
  
  console.log('🟢 Green (findOverlaps):', {
    count: overlaps.length,
    samples: overlaps.slice(0, 3).map(i => ({ text: i.text?.substring(0, 50), start: i.start, end: i.end }))
  });
  
  // Сортуємо за часом початку
  overlaps.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return overlaps;
}

/**
 * RED: Знаходить фрази з markdown table, які відсутні і в primary, і в voice tracks
 * Це галюцинації - фрази, які LLM додав, але яких немає ні в initial діаризації, ні в voice tracks
 * 
 * ВАЖЛИВО: Фрази, які є в voice tracks (навіть якщо немає в primary), НЕ мають потрапляти в Red
 * Такі фрази мають бути в Green
 */
function findDiscrepancies(payload) {
  const discrepancies = [];
  
  // Перевіряємо наявність необхідних даних
  if (!payload || !payload.general || (!payload.speaker1 && !payload.speaker2)) {
    return {
      error: 'Missing required fields: general, speaker1, or speaker2',
      discrepancies: []
    };
  }
  
  // Отримуємо сегменти з primary транскрайбу
  const primarySegments = payload.general.segments || 
                         payload.general.speechmatics?.segments || 
                         [];
  
  // Отримуємо сегменти з voice tracks
  const speaker1Segments = payload.speaker1?.segments || 
                          payload.speaker1?.speechmatics?.segments || 
                          [];
  const speaker2Segments = payload.speaker2?.segments || 
                          payload.speaker2?.speechmatics?.segments || 
                          [];
  
  // Об'єднуємо всі voice track сегменти
  const allVoiceTrackSegments = [...speaker1Segments, ...speaker2Segments];
  
  // Функція для перевірки, чи існує текст в primary
  // Для Red використовуємо більш строгу перевірку, щоб не помічати як галюцинації те, що частково є
  function existsInPrimaryCheck(textToCheck, primarySegments, startTime = null, endTime = null) {
    if (!textToCheck || !textToCheck.trim()) return true; // Порожній текст вважаємо існуючим
    
    const normalizedTextToCheck = normalizeText(textToCheck);
    if (!normalizedTextToCheck || normalizedTextToCheck.length < 1) return true; // Знижено з 3
    
    // Валідація часу
    const start = startTime !== null ? parseFloat(startTime) : null;
    const end = endTime !== null ? parseFloat(endTime) : null;
    if (start !== null && (isNaN(start) || (end !== null && (isNaN(end) || end < start)))) {
      return true; // Невірний час - вважаємо існуючим для безпеки
    }
    
    const testSegment = { 
      text: textToCheck,
      start: start !== null && !isNaN(start) ? start : 0,
      end: end !== null && !isNaN(end) ? end : (start !== null && !isNaN(start) ? start : 0)
    };
      
    // Для коротких фраз використовуємо більш гнучкий поріг
    const threshold = normalizedTextToCheck.length < 3 ? 0.45 : 0.55;
    // Time tolerance 2.5 секунди для врахування часу
    const result = existsInPrimary(testSegment, primarySegments, threshold, 2.5);
    return result && result.found && result.similarity >= threshold;
      }
      
  // Функція для перевірки, чи існує текст в voice tracks
  // Для Red використовуємо більш строгу перевірку
  function existsInVoiceTracks(textToCheck, voiceTrackSegments, startTime = null, endTime = null) {
    if (!textToCheck || !textToCheck.trim()) return true; // Порожній текст вважаємо існуючим
    
    const normalizedTextToCheck = normalizeText(textToCheck);
    if (!normalizedTextToCheck || normalizedTextToCheck.length < 1) return true; // Знижено з 3
    
    // Валідація часу
    const start = startTime !== null ? parseFloat(startTime) : null;
    const end = endTime !== null ? parseFloat(endTime) : null;
    if (start !== null && (isNaN(start) || (end !== null && (isNaN(end) || end < start)))) {
      return true; // Невірний час - вважаємо існуючим для безпеки
    }
    
    const testSegment = { 
      text: textToCheck,
      start: start !== null && !isNaN(start) ? start : 0,
      end: end !== null && !isNaN(end) ? end : (start !== null && !isNaN(start) ? start : 0)
    };
    
    // Для коротких фраз використовуємо більш гнучкий поріг
    const threshold = normalizedTextToCheck.length < 3 ? 0.45 : 0.55;
    // Time tolerance 2.5 секунди для врахування часу
    const result = existsInPrimary(testSegment, voiceTrackSegments, threshold, 2.5);
    return result && result.found && result.similarity >= threshold;
  }
  
  // Парсимо markdown table (якщо є) і знаходимо тексти, які відсутні і в primary, і в voice tracks
  // ВАЖЛИВО: Спочатку отримуємо список Green фраз, щоб виключити їх з Red
  const greenTexts = new Set();
  if (payload.markdown) {
    // Отримуємо Green фрази з попереднього кроку (якщо вони вже обчислені)
    // Але оскільки analyzeText викликає функції послідовно, ми не маємо доступу до Green тут
    // Тому перевіряємо безпосередньо: якщо текст є в voice tracks, він не має бути в Red
  }
  
  if (payload.markdown) {
    const markdownLines = payload.markdown.split('\n');
    
    for (const line of markdownLines) {
      if (!line.trim().startsWith('|')) continue;
      
      // Пропускаємо separator row (|---|---|)
      if (line.trim().match(/^\|\s*:?-+:?\s*\|/)) continue;
    
      const cells = line.split('|').map(c => c.trim()).filter(c => c.length > 0);
      // Припускаємо формат: Segment ID | Speaker | Text | Start Time | End Time
      if (cells.length >= 5) {
        const text = cells[2]; // Text column
        const startTime = parseFloat(cells[3]) || 0;
        const endTime = parseFloat(cells[4]) || startTime;
        
        if (!text || !text.trim()) continue;
    
        // Перевіряємо, чи цей текст відсутній і в primary, і в voice tracks
        // Якщо текст є хоча б в одному з них, він НЕ має потрапляти в Red
        // Передаємо час для більш точної перевірки
        const existsInPrimaryResult = existsInPrimaryCheck(text, primarySegments, startTime, endTime);
        const existsInVoiceTracksResult = existsInVoiceTracks(text, allVoiceTrackSegments, startTime, endTime);
        
        // Діагностичне логування для перших кількох фраз
        if (discrepancies.length < 3) {
          console.log('🔍 Checking markdown text for Red:', {
            text: text.substring(0, 50),
            existsInPrimary: existsInPrimaryResult,
            existsInVoiceTracks: existsInVoiceTracksResult,
            willBeRed: !existsInPrimaryResult && !existsInVoiceTracksResult
          });
        }
        
        // Отримуємо спікера з markdown таблиці
        const markdownSpeaker = cells.length >= 2 ? cells[1].trim() : null; // Speaker column
        
        // Red: текст відсутній і в primary, і в voice tracks (галюцинація LLM)
        // Якщо текст є в voice tracks (навіть якщо немає в primary), він має бути в Green, а не в Red
        // Якщо текст є в primary (навіть якщо немає в voice tracks), він має бути в Blue, а не в Red
        // ВАЖЛИВО: Якщо текст є і в primary, і в voice tracks, він точно не має бути в Red
        if (!existsInPrimaryResult && !existsInVoiceTracksResult) {
      discrepancies.push({
            text: text.trim(),
            start: startTime,
            end: endTime,
            speaker: markdownSpeaker // Додаємо інформацію про спікера з markdown
      });
        } else {
          // Діагностичне логування: чому фраза не потрапила в Red
          if (discrepancies.length < 3) {
            console.log('✅ Text excluded from Red:', {
              text: text.substring(0, 50),
              reason: existsInPrimaryResult ? 'exists in primary' : 'exists in voice tracks'
      });
          }
        }
      }
    }
  }
  
  console.log('🔴 Red (findDiscrepancies):', {
    count: discrepancies.length,
    samples: discrepancies.slice(0, 3).map(i => ({ text: i.text?.substring(0, 50), start: i.start, end: i.end }))
  });
  
  // Сортуємо за часом початку
  discrepancies.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return discrepancies;
}

/**
 * Знаходить перекриття в markdown таблиці (End попереднього > Start наступного)
 * Повертає масив об'єктів з інформацією про перекриваючі сегменти
 */
function findOverlappingSegments(markdown) {
  if (!markdown || typeof markdown !== 'string') return [];
  
  const segments = [];
  const lines = markdown.split('\n');
  let inTable = false;
  let headers = [];
  let headerProcessed = false;
  
  for (const line of lines) {
    const trimmedLine = line.trim();
    
    if (trimmedLine.startsWith('|') && trimmedLine.endsWith('|')) {
      if (!inTable) {
        inTable = true;
        headerProcessed = false;
        headers = [];
      }
      
      const cells = trimmedLine.split('|').map(c => c.trim()).filter(c => c.length > 0);
      
      if (!headerProcessed) {
        headers = cells;
        headerProcessed = true;
        continue;
      }
      
      // Парсимо рядок таблиці
      if (cells.length >= 5) {
        // Очікуваний формат: Segment ID | Speaker | Text | Start Time | End Time
        const segmentId = cells[0];
        const speaker = cells[1];
        const text = cells[2];
        const startTime = parseFloat(cells[3]);
        const endTime = parseFloat(cells[4]);
        
        if (!isNaN(startTime) && !isNaN(endTime) && endTime > startTime && text && text.trim()) {
          segments.push({
            segmentId,
            speaker: speaker.trim(),
            text: text.trim(),
            start: startTime,
            end: endTime
          });
        }
      }
    } else if (inTable && trimmedLine === '') {
      // Порожній рядок після таблиці - можливо кінець
      continue;
    } else if (inTable && !trimmedLine.startsWith('|')) {
      // Не табличний рядок після початку таблиці - можливо кінець
      break;
    }
  }
  
  // Сортуємо за часом початку
  segments.sort((a, b) => a.start - b.start);
  
  // Знаходимо перекриття: End попереднього > Start наступного
  const overlappingSegments = [];
  for (let i = 0; i < segments.length - 1; i++) {
    const current = segments[i];
    const next = segments[i + 1];
    
    if (current.end > next.start) {
      // Знайдено перекриття
      overlappingSegments.push({
        previous: current,
        next: next,
        overlapStart: next.start,
        overlapEnd: Math.min(current.end, next.end)
      });
    }
  }
  
  return overlappingSegments;
}

/**
 * Перевіряє, чи існує фрагмент в voice tracks для конкретного спікера
 */
function existsInVoiceTracksForSpeaker(text, startTime, endTime, speaker, voiceTrackSegments) {
  if (!text || !text.trim() || !voiceTrackSegments || voiceTrackSegments.length === 0) {
    return false;
  }
  
  const normalizedText = normalizeText(text);
  if (normalizedText.length < 1) return false;
  
  const threshold = normalizedText.length < 3 ? 0.4 : 0.55;
  const timeTolerance = 4.0;
  
  const testSegment = {
    text: text,
    start: parseFloat(startTime) || 0,
    end: parseFloat(endTime) || 0
  };
  
  // Фільтруємо сегменти за спікером
  const speakerSegments = voiceTrackSegments.filter(seg => {
    if (!seg || !seg.speaker) return false;
    const segSpeaker = seg.speaker.trim().toLowerCase();
    const targetSpeaker = speaker.trim().toLowerCase();
    return segSpeaker === targetSpeaker || 
           (segSpeaker.includes('agent') && targetSpeaker.includes('agent')) ||
           (segSpeaker.includes('client') && targetSpeaker.includes('client'));
  });
  
  if (speakerSegments.length === 0) return false;
  
  const result = existsInPrimary(testSegment, speakerSegments, threshold, timeTolerance);
  return result && result.found && result.similarity >= threshold;
}

/**
 * Перевіряє, чи існує фрагмент в general (primary) для конкретного спікера
 */
function existsInGeneralForSpeaker(text, startTime, endTime, speaker, generalSegments) {
  if (!text || !text.trim() || !generalSegments || generalSegments.length === 0) {
    return false;
  }
  
  const normalizedText = normalizeText(text);
  if (normalizedText.length < 1) return false;
  
  const threshold = normalizedText.length < 3 ? 0.5 : 0.65;
  const timeTolerance = 2.5;
  
  const testSegment = {
    text: text,
    start: parseFloat(startTime) || 0,
    end: parseFloat(endTime) || 0
  };
  
  // Фільтруємо сегменти за спікером (якщо є інформація про спікера)
  let filteredSegments = generalSegments;
  if (speaker) {
    filteredSegments = generalSegments.filter(seg => {
      if (!seg || !seg.speaker) return true; // Якщо немає спікера, включаємо
      const segSpeaker = seg.speaker.trim().toLowerCase();
      const targetSpeaker = speaker.trim().toLowerCase();
      return segSpeaker === targetSpeaker || 
             (segSpeaker.includes('agent') && targetSpeaker.includes('agent')) ||
             (segSpeaker.includes('client') && targetSpeaker.includes('client'));
    });
  }
  
  if (filteredSegments.length === 0) return false;
  
  const result = existsInPrimary(testSegment, filteredSegments, threshold, timeTolerance);
  return result && result.found && result.similarity >= threshold;
}

/**
 * Головна функція для аналізу тексту
 * Повертає об'єкт з Blue, Green, Red
 * ВАЖЛИВО: Виконує дедуплікацію між категоріями для уникнення перекриття
 * 
 * НОВА ЛОГІКА:
 * 1. Знаходимо перекриття в markdown таблиці (End попереднього > Start наступного)
 * 2. Для перекриваючих фрагментів:
 *    - Якщо є в voice tracks, але немає в general - Green
 *    - Якщо немає в voice tracks - Red
 *    - Тільки для правильного спікера
 * 3. Для неперекриваючих фрагментів:
 *    - Якщо фраза належить цьому спікеру в general - Blue
 */
/**
 * Простий підхід: нормалізуємо текст і перевіряємо через includes()
 */
function normalizeTextSimple(text) {
  if (!text || typeof text !== 'string') return '';
  return text.toLowerCase().replace(/[,.!?]/g, '').trim();
}

/**
 * Парсить markdown таблицю для отримання сегментів
 */
function parseMarkdownSegments(markdown) {
  if (!markdown || typeof markdown !== 'string') return [];
  
  const segments = [];
  const lines = markdown.split('\n');
  let headerProcessed = false;
  
  for (const line of lines) {
    const trimmedLine = line.trim();
    
    // Пропускаємо header та separator
    if (trimmedLine.includes('---') || trimmedLine.toLowerCase().includes('segment id')) {
      headerProcessed = true;
      continue;
    }
    
    if (trimmedLine.startsWith('|') && headerProcessed) {
      const cells = trimmedLine.split('|').map(c => c.trim()).filter(c => c);
      
      if (cells.length >= 5) {
        // Format: Segment ID | Speaker | Text | Start Time | End Time
        const segmentId = cells[0];
        const speaker = cells[1];
        const text = cells[2];
        const startTime = parseFloat(cells[3]) || 0;
        const endTime = parseFloat(cells[4]) || 0;
        
        if (text && text.length > 0) {
          segments.push({
            id: segmentId,
            speaker: speaker,
            text: text,
            start: startTime,
            end: endTime
          });
        }
      }
    }
  }
  
  return segments;
}

function analyzeText(payload) {
  try {
    // Валідація вхідних даних
    if (!payload) {
      console.warn('⚠️ analyzeText: payload is null or undefined');
      return { Blue: [], Green: [], Red: [] };
    }
    
    // Отримуємо сегменти з джерел
    const generalSegments = payload.general?.segments || 
                           payload.general?.speechmatics?.segments || 
                           [];
    const speaker1Segments = payload.speaker1?.segments || 
                            payload.speaker1?.speechmatics?.segments || 
                            [];
    const speaker2Segments = payload.speaker2?.segments || 
                            payload.speaker2?.speechmatics?.segments || 
                            [];
    
    // Парсимо markdown таблицю
    const markdownSegments = parseMarkdownSegments(payload.markdown || '');
    
    if (markdownSegments.length === 0) {
      console.warn('⚠️ analyzeText: No segments found in markdown');
      return { Blue: [], Green: [], Red: [] };
    }
    
    // Результати
    const blue = [];
    const green = [];
    const red = [];
    
    // Для кожного сегмента з markdown
    for (const mdSegment of markdownSegments) {
      const normalizedMd = normalizeTextSimple(mdSegment.text);
      if (!normalizedMd) continue;
      
      // Перевіряємо наявність у конкретних джерелах
      let foundInGeneral = false;
      let foundInSpeaker1 = false;
      let foundInSpeaker2 = false;
      
      // Перевірка в general
      for (const seg of generalSegments) {
        if (!seg.text) continue;
        const normalizedSeg = normalizeTextSimple(seg.text);
        if (normalizedSeg.includes(normalizedMd) || normalizedMd.includes(normalizedSeg)) {
          foundInGeneral = true;
          break;
        }
      }
      
      // Перевірка в speaker1
      for (const seg of speaker1Segments) {
        if (!seg.text) continue;
        const normalizedSeg = normalizeTextSimple(seg.text);
        if (normalizedSeg.includes(normalizedMd) || normalizedMd.includes(normalizedSeg)) {
          foundInSpeaker1 = true;
          break;
        }
      }
      
      // Перевірка в speaker2
      for (const seg of speaker2Segments) {
        if (!seg.text) continue;
        const normalizedSeg = normalizeTextSimple(seg.text);
        if (normalizedSeg.includes(normalizedMd) || normalizedMd.includes(normalizedSeg)) {
          foundInSpeaker2 = true;
          break;
        }
      }
      
      // Класифікуємо результат
      const segmentResult = {
        text: mdSegment.text,
        start: mdSegment.start,
        end: mdSegment.end,
        speaker: mdSegment.speaker
      };
      
      // Blue: є в general І (є в speaker1 АБО speaker2)
      if (foundInGeneral && (foundInSpeaker1 || foundInSpeaker2)) {
        blue.push(segmentResult);
      }
      // Green: є в speaker1 АБО speaker2, але НЕ в general
      else if ((foundInSpeaker1 || foundInSpeaker2) && !foundInGeneral) {
        green.push(segmentResult);
      }
      // Red: немає в жодному джерелі
      else {
        red.push(segmentResult);
      }
    }
    
    const result = {
      Blue: blue,
      Green: green,
      Red: red
    };
    
    // Логування результатів
    console.log('📊 Text Analysis Results (Simple):', {
      Blue: { count: result.Blue.length, samples: result.Blue.slice(0, 3).map(i => i.text?.substring(0, 50)) },
      Green: { count: result.Green.length, samples: result.Green.slice(0, 3).map(i => i.text?.substring(0, 50)) },
      Red: { count: result.Red.length, samples: result.Red.slice(0, 3).map(i => i.text?.substring(0, 50)) },
      totalSegments: markdownSegments.length
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

/**
 * Аналізує текст за допомогою LLM для класифікації фрагментів
 * LLM отримує markdown таблицю та сегменти з general, speaker1, speaker2
 * і повертає класифікацію: Blue (звичайна діаризація), Green (overlap), Red (галюцинації)
 * 
 * ВАЖЛИВО: Ця функція НЕ використовує кеш, оскільки аналіз фраз для маркування
 * має відбуватися кожен раз заново для точного визначення кольорів.
 * Кеш використовується тільки для аналізу ролей та генерації markdown таблиці.
 */
async function analyzeTextWithLLM(payload, llmModel, apiUrl, apiKey, useLocalLLM = false, mode = 'smart') {
  try {
    if (!payload) {
      console.warn('⚠️ analyzeTextWithLLM: payload is missing');
      return { Blue: [], Green: [], Red: [] };
    }
    
    // ВАЖЛИВО: Не використовуємо кеш для text analysis
    // Кеш використовується тільки для аналізу ролей та markdown таблиці
    // Маркування фраз через LLM має виконуватися кожен раз заново для точного визначення кольорів
    console.log('🤖 ============================================');
    console.log('🤖 Calling LLM for text analysis classification');
    console.log('🤖 ВАЖЛИВО: Без кешування - кожен раз заново!');
    console.log('🤖 ============================================');
    console.log('📋 LLM text analysis parameters:', {
      llmModel: llmModel,
      apiUrl: apiUrl,
      useLocalLLM: useLocalLLM,
      mode: mode,
      hasMarkdown: !!payload.markdown,
      markdownLength: payload.markdown?.length || 0,
      hasGeneral: !!payload.general,
      hasSpeaker1: !!payload.speaker1,
      hasSpeaker2: !!payload.speaker2
    });
    
    // Підготовка даних для LLM
    const generalSegments = payload.general?.segments || 
                           payload.general?.speechmatics?.segments || 
                           [];
    const speaker1Segments = payload.speaker1?.segments || 
                            payload.speaker1?.speechmatics?.segments || 
                            [];
    const speaker2Segments = payload.speaker2?.segments || 
                            payload.speaker2?.speechmatics?.segments || 
                            [];
    
    // Створюємо JSON з сегментами для LLM (повні дані, не обмежуємо)
    const segmentsData = {
      general: {
        speechmatics: { segments: generalSegments },
        segments: generalSegments
      },
      speaker1: {
        speechmatics: { segments: speaker1Segments },
        segments: speaker1Segments
      },
      speaker2: {
        speechmatics: { segments: speaker2Segments },
        segments: speaker2Segments
      }
    };
    
    // Читаємо три окремі промпти з файлів
    const promptsDir = path.join(__dirname, '..', 'docs');
    const bluePromptPath = path.join(promptsDir, 'n8n_ai_blue_repeated_phrases_prompt.txt');
    const greenPromptPath = path.join(promptsDir, 'n8n_ai_green_overlaps_prompt.txt');
    const redPromptPath = path.join(promptsDir, 'n8n_ai_red_discrepancies_prompt.txt');
    
    let bluePrompt = '';
    let greenPrompt = '';
    let redPrompt = '';
    
    try {
      bluePrompt = fs.readFileSync(bluePromptPath, 'utf8');
      console.log('✅ Loaded Blue prompt from:', bluePromptPath);
    } catch (err) {
      console.error('❌ Failed to load Blue prompt:', err.message);
      bluePrompt = 'Find segments from general that have matching text in speaker1 or speaker2 tracks. Return JSON array.';
    }
    
    try {
      greenPrompt = fs.readFileSync(greenPromptPath, 'utf8');
      console.log('✅ Loaded Green prompt from:', greenPromptPath);
    } catch (err) {
      console.error('❌ Failed to load Green prompt:', err.message);
      greenPrompt = 'Find overlapping speech segments between speaker1 and speaker2. Return JSON array.';
    }
    
    try {
      redPrompt = fs.readFileSync(redPromptPath, 'utf8');
      console.log('✅ Loaded Red prompt from:', redPromptPath);
    } catch (err) {
      console.error('❌ Failed to load Red prompt:', err.message);
      redPrompt = 'Find discrepancies and missing phrases. Return JSON array.';
    }
    
    // Замінюємо плейсхолдер {{ $json.body }} на реальні дані
    const replacePlaceholder = (promptText) => {
      return promptText.replace(/\{\{\s*\$json\.body\s*\}\}/g, JSON.stringify(segmentsData, null, 2));
    };
    
    const bluePromptFinal = replacePlaceholder(bluePrompt);
    const greenPromptFinal = replacePlaceholder(greenPrompt);
    const redPromptFinal = replacePlaceholder(redPrompt);
    
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (useLocalLLM) {
      if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
      }
    } else {
      headers['Authorization'] = `Bearer ${apiKey}`;
      headers['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
      headers['X-Title'] = 'Text Analysis Classification';
    }

    const axios = require('axios');
    const timeout = useLocalLLM ? 1800000 : 60000; // 30 хвилин для локальної, 1 хвилина для віддаленої
    
    // Функція для виклику LLM з промптом
    const callLLMWithPrompt = async (promptText, colorName) => {
      const llmPayload = {
        model: llmModel,
        messages: [
          {
            role: 'system',
            content: `You are an expert in speaker diarization analysis. Analyze the provided data and return ONLY a valid JSON array. No explanations, no markdown, no code blocks.`
          },
          {
            role: 'user',
            content: promptText
          }
        ],
        temperature: 0
      };
      
      // Для локальних моделей не використовуємо response_format: json_object
      if (!useLocalLLM) {
        llmPayload.response_format = { type: 'json_object' };
      }
      
      // Add reasoning effort if needed
      const shouldUseHighReasoning = mode === 'smart' || mode === 'fast' || mode === 'local' || 
                                      llmModel?.includes('gpt-4') || llmModel?.includes('gpt-5') || 
                                      llmModel?.includes('o1') || llmModel?.includes('o3');
      if (!useLocalLLM && shouldUseHighReasoning) {
        llmPayload.reasoning = { effort: 'high' };
      }
      
      console.log(`🔵 [${colorName}] Calling LLM...`);
      const response = await axios.post(apiUrl, llmPayload, { headers, timeout });
      
      if (response.data && response.data.choices && response.data.choices[0]) {
        const content = response.data.choices[0].message?.content;
        if (content) {
          try {
            // Спробуємо витягти JSON масив
            let jsonStr = content;
            
            // Видаляємо markdown code blocks якщо є
            const codeBlockMatch = content.match(/```(?:json)?\s*(\[[\s\S]*\])\s*```/);
            if (codeBlockMatch) {
              jsonStr = codeBlockMatch[1];
            } else {
              // Спробуємо знайти JSON масив безпосередньо
              const arrayMatch = content.match(/(\[[\s\S]*\])/);
              if (arrayMatch) {
                jsonStr = arrayMatch[1];
              }
            }
            
            const parsed = JSON.parse(jsonStr);
            
            // Перевіряємо, чи це масив
            if (Array.isArray(parsed)) {
              return parsed;
            } else if (parsed && typeof parsed === 'object') {
              // Якщо це об'єкт, спробуємо знайти масив всередині
              const arrayKeys = Object.keys(parsed).filter(key => Array.isArray(parsed[key]));
              if (arrayKeys.length > 0) {
                return parsed[arrayKeys[0]];
              }
            }
            
            return [];
          } catch (parseError) {
            console.error(`❌ [${colorName}] Failed to parse LLM response:`, parseError.message);
            console.error(`❌ [${colorName}] Response content (first 500 chars):`, content.substring(0, 500));
            return [];
          }
        }
      }
      
      return [];
    };
    
    // Викликаємо три окремі LLM запити паралельно
    console.log('🚀 Starting three parallel LLM calls for Blue, Green, Red...');
    const [blueResult, greenResult, redResult] = await Promise.all([
      callLLMWithPrompt(bluePromptFinal, 'Blue'),
      callLLMWithPrompt(greenPromptFinal, 'Green'),
      callLLMWithPrompt(redPromptFinal, 'Red')
    ]);
    
    const result = {
      Blue: Array.isArray(blueResult) ? blueResult : [],
      Green: Array.isArray(greenResult) ? greenResult : [],
      Red: Array.isArray(redResult) ? redResult : []
    };
    
    console.log('✅ LLM text analysis completed:', {
      blueCount: result.Blue.length,
      greenCount: result.Green.length,
      redCount: result.Red.length
    });
    
    console.log('✅ [analyzeTextWithLLM] Returning result:', {
      blueCount: result.Blue.length,
      greenCount: result.Green.length,
      redCount: result.Red.length
    });
    
    return result;
  } catch (error) {
    console.error('❌ [analyzeTextWithLLM] ============================================');
    console.error('❌ [analyzeTextWithLLM] LLM text analysis error:', error.message);
    console.error('❌ [analyzeTextWithLLM] Error type:', error.constructor.name);
    console.error('❌ [analyzeTextWithLLM] Error details:', error);
    if (error.response) {
      console.error('❌ [analyzeTextWithLLM] Response status:', error.response.status);
      console.error('❌ [analyzeTextWithLLM] Response data:', error.response.data);
    }
    console.error('❌ [analyzeTextWithLLM] ============================================');
    
    // Прокидаємо помилку далі, щоб сервер міг її обробити
    throw error;
  }
}

// Helper function to check if high reasoning effort should be used
function shouldUseHighReasoningEffort(mode, model) {
  // Використовуємо high reasoning для складних моделей
  return mode === 'smart' || mode === 'fast' || mode === 'local' || 
         model?.includes('gpt-4') || model?.includes('gpt-5') || 
         model?.includes('o1') || model?.includes('o3');
}

module.exports = {
  findRepeatedPhrases,
  findOverlaps,
  findDiscrepancies,
  analyzeText,
  analyzeTextWithLLM
};

