/**
 * n8n Function Node: Red - Discrepancies and Transcription Errors
 * 
 * Знаходить розбіжності та помилки транскрибації:
 * 1. Фрази, які є в окремих доріжках (speaker1/speaker2), але відсутні в general
 * 2. Фрази з різним текстом між general та окремими доріжками (помилки транскрибації)
 * 
 * Input: Webhook payload з полями general, speaker1, speaker2
 * Output: Масив об'єктів з розбіжностями та помилками
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

// Функція для знаходження найближчого сегменту в primary за часом
function findClosestPrimarySegmentByTime(segment, primarySegments, timeTolerance = 2.0) {
  const segStart = parseFloat(segment.start) || 0;
  const segEnd = parseFloat(segment.end) || segStart;
  const segMid = (segStart + segEnd) / 2;
  
  let closest = null;
  let minDistance = Infinity;
  
  for (const primarySeg of primarySegments) {
    const primaryStart = parseFloat(primarySeg.start) || 0;
    const primaryEnd = parseFloat(primarySeg.end) || primaryStart;
    const primaryMid = (primaryStart + primaryEnd) / 2;
    
    const distance = Math.abs(segMid - primaryMid);
    
    // Перевіряємо, чи сегменти перекриваються за часом
    const timeOverlap = Math.max(segStart, primaryStart) <= Math.min(segEnd, primaryEnd);
    
    if (timeOverlap || distance <= timeTolerance) {
      if (distance < minDistance) {
        minDistance = distance;
        closest = primarySeg;
      }
    }
  }
  
  return closest;
}

// Функція для перевірки, чи існує сегмент в primary транскрайбі
function existsInPrimary(segment, primarySegments, similarityThreshold = 0.85) {
  const segmentText = normalizeText(segment.text || '');
  if (!segmentText || segmentText.length < 3) return { exists: true, match: null };
  
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

// Основна функція для знаходження розбіжностей та помилок
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
  const speaker1Role = payload.speaker1.role || 'unknown';
  const speaker1Label = speaker1Role === 'operator' || speaker1Role === 'agent' 
    ? 'Agent' 
    : speaker1Role === 'client' || speaker1Role === 'customer' 
    ? 'Client' 
    : 'Speaker 1';
  
  // Обробляємо speaker2
  const speaker2Segments = payload.speaker2.segments || 
                          payload.speaker2.speechmatics?.segments || 
                          [];
  const speaker2Role = payload.speaker2.role || 'unknown';
  const speaker2Label = speaker2Role === 'operator' || speaker2Role === 'agent' 
    ? 'Agent' 
    : speaker2Role === 'client' || speaker2Role === 'customer' 
    ? 'Client' 
    : 'Speaker 2';
  
  // Знаходимо розбіжності для speaker1
  const seenTexts = new Set();
  for (const segment of speaker1Segments) {
    if (!segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 3) continue;
    if (seenTexts.has(normalizedText)) continue;
    seenTexts.add(normalizedText);
    
    const checkResult = existsInPrimary(segment, primarySegments);
    
    if (!checkResult.exists) {
      // Фраза відсутня в primary - це помилка
      discrepancies.push({
        speaker: payload.speaker1.speaker || 'SPEAKER_00',
        role: speaker1Role,
        roleLabel: speaker1Label,
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0,
        source: 'speaker1',
        confidence: segment.confidence || null,
        errorType: 'missing',
        description: 'Phrase present in speaker track but missing from primary transcript'
      });
    } else if (checkResult.match && checkResult.similarity) {
      // Перевіряємо, чи текст відрізняється (помилка транскрибації)
      // Якщо схожість менша за високий поріг - можлива помилка
      if (checkResult.similarity < 0.95) {
        discrepancies.push({
          speaker: payload.speaker1.speaker || 'SPEAKER_00',
          role: speaker1Role,
          roleLabel: speaker1Label,
          text: segment.text.trim(),
          primaryText: checkResult.match.text.trim(),
          start: segment.start || 0,
          end: segment.end || segment.start || 0,
          primaryStart: checkResult.match.start || 0,
          primaryEnd: checkResult.match.end || checkResult.match.start || 0,
          source: 'speaker1',
          confidence: segment.confidence || null,
          similarity: checkResult.similarity,
          errorType: 'transcription_error',
          description: 'Text differs between speaker track and primary transcript'
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
    
    const checkResult = existsInPrimary(segment, primarySegments);
    
    if (!checkResult.exists) {
      // Фраза відсутня в primary - це помилка
      discrepancies.push({
        speaker: payload.speaker2.speaker || 'SPEAKER_01',
        role: speaker2Role,
        roleLabel: speaker2Label,
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0,
        source: 'speaker2',
        confidence: segment.confidence || null,
        errorType: 'missing',
        description: 'Phrase present in speaker track but missing from primary transcript'
      });
    } else if (checkResult.match && checkResult.similarity) {
      // Перевіряємо, чи текст відрізняється (помилка транскрибації)
      // Якщо схожість менша за високий поріг - можлива помилка
      if (checkResult.similarity < 0.95) {
        discrepancies.push({
          speaker: payload.speaker2.speaker || 'SPEAKER_01',
          role: speaker2Role,
          roleLabel: speaker2Label,
          text: segment.text.trim(),
          primaryText: checkResult.match.text.trim(),
          start: segment.start || 0,
          end: segment.end || segment.start || 0,
          primaryStart: checkResult.match.start || 0,
          primaryEnd: checkResult.match.end || checkResult.match.start || 0,
          source: 'speaker2',
          confidence: segment.confidence || null,
          similarity: checkResult.similarity,
          errorType: 'transcription_error',
          description: 'Text differs between speaker track and primary transcript'
        });
      }
    }
  }
  
  // Сортуємо за часом початку
  discrepancies.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return {
    success: true,
    totalDiscrepancies: discrepancies.length,
    discrepancies: discrepancies,
    summary: {
      missing: discrepancies.filter(d => d.errorType === 'missing').length,
      transcriptionErrors: discrepancies.filter(d => d.errorType === 'transcription_error').length,
      agentDiscrepancies: discrepancies.filter(d => d.roleLabel === 'Agent').length,
      clientDiscrepancies: discrepancies.filter(d => d.roleLabel === 'Client').length,
      unknownDiscrepancies: discrepancies.filter(d => d.roleLabel !== 'Agent' && d.roleLabel !== 'Client').length
    }
  };
}

// n8n Function Node код
// Отримуємо дані з Webhook (навіть якщо нода підключена після інших)
let inputData = null;
let dataSource = 'unknown';

// Варіант 1: Спробуємо знайти Webhook ноду за різними назвами (найнадійніший спосіб)
if (!inputData) {
  const webhookNames = ['Webhook', 'Webhook 1', 'Webhook 2', 'webhook', 'WEBHOOK'];
  for (const name of webhookNames) {
    try {
      const webhookData = $(name).item.json;
      if (webhookData) {
        // Перевіряємо body
        if (webhookData.body && (webhookData.body.general || webhookData.body.speaker1)) {
          inputData = webhookData.body;
          dataSource = `webhook_${name}_body`;
          break;
        }
        // Або безпосередньо в json
        if (webhookData.general || webhookData.speaker1) {
          inputData = webhookData;
          dataSource = `webhook_${name}_direct`;
          break;
        }
      }
    } catch (e) {
      // Продовжуємо спробувати інші назви
    }
  }
}

// Варіант 2: Перевіряємо всі вхідні елементи (може містити дані з Webhook)
if (!inputData) {
  try {
    const allInputs = $input.all();
    for (const item of allInputs) {
      const itemData = item.json;
      // Перевіряємо body з general/speaker1/speaker2
      if (itemData && itemData.body) {
        if (itemData.body.general || itemData.body.speaker1 || itemData.body.speaker2) {
          inputData = itemData.body;
          dataSource = 'input_all_body';
          break;
        }
      }
      // Або безпосередньо в json
      if (itemData && (itemData.general || itemData.speaker1 || itemData.speaker2)) {
        inputData = itemData;
        dataSource = 'input_all_direct';
        break;
      }
    }
  } catch (e) {
    // Ігноруємо помилку
  }
}

// Варіант 3: З Set ноди (якщо використовується)
if (!inputData) {
  try {
    const currentData = $input.item.json;
    if (currentData && currentData.originalPayload) {
      inputData = currentData.originalPayload;
      dataSource = 'set_node';
    }
  } catch (e) {
    // Ігноруємо помилку
  }
}

// Варіант 4: З поточної ноди, якщо це не результат попередньої
if (!inputData) {
  try {
    const currentData = $input.item.json;
    // Перевіряємо, чи це не результат попередньої ноди
    const isPreviousNodeResult = currentData.text && currentData.start !== undefined && 
                                 currentData.end !== undefined && 
                                 !currentData.general && !currentData.speaker1;
    
    if (!isPreviousNodeResult) {
      if (currentData.body && (currentData.body.general || currentData.body.speaker1)) {
        inputData = currentData.body;
        dataSource = 'current_body';
      } else if (currentData.general || currentData.speaker1) {
        inputData = currentData;
        dataSource = 'current_direct';
      }
    }
  } catch (e) {
    // Ігноруємо помилку
  }
}

if (!inputData) {
  // Якщо нічого не знайшли, повертаємо помилку з діагностикою
  return [{
    json: {
      error: 'Could not find payload data',
      text: `Error: Could not find payload data. Tried: ${dataSource}. Available keys: ${Object.keys($input.item?.json || {}).join(', ')}`,
      start: 0,
      end: 0,
      debug: {
        source: dataSource,
        hasInput: !!$input.item,
        inputKeys: $input.item?.json ? Object.keys($input.item.json) : [],
        inputType: typeof $input.item?.json
      }
    }
  }];
}

// ДІАГНОСТИКА ДАНИХ: Встановіть true для перегляду структури вхідних даних
const DEBUG_DATA = false;

// ДІАГНОСТИКА ДАНИХ: показуємо структуру даних
if (DEBUG_DATA) {
  return [{
    json: {
      text: `DEBUG: Data source: ${dataSource}. Has general: ${!!inputData?.general}. Has speaker1: ${!!inputData?.speaker1}. Has speaker2: ${!!inputData?.speaker2}. Keys: ${Object.keys(inputData || {}).join(', ')}`,
      start: 0,
      end: 0,
      debug: {
        dataSource: dataSource,
        hasGeneral: !!inputData?.general,
        hasSpeaker1: !!inputData?.speaker1,
        hasSpeaker2: !!inputData?.speaker2,
        inputKeys: Object.keys(inputData || {}),
        generalKeys: inputData?.general ? Object.keys(inputData.general) : [],
        speaker1Keys: inputData?.speaker1 ? Object.keys(inputData.speaker1) : [],
        speaker2Keys: inputData?.speaker2 ? Object.keys(inputData.speaker2) : []
      }
    }
  }];
}

const result = findDiscrepancies(inputData);

// ДІАГНОСТИКА: Встановіть true для перевірки роботи функції
const DEBUG_MODE = false;

// Повертаємо просто масив блоків тексту
if (result.success && result.discrepancies && result.discrepancies.length > 0) {
  return result.discrepancies.map(discrepancy => ({
    json: {
      text: discrepancy.text,
      start: discrepancy.start,
      end: discrepancy.end
    }
  }));
} else if (result.error) {
  // Якщо є помилка, повертаємо її для діагностики
  return [{
    json: {
      error: result.error,
      text: `Error: ${result.error}`,
      start: 0,
      end: 0
    }
  }];
} else {
  // Якщо все добре і немає розбіжностей - це нормально!
  // Red нода має повертати порожній масив, якщо помилок немає
  if (DEBUG_MODE) {
    // Діагностичний режим: повертаємо інформацію про перевірку
    const speaker1Count = inputData.speaker1?.speechmatics?.segments?.length || 
                         inputData.speaker1?.segments?.length || 0;
    const speaker2Count = inputData.speaker2?.speechmatics?.segments?.length || 
                         inputData.speaker2?.segments?.length || 0;
    const primaryCount = inputData.general?.speechmatics?.segments?.length || 
                        inputData.general?.segments?.length || 0;
    
    return [{
      json: {
        text: `✓ No discrepancies found. Checked ${speaker1Count + speaker2Count} speaker segments against ${primaryCount} primary segments.`,
        start: 0,
        end: 0,
        status: 'success',
        summary: {
          speaker1Segments: speaker1Count,
          speaker2Segments: speaker2Count,
          primarySegments: primaryCount,
          totalChecked: speaker1Count + speaker2Count
        }
      }
    }];
  }
  // Нормальний режим: повертаємо порожній масив (все добре!)
  return [];
}


