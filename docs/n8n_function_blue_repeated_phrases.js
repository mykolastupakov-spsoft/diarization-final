/**
 * n8n Function Node: Blue - Repeated Phrases
 * 
 * Знаходить фрази, які повторюються в тексті та першому JSON (general).
 * Тобто фрази, які присутні і в general транскрайбі, і в окремих доріжках (speaker1/speaker2).
 * 
 * Input: Webhook payload з полями general, speaker1, speaker2
 * Output: Масив об'єктів з повторюваними фразами
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

// Основна функція для знаходження повторюваних фраз
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
        speaker: payload.speaker1.speaker || 'SPEAKER_00',
        role: speaker1Role,
        roleLabel: speaker1Label,
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0,
        source: 'speaker1',
        confidence: segment.confidence || null,
        type: 'repeated'
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
        speaker: payload.speaker2.speaker || 'SPEAKER_01',
        role: speaker2Role,
        roleLabel: speaker2Label,
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0,
        source: 'speaker2',
        confidence: segment.confidence || null,
        type: 'repeated'
      });
    }
  }
  
  // Сортуємо за часом початку
  repeatedPhrases.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return {
    success: true,
    totalRepeated: repeatedPhrases.length,
    repeatedPhrases: repeatedPhrases,
    summary: {
      agentRepeated: repeatedPhrases.filter(p => p.roleLabel === 'Agent').length,
      clientRepeated: repeatedPhrases.filter(p => p.roleLabel === 'Client').length,
      unknownRepeated: repeatedPhrases.filter(p => p.roleLabel !== 'Agent' && p.roleLabel !== 'Client').length
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

// Варіант 3: З поточної ноди, якщо це не результат попередньої
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
  // Якщо нічого не знайшли, повертаємо помилку
  return [{
    json: {
      error: 'Could not find payload data',
      text: `Error: Could not find payload data. Tried: ${dataSource}`,
      start: 0,
      end: 0
    }
  }];
}

const result = findRepeatedPhrases(inputData);

// ДІАГНОСТИКА: Встановіть true для перевірки роботи функції
const DEBUG_MODE = false;

// Повертаємо просто масив блоків тексту
if (result.success && result.repeatedPhrases.length > 0) {
  return result.repeatedPhrases.map(phrase => ({
    json: {
      text: phrase.text,
      start: phrase.start,
      end: phrase.end
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
  // Якщо немає повторюваних фраз - це нормально
  if (DEBUG_MODE) {
    const speaker1Count = inputData.speaker1?.speechmatics?.segments?.length || 
                         inputData.speaker1?.segments?.length || 0;
    const speaker2Count = inputData.speaker2?.speechmatics?.segments?.length || 
                         inputData.speaker2?.segments?.length || 0;
    return [{
      json: {
        text: `✓ No repeated phrases found. Checked ${speaker1Count + speaker2Count} segments.`,
        start: 0,
        end: 0,
        status: 'success'
      }
    }];
  }
  return [];
}

