/**
 * n8n Function Node: Find Missing Phrases
 * 
 * Знаходить фрази, які є в окремих доріжках (speaker1/speaker2),
 * але відсутні в первинному транскрайбі (general)
 * 
 * Input: Webhook payload з полями general, speaker1, speaker2
 * Output: Масив об'єктів з відсутніми фразами
 */

// Функція для нормалізації тексту (прибирає зайві пробіли, нижній регістр)
function normalizeText(text) {
  if (!text || typeof text !== 'string') return '';
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ') // Замінюємо пунктуацію на пробіли
    .replace(/\s+/g, ' ')
    .trim();
}

// Функція для обчислення схожості текстів (Jaccard similarity + substring matching)
function computeTextSimilarity(text1, text2) {
  const normalized1 = normalizeText(text1);
  const normalized2 = normalizeText(text2);
  
  if (!normalized1 || !normalized2) return 0;
  if (normalized1 === normalized2) return 1;
  
  // Перевірка на включення (якщо один текст містить інший)
  if (normalized1.includes(normalized2) || normalized2.includes(normalized1)) {
    const shorter = normalized1.length < normalized2.length ? normalized1 : normalized2;
    const longer = normalized1.length >= normalized2.length ? normalized1 : normalized2;
    const ratio = shorter.length / longer.length;
    // Якщо коротший текст становить > 80% довшого, вважаємо дуже схожим
    if (ratio >= 0.8) return 0.95;
    // Інакше повертаємо пропорційну схожість
    return ratio;
  }
  
  // Jaccard similarity на основі спільних слів
  const words1 = normalized1.split(/\s+/).filter(w => w.length > 2);
  const words2 = normalized2.split(/\s+/).filter(w => w.length > 2);
  
  if (words1.length === 0 || words2.length === 0) return 0;
  
  const set1 = new Set(words1);
  const set2 = new Set(words2);
  
  const intersection = [...set1].filter(word => set2.has(word));
  const union = new Set([...words1, ...words2]);
  
  if (union.size === 0) return 0;
  return intersection.length / union.size;
}

// Функція для перевірки, чи існує сегмент в primary транскрайбі
function existsInPrimary(segment, primarySegments, similarityThreshold = 0.65) {
  const segmentText = normalizeText(segment.text || '');
  if (!segmentText || segmentText.length < 3) return true; // Ігноруємо дуже короткі сегменти
  
  // Перевіряємо за текстом
  for (const primarySeg of primarySegments) {
    const primaryText = normalizeText(primarySeg.text || '');
    if (!primaryText) continue;
    
    // Якщо тексти ідентичні
    if (segmentText === primaryText) return true;
    
    // Якщо один текст містить інший (для часткових збігів)
    if (segmentText.includes(primaryText) || primaryText.includes(segmentText)) {
      const shorter = segmentText.length < primaryText.length ? segmentText : primaryText;
      const longer = segmentText.length >= primaryText.length ? segmentText : primaryText;
      // Якщо коротший текст становить > 75% довшого, вважаємо збігом
      if (shorter.length / longer.length >= 0.75) return true;
    }
    
    // Перевірка схожості через Jaccard similarity
    const similarity = computeTextSimilarity(segmentText, primaryText);
    if (similarity >= similarityThreshold) return true;
  }
  
  return false;
}

// Основна функція для знаходження відсутніх фраз
function findMissingPhrases(payload) {
  const missingPhrases = [];
  
  // Перевіряємо наявність необхідних даних
  if (!payload || !payload.general || !payload.speaker1 || !payload.speaker2) {
    return {
      error: 'Missing required fields: general, speaker1, or speaker2',
      missingPhrases: []
    };
  }
  
  // Отримуємо сегменти з primary транскрайбу
  const primarySegments = payload.general.segments || 
                         payload.general.speechmatics?.segments || 
                         [];
  
  if (primarySegments.length === 0) {
    return {
      error: 'No segments found in primary diarization',
      missingPhrases: []
    };
  }
  
  // Обробляємо speaker1 (агент)
  const speaker1Segments = payload.speaker1.segments || 
                          payload.speaker1.speechmatics?.segments || 
                          [];
  const speaker1Role = payload.speaker1.role || 'unknown';
  const speaker1Label = speaker1Role === 'operator' || speaker1Role === 'agent' 
    ? 'Agent' 
    : speaker1Role === 'client' || speaker1Role === 'customer' 
    ? 'Client' 
    : 'Speaker 1';
  
  // Обробляємо speaker2 (клієнт)
  const speaker2Segments = payload.speaker2.segments || 
                          payload.speaker2.speechmatics?.segments || 
                          [];
  const speaker2Role = payload.speaker2.role || 'unknown';
  const speaker2Label = speaker2Role === 'operator' || speaker2Role === 'agent' 
    ? 'Agent' 
    : speaker2Role === 'client' || speaker2Role === 'customer' 
    ? 'Client' 
    : 'Speaker 2';
  
  // Знаходимо відсутні фрази для speaker1
  const seenTexts = new Set();
  for (const segment of speaker1Segments) {
    if (!segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 3) continue; // Ігноруємо дуже короткі фрази
    if (seenTexts.has(normalizedText)) continue; // Уникаємо дублікатів
    seenTexts.add(normalizedText);
    
    if (!existsInPrimary(segment, primarySegments)) {
      missingPhrases.push({
        speaker: payload.speaker1.speaker || 'SPEAKER_00',
        role: speaker1Role,
        roleLabel: speaker1Label,
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0,
        source: 'speaker1',
        confidence: segment.confidence || null
      });
    }
  }
  
  // Знаходимо відсутні фрази для speaker2
  for (const segment of speaker2Segments) {
    if (!segment.text || !segment.text.trim()) continue;
    
    const normalizedText = normalizeText(segment.text);
    if (normalizedText.length < 3) continue; // Ігноруємо дуже короткі фрази
    if (seenTexts.has(normalizedText)) continue; // Уникаємо дублікатів
    seenTexts.add(normalizedText);
    
    if (!existsInPrimary(segment, primarySegments)) {
      missingPhrases.push({
        speaker: payload.speaker2.speaker || 'SPEAKER_01',
        role: speaker2Role,
        roleLabel: speaker2Label,
        text: segment.text.trim(),
        start: segment.start || 0,
        end: segment.end || segment.start || 0,
        source: 'speaker2',
        confidence: segment.confidence || null
      });
    }
  }
  
  // Сортуємо за часом початку
  missingPhrases.sort((a, b) => (a.start || 0) - (b.start || 0));
  
  return {
    success: true,
    totalMissing: missingPhrases.length,
    missingPhrases: missingPhrases,
    summary: {
      agentMissing: missingPhrases.filter(p => p.roleLabel === 'Agent').length,
      clientMissing: missingPhrases.filter(p => p.roleLabel === 'Client').length,
      unknownMissing: missingPhrases.filter(p => p.roleLabel !== 'Agent' && p.roleLabel !== 'Client').length
    }
  };
}

// n8n Function Node код
// Отримуємо дані з вхідного елементу
// n8n Webhook повертає дані в body, тому перевіряємо обидва варіанти для сумісності
const inputData = $input.item.json.body || $input.item.json;

// Виконуємо аналіз
const result = findMissingPhrases(inputData);

// Повертаємо результат
// n8n очікує масив об'єктів, тому повертаємо кожну відсутню фразу як окремий елемент
if (result.success && result.missingPhrases.length > 0) {
  // Повертаємо масив об'єктів - кожна відсутня фраза стане окремим елементом
  return result.missingPhrases.map(phrase => ({
    json: {
      ...phrase,
      metadata: {
        totalMissing: result.totalMissing,
        summary: result.summary
      }
    }
  }));
} else {
  // Якщо немає відсутніх фраз або є помилка, повертаємо результат аналізу
  return [{
    json: result
  }];
}

