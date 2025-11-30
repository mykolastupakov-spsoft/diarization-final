/**
 * n8n Function Node: Green - Overlaps
 * 
 * Знаходить фрази з накладками (overlaps) - моменти, коли два спікери говорили одночасно.
 * Порівнює таймстемпи з обох JSON-файлів (speaker1 та speaker2) для виявлення одночасної мови.
 * 
 * Input: Webhook payload з полями general, speaker1, speaker2
 * Output: Масив об'єктів з перекриваючимися фразами
 */

// Функція для перевірки перекриття часових інтервалів
function timeRangesOverlap(start1, end1, start2, end2, minOverlapSeconds = 0.1) {
  // Перевіряємо, чи перекриваються інтервали
  const overlapStart = Math.max(start1, start2);
  const overlapEnd = Math.min(end1, end2);
  const overlapDuration = overlapEnd - overlapStart;
  
  // Якщо перекриття більше мінімального порогу, вважаємо це overlap
  return overlapDuration >= minOverlapSeconds;
}

// Функція для знаходження перекриваючихся сегментів між двома спикерами
function findOverlappingSegments(segments1, segments2, speaker1Info, speaker2Info) {
  const overlaps = [];
  
  for (const seg1 of segments1) {
    if (!seg1.text || !seg1.text.trim()) continue;
    if (!seg1.start && seg1.start !== 0) continue;
    if (!seg1.end && seg1.end !== 0) continue;
    
    const start1 = parseFloat(seg1.start) || 0;
    const end1 = parseFloat(seg1.end) || start1;
    
    if (end1 <= start1) continue; // Пропускаємо невалідні інтервали
    
    for (const seg2 of segments2) {
      if (!seg2.text || !seg2.text.trim()) continue;
      if (!seg2.start && seg2.start !== 0) continue;
      if (!seg2.end && seg2.end !== 0) continue;
      
      const start2 = parseFloat(seg2.start) || 0;
      const end2 = parseFloat(seg2.end) || start2;
      
      if (end2 <= start2) continue; // Пропускаємо невалідні інтервали
      
      // Перевіряємо перекриття
      if (timeRangesOverlap(start1, end1, start2, end2)) {
        const overlapStart = Math.max(start1, start2);
        const overlapEnd = Math.min(end1, end2);
        const overlapDuration = overlapEnd - overlapStart;
        
        // Додаємо обидва сегменти як перекриваючі
        overlaps.push({
          speaker1: {
            speaker: speaker1Info.speaker || 'SPEAKER_00',
            role: speaker1Info.role || 'unknown',
            roleLabel: speaker1Info.roleLabel || 'Speaker 1',
            text: seg1.text.trim(),
            start: start1,
            end: end1,
            source: 'speaker1'
          },
          speaker2: {
            speaker: speaker2Info.speaker || 'SPEAKER_01',
            role: speaker2Info.role || 'unknown',
            roleLabel: speaker2Info.roleLabel || 'Speaker 2',
            text: seg2.text.trim(),
            start: start2,
            end: end2,
            source: 'speaker2'
          },
          overlap: {
            start: overlapStart,
            end: overlapEnd,
            duration: overlapDuration
          },
          type: 'overlap'
        });
      }
    }
  }
  
  return overlaps;
}

// Основна функція для знаходження накладок
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
  const speaker1Role = payload.speaker1.role || 'unknown';
  const speaker1Label = speaker1Role === 'operator' || speaker1Role === 'agent' 
    ? 'Agent' 
    : speaker1Role === 'client' || speaker1Role === 'customer' 
    ? 'Client' 
    : 'Speaker 1';
  
  const speaker2Segments = payload.speaker2.segments || 
                          payload.speaker2.speechmatics?.segments || 
                          [];
  const speaker2Role = payload.speaker2.role || 'unknown';
  const speaker2Label = speaker2Role === 'operator' || speaker2Role === 'agent' 
    ? 'Agent' 
    : speaker2Role === 'client' || speaker2Role === 'customer' 
    ? 'Client' 
    : 'Speaker 2';
  
  if (speaker1Segments.length === 0 || speaker2Segments.length === 0) {
    return {
      success: true,
      totalOverlaps: 0,
      overlaps: [],
      summary: {
        message: 'No segments found in one or both speaker tracks'
      }
    };
  }
  
  // Знаходимо перекриваючі сегменти
  const speaker1Info = {
    speaker: payload.speaker1.speaker || 'SPEAKER_00',
    role: speaker1Role,
    roleLabel: speaker1Label
  };
  
  const speaker2Info = {
    speaker: payload.speaker2.speaker || 'SPEAKER_01',
    role: speaker2Role,
    roleLabel: speaker2Label
  };
  
  const foundOverlaps = findOverlappingSegments(
    speaker1Segments,
    speaker2Segments,
    speaker1Info,
    speaker2Info
  );
  
  // Сортуємо за часом початку перекриття
  foundOverlaps.sort((a, b) => (a.overlap.start || 0) - (b.overlap.start || 0));
  
  return {
    success: true,
    totalOverlaps: foundOverlaps.length,
    overlaps: foundOverlaps,
    summary: {
      agentOverlaps: foundOverlaps.filter(o => 
        o.speaker1.roleLabel === 'Agent' || o.speaker2.roleLabel === 'Agent'
      ).length,
      clientOverlaps: foundOverlaps.filter(o => 
        o.speaker1.roleLabel === 'Client' || o.speaker2.roleLabel === 'Client'
      ).length,
      totalOverlapDuration: foundOverlaps.reduce((sum, o) => sum + (o.overlap.duration || 0), 0)
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

const result = findOverlaps(inputData);

// ДІАГНОСТИКА: Встановіть true для перевірки роботи функції
const DEBUG_MODE = false;

// Повертаємо просто масив блоків тексту
if (result.success && result.overlaps.length > 0) {
  return result.overlaps.map(overlap => ({
    json: {
      text: `${overlap.speaker1.text} | ${overlap.speaker2.text}`,
      start: overlap.overlap.start,
      end: overlap.overlap.end
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
  // Якщо немає overlaps - це нормально (спікери не говорили одночасно)
  if (DEBUG_MODE) {
    const speaker1Count = inputData.speaker1?.speechmatics?.segments?.length || 
                         inputData.speaker1?.segments?.length || 0;
    const speaker2Count = inputData.speaker2?.speechmatics?.segments?.length || 
                         inputData.speaker2?.segments?.length || 0;
    return [{
      json: {
        text: `✓ No overlaps found. Checked ${speaker1Count} vs ${speaker2Count} segments.`,
        start: 0,
        end: 0,
        status: 'success'
      }
    }];
  }
  return [];
}

