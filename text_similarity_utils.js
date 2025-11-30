/* eslint-disable max-lines */
/**
 * Утиліти для порівняння тексту з використанням множинних метрик
 * Використовується для дедуплікації сегментів в overlap merge
 */

function normalizeText(text) {
  if (!text || typeof text !== 'string') return '';
  // Видаляємо пунктуацію, lowercase, нормалізуємо пробіли
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')  // Пунктуація -> пробіли
    .replace(/\s+/g, ' ')       // Множинні пробіли -> один
    .trim();
}

function levenshteinDistance(str1, str2) {
  // Ефективна реалізація (O(n) по памяти)
  const m = str1.length;
  const n = str2.length;
  
  if (m === 0) return n;
  if (n === 0) return m;
  
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  
  for (let i = 1; i <= m; i++) {
    let curr = [i];
    
    for (let j = 1; j <= n; j++) {
      const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        curr[j - 1] + 1,      // Вставка
        prev[j] + 1,          // Видалення
        prev[j - 1] + cost    // Заміна
      );
    }
    
    prev = curr;
  }
  
  return prev[n];
}

function levenshteinSimilarity(str1, str2) {
  // Нормалізуємо до 0-1, де 1 = ідентичні
  const distance = levenshteinDistance(str1, str2);
  const maxLen = Math.max(str1.length, str2.length);
  if (maxLen === 0) return 1;
  return 1 - (distance / maxLen);
}

function jaccardSimilarity(tokens1, tokens2) {
  // Jaccard = intersection / union
  if (!tokens1.length || !tokens2.length) return 0;
  
  const set1 = new Set(tokens1);
  const set2 = new Set(tokens2);
  
  const intersection = [...set1].filter(token => set2.has(token)).length;
  const union = new Set([...set1, ...set2]).size;
  
  return union === 0 ? 0 : intersection / union;
}

function areTextsSimilar(text1, text2, thresholdConfig = {}) {
  // Тепер функція приймає конфіг з порогами
  const {
    minLevenshteinSim = 0.88,  // Підвищено з 0.72
    minJaccardSim = 0.80,       // Новий поріг
    minSubstringMatch = 0.75,   // Новий поріг
    ignoreContent = false       // Для дебугу
  } = thresholdConfig;
  
  if (!text1 || !text2) return false;
  
  const norm1 = normalizeText(text1);
  const norm2 = normalizeText(text2);
  
  if (norm1 === norm2) {
    if (ignoreContent) console.log('✓ Matched via exact normalized');
    return true;
  }
  
  // Метрика 1: Substring matching (найбільш строгий)
  if (norm1.includes(norm2) || norm2.includes(norm1)) {
    // Перевіряємо, чи один не занадто короткий (щоб не ловити "Hi" в "Hi there")
    const ratio = Math.min(norm1.length, norm2.length) / Math.max(norm1.length, norm2.length);
    if (ratio >= minSubstringMatch) {
      if (ignoreContent) console.log('✓ Matched via substring');
      return true;
    }
  }
  
  // Метрика 2: Levenshtein similarity (для typos/варіацій)
  const levSim = levenshteinSimilarity(norm1, norm2);
  if (levSim >= minLevenshteinSim) {
    if (ignoreContent) console.log(`✓ Matched via Levenshtein (${levSim.toFixed(3)})`);
    return true;
  }
  
  // Метрика 3: Jaccard similarity на tokenized тексті
  const tokens1 = norm1.split(/\s+/).filter(t => t.length > 0);
  const tokens2 = norm2.split(/\s+/).filter(t => t.length > 0);
  
  if (tokens1.length === 0 || tokens2.length === 0) return false;
  
  const jacSim = jaccardSimilarity(tokens1, tokens2);
  
  if (jacSim >= minJaccardSim) {
    if (ignoreContent) console.log(`✓ Matched via Jaccard (${jacSim.toFixed(3)})`);
    return true;
  }
  
  // Метрика 4: Комбінована метрика (якщо 2+ метрики близькі)
  // Вважаємо дублікатом, якщо:
  // - (Levenshtein > 0.82 И Jaccard > 0.70) ИЛИ
  // - (Levenshtein > 0.85 ИЛИ Jaccard > 0.75)
  const combinedScore = (levSim * 0.6 + jacSim * 0.4);
  if (combinedScore >= 0.82) {
    if (ignoreContent) console.log(`✓ Matched via combined (Lev:${levSim.toFixed(3)} Jac:${jacSim.toFixed(3)} Comb:${combinedScore.toFixed(3)})`);
    return true;
  }
  
  // Додаткова перевірка: якщо обидві метрики досить високі
  if ((levSim > 0.82 && jacSim > 0.70) || (levSim > 0.85 || jacSim > 0.75)) {
    if (ignoreContent) console.log(`✓ Matched via dual threshold (Lev:${levSim.toFixed(3)} Jac:${jacSim.toFixed(3)})`);
    return true;
  }
  
  if (ignoreContent) console.log(`✗ Not matched (Lev:${levSim.toFixed(3)} Jac:${jacSim.toFixed(3)} Comb:${combinedScore.toFixed(3)})`);
  return false;
}

// Експорт для Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    normalizeText,
    levenshteinDistance,
    levenshteinSimilarity,
    jaccardSimilarity,
    areTextsSimilar
  };
}

// Експорт для браузера
if (typeof window !== 'undefined') {
  window.TextSimilarityUtils = {
    normalizeText,
    levenshteinDistance,
    levenshteinSimilarity,
    jaccardSimilarity,
    areTextsSimilar
  };
}

