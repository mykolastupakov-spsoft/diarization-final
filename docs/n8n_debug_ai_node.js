/**
 * n8n Function Node: Debug AI Response
 * 
 * Діагностична нода для перевірки, що повертає AI нода.
 * Додайте цю ноду ПІСЛЯ AI ноди, щоб побачити структуру відповіді.
 */

const aiResponse = $input.item.json;

// Збираємо всю діагностичну інформацію
const debugInfo = {
  // Тип даних
  type: typeof aiResponse,
  isArray: Array.isArray(aiResponse),
  isObject: typeof aiResponse === 'object' && aiResponse !== null,
  isString: typeof aiResponse === 'string',
  
  // Ключі об'єкта
  keys: typeof aiResponse === 'object' && aiResponse !== null ? Object.keys(aiResponse) : [],
  
  // Перевірка OpenAI формату
  hasChoices: !!(aiResponse.choices),
  hasMessage: !!(aiResponse.choices && aiResponse.choices[0] && aiResponse.choices[0].message),
  hasContent: !!(aiResponse.content),
  hasOutput: !!(aiResponse.output),
  
  // Довжина (якщо масив або рядок)
  length: Array.isArray(aiResponse) ? aiResponse.length : (typeof aiResponse === 'string' ? aiResponse.length : null),
  
  // Перші 500 символів (якщо рядок)
  firstChars: typeof aiResponse === 'string' ? aiResponse.substring(0, 500) : null,
  
  // Перший елемент (якщо масив)
  firstItem: Array.isArray(aiResponse) && aiResponse.length > 0 ? aiResponse[0] : null,
  
  // Повна структура (обмежена)
  fullStructure: JSON.stringify(aiResponse, null, 2).substring(0, 2000)
};

// Спробуємо витягти текст з різних форматів
let extractedText = null;

if (aiResponse.choices && aiResponse.choices[0] && aiResponse.choices[0].message) {
  extractedText = aiResponse.choices[0].message.content;
  debugInfo.extractedFrom = 'choices[0].message.content';
} else if (aiResponse.content && Array.isArray(aiResponse.content)) {
  extractedText = aiResponse.content.map(c => c.text).join('');
  debugInfo.extractedFrom = 'content[].text';
} else if (aiResponse.content && typeof aiResponse.content === 'string') {
  extractedText = aiResponse.content;
  debugInfo.extractedFrom = 'content (string)';
} else if (aiResponse.output) {
  extractedText = typeof aiResponse.output === 'string' ? aiResponse.output : JSON.stringify(aiResponse.output);
  debugInfo.extractedFrom = 'output';
} else if (typeof aiResponse === 'string') {
  extractedText = aiResponse;
  debugInfo.extractedFrom = 'direct string';
} else if (Array.isArray(aiResponse)) {
  extractedText = JSON.stringify(aiResponse);
  debugInfo.extractedFrom = 'array (stringified)';
} else {
  extractedText = JSON.stringify(aiResponse);
  debugInfo.extractedFrom = 'object (stringified)';
}

debugInfo.extractedText = extractedText ? extractedText.substring(0, 1000) : null;
debugInfo.extractedTextLength = extractedText ? extractedText.length : 0;

// Повертаємо діагностичну інформацію
return [{
  json: {
    debug: debugInfo,
    message: 'Check the debug object above to see what the AI node returned',
    recommendation: extractedText ? 'Text was extracted. Check if it\'s valid JSON.' : 'No text could be extracted. Check the structure above.'
  }
}];

