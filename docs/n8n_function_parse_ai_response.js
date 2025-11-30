/**
 * n8n Function Node: Parse AI Response
 * 
 * Парсить відповідь від AI ноди (OpenAI/Anthropic) і конвертує її в формат,
 * який очікує Set нода (масив об'єктів з text, start, end).
 * 
 * Input: Відповідь від AI ноди
 * Output: Масив об'єктів з розбіжностями
 */

// Отримуємо відповідь від AI ноди
const aiResponse = $input.item.json;

// Спробуємо отримати текст відповіді з різних форматів
let responseText = '';

// Формат OpenAI
if (aiResponse.choices && aiResponse.choices[0] && aiResponse.choices[0].message) {
  responseText = aiResponse.choices[0].message.content;
}
// Формат Anthropic
else if (aiResponse.content && Array.isArray(aiResponse.content)) {
  responseText = aiResponse.content.map(c => c.text).join('');
}
// Формат Anthropic (новий)
else if (aiResponse.content && typeof aiResponse.content === 'string') {
  responseText = aiResponse.content;
}
// Якщо AI обгорнув відповідь в об'єкт з "output" (помилка AI)
else if (aiResponse.output && typeof aiResponse.output === 'string') {
  responseText = aiResponse.output;
}
// Якщо це вже рядок
else if (typeof aiResponse === 'string') {
  responseText = aiResponse;
}
// Якщо це вже масив (AI повернув готовий JSON)
else if (Array.isArray(aiResponse)) {
  // Перевіряємо, чи це вже правильний формат
  if (aiResponse.length > 0 && aiResponse[0].text !== undefined) {
    return aiResponse.map(item => ({
      json: {
        text: item.text || '',
        start: parseFloat(item.start) || 0,
        end: parseFloat(item.end) || 0
      }
    }));
  }
  // Якщо ні, конвертуємо в рядок для парсингу
  responseText = JSON.stringify(aiResponse);
}
// Якщо це об'єкт з "output" як масив
else if (aiResponse.output && Array.isArray(aiResponse.output)) {
  // Беремо масив з output
  const outputArray = aiResponse.output;
  if (outputArray.length > 0 && outputArray[0].text !== undefined) {
    return outputArray.map(item => ({
      json: {
        text: item.text || '',
        start: parseFloat(item.start) || 0,
        end: parseFloat(item.end) || 0
      }
    }));
  }
  // Якщо output - це рядок з JSON
  if (typeof outputArray[0] === 'string') {
    responseText = outputArray[0];
  } else {
    responseText = JSON.stringify(outputArray);
  }
}
// Інший формат - спробуємо конвертувати в рядок
else {
  responseText = JSON.stringify(aiResponse);
}

// Очищаємо від markdown code blocks якщо є
responseText = responseText
  .replace(/```json\n?/g, '')
  .replace(/```\n?/g, '')
  .replace(/^```/g, '')
  .replace(/```$/g, '')
  .trim();

// Якщо рядок порожній, повертаємо порожній масив
if (!responseText || responseText.length === 0) {
  return [];
}

// Парсимо JSON
let discrepancies = [];
try {
  discrepancies = JSON.parse(responseText);
  
  // Перевіряємо, чи це масив
  if (!Array.isArray(discrepancies)) {
    // Якщо це об'єкт з масивом всередині, спробуємо витягти
    if (discrepancies.discrepancies && Array.isArray(discrepancies.discrepancies)) {
      discrepancies = discrepancies.discrepancies;
    } else if (discrepancies.results && Array.isArray(discrepancies.results)) {
      discrepancies = discrepancies.results;
    } else {
      // Якщо це один об'єкт, обгортаємо в масив
      discrepancies = [discrepancies];
    }
  }
} catch (e) {
  // Якщо не вдалося розпарсити, повертаємо помилку для діагностики
  return [{
    json: {
      error: 'Failed to parse AI response',
      text: `Error: ${e.message}. Response: ${responseText.substring(0, 200)}...`,
      start: 0,
      end: 0,
      debug: {
        originalResponse: responseText.substring(0, 500),
        error: e.message
      }
    }
  }];
}

// Валідуємо та конвертуємо в потрібний формат
const result = discrepancies
  .filter(item => item && (item.text || item.text === '')) // Фільтруємо валідні елементи
  .map(item => ({
    json: {
      text: String(item.text || ''),
      start: parseFloat(item.start) || 0,
      end: parseFloat(item.end) || parseFloat(item.start) || 0
    }
  }));

// Повертаємо результат
return result.length > 0 ? result : [];

