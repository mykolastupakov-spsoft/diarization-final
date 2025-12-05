/**
 * Parameter Optimizer for SpeechBrain Separation
 * 
 * Використовує hill climbing алгоритм для пошуку оптимальних параметрів
 * на основі фідбеку користувача (краще/гірше)
 */

const fs = require('fs');
const path = require('path');
const axios = require('axios');

class ParameterOptimizer {
  constructor(historyFile = 'cache/parameter_optimization_history.json') {
    this.historyFile = historyFile;
    this.history = this.loadHistory();
    
    // Діапазони параметрів для оптимізації
    this.paramRanges = {
      chunkSeconds: { min: 5, max: 30, step: 1, type: 'float' },
      overlapSeconds: { min: 0.5, max: 5, step: 0.5, type: 'float' },
      sampleRate: { values: [8000, 16000, 22050, 44100], type: 'discrete' },
      segmentOverlap: { min: 0.1, max: 2.0, step: 0.1, type: 'float' },
      minIntersegmentGap: { min: 0.05, max: 0.5, step: 0.05, type: 'float' },
      vadThreshold: { min: 0.3, max: 0.9, step: 0.1, type: 'float' },
      strictMode: { values: [true, false], type: 'boolean' },
      batchSize: { min: 1, max: 8, step: 1, type: 'int' },
      normalizationMethod: { values: ['peak', 'rms'], type: 'discrete' },
      normalizationLevel: { min: 0.5, max: 0.95, step: 0.05, type: 'float' }
    };
    
    // Поточний стан оптимізації
    this.currentState = null;
    this.bestState = null;
  }
  
  loadHistory() {
    try {
      const historyPath = path.join(process.cwd(), this.historyFile);
      if (fs.existsSync(historyPath)) {
        const data = fs.readFileSync(historyPath, 'utf8');
        return JSON.parse(data);
      }
    } catch (error) {
      console.warn('Failed to load optimization history:', error.message);
    }
    
    return {
      tests: [],
      bestConfig: null,
      currentIteration: 0,
      sessionId: null
    };
  }
  
  saveHistory() {
    try {
      const historyPath = path.join(process.cwd(), this.historyFile);
      const dir = path.dirname(historyPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(historyPath, JSON.stringify(this.history, null, 2), 'utf8');
    } catch (error) {
      console.error('Failed to save optimization history:', error);
    }
  }
  
  /**
   * Генерує початкову конфігурацію параметрів
   * device та sampleRate завжди фіксовані
   */
  generateInitialConfig() {
    return {
      chunkSeconds: 10.0,
      overlapSeconds: 2.0,
      sampleRate: 16000, // ФІКСОВАНО: завжди 16000 для MacBook Pro M4
      segmentOverlap: 0.5,
      minIntersegmentGap: 0.1,
      vadThreshold: 0.7,
      strictMode: true,
      batchSize: 4,
      numSpeakers: 2,
      normalizationMethod: 'peak',
      normalizationLevel: 0.80,
      device: 'mps' // ФІКСОВАНО: завжди MPS для MacBook Pro M4
    };
  }
  
  /**
   * Генерує нову конфігурацію на основі поточної (hill climbing)
   * device та sampleRate завжди залишаються фіксованими
   */
  generateNextConfig(currentConfig, direction = 'explore', stepSize = 1.0) {
    const newConfig = { ...currentConfig };
    // Фіксовані параметри - не оптимізуємо
    const fixedParams = ['device', 'sampleRate'];
    const paramsToOptimize = Object.keys(this.paramRanges).filter(p => !fixedParams.includes(p));
    
    if (direction === 'explore') {
      // Випадково змінюємо 1-3 параметри (менше для 'same')
      const numChanges = stepSize < 0.5 ? 1 : Math.floor(Math.random() * 3) + 1;
      const paramsToChange = this.shuffleArray([...paramsToOptimize]).slice(0, numChanges);
      
      for (const param of paramsToChange) {
        newConfig[param] = this.mutateParameter(param, currentConfig[param], stepSize);
      }
    } else if (direction === 'exploit') {
      // Змінюємо параметри в напрямку кращого результату
      const param = paramsToOptimize[Math.floor(Math.random() * paramsToOptimize.length)];
      newConfig[param] = this.mutateParameter(param, currentConfig[param], stepSize * 0.5); // Менший крок
    }
    
    // Завжди встановлюємо фіксовані параметри
    newConfig.device = 'mps';
    newConfig.sampleRate = 16000;
    
    return newConfig;
  }
  
  /**
   * Мутує один параметр
   */
  mutateParameter(paramName, currentValue, stepSize = 1.0) {
    const range = this.paramRanges[paramName];
    if (!range) return currentValue;
    
    if (range.type === 'discrete') {
      // Випадковий вибір з доступних значень
      const availableValues = range.values.filter(v => v !== currentValue);
      if (availableValues.length === 0) return currentValue;
      return availableValues[Math.floor(Math.random() * availableValues.length)];
    } else if (range.type === 'boolean') {
      return !currentValue;
    } else if (range.type === 'float' || range.type === 'int') {
      // Додаємо або віднімаємо крок
      const step = range.step * stepSize;
      const direction = Math.random() > 0.5 ? 1 : -1;
      let newValue = currentValue + (direction * step);
      
      // Обмежуємо діапазоном
      newValue = Math.max(range.min, Math.min(range.max, newValue));
      
      if (range.type === 'int') {
        newValue = Math.round(newValue);
      }
      
      return newValue;
    }
    
    return currentValue;
  }
  
  /**
   * Перемішує масив
   */
  shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }
  
  /**
   * Починає нову сесію оптимізації
   */
  startNewSession(audioFile = null) {
    const sessionId = `session_${Date.now()}`;
    this.history.sessionId = sessionId;
    this.history.currentIteration = 0;
    this.history.audioFile = audioFile;
    
    const initialConfig = this.generateInitialConfig();
    this.currentState = {
      config: initialConfig,
      iteration: 0,
      sessionId: sessionId,
      timestamp: new Date().toISOString()
    };
    
    this.saveHistory();
    
    return {
      sessionId: sessionId,
      config: initialConfig,
      iteration: 0
    };
  }
  
  /**
   * Отримує поточну конфігурацію для тестування
   */
  getCurrentConfig() {
    if (!this.currentState) {
      return this.startNewSession();
    }
    
    return {
      sessionId: this.currentState.sessionId,
      config: this.currentState.config,
      iteration: this.currentState.iteration,
      generationMethod: this.currentState.lastGenerationMethod || 'initial'
    };
  }
  
  /**
   * Генерує детальний промпт для LLM про параметри
   */
  getParameterContextPrompt() {
    return `Ти експерт з оптимізації параметрів для SpeechBrain SepFormer WHAMR16k - моделі розділення голосів спікерів.

КОНТЕКСТ ПАРАМЕТРІВ:

1. chunkSeconds (5-30 сек, float):
   - Визначає розмір сегмента для sliding window обробки
   - Менше (5-10 сек): краща якість для коротких сегментів, менше змішування спікерів, але повільніша обробка, більше пам'яті
   - Оптимальне (10-15 сек): баланс між якістю та швидкістю, рекомендовано для більшості випадків
   - Більше (20-30 сек): швидша обробка, але може бути гірша якість через змішування спікерів на довгих сегментах
   - Впливає на: якість розділення, швидкість обробки, використання пам'яті

2. overlapSeconds (0.5-5 сек, float):
   - Overlap між чанками для плавного зшивання
   - Менше (0.5-1 сек): швидша обробка, але можливі артефакти на стиках
   - Оптимальне (2-3 сек): хороший баланс, рекомендовано
   - Більше (4-5 сек): дуже плавне зшивання, але повільніше
   - Впливає на: плавність зшивання чанків, артефакти на стиках

3. sampleRate: ⚠️ ФІКСОВАНО НА 16000 Hz
   - Завжди 16000 Hz для MacBook Pro M4
   - Модель навчалася на 16kHz, оптимальна якість для транскрибації Whisper
   - НЕ ЗМІНЮЙ цей параметр!

4. segmentOverlap (0.1-2.0 сек, float):
   - Overlap між сегментами для кращого зшивання
   - Менше (0.1-0.3 сек): швидша обробка, але можливі залишки спікерів
   - Оптимальне (0.5-1.0 сек): хороший баланс, рекомендовано
   - Більше (1.5-2.0 сек): краще зшивання, але повільніше
   - Впливає на: залишки спікерів, якість зшивання сегментів

5. minIntersegmentGap (0.05-0.5 сек, float):
   - Мінімальний проміжок між сегментами
   - Менше (0.05-0.1 сек): більше сегментів, краща якість для перекривань
   - Оптимальне (0.1-0.2 сек): баланс, рекомендовано
   - Більше (0.3-0.5 сек): менше сегментів, але може гірше обробляти перекриття
   - Впливає на: обробку перекриваючих сегментів, "слипання" голосів

6. vadThreshold (0.3-0.9, float):
   - Поріг чутливості VAD (Voice Activity Detection)
   - Менше (0.3-0.5): більш чутливий, виявляє більше активності
   - Оптимальне (0.6-0.7): баланс, рекомендовано
   - Більше (0.8-0.9): менш чутливий, тільки явна активність
   - Впливає на: виявлення активності голосу, фільтрацію шуму

7. strictMode (true/false, boolean):
   - Строгий режим обробки
   - true: краща якість, більше перевірок, повільніше
   - false: швидша обробка, але може гірша якість
   - Впливає на: якість розділення, швидкість обробки

8. batchSize (1-8, int):
   - Розмір батча для обробки
   - Менше (1-2): менше пам'яті, стабільніша робота, але повільніше
   - Оптимальне (4): баланс, рекомендовано
   - Більше (6-8): швидша обробка, але може викликати memory overflow
   - Впливає на: швидкість обробки, використання пам'яті

9. normalizationMethod ('peak' або 'rms', string):
   - Метод нормалізації гучності перед обробкою
   - 'peak': вирівнює до максимального значення (рекомендовано для більшості випадків)
   - 'rms': вирівнює середньоквадратичне значення (краще для записів з різною динамікою)
   - Впливає на: вирівнювання гучності, уникнення плутанини спікерів через перепади гучності

10. normalizationLevel (0.5-0.95, float):
    - Цільовий рівень нормалізації
    - Для 'peak': 0.0-1.0 (0.80 рекомендовано для агресивної нормалізації)
    - Для 'rms': може бути дБ або лінійне значення
    - Менше (0.5-0.7): менш агресивна нормалізація, зберігає більше динаміки
    - Оптимальне (0.75-0.85): баланс між вирівнюванням та збереженням якості
    - Більше (0.9-0.95): дуже агресивна нормалізація, максимальне вирівнювання
    - Впливає на: вирівнювання гучності протягом запису, уникнення перепадів гучності

ФІКСОВАНІ ПАРАМЕТРИ (НЕ ЗМІНЮЮТЬСЯ):
- device: ЗАВЖДИ "mps" (MacBook Pro M4 з Apple Silicon)
- sampleRate: ЗАВЖДИ 16000 Hz (модель навчалася на 16kHz)

ПАРАМЕТРИ ДЛЯ ОПТИМІЗАЦІЇ (можна змінювати):
- chunkSeconds, overlapSeconds, segmentOverlap, minIntersegmentGap, vadThreshold, strictMode, batchSize, normalizationMethod, normalizationLevel

ВАЖЛИВО:
- chunkSeconds=10 та overlapSeconds=2.0 - хороші значення за замовчуванням
- strictMode=true критично для якості
- Всі параметри взаємопов'язані - зміна одного впливає на інші
- Аналізуй всю історію тестів для виявлення паттернів`;
  }
  
  /**
   * Генерує нову конфігурацію через LLM на основі фідбеку
   * Використовує ВСЮ історію змін для контексту
   * Використовує ті самі налаштування, що для діаризації
   */
  async generateConfigWithLLM(feedback, notes, currentConfig, bestConfig, history, llmApiKey, llmApiUrl, llmModel, llmHeaders = null, useLocalLLM = false) {
    console.log('🚀 [OPTIMIZATION] ===== generateConfigWithLLM START =====');
    console.log('🚀 [OPTIMIZATION] Parameters:', {
      feedback,
      hasNotes: !!notes,
      hasCurrentConfig: !!currentConfig,
      hasBestConfig: !!bestConfig,
      historyLength: history?.length || 0,
      hasApiKey: !!llmApiKey,
      apiUrl: llmApiUrl,
      model: llmModel,
      hasHeaders: !!llmHeaders,
      useLocalLLM
    });
    
    // Використовуємо ВСЮ історію, а не тільки останні 5
    const fullHistory = history || [];
    
    console.log('🤖 [OPTIMIZATION] Generating config with LLM:', {
      model: llmModel,
      apiUrl: llmApiUrl,
      historyLength: fullHistory.length,
      feedback: feedback
    });
    
    const prompt = `${this.getParameterContextPrompt()}

ВАЖЛИВО - ФІКСОВАНІ ПАРАМЕТРИ (НЕ ЗМІНЮЙ ЇХ):
- device: ЗАВЖДИ "mps" (MacBook Pro M4 з Apple Silicon)
- sampleRate: ЗАВЖДИ 16000 (модель навчалася на 16kHz)

ПОТОЧНА СИТУАЦІЯ:
- Поточна конфігурація: ${JSON.stringify(currentConfig, null, 2)}
${bestConfig ? `- Найкраща конфігурація (ітерація ${this.bestState?.iteration || 'N/A'}): ${JSON.stringify(bestConfig, null, 2)}` : '- Найкращої конфігурації ще немає'}
- Фідбек: ${feedback === 'better' ? '✅ КРАЩЕ - ця конфігурація покращила результат' : feedback === 'same' ? '⚖️ ТАК САМО - ця конфігурація дала такий самий результат' : '❌ ГІРШЕ - ця конфігурація погіршила результат'}
${notes ? `- Нотатки користувача: ${notes}` : ''}

ПОВНА ІСТОРІЯ ВСІХ ТЕСТІВ (${fullHistory.length} тестів):
${fullHistory.length > 0 
  ? fullHistory.map((test, idx) => 
      `Ітерація ${test.iteration} (${new Date(test.timestamp).toLocaleString()}): ${test.feedback === 'better' ? '✅ КРАЩЕ' : test.feedback === 'same' ? '⚖️ ТАК САМО' : '❌ ГІРШЕ'}
  Конфігурація: ${JSON.stringify(test.config, null, 2)}
  ${test.notes ? `Нотатки: ${test.notes}` : ''}
  ---`
    ).join('\n\n')
  : 'Історія порожня - це перший тест'
}

АНАЛІЗ ІСТОРІЇ:
Проаналізуй всю історію тестів вище. Зверни увагу на:
- Які конфігурації дали "✅ КРАЩЕ" - що в них спільного?
- Які конфігурації дали "❌ ГІРШЕ" - які параметри викликали проблеми?
- Які конфігурації дали "⚖️ ТАК САМО" - які параметри дали нейтральний результат?
- Які тренди в нотатках користувача?
- Як еволюціонували параметри від ітерації до ітерації?

ЗАВДАННЯ:
${feedback === 'better' 
  ? 'Генеруй нову конфігурацію, яка ПІДВИЩИТЬ якість ще більше. Використовуй поточну конфігурацію як базу, але вноси покращення на основі ВСІЄЇ історії тестів. Враховуй паттерни, які працювали в минулому.'
  : feedback === 'same'
  ? 'Генеруй нову конфігурацію з МІКРО-НАЛАШТУВАННЯМИ. Поточна конфігурація дала такий самий результат - зроби невеликі зміни (1-2 параметри), щоб знайти кращий варіант. Використовуй поточну конфігурацію як базу, але внеси мінімальні покращення.'
  : 'Генеруй нову конфігурацію, яка ВИПРАВИТЬ проблеми. Проаналізуй історію - які конфігурації працювали краще? Використовуй їх як основу, але уникай параметрів, які викликали проблеми в історії.'
}

КРИТИЧНІ ВИМОГИ:
1. Поверни ТІЛЬКИ валідний JSON об'єкт з параметрами (без markdown, без пояснень, без code blocks)
2. device: ЗАВЖДИ "mps" (не змінюй!)
3. sampleRate: ЗАВЖДИ 16000 (не змінюй!)
4. Всі інші значення повинні бути в дозволених діапазонах
5. strictMode повинен бути boolean (true/false)
6. batchSize повинен бути цілим числом від 1 до 8
7. Всі float значення повинні мати розумну кількість знаків після коми (1-2)
8. Враховуй ВСЮ історію тестів - використовуй паттерни, які працювали
9. Якщо feedback="better", покращуй параметри поступово на основі успішних конфігурацій з історії
10. Якщо feedback="worse", повертайся до параметрів, які давали "✅ КРАЩЕ" в історії
11. Якщо feedback="same", роби МІКРО-НАЛАШТУВАННЯ (зміни 1-2 параметрів на невеликі значення) для пошуку кращого варіанту
12. Враховуй нотатки користувача - вони містять важливу інформацію про проблеми

ФОРМАТ ВІДПОВІДІ (тільки JSON, без іншого тексту):
{
  "chunkSeconds": 10.0,
  "overlapSeconds": 2.0,
  "sampleRate": 16000,
  "segmentOverlap": 0.5,
  "minIntersegmentGap": 0.1,
  "vadThreshold": 0.7,
  "strictMode": true,
  "batchSize": 4,
  "numSpeakers": 2,
  "normalizationMethod": "peak",
  "normalizationLevel": 0.80,
  "device": "mps"
}`;

    // Використовуємо готові заголовки, якщо передані (з server.js)
    // Інакше формуємо їх самостійно (для сумісності)
    const headers = llmHeaders || (() => {
      const h = { 'Content-Type': 'application/json' };
      if (llmApiKey && llmApiKey !== 'not-needed') {
        h['Authorization'] = `Bearer ${llmApiKey}`;
      }
      if (!useLocalLLM && llmApiUrl.includes('openrouter.ai')) {
        h['HTTP-Referer'] = process.env.APP_URL || 'http://localhost:3000';
        h['X-Title'] = 'Parameter Optimization';
      }
      return h;
    })();
    
    const payload = {
      model: llmModel,
      messages: [
        {
          role: 'system',
          content: 'Ти експерт з оптимізації параметрів для аудіо обробки. Повертай ТІЛЬКИ валідний JSON без markdown, без пояснень, без code blocks.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: 0.3 // Низька температура для більш детермінованих результатів
    };
    
    // response_format підтримується не всіма моделями (особливо локальними)
    if (!useLocalLLM) {
      payload.response_format = { type: 'json_object' };
    }
    
    try {
      
      console.log('🤖 [OPTIMIZATION] Sending LLM request (same config as diarization):', {
        model: llmModel,
        url: llmApiUrl,
        useLocalLLM: useLocalLLM,
        hasResponseFormat: !!payload.response_format,
        promptLength: prompt.length,
        headersProvided: !!llmHeaders,
        headersKeys: llmHeaders ? Object.keys(llmHeaders) : [],
        payloadPreview: JSON.stringify(payload).substring(0, 200)
      });
      
      console.log('📤 [OPTIMIZATION] Full request to LM Studio:', {
        url: llmApiUrl,
        method: 'POST',
        headers: headers,
        payload: JSON.stringify(payload, null, 2)
      });
      
      let response;
      try {
        const startTime = Date.now();
        response = await axios.post(llmApiUrl, payload, {
          headers: headers,
          timeout: 60000 // Збільшено timeout до 60 секунд
        });
        const duration = Date.now() - startTime;
        console.log(`✅ [OPTIMIZATION] LM Studio responded in ${duration}ms`);
      } catch (axiosError) {
        // Детальне логування помилки від API
        console.error('❌ [OPTIMIZATION] Axios error details:', {
          message: axiosError.message,
          status: axiosError.response?.status,
          statusText: axiosError.response?.statusText,
          data: axiosError.response?.data,
          config: {
            url: axiosError.config?.url,
            method: axiosError.config?.method,
            model: payload.model
          }
        });
        throw axiosError;
      }

      console.log('📥 [OPTIMIZATION] LLM response received from LM Studio:', {
        status: response.status,
        statusText: response.statusText,
        hasChoices: !!response.data?.choices,
        choicesLength: response.data?.choices?.length || 0,
        responseData: JSON.stringify(response.data).substring(0, 500)
      });

      const content = response.data.choices?.[0]?.message?.content || 
                     response.data.choices?.[0]?.message?.text || 
                     response.data.content || 
                     '';
      
      if (!content) {
        throw new Error('Empty response from LLM');
      }
      
      console.log('🤖 [OPTIMIZATION] LLM content length:', content.length);
      console.log('🤖 [OPTIMIZATION] LLM content preview:', content.substring(0, 200));
      
      let config = {};
      
      // Парсимо JSON (може бути обгорнутий в markdown)
      try {
        // Спробуємо знайти JSON в тексті
        const jsonMatch = content.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          config = JSON.parse(jsonMatch[0]);
          console.log('✅ [OPTIMIZATION] Successfully parsed JSON from LLM response');
        } else {
          // Якщо не знайдено, спробуємо парсити весь контент
          config = JSON.parse(content);
          console.log('✅ [OPTIMIZATION] Successfully parsed entire content as JSON');
        }
      } catch (parseError) {
        console.error('❌ [OPTIMIZATION] Failed to parse LLM response:', parseError.message);
        console.error('❌ [OPTIMIZATION] Content that failed to parse:', content.substring(0, 500));
        throw new Error(`Failed to parse LLM response: ${parseError.message}`);
      }

      // Валідація та нормалізація параметрів
      const validatedConfig = this.validateAndNormalizeConfig(config);
      console.log('✅ [OPTIMIZATION] Generated and validated config:', validatedConfig);
      console.log('🚀 [OPTIMIZATION] ===== generateConfigWithLLM END (SUCCESS) =====');
      return validatedConfig;
    } catch (error) {
      // Детальне логування помилки
      const errorDetails = {
        error: error.message,
        stack: error.stack
      };
      
      if (error.response) {
        errorDetails.response = {
          status: error.response.status,
          statusText: error.response.statusText,
          data: error.response.data,
          headers: error.response.headers
        };
      }
      
      if (error.config) {
        errorDetails.request = {
          url: error.config.url,
          method: error.config.method,
          model: payload?.model,
          hasApiKey: !!llmApiKey
        };
      }
      
      console.error('❌ [OPTIMIZATION] LLM generation failed:', JSON.stringify(errorDetails, null, 2));
      console.log('🚀 [OPTIMIZATION] ===== generateConfigWithLLM END (ERROR) =====');
      
      // Якщо помилка через невалідну модель, повертаємо більш зрозуміле повідомлення
      if (error.response?.status === 400 && error.response?.data?.error?.message?.includes('not a valid model')) {
        console.error('❌ [OPTIMIZATION] Invalid model ID:', payload?.model);
        console.error('❌ [OPTIMIZATION] Please check SMART_MODEL_ID in .env file');
      }
      
      // Fallback до звичайної генерації
      console.log('🔄 [OPTIMIZATION] Falling back to hill climbing algorithm');
      return this.generateNextConfig(
        feedback === 'better' && bestConfig ? bestConfig : currentConfig,
        feedback === 'better' ? 'exploit' : 'explore'
      );
    }
  }
  
  /**
   * Валідує та нормалізує конфігурацію
   * device та sampleRate завжди фіксовані
   */
  validateAndNormalizeConfig(config) {
    const validated = { ...this.generateInitialConfig() };
    
    // ФІКСОВАНІ ПАРАМЕТРИ (не можуть бути змінені)
    validated.sampleRate = 16000; // Завжди 16000 для MacBook Pro M4
    validated.device = 'mps'; // Завжди MPS для MacBook Pro M4
    
    // Валідація параметрів, які можна змінювати
    if (config.chunkSeconds !== undefined) {
      validated.chunkSeconds = Math.max(5, Math.min(30, parseFloat(config.chunkSeconds) || 10));
    }
    if (config.overlapSeconds !== undefined) {
      validated.overlapSeconds = Math.max(0.5, Math.min(5, parseFloat(config.overlapSeconds) || 2));
    }
    if (config.segmentOverlap !== undefined) {
      validated.segmentOverlap = Math.max(0.1, Math.min(2.0, parseFloat(config.segmentOverlap) || 0.5));
    }
    if (config.minIntersegmentGap !== undefined) {
      validated.minIntersegmentGap = Math.max(0.05, Math.min(0.5, parseFloat(config.minIntersegmentGap) || 0.1));
    }
    if (config.vadThreshold !== undefined) {
      validated.vadThreshold = Math.max(0.3, Math.min(0.9, parseFloat(config.vadThreshold) || 0.7));
    }
    if (config.strictMode !== undefined) {
      validated.strictMode = config.strictMode === true || config.strictMode === 'true';
    }
    if (config.batchSize !== undefined) {
      validated.batchSize = Math.max(1, Math.min(8, parseInt(config.batchSize) || 4));
    }
    if (config.numSpeakers !== undefined) {
      validated.numSpeakers = Math.max(1, Math.min(10, parseInt(config.numSpeakers) || 2));
    }
    if (config.normalizationMethod !== undefined) {
      const validMethods = ['peak', 'rms'];
      validated.normalizationMethod = validMethods.includes(config.normalizationMethod) 
        ? config.normalizationMethod 
        : 'peak';
    }
    if (config.normalizationLevel !== undefined) {
      validated.normalizationLevel = Math.max(0.5, Math.min(0.95, parseFloat(config.normalizationLevel) || 0.80));
    }
    
    return validated;
  }
  
  /**
   * Обробляє фідбек користувача (краще/гірше) з використанням LLM
   */
  async processFeedback(feedback, notes = '', useLLM = false, llmApiKey = null, llmApiUrl = null, llmModel = null, llmHeaders = null, useLocalLLM = false) {
    if (!this.currentState) {
      throw new Error('No active optimization session. Start a new session first.');
    }
    
    const testResult = {
      sessionId: this.currentState.sessionId,
      iteration: this.currentState.iteration,
      config: { ...this.currentState.config },
      feedback: feedback, // 'better' or 'worse'
      notes: notes,
      timestamp: new Date().toISOString()
    };
    
    this.history.tests.push(testResult);
    
    // Оновлюємо найкращу конфігурацію
    if (feedback === 'better') {
      if (!this.bestState || this.currentState.iteration > this.bestState.iteration) {
        this.bestState = {
          config: { ...this.currentState.config },
          iteration: this.currentState.iteration,
          sessionId: this.currentState.sessionId
        };
        this.history.bestConfig = this.bestState;
      }
    }
    
    // Генеруємо нову конфігурацію
    let nextConfig;
    
    // Для локальної LLM API ключ не обов'язковий
    const shouldUseLLM = useLLM && llmApiUrl && llmModel && (useLocalLLM || llmApiKey);
    console.log('🔍 [OPTIMIZATION] Checking LLM conditions:', {
      useLLM,
      hasApiUrl: !!llmApiUrl,
      hasModel: !!llmModel,
      hasApiKey: !!llmApiKey,
      useLocalLLM,
      shouldUseLLM,
      feedback: feedback
    });
    
    if (shouldUseLLM) {
      console.log('🔵 [OPTIMIZATION] About to call generateConfigWithLLM with:', {
        hasApiKey: !!llmApiKey,
        apiUrl: llmApiUrl,
        model: llmModel,
        hasHeaders: !!llmHeaders,
        useLocalLLM: useLocalLLM,
        feedback: feedback
      });
      // Використовуємо LLM для генерації (з тими самими налаштуваннями, що для діаризації)
      // LLM підтримує всі типи feedback: 'better', 'worse', 'same'
      try {
        nextConfig = await this.generateConfigWithLLM(
          feedback,
          notes,
          this.currentState.config,
          this.bestState ? this.bestState.config : null,
          this.history.tests,
          llmApiKey,
          llmApiUrl,
          llmModel,
          llmHeaders,  // Передаємо готові заголовки
          useLocalLLM  // Передаємо флаг локальної LLM
        );
        console.log('🟢 [OPTIMIZATION] generateConfigWithLLM returned:', nextConfig);
      } catch (err) {
        console.error('🔴 [OPTIMIZATION] generateConfigWithLLM threw error:', err.message);
        throw err;
      }
    } else {
      // Використовуємо звичайний hill climbing (fallback)
      if (feedback === 'better') {
        nextConfig = this.generateNextConfig(this.currentState.config, 'exploit');
      } else if (feedback === 'same') {
        // Мікро-налаштування: невеликі випадкові зміни для пошуку кращого варіанту
        nextConfig = this.generateNextConfig(this.currentState.config, 'explore', 0.3); // Менший крок
      } else {
        // feedback === 'worse'
        if (this.bestState) {
          nextConfig = this.generateNextConfig(this.bestState.config, 'explore');
        } else {
          nextConfig = this.generateNextConfig(this.currentState.config, 'explore');
        }
      }
    }
    
    // Зберігаємо попередню конфігурацію для порівняння
    const previousConfig = { ...this.currentState.config };
    
    this.currentState.config = nextConfig;
    
    // Збільшуємо ітерацію
    this.currentState.iteration++;
    this.currentState.timestamp = new Date().toISOString();
    
    // Зберігаємо інформацію про метод генерації
    this.currentState.lastGenerationMethod = useLLM && shouldUseLLM ? 'llm' : 'hill-climbing';
    
    this.saveHistory();
    
    return {
      nextConfig: this.currentState.config,
      iteration: this.currentState.iteration,
      bestConfig: this.bestState ? this.bestState.config : null,
      history: this.getRecentHistory(5),
      generationMethod: this.currentState.lastGenerationMethod || 'hill-climbing',
      previousConfig: previousConfig // Для порівняння в UI
    };
  }
  
  /**
   * Отримує останні N тестів
   */
  getRecentHistory(limit = 10) {
    return this.history.tests.slice(-limit);
  }
  
  /**
   * Отримує статистику оптимізації
   */
  getStatistics() {
    const tests = this.history.tests;
    const betterCount = tests.filter(t => t.feedback === 'better').length;
    const worseCount = tests.filter(t => t.feedback === 'worse').length;
    const sameCount = tests.filter(t => t.feedback === 'same').length;
    
    return {
      totalTests: tests.length,
      betterCount: betterCount,
      worseCount: worseCount,
      sameCount: sameCount,
      bestConfig: this.history.bestConfig,
      currentIteration: this.currentState ? this.currentState.iteration : 0,
      sessionId: this.history.sessionId
    };
  }
  
  /**
   * Скидає поточну сесію
   */
  resetSession() {
    this.currentState = null;
    this.bestState = null;
    this.history.sessionId = null;
    this.history.currentIteration = 0;
    this.saveHistory();
  }
  
  /**
   * Експортує найкращу конфігурацію як JSON
   */
  exportBestConfig() {
    if (!this.history.bestConfig) {
      return null;
    }
    
    return {
      ...this.history.bestConfig.config,
      _metadata: {
        iteration: this.history.bestConfig.iteration,
        sessionId: this.history.bestConfig.sessionId,
        exportedAt: new Date().toISOString()
      }
    };
  }
}

module.exports = ParameterOptimizer;

