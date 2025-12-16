const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const BASE_URL = 'http://localhost:3000';
const TEST_FILE = path.join(__dirname, 'audio examples', 'Screen Recording 2025-12-05 at 07.29.15.m4a');

async function testFragmentMerge() {
  console.log('🧪 Тестування обробки розірваних фраз\n');
  console.log(`📁 Тестовий файл: ${TEST_FILE}`);
  
  if (!fs.existsSync(TEST_FILE)) {
    console.error(`❌ Файл не знайдено: ${TEST_FILE}`);
    return;
  }
  
  console.log('\n📤 Завантаження файлу через /api/diarize-overlap...');
  
  const formData = new FormData();
  formData.append('audio', fs.createReadStream(TEST_FILE));
  formData.append('language', 'en');
  formData.append('speakerCount', '2');
  formData.append('pipelineMode', 'mode3');
  formData.append('mode', 'smart');
  formData.append('engine', 'speechmatics');
  
  try {
    const response = await axios.post(`${BASE_URL}/api/diarize-overlap`, formData, {
      headers: formData.getHeaders(),
      timeout: 600000, // 10 хвилин
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    
    console.log('\n✅ Діаризація завершена');
    console.log('\n📊 Результат:');
    console.log(`   - Тип: ${response.data.type || 'unknown'}`);
    console.log(`   - Успіх: ${response.data.success || false}`);
    
    if (response.data.correctedDiarization) {
      const segments = response.data.correctedDiarization?.recordings?.[0]?.results?.['overlap-corrected']?.segments || [];
      console.log(`   - Кількість сегментів: ${segments.length}`);
      
      // Шукаємо приклади розірваних фраз
      console.log('\n🔍 Перевірка на розірвані фрази:');
      for (let i = 0; i < segments.length - 1; i++) {
        const current = segments[i];
        const next = segments[i + 1];
        const gap = (parseFloat(next.start) || 0) - (parseFloat(current.end) || 0);
        
        if (gap >= 0 && gap <= 3.0) {
          const currentText = (current.text || '').trim();
          const nextText = (next.text || '').trim();
          
          // Перевіряємо, чи виглядає як розірвана фраза
          const looksIncomplete = !/[.!?]$/.test(currentText) && 
                                  (/\b(to|and|or|but|did\s+you|can\s+you|try\s+to)\s*$/i.test(currentText) ||
                                   currentText.length < 20);
          
          if (looksIncomplete) {
            console.log(`\n   ⚠️ Підозріла пара (gap: ${gap.toFixed(2)}s):`);
            console.log(`      [${current.start.toFixed(2)}s-${current.end.toFixed(2)}s] ${current.speaker}: "${currentText}"`);
            console.log(`      [${next.start.toFixed(2)}s-${next.end.toFixed(2)}s] ${next.speaker}: "${nextText}"`);
          }
        }
      }
    }
    
    if (response.data.markdownTable) {
      console.log('\n📋 Markdown таблиця отримана');
      const lines = response.data.markdownTable.split('\n').filter(l => l.trim());
      console.log(`   - Рядків: ${lines.length}`);
      
      // Показуємо перші 10 рядків
      console.log('\n   Перші 10 рядків:');
      lines.slice(0, 10).forEach((line, idx) => {
        if (line.includes('|')) {
          console.log(`   ${idx + 1}. ${line.substring(0, 100)}`);
        }
      });
    }
    
  } catch (error) {
    console.error('\n❌ Помилка:', error.message);
    if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', JSON.stringify(error.response.data, null, 2).substring(0, 500));
    }
  }
}

// Запускаємо тест
testFragmentMerge().catch(console.error);



