const axios = require('axios');

const BASE_URL = 'http://localhost:3000';

// Тестові дані з прикладу користувача
const testData = {
  agentTranscript: {
    segments: [
      {
        text: "And did you try to",
        start: 7.28,
        end: 8.56,
        speaker: "SPEAKER_00"
      },
      {
        text: "reset your modem",
        start: 9.40,
        end: 10.84,
        speaker: "SPEAKER_00"
      }
    ]
  },
  clientTranscript: {
    segments: [
      {
        text: "I have a problem with my internet connection is still dropping",
        start: 0.32,
        end: 5.24,
        speaker: "SPEAKER_01"
      }
    ]
  },
  mode: 'smart',
  recordingId: 'test_fragment_merge'
};

async function testFragmentMergeDirect() {
  console.log('🧪 Тестування обробки розірваних фраз через /api/apply-markdown-fixes\n');
  console.log('📤 Відправка тестових даних...\n');
  console.log('📋 Вхідні дані:');
  console.log('   Agent segments:', JSON.stringify(testData.agentTranscript.segments, null, 2));
  console.log('   Client segments:', JSON.stringify(testData.clientTranscript.segments, null, 2));
  
  try {
    const response = await axios.post(`${BASE_URL}/api/apply-markdown-fixes`, testData, {
      headers: {
        'Content-Type': 'application/json'
      },
      timeout: 120000 // 2 хвилини
    });
    
    console.log('\n✅ Відповідь отримана');
    console.log('\n📥 Повернуті дані:');
    console.log('   Success:', response.data.success);
    console.log('   Cached:', response.data.cached || false);
    
    if (response.data.markdown) {
      const lines = response.data.markdown.split('\n').filter(l => l.trim() && l.includes('|'));
      console.log(`\n📋 Markdown таблиця (${lines.length} рядків):`);
      lines.forEach((line, idx) => {
        if (idx < 10) { // Перші 10 рядків
          console.log(`   ${line}`);
        }
      });
      
      // Перевіряємо, чи об'єднані фрази
      const mergedLine = lines.find(line => 
        line.includes('And did you try to') && 
        line.includes('reset your modem')
      );
      
      if (mergedLine) {
        console.log('\n✅ Розірвана фраза об\'єднана!');
        console.log(`   Рядок: ${mergedLine}`);
      } else {
        console.log('\n⚠️ Розірвана фраза НЕ об\'єднана');
        console.log('   Шукаємо окремі сегменти...');
        const firstPart = lines.find(line => line.includes('And did you try to'));
        const secondPart = lines.find(line => line.includes('reset your modem'));
        if (firstPart) console.log(`   Перша частина: ${firstPart}`);
        if (secondPart) console.log(`   Друга частина: ${secondPart}`);
      }
    }
    
  } catch (error) {
    console.error('\n❌ Помилка:', error.message);
    if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', JSON.stringify(error.response.data, null, 2).substring(0, 1000));
    }
  }
}

testFragmentMergeDirect().catch(console.error);



