/**
 * Скрипт для автоматичної діаризації та збереження результатів в Debug
 */

const fs = require('fs');
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');

const AUDIO_FILE = 'Call centre example.MP3';
const DEBUG_DIR = path.join(__dirname, 'Debug');
const API_URL = 'http://localhost:3000';

async function runDiarization() {
  try {
    console.log('🎵 Starting diarization for:', AUDIO_FILE);
    
    // Перевіряємо наявність файлу
    const audioPath = path.join(__dirname, AUDIO_FILE);
    if (!fs.existsSync(audioPath)) {
      throw new Error(`Audio file not found: ${audioPath}`);
    }
    
    // Створюємо FormData
    const formData = new FormData();
    formData.append('audio', fs.createReadStream(audioPath));
    formData.append('language', 'en');
    formData.append('speakerCount', '2');
    formData.append('mode', 'local');
    formData.append('pipelineMode', 'mode3');
    formData.append('textAnalysisMode', 'script');
    
    console.log('📤 Sending request to /api/diarize-overlap...');
    
    // Відправляємо запит (mode3 використовує SSE, тому потрібно обробляти потік)
    let sseData = '';
    
    const response = await axios.post(`${API_URL}/api/diarize-overlap`, formData, {
      headers: formData.getHeaders(),
      timeout: 600000, // 10 хвилин
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      responseType: 'stream' // Для SSE
    });
    
    // Обробляємо SSE потік
    let result = null;
    await new Promise((resolve, reject) => {
      response.data.on('data', (chunk) => {
        sseData += chunk.toString();
        const lines = sseData.split('\n\n');
        sseData = lines.pop() || ''; // Залишаємо останній неповний блок
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              if (data.type === 'final-result') {
                result = data;
              }
            } catch (e) {
              // Ігноруємо помилки парсингу проміжних повідомлень
            }
          }
        }
      });
      
      response.data.on('end', () => {
        // Обробляємо останній блок
        if (sseData) {
          const lines = sseData.split('\n\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6));
                if (data.type === 'final-result') {
                  result = data;
                }
              } catch (e) {
                // Ігноруємо помилки
              }
            }
          }
        }
        
        if (!result) {
          reject(new Error('No final result received from SSE stream'));
          return;
        }
        
        console.log('✅ Diarization completed');
        resolve(result);
      });
      
      response.data.on('error', (error) => {
        reject(error);
      });
    });
    
    if (!result) {
      throw new Error('No result received');
    }
    
    processResult(result);
    
  } catch (error) {
    console.error('❌ Error:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      if (error.response.data && typeof error.response.data === 'string') {
        console.error('Response data (first 500 chars):', error.response.data.substring(0, 500));
      } else {
        console.error('Response data:', JSON.stringify(error.response.data, null, 2));
      }
    }
    process.exit(1);
  }
}

function processResult(result) {
  try {
    // Створюємо структуру для збереження
    // Структура відповіді: { primaryDiarization, voiceTracks, markdownTable, correctedDiarization, textAnalysis, ... }
    const primaryDiarization = result.primaryDiarization;
    const voiceTracks = result.voiceTracks || [];
    const markdownTable = result.markdownTable;
    const textAnalysis = result.textAnalysis;
    
    // Витягуємо general, speaker1, speaker2
    let general = null;
    let speaker1 = null;
    let speaker2 = null;
    
    if (primaryDiarization) {
      general = {
        segments: primaryDiarization.segments || primaryDiarization.speechmatics?.segments || [],
        speechmatics: primaryDiarization.speechmatics || primaryDiarization
      };
    }
    
    if (Array.isArray(voiceTracks) && voiceTracks.length > 0) {
      // voiceTracks - масив об'єктів з полями speaker, role, transcription, segments
      voiceTracks.forEach((track) => {
        const trackData = track.transcription || track;
        const segments = trackData.segments || trackData.speechmatics?.segments || [];
        
        if (track.speaker === 'SPEAKER_00' || (track.role === 'operator' || track.role === 'agent')) {
          speaker1 = {
            segments: segments,
            speechmatics: trackData.speechmatics || trackData,
            speaker: track.speaker,
            role: track.role
          };
        } else if (track.speaker === 'SPEAKER_01' || (track.role === 'client' || track.role === 'customer')) {
          speaker2 = {
            segments: segments,
            speechmatics: trackData.speechmatics || trackData,
            speaker: track.speaker,
            role: track.role
          };
        }
      });
    }
    
    const debugData = {
      timestamp: new Date().toISOString(),
      audioFile: AUDIO_FILE,
      general: general,
      speaker1: speaker1,
      speaker2: speaker2,
      markdown: markdownTable,
      textAnalysis: textAnalysis,
      fullResult: result
    };
    
    // Зберігаємо повний результат
    const fullResultPath = path.join(DEBUG_DIR, 'diarization_full_result.json');
    fs.writeFileSync(fullResultPath, JSON.stringify(debugData, null, 2));
    console.log('💾 Saved full result to:', fullResultPath);
    
    // Зберігаємо окремі JSON файли
    if (debugData.general) {
      const generalPath = path.join(DEBUG_DIR, 'general_segments.json');
      fs.writeFileSync(generalPath, JSON.stringify(debugData.general, null, 2));
      console.log('💾 Saved general segments to:', generalPath);
    }
    
    if (debugData.speaker1) {
      const speaker1Path = path.join(DEBUG_DIR, 'speaker1_segments.json');
      fs.writeFileSync(speaker1Path, JSON.stringify(debugData.speaker1, null, 2));
      console.log('💾 Saved speaker1 segments to:', speaker1Path);
    }
    
    if (debugData.speaker2) {
      const speaker2Path = path.join(DEBUG_DIR, 'speaker2_segments.json');
      fs.writeFileSync(speaker2Path, JSON.stringify(debugData.speaker2, null, 2));
      console.log('💾 Saved speaker2 segments to:', speaker2Path);
    }
    
    // Створюємо діагностичний файл з прикладами
    createDiagnosticFile(debugData);
    
    console.log('✅ All files saved to Debug directory');
    
  } catch (error) {
    console.error('❌ Error processing result:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
  }
}

function createDiagnosticFile(data) {
  const diagnosticPath = path.join(DEBUG_DIR, 'diagnostic_examples.md');
  
  // Отримуємо приклади сегментів
  const generalSegments = data.general?.segments || data.general?.speechmatics?.segments || [];
  const speaker1Segments = data.speaker1?.segments || data.speaker1?.speechmatics?.segments || [];
  const speaker2Segments = data.speaker2?.segments || data.speaker2?.speechmatics?.segments || [];
  
  // Парсимо markdown для отримання прикладів фраз
  const markdownLines = (data.markdown || '').split('\n');
  const markdownSegments = [];
  let headerProcessed = false;
  
  for (const line of markdownLines) {
    const trimmed = line.trim();
    if (trimmed.includes('---') || trimmed.toLowerCase().includes('segment id')) {
      headerProcessed = true;
      continue;
    }
    if (trimmed.startsWith('|') && headerProcessed) {
      const cells = trimmed.split('|').map(c => c.trim()).filter(c => c);
      if (cells.length >= 5) {
        markdownSegments.push({
          id: cells[0],
          speaker: cells[1],
          text: cells[2],
          start: parseFloat(cells[3]) || 0,
          end: parseFloat(cells[4]) || 0
        });
      }
    }
  }
  
  // Створюємо діагностичний документ
  let content = `# Діагностичні дані для аналізу класифікації\n\n`;
  content += `**Дата:** ${new Date().toISOString()}\n`;
  content += `**Аудіо файл:** ${AUDIO_FILE}\n\n`;
  
  content += `## 1. Структура JSON\n\n`;
  content += `### General segments (${generalSegments.length} сегментів)\n`;
  content += `\`\`\`json\n`;
  content += JSON.stringify(generalSegments.slice(0, 3), null, 2);
  content += `\n\`\`\`\n\n`;
  
  content += `### Speaker1 segments (${speaker1Segments.length} сегментів)\n`;
  content += `\`\`\`json\n`;
  content += JSON.stringify(speaker1Segments.slice(0, 3), null, 2);
  content += `\n\`\`\`\n\n`;
  
  content += `### Speaker2 segments (${speaker2Segments.length} сегментів)\n`;
  content += `\`\`\`json\n`;
  content += JSON.stringify(speaker2Segments.slice(0, 3), null, 2);
  content += `\n\`\`\`\n\n`;
  
  content += `## 2. Приклади Markdown фраз\n\n`;
  markdownSegments.slice(0, 5).forEach((seg, idx) => {
    content += `### Фраза ${idx + 1}\n`;
    content += `- **Text:** "${seg.text}"\n`;
    content += `- **Speaker:** ${seg.speaker}\n`;
    content += `- **Time:** ${seg.start}s - ${seg.end}s\n\n`;
  });
  
  content += `## 3. Результати Text Analysis\n\n`;
  if (data.textAnalysis) {
    content += `- **Blue:** ${data.textAnalysis.Blue?.length || 0} сегментів\n`;
    content += `- **Green:** ${data.textAnalysis.Green?.length || 0} сегментів\n`;
    content += `- **Red:** ${data.textAnalysis.Red?.length || 0} сегментів\n\n`;
    
    if (data.textAnalysis.Blue?.length > 0) {
      content += `### Приклади Blue:\n`;
      data.textAnalysis.Blue.slice(0, 3).forEach((item, idx) => {
        content += `${idx + 1}. "${item.text}" (${item.start}s - ${item.end}s)\n`;
      });
      content += `\n`;
    }
    
    if (data.textAnalysis.Green?.length > 0) {
      content += `### Приклади Green:\n`;
      data.textAnalysis.Green.slice(0, 3).forEach((item, idx) => {
        content += `${idx + 1}. "${item.text}" (${item.start}s - ${item.end}s)\n`;
      });
      content += `\n`;
    }
    
    if (data.textAnalysis.Red?.length > 0) {
      content += `### Приклади Red:\n`;
      data.textAnalysis.Red.slice(0, 3).forEach((item, idx) => {
        content += `${idx + 1}. "${item.text}" (${item.start}s - ${item.end}s)\n`;
      });
      content += `\n`;
    }
  }
  
  content += `## 4. Очікуваний результат\n\n`;
  content += `Для кожної фрази з markdown:\n`;
  content += `- **Blue:** має бути в general І (speaker1 АБО speaker2)\n`;
  content += `- **Green:** має бути в (speaker1 АБО speaker2), але НЕ в general\n`;
  content += `- **Red:** немає в жодному джерелі\n\n`;
  
  fs.writeFileSync(diagnosticPath, content);
  console.log('💾 Saved diagnostic file to:', diagnosticPath);
}

// Запускаємо
runDiarization();

