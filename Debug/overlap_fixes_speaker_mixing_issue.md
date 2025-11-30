# Проблема: Змішування реплік між спікерами після застосування Overlap Fixes

## Дата: 2025-11-26

## Опис проблеми

Після застосування overlap fixes через кнопку "✅ Apply Overlap Fixes", структура діалогу руйнується: фрази з різних спікерів об'єднуються в один рядок. Конкретно:

- Фрази, сказані другим спікером (SPEAKER_01), потрапляють до першого спікера (SPEAKER_00)
- Це призводить до того, що "їде вся структура" - діалог стає некоректним
- Проблема виникає після обробки voice tracks та їх об'єднання з primary diarization

## Контекст системи

### Архітектура Overlap Diarization

Система використовує наступний pipeline:

1. **Step 1**: Primary diarization (Speechmatics/Azure) - базовий транскрипт з визначенням спікерів
2. **Step 2**: Speaker separation - розділення аудіо на окремі voice tracks для кожного спікера
   - Mode 1: AudioShake API
   - Mode 2: PyAnnote (локально)
   - Mode 3: SpeechBrain SepFormer (локально)
3. **Step 3**: Transcription of voice tracks - транскрипція кожного voice track окремо
4. **Step 4**: Role analysis - визначення ролі (operator/client)
5. **Step 5**: Overlap correction - об'єднання primary diarization з voice tracks

### Ключові файли

- `app.js`: Frontend логіка, обробка overlap fixes
- `overlap_merge_utils.js`: Core логіка збору та об'єднання сегментів
- `server.js`: Backend API для overlap diarization

## Спроби виправлення

### 1. Фільтрація residual audio в voice tracks

**Проблема**: Voice tracks містять residual audio від інших спікерів, які Speechmatics визначає як окремих спікерів.

**Рішення**:
- Визначення основного спікера в кожному voice track за тривалістю та кількістю сегментів
- Фільтрація всіх сегментів, які не належать основному спікеру
- Додано перевірку: основний спікер має мінімум 60% загальної тривалості voice track

**Файл**: `overlap_merge_utils.js`, функція `collectVoiceTrackSegments`

### 2. Перевірка перекриття з іншими спікерами

**Проблема**: Primary segments можуть перекриватися з voice tracks іншого спікера, що призводить до змішування.

**Рішення**:
- Додано перевірку: якщо primary segment перекривається більш ніж на 30% з voice track іншого спікера, він пропускається
- Це запобігає прив'язуванню реплік до неправильного спікера

**Файли**: 
- `app.js`, функція `mergeVoiceTrackSegments` (STEP 3)
- `overlap_merge_utils.js`, функція `shouldAddPrimarySegment`

### 3. Фінальна валідація сегментів

**Проблема**: Можливі сегменти з неправильними speaker labels.

**Рішення**:
- Додано фінальну перевірку: всі сегменти з одного voice track мають правильний спікер
- Автоматичне виправлення сегментів з неправильними labels
- Детальне логування для діагностики

**Файл**: `overlap_merge_utils.js`, функція `collectVoiceTrackSegments`

### 4. Перевірки при об'єднанні послідовних сегментів

**Проблема**: Можливе об'єднання сегментів різних спікерів.

**Рішення**:
- Додано перевірку: сегменти об'єднуються тільки якщо спікери співпадають точно
- Перевірка валідності speaker labels перед обробкою

**Файл**: `app.js`, функція `mergeVoiceTrackSegments` (STEP 6)

## Поточна ситуація

**Статус**: Проблема залишається невирішеною після всіх виправлень.

**Сприйняття**: "нічого не змінилося в результатах"

## Можливі причини, що залишилися

### 1. Проблема в логіці визначення основного спікера

**Гіпотеза**: Voice tracks можуть містити більше residual audio, ніж очікується, і основний спікер визначається неправильно.

**Перевірка потрібна**:
- Логування розподілу спікерів у кожному voice track
- Перевірка, чи основний спікер дійсно має >60% тривалості
- Аналіз випадків, коли основний спікер має <50% тривалості

### 2. Проблема в логіці об'єднання сегментів

**Гіпотеза**: Сегменти можуть об'єднуватися неправильно через помилки в тексті або часових мітках.

**Перевірка потрібна**:
- Логування всіх операцій об'єднання
- Перевірка часових міток перед об'єднанням
- Аналіз випадків, коли сегменти різних спікерів об'єднуються

### 3. Проблема в primary diarization

**Гіпотеза**: Primary diarization може містити помилки в призначенні спікерів, які потім поширюються на voice tracks.

**Перевірка потрібна**:
- Порівняння primary diarization з voice tracks
- Аналіз розбіжностей у призначенні спікерів

### 4. Проблема в самій separation

**Гіпотеза**: Speaker separation може створювати voice tracks, які містять audio від обох спікерів, а не тільки від одного.

**Перевірка потрібна**:
- Аналіз якості separation (чи дійсно voice tracks містять тільки одного спікера)
- Порівняння voice tracks з primary diarization

## Рекомендації для діагностики

### 1. Додати детальне логування

```javascript
// У overlap_merge_utils.js, функція collectVoiceTrackSegments
logger.log(`📊 Voice track ${trackSpeaker} analysis:`, {
  totalSegments: speechmaticsSegments.length,
  speakersDetected: Object.keys(segmentsByDetectedSpeaker),
  mainSpeaker: mainDetectedSpeaker,
  mainSpeakerDuration: maxDuration,
  mainSpeakerPercent: finalDurationPercent,
  skippedSegments: skippedResidualSegments,
  acceptedSegments: segments.filter(s => s.originalTrackSpeaker === trackSpeaker).length
});
```

### 2. Створити тестовий набір

- Взяти конкретний випадок з проблемою
- Зберегти всі проміжні результати:
  - Primary diarization segments
  - Voice tracks segments (до та після фільтрації)
  - Merged segments (після кожного кроку)
- Порівняти очікуваний результат з фактичним

### 3. Візуалізація проблеми

- Створити timeline візуалізацію:
  - Primary segments (з кольорами для кожного спікера)
  - Voice track segments (з кольорами для кожного спікера)
  - Merged segments (з кольорами для кожного спікера)
- Виділити місця, де відбувається змішування

### 4. Перевірити логіку в app.js

**Проблема**: У `app.js` є своя версія `mergeVoiceTrackSegments`, яка може конфліктувати з версією в `overlap_merge_utils.js`.

**Перевірка потрібна**:
- Чи використовується правильна функція?
- Чи є різниця в логіці між двома версіями?

## Наступні кроки

1. **Додати детальне логування** для відстеження кожного кроку обробки
2. **Створити тестовий випадок** з конкретним прикладом проблеми
3. **Порівняти primary diarization з voice tracks** для виявлення розбіжностей
4. **Перевірити якість separation** - чи дійсно voice tracks містять тільки одного спікера
5. **Аналізувати merged segments** - знайти місця, де відбувається змішування

## Технічні деталі

### Ключові функції

1. `collectVoiceTrackSegments` (overlap_merge_utils.js)
   - Збирає сегменти з voice tracks
   - Фільтрує residual audio
   - Визначає основного спікера

2. `mergeVoiceTrackSegments` (app.js та overlap_merge_utils.js)
   - Об'єднує voice track segments з primary segments
   - Виконує дедуплікацію
   - Об'єднує послідовні сегменти

3. `shouldAddPrimarySegment` (overlap_merge_utils.js)
   - Визначає, чи потрібно додавати primary segment
   - Перевіряє перекриття з voice tracks

### Пороги та параметри

- **Мінімальний відсоток основного спікера**: 60%
- **Максимальне перекриття з іншим спікером**: 30%
- **Мінімальна тривалість сегмента**: 0.3s
- **Поріг для об'єднання послідовних сегментів**: 0.5s пауза

## Детальний план дій для вирішення проблеми

### Кроки розв'язання для 5-річної дитини (без технічних знань)

1. **Уяви дві книги**: у першій книзі – історія про дівчинку, а в другій про хлопчика.
2. **Перевір, чи правильно написані сторінки**: кожен розділ має бути в правильній книзі.
3. **Переконайся, що сторінки не перемішані**: якщо на одній сторінці вмістить текст про дівчинку і хлопчика, це помилка.
4. **Підкажи дитині знову перевірити книги**: підготуй список сторінок, щоб зрозуміти, де саме з'явилася помилка.
5. **Візуалізуй**: намальуй два колірних стовпчики – один для дівчинки, інший для хлопчика. Підкажи дитині ставити позначку (наприклад, сердечко), коли знайдеш сторінку з неправильним текстом.
6. **Повторно прочитай історію**: після виправлення перевір, що кожен розділ тепер у своїй книзі.

---

## Детальні інструкції для розробників

### 1. Додати детальне логування

#### `collectVoiceTrackSegments` (overlap_merge_utils.js)

```javascript
logger.log(`📊 Voice track ${trackSpeaker} analysis:`, {
  totalSegments: speechmaticsSegments.length,
  speakersDetected: Object.keys(segmentsByDetectedSpeaker),
  mainSpeaker: mainDetectedSpeaker,
  mainSpeakerDuration: maxDuration,
  mainSpeakerPercent: finalDurationPercent,
  skippedSegments: skippedResidualSegments,
  acceptedSegments: segments.filter(s => s.originalTrackSpeaker === trackSpeaker).length
});
```

#### `mergeVoiceTrackSegments`

Логувати кожен крок:
- Додавання сегментів
- Перевірки перекриття
- Об'єднання послідовних сегментів
- Фінальна дедуплікація

```javascript
logger.log(`🔄 Merge step:`, {
  step: 'adding_primary',
  primarySegment: { speaker: pSpeaker, start: pStart, end: pEnd, text: pText.substring(0, 50) },
  overlapsSameSpeaker: overlapsSameSpeaker,
  overlapsDifferentSpeaker: overlapsDifferentSpeaker,
  decision: overlapsSameSpeaker || overlapsDifferentSpeaker ? 'skip' : 'add'
});
```

#### `shouldAddPrimarySegment`

Логувати рішення «пропустити» або «додати»:

```javascript
logger.log(`🔍 Primary segment check:`, {
  segment: { speaker: pSpeaker, start: pStart, end: pEnd, text: pText.substring(0, 50) },
  overlapsDifferentSpeaker: overlapsDifferentSpeaker,
  overlapsSameSpeaker: overlapsSameSpeaker,
  decision: shouldAdd ? 'add' : 'skip',
  reason: shouldAdd ? 'no_overlap' : (overlapsDifferentSpeaker ? 'overlaps_different_speaker' : 'overlaps_same_speaker')
});
```

### 2. Створити репрезентативний тестовий випадок

1. **Вибрати конкретну розмову**, де з'являється помилка.
2. **Зберегти проміжні результати**:
   - Primary diarization сегменти (JSON)
   - Voice-track сегменти до/після фільтрації
   - Сегменти після кожного кроку об'єднання (Step 3, Step 6)
3. **Порівняти очікуваний результат** (коректна розмірка спікерів) з фактичним.
4. **Виявити етап**, на якому виникає відмінність.

**Структура тестового випадку:**

```json
{
  "testCase": "speaker_mixing_issue_001",
  "audioFile": "path/to/test/audio.wav",
  "expectedResult": {
    "SPEAKER_00": ["text1", "text2", ...],
    "SPEAKER_01": ["text3", "text4", ...]
  },
  "stages": {
    "primaryDiarization": [...],
    "voiceTracksBeforeFilter": [...],
    "voiceTracksAfterFilter": [...],
    "mergedAfterStep3": [...],
    "mergedAfterStep6": [...],
    "finalResult": [...]
  }
}
```

### 3. Порівняти primary diarization з voice tracks

Для кожного `segment` в primary діагностиці порівняти:

- `speakerId`
- Часові мітки (`start`, `end`)
- Текст (`text`)

**Алгоритм перевірки:**

```javascript
function comparePrimaryWithVoiceTracks(primarySegments, voiceTrackSegments) {
  const discrepancies = [];
  
  primarySegments.forEach(pSeg => {
    const overlappingVoiceTracks = voiceTrackSegments.filter(vtSeg => {
      const overlap = calculateOverlap(pSeg, vtSeg);
      return overlap > 0.1; // 100ms minimum
    });
    
    overlappingVoiceTracks.forEach(vtSeg => {
      if (pSeg.speaker !== vtSeg.speaker) {
        discrepancies.push({
          primary: pSeg,
          voiceTrack: vtSeg,
          overlap: calculateOverlap(pSeg, vtSeg),
          issue: 'speaker_mismatch'
        });
      }
    });
  });
  
  return discrepancies;
}
```

**Якщо знайдуться значні розбіжності**, перевірити:
- Чи було застосовано `shouldAddPrimarySegment` правильно
- Чи не підштовхнуло інший процес (наприклад, `mergeVoiceTrackSegments` у `app.js`)

### 4. Перевірити якість speaker separation

**Кроки перевірки:**

1. Після `pyannote`/`AudioShake`/`SpeechBrain` отримати аудіо-файли voice tracks
2. Відтворити і прослухати кілька випадків вручну: чи вийшов лише один голос?
3. Якщо residual audio > 20% – це сигнал про помилку в separation

**Автоматична перевірка:**

```javascript
function analyzeSeparationQuality(voiceTrack, primaryDiarization) {
  const voiceTrackSegments = transcribeVoiceTrack(voiceTrack);
  const detectedSpeakers = new Set(voiceTrackSegments.map(s => s.speaker));
  
  if (detectedSpeakers.size > 1) {
    const speakerDurations = {};
    voiceTrackSegments.forEach(seg => {
      const duration = seg.end - seg.start;
      speakerDurations[seg.speaker] = (speakerDurations[seg.speaker] || 0) + duration;
    });
    
    const totalDuration = Object.values(speakerDurations).reduce((a, b) => a + b, 0);
    const mainSpeakerPercent = (Math.max(...Object.values(speakerDurations)) / totalDuration) * 100;
    
    return {
      quality: mainSpeakerPercent > 80 ? 'good' : mainSpeakerPercent > 60 ? 'acceptable' : 'poor',
      mainSpeakerPercent,
      detectedSpeakers: Array.from(detectedSpeakers),
      warning: mainSpeakerPercent < 60 ? 'High residual audio detected' : null
    };
  }
  
  return { quality: 'good', mainSpeakerPercent: 100 };
}
```

### 5. Аналізувати логіку об'єднання сегментів

#### Перевірка перекриття (`shouldAddPrimarySegment`)

- Переконатися, що перевірка `overlap > 30%` не застосовується до сегментів, які вже належать іншому спікеру
- Додати перевірку на часові мітки перед перевіркою перекриття

```javascript
// CRITICAL: Check temporal overlap first
const overlapStart = Math.max(pStart, vStart);
const overlapEnd = Math.min(pEnd, vEnd);
const overlapDuration = Math.max(0, overlapEnd - overlapStart);

if (overlapDuration <= 0) continue; // No temporal overlap

// Then check percentage
const overlapPercent = pDuration > 0 ? (overlapDuration / pDuration) * 100 : 0;
```

#### Об'єднання послідовних сегментів (`mergeVoiceTrackSegments`)

- Переконатися, що `speakerId` дорівнює передньому сегменту
- Перевірити, що `gap < 0.5s` (або заданий поріг)
- Додати перевірку на текст перед об'єднанням (щоб не об'єднувати різні репліки)

```javascript
// CRITICAL: Verify speakers match EXACTLY
if (currentSpeaker !== nextSpeaker) {
  console.warn(`⚠️ Cannot merge: different speakers (${currentSpeaker} vs ${nextSpeaker})`);
  break;
}

// Check gap
const gap = nextStart - currentEnd;
if (gap >= maxPauseForMerge) {
  break; // Too large gap
}

// Check text similarity (optional but recommended)
const isDuplicate = areTextsSimilar(mergedSegment.text, next.text, {
  minLevenshteinSim: 0.8,
  minJaccardSim: 0.6
});

if (isDuplicate) {
  // Keep longer text
  if (next.text.length > mergedSegment.text.length) {
    mergedSegment = { ...next };
  }
  continue;
}
```

### 6. Перевірити конфлікти між `app.js` і `overlap_merge_utils.js`

**Проблема**: У `app.js` оголошена функція `mergeVoiceTrackSegments`; переконатися, що саме вона викликається.

**Кроки перевірки:**

1. Знайти всі місця, де викликається `mergeVoiceTrackSegments`:

```bash
grep -r "mergeVoiceTrackSegments" app.js overlap_merge_utils.js
```

2. Перевірити, яка версія використовується:

```javascript
// У app.js
const { mergeVoiceTrackSegments } = require('./overlap_merge_utils');
// АБО
function mergeVoiceTrackSegments(...) { ... } // Локальна версія
```

3. Якщо в обох файлах різна логіка:
   - Визначити, яка версія потрібна
   - Видалити зайву або об'єднати логіку
   - Внести відповідні `export`/`require`

4. Додати коментарі для ясності:

```javascript
// IMPORTANT: This function is defined in overlap_merge_utils.js
// Do not create a local version in app.js - use the imported one
const { mergeVoiceTrackSegments } = require('./overlap_merge_utils');
```

### 7. Налаштувати пороги

**Поточні налаштування:**

- **Мінімальний відсоток основного спікера**: 60%
  - Якщо виявляється < 50%, логувати як *critical*
  - Якщо 50-60%, логувати як *warning*

- **Максимальне перекриття з іншим спікером**: 30%
  - Якщо перевищено, фільтрувати сегмент
  - Логувати всі випадки перевищення

- **Мінімальна тривалість сегмента**: 0.3s
  - Якщо < 0.2s – пропускати
  - Якщо 0.2-0.3s – логувати як *warning*

- **Поріг об'єднання**: 0.5s пауза між сегментами
  - Налаштовується залежно від типу діалогу
  - Для швидких діалогів можна зменшити до 0.3s

**Конфігураційний файл:**

```javascript
// config/overlap_merge_config.js
module.exports = {
  thresholds: {
    mainSpeakerMinPercent: 60,
    mainSpeakerWarningPercent: 50,
    maxOverlapWithDifferentSpeaker: 30,
    minSegmentDuration: 0.3,
    minSegmentDurationWarning: 0.2,
    maxPauseForMerge: 0.5,
    maxPauseForFastDialogue: 0.3
  },
  logging: {
    logAllOverlaps: true,
    logAllMerges: true,
    logAllSkips: true
  }
};
```

### 8. Візуалізація

**Створити timeline візуалізацію** (наприклад, з `vis.js` або `plotly`):

- **Primary segments** – колір за спікером (наприклад, синій для SPEAKER_00, червоний для SPEAKER_01)
- **Voice-track сегменти** – інший колір (наприклад, світло-синій для SPEAKER_00, світло-червоний для SPEAKER_01)
- **Після об'єднання** – третій колір (наприклад, темно-синій для SPEAKER_00, темно-червоний для SPEAKER_01)

**Виділити місця, де спікери змішуються** (перетини кольорів):

```javascript
function visualizeSegments(primarySegments, voiceTrackSegments, mergedSegments) {
  const timeline = {
    primary: primarySegments.map(seg => ({
      speaker: seg.speaker,
      start: seg.start,
      end: seg.end,
      text: seg.text.substring(0, 30),
      color: seg.speaker === 'SPEAKER_00' ? '#3498db' : '#e74c3c'
    })),
    voiceTracks: voiceTrackSegments.map(seg => ({
      speaker: seg.speaker,
      start: seg.start,
      end: seg.end,
      text: seg.text.substring(0, 30),
      color: seg.speaker === 'SPEAKER_00' ? '#85c1e2' : '#f1948a'
    })),
    merged: mergedSegments.map(seg => ({
      speaker: seg.speaker,
      start: seg.start,
      end: seg.end,
      text: seg.text.substring(0, 30),
      color: seg.speaker === 'SPEAKER_00' ? '#1b4f72' : '#922b21'
    }))
  };
  
  // Виявити змішування
  const mixingPoints = detectMixing(mergedSegments);
  
  return { timeline, mixingPoints };
}
```

### 9. Тестування в CI

**Додати unit-тести** для ключових функцій:

```javascript
// tests/overlap_merge.test.js
describe('collectVoiceTrackSegments', () => {
  it('should filter residual audio correctly', () => {
    const voiceTrack = {
      speaker: 'SPEAKER_00',
      transcription: {
        segments: [
          { speaker: 'SPEAKER_00', start: 0, end: 5, text: 'Hello' },
          { speaker: 'SPEAKER_01', start: 5, end: 6, text: 'Hi' }, // Residual
          { speaker: 'SPEAKER_00', start: 6, end: 10, text: 'How are you' }
        ]
      }
    };
    
    const result = collectVoiceTrackSegments([voiceTrack]);
    
    expect(result).toHaveLength(2);
    expect(result.every(s => s.speaker === 'SPEAKER_00')).toBe(true);
  });
});

describe('shouldAddPrimarySegment', () => {
  it('should skip primary segment that overlaps with different speaker voice track', () => {
    const primarySeg = { speaker: 'SPEAKER_00', start: 0, end: 5, text: 'Hello' };
    const voiceTrackSegs = [
      { speaker: 'SPEAKER_01', start: 0, end: 5, text: 'Hi there' }
    ];
    
    const result = shouldAddPrimarySegment(primarySeg, voiceTrackSegs);
    
    expect(result).toBe(false);
  });
});
```

**Тестувати на різних сценаріях:**
- Чистий аудіо (без residual audio)
- Residual audio < 20%
- Residual audio > 20%
- Перекриття > 30%
- Перекриття < 30%

### 10. Документація

**Оновити README/Doc**, описуючи:

- **Кроки обробки**: детальний опис кожного кроку pipeline
- **Порогові значення**: що вони означають і як їх налаштувати
- **Як слід реагувати на логування**: що робити при різних типах попереджень

**Приклад документації:**

```markdown
## Overlap Merge Configuration

### Thresholds

- `mainSpeakerMinPercent` (60%): Minimum percentage of voice track duration that must belong to main speaker
  - If below 50%: Critical warning - separation quality is poor
  - If 50-60%: Warning - possible residual audio issues

- `maxOverlapWithDifferentSpeaker` (30%): Maximum allowed overlap between primary segment and different speaker's voice track
  - If exceeded: Segment is skipped to prevent speaker mixing

### Logging

- `logAllOverlaps`: Log all overlap checks (useful for debugging)
- `logAllMerges`: Log all merge operations (useful for debugging)
- `logAllSkips`: Log all skipped segments (useful for debugging)
```

## Висновок

Проблема змішування реплік між спікерами залишається невирішеною, незважаючи на численні спроби виправлення. Потрібна глибша діагностика для виявлення кореневої причини. Можливі причини:

1. Неправильне визначення основного спікера в voice tracks
2. Помилки в логіці об'єднання сегментів
3. Проблеми в primary diarization
4. Низька якість speaker separation

**План дій:**

1. ✅ Додати розширене логування у ключових функціях
2. ✅ Створити конкретний тестовий кейс і порівняти результат на кожному етапі
3. ✅ Порівняти primary diarization та voice tracks – це допоможе визначити, чи помилка в первинному розпізнаванні
4. ✅ Перевірити якість separation – залишок іншого голосу в track'і може бути причиною
5. ✅ Верифікувати логіку об'єднання – переконатися, що немає конфліктів між двома реалізаціями
6. ✅ Візуалізувати – це допоможе швидко локалізувати помилку

Після виконання цих кроків ми зможемо визначити, де саме виникає змішування реплік і виправити проблему.

