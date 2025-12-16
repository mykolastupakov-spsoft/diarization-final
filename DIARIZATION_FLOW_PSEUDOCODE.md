# Псевдокод повного flow діаризації

## Від завантаження файлу до отримання діаризованого результату

```
FUNCTION diarization_pipeline():
    // ============================================
    // КРОК 0: ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА
    // ============================================
    
    user_selects_audio_file()
    file = upload_file_to_server()  // POST /api/diarize-overlap з multipart/form-data
    requestId = generate_unique_id()
    
    // Параметри з запиту
    language = request.body.language || 'auto'
    speakerCount = request.body.speakerCount || 'auto'
    mode = request.body.mode || 'smart'  // LLM режим: smart/fast/local/test/test2
    pipelineMode = request.body.pipelineMode || 'mode1'  // mode1=AudioShake, mode2=PyAnnote, mode3=SpeechBrain
    engine = request.body.engine || 'speechmatics'  // speechmatics/azure/azure_realtime
    textAnalysisMode = request.body.textAnalysisMode || 'script'  // script/llm
    
    // Перевірка API ключів
    IF pipelineMode == 'mode1':
        CHECK AUDIOSHAKE_API_KEY exists
        CHECK tunnelUrl OR PUBLIC_URL exists (для HTTPS доступу до файлу)
    ELSE IF pipelineMode == 'mode2':
        CHECK HUGGINGFACE_TOKEN exists
    
    // Налаштування SSE (Server-Sent Events) для mode1/mode3
    IF pipelineMode == 'mode1' OR pipelineMode == 'mode3':
        SET response headers для SSE:
            Content-Type: text/event-stream
            Cache-Control: no-cache
            Connection: keep-alive
        START keep-alive ping interval (кожні 30 секунд)
    
    // ============================================
    // КРОК 1: ПОЧАТКОВА ДІАРИЗАЦІЯ АУДІО
    // ============================================
    
    SEND SSE update: "STEP 1: Початок аналізу аудіо..."
    
    // Підготовка файлу
    IF uploadedFile exists:
        sourcePath = uploadedFile.path
    ELSE IF url provided:
        sourcePath = download_audio_to_temp(url)
    
    // Перевірка кешу діаризації
    cacheKey = build_diarization_cache_key(sourcePath, language, speakerCount, engine)
    IF cache exists AND not expired:
        primaryDiarization = read_from_cache(cacheKey)
        SEND SSE update: "STEP 1: Використано кешований результат"
    ELSE:
        // Виклик Python скрипта для діаризації
        primaryDiarization = run_python_diarization({
            filePath: sourcePath,
            url: url (якщо є),
            language: language,
            speakerCount: speakerCount,
            engine: engine,  // speechmatics/azure/azure_realtime
            onProgress: progress_callback (для Azure)
        })
        
        // Збереження в кеш
        write_to_cache(cacheKey, primaryDiarization)
    
    segmentsCount = primaryDiarization.recordings[0].results[engine].segments.length
    SEND SSE update: "STEP 1: Завершено. Знайдено N сегментів"
    
    // ============================================
    // КРОК 1.5: LLM ДІАРИЗАЦІЯ (ОПЦІОНАЛЬНО)
    // ============================================
    
    IF GOOGLE_GEMINI_API_KEY OR OPENROUTER_API_KEY exists:
        SEND SSE update: "STEP 1.5: LLM діаризація (Gemini 2.5 Pro)..."
        
        // Витягуємо простий транскрипт з Speechmatics для контексту
        plainTranscript = extract_plain_transcript(primaryDiarization)
        
        geminiDiarization = call_gemini_multimodal(
            audioFilePath: sourcePath,
            transcript: plainTranscript,
            language: language
        )
        
        SEND SSE update: "STEP 1.5: LLM діаризація завершена"
    ELSE:
        SKIP STEP 1.5
    
    // ============================================
    // КРОК 2: РОЗДІЛЕННЯ СПІКЕРІВ
    // ============================================
    
    SEND SSE update: "STEP 2: Початок розділення спікерів..."
    
    // Підготовка файлу для розділення
    IF pipelineMode == 'mode1':  // AudioShake
        // Потрібен HTTPS URL
        storedFile = persist_file_to_uploads(sourcePath, originalName)
        publicUrl = get_public_file_url(storedFile.filename)
        
        // Перевірка доступності файлу
        ensure_file_accessible(publicUrl)
        
        // Перевірка кешу розділення
        separationCacheKey = build_separation_cache_key(storedFile.filename, 'mode1')
        IF cache exists:
            separation = read_separation_cache(separationCacheKey)
            regenerate_urls_for_speakers(separation.speakers)
        ELSE:
            separation = separate_speakers_with_audioshake(
                audioUrl: publicUrl,
                language: language
            )
            write_separation_cache(separationCacheKey, separation)
    
    ELSE IF pipelineMode == 'mode2':  // PyAnnote
        // Використовуємо локальний файл
        separationCacheKey = build_separation_cache_key(originalName, 'mode2')
        IF cache exists:
            separation = read_separation_cache(separationCacheKey)
        ELSE:
            separation = separate_speakers_with_pyannote(
                filePath: sourcePath,
                language: language
            )
            write_separation_cache(separationCacheKey, separation)
    
    ELSE IF pipelineMode == 'mode3':  // SpeechBrain
        // Використовуємо локальний файл
        separationCacheKey = build_separation_cache_key(originalName, 'mode3')
        IF cache exists:
            separation = read_separation_cache(separationCacheKey)
        ELSE:
            SEND SSE update: "STEP 2: Завантаження моделі SpeechBrain..."
            separation = separate_speakers_with_speechbrain(
                filePath: sourcePath,
                requestId: requestId,
                progressCallback: sendSSEUpdate
            )
            write_separation_cache(separationCacheKey, separation)
    
    speakersCount = separation.speakers.length
    SEND SSE update: "STEP 2: Розділення завершено. Знайдено N спікерів"
    
    // ============================================
    // КРОК 3: ТРАНСКРИПЦІЯ РОЗДІЛЕНИХ ДОРІЖОК
    // ============================================
    
    SEND SSE update: "STEP 3: Початок транскрипції розділених доріжок..."
    
    voiceTracks = []
    FOR EACH speaker IN separation.speakers:
        IF speaker.isBackground:
            CONTINUE  // Пропускаємо фонові доріжки
        
        SEND SSE update: "STEP 3: Транскрипція спікера {speaker.name}..."
        
        // Визначення шляху до аудіо файлу
        audioPath = speaker.local_path OR download_speaker_audio(speaker.url)
        
        // Транскрипція через Python скрипт
        transcription = run_python_diarization({
            filePath: audioPath,
            language: language,
            speakerCount: 1,  // Одна доріжка = один спікер
            engine: engine,
            diarizationMode: 'channel'  // КРИТИЧНО: channel mode для розділених доріжок
        })
        
        // Аналіз ролі спікера (Agent/Client)
        roleAnalysis = analyze_speaker_role(
            transcript: transcription,
            trackId: speaker.name,
            mode: mode
        )
        
        voiceTrack = {
            speaker: speaker.name,
            role: roleAnalysis.role,  // Agent/Client
            confidence: roleAnalysis.confidence,
            transcription: transcription,
            segments: transcription.recordings[0].results[engine].segments,
            audioPath: audioPath,
            downloadUrl: speaker.url
        }
        
        voiceTracks.append(voiceTrack)
    
    SEND SSE update: "STEP 3: Транскрипція завершена"
    
    // ============================================
    // КРОК 4: КОРЕКЦІЯ ТА ГЕНЕРАЦІЯ MARKDOWN
    // ============================================
    
    SEND SSE update: "STEP 4: Початок корекції та генерації таблиці..."
    
    // Побудова контексту для LLM
    promptContext = build_dialogue_prompt_context({
        primaryDiarization: primaryDiarization,
        geminiDiarization: geminiDiarization,  // Якщо є
        voiceTracks: voiceTracks,
        groundTruthText: load_ground_truth_if_exists(uploadedFile.originalname)
    })
    
    // Визначення ролей для спікерів
    assignedRoles = assign_speaker_roles(voiceTracks, promptContext)
    
    // Корекція primary діаризації з використанням voice tracks
    IF pipelineMode == 'mode3':
        correctionResult = correct_primary_diarization_with_tracks(
            primaryDiarization: primaryDiarization,
            voiceTracks: voiceTracks,
            language: language,
            mode: mode,
            requestId: requestId,
            promptContext: promptContext
        )
    ELSE:
        correctionResult = generate_overlap_correction_result(
            primaryDiarization: primaryDiarization,
            voiceTracks: voiceTracks,
            language: language,
            mode: mode
        )
    
    // Генерація markdown таблиці
    SEND SSE update: "STEP 5: Генерація Markdown таблиці..."
    
    // Перевірка кешу markdown
    markdownCacheKey = build_llm_cache_key(
        filename: uploadedFile.originalname,
        prompt: promptContext,
        model: get_model_id(mode),
        mode: mode,
        type: 'markdown-fixes'
    )
    
    IF markdownCacheKey exists in cache:
        markdownTable = read_llm_cache(markdownCacheKey).rawMarkdown
        markdownSource = 'cache'
    ELSE:
        // Генерація через LLM
        IF use_multi_step_processing:
            // Багатокрокова обробка (Step 1-6)
            markdownTable = multi_step_llm_processing(
                promptContext: promptContext,
                mode: mode,
                requestId: requestId
            )
            // Step 1: Validate Replicas
            // Step 2: Assign Roles
            // Step 3: Remove Duplicates
            // Step 4: Format Table
            // Step 5: Verify Result
            // Step 6: Ground Truth Analysis (якщо є)
        ELSE:
            // Одноразовий виклик LLM
            markdownTable = call_llm_for_markdown(
                prompt: build_markdown_prompt(promptContext),
                model: get_model_id(mode),
                mode: mode,
                useLocalLLM: (mode == 'local' OR mode == 'test' OR mode == 'test2')
            )
        
        // Пост-обробка markdown
        markdownTable = remove_filler_words(markdownTable)
        markdownTable = merge_consecutive_speaker_segments(markdownTable, maxGapSeconds=2.0)
        
        // Збереження в кеш
        write_llm_cache(markdownCacheKey, {
            rawMarkdown: markdownTable,
            timestamp: now()
        })
        markdownSource = 'llm-generated'
    
    // Перевірка розподілу ролей
    roleDistribution = analyze_role_distribution(markdownTable)
    IF roleDistribution.agentCount == 0 OR roleDistribution.clientCount == 0:
        WARN "Всі сегменти мають одну роль!"
    
    SEND SSE update: "STEP 5: Markdown таблицю згенеровано"
    
    // ============================================
    // КРОК 5: ТЕКСТОВИЙ АНАЛІЗ (BLUE/GREEN/RED)
    // ============================================
    
    SEND SSE update: "STEP 6: Текстовий аналіз..."
    
    // Побудова webhook payload для аналізу
    webhookPayload = {
        general: {
            speechmatics: primaryDiarization.recordings[0].results[engine],
            segments: primaryDiarization.recordings[0].results[engine].segments
        },
        speaker1: {
            speaker: voiceTracks[0].speaker,
            role: assignedRoles[0],
            speechmatics: voiceTracks[0].transcription.recordings[0].results[engine],
            segments: voiceTracks[0].segments
        },
        speaker2: {
            speaker: voiceTracks[1].speaker,
            role: assignedRoles[1],
            speechmatics: voiceTracks[1].transcription.recordings[0].results[engine],
            segments: voiceTracks[1].segments
        },
        markdown: markdownTable
    }
    
    // Аналіз тексту (script або LLM режим)
    IF textAnalysisMode == 'llm' OR TEXT_ANALYSIS_MODE env == 'llm':
        textAnalysisResults = analyze_text_with_llm(
            webhookPayload: webhookPayload,
            model: get_model_id(mode),
            mode: mode
        )
    ELSE:
        textAnalysisResults = analyze_text_with_script(webhookPayload)
    
    // textAnalysisResults містить:
    // - Blue: сегменти, які є тільки в primary (втрачені після розділення)
    // - Green: сегменти, які є в обох (правильно оброблені)
    // - Red: сегменти, які є тільки в voice tracks (нові/виправлені)
    
    SEND SSE update: "STEP 6: Текстовий аналіз завершено"
    
    // ============================================
    // КРОК 6: РОЗРАХУНОК МЕТРИК (ЯКЩО Є GROUND TRUTH)
    // ============================================
    
    groundTruthMetrics = null
    IF promptContext.groundTruthText exists:
        groundTruthMetrics = calculate_ground_truth_match(
            markdownTable: markdownTable,
            groundTruthText: promptContext.groundTruthText,
            primaryDiarization: primaryDiarization
        )
        // Метрики містять:
        // - nextLevel.matchPercent (відсоток збігу з NextLevel)
        // - speechmatics.matchPercent (відсоток збігу з Speechmatics)
        // - comparison.improvement (покращення порівняно з Speechmatics)
    
    // ============================================
    // КРОК 7: ФОРМУВАННЯ ФІНАЛЬНОГО РЕЗУЛЬТАТУ
    // ============================================
    
    finalResult = {
        type: 'final-result',
        success: true,
        requestId: requestId,
        pipelineMode: pipelineMode,
        
        // Діаризаційні дані
        primaryDiarization: primaryDiarization,
        geminiDiarization: geminiDiarization,  // Якщо є
        correctedDiarization: correctionResult,
        
        // Розділення спікерів
        separation: {
            taskId: separation.taskId,
            speakers: separation.speakers,
            cost: separation.cost,
            duration: separation.duration
        },
        
        // Voice tracks з транскрипціями
        voiceTracks: voiceTracks,
        
        // Фінальна markdown таблиця
        markdownTable: markdownTable,
        
        // Текстовий аналіз
        textAnalysis: textAnalysisResults,
        
        // Метрики ground truth (якщо є)
        groundTruthMetrics: groundTruthMetrics,
        
        // Діагностика
        diagnostics: {
            combinedTranscript: build_combined_transcript(voiceTracks),
            llmDiarization: geminiDiarization,
            comparison: comparisonAnalysis,
            missingReplicas: find_missing_replicas(primaryDiarization, voiceTracks)
        },
        
        // Час виконання
        steps: {
            step1: { name: 'Initial Audio Analysis', duration: step1Duration },
            step1_5: { name: 'LLM Diarization', duration: step1_5Duration },
            step2: { name: 'Speaker Separation', duration: step2Duration },
            step3: { name: 'Voice Track Transcription', duration: step3Duration },
            step4: { name: 'Correction', duration: step4Duration },
            step5: { name: 'Markdown Generation', duration: step5Duration },
            step6: { name: 'Text Analysis', duration: step6Duration }
        },
        totalDuration: totalTime
    }
    
    // Санітизація результату (видалення внутрішніх ключів)
    sanitizedResult = sanitize_diarization_response(finalResult)
    
    // ============================================
    // КРОК 8: ВІДПРАВКА РЕЗУЛЬТАТУ КЛІЄНТУ
    // ============================================
    
    IF pipelineMode == 'mode1' OR pipelineMode == 'mode3':  // SSE режим
        SEND SSE event: {
            type: 'final-result',
            data: sanitizedResult
        }
        CLOSE SSE connection
        CLEANUP keep-alive interval
    ELSE:  // JSON режим
        SEND JSON response: sanitizedResult
    
    // Очищення тимчасових файлів
    IF tempDownloadedAudioPath exists:
        DELETE tempDownloadedAudioPath
    
    RETURN success

END FUNCTION
```

## Додаткові функції та деталі

### Функція run_python_diarization()
```
FUNCTION run_python_diarization(params):
    pythonScript = 'process_audio_temp.py'
    
    // Підготовка аргументів для Python скрипта
    args = [
        '--file-path', params.filePath,
        '--language', params.language,
        '--speaker-count', params.speakerCount,
        '--engine', params.engine
    ]
    
    IF params.url:
        args.append('--url', params.url)
    
    IF params.diarizationMode == 'channel':
        args.append('--diarization-mode', 'channel')
    
    // Запуск Python процесу
    pythonProcess = spawn(PYTHON_BIN, [pythonScript, ...args], {
        env: {
            ...process.env,
            SPEECHMATICS_API_KEY: process.env.SPEECHMATICS_API_KEY,
            AZURE_SPEECH_KEY: process.env.AZURE_SPEECH_KEY,
            AZURE_SPEECH_REGION: process.env.AZURE_SPEECH_REGION
        }
    })
    
    // Обробка прогресу (для Azure)
    IF params.onProgress:
        pythonProcess.stdout.on('data', (data) => {
            parse_progress_updates(data, params.onProgress)
        })
    
    // Очікування завершення
    result = await pythonProcess.complete()
    RETURN parse_json_result(result.stdout)
END FUNCTION
```

### Функція build_dialogue_prompt_context()
```
FUNCTION build_dialogue_prompt_context(data):
    // Витягування діалогів з різних джерел
    generalDialog = extract_dialogue_from_segments(
        primaryDiarization.recordings[0].results[engine].segments
    )
    
    agentDialog = extract_dialogue_from_segments(
        voiceTracks.find(track => track.role == 'Agent').segments
    )
    
    clientDialog = extract_dialogue_from_segments(
        voiceTracks.find(track => track.role == 'Client').segments
    )
    
    // Побудова timestamp mappings
    segmentTimestamps = build_timestamp_mappings(
        primaryDiarization,
        voiceTracks
    )
    
    // Role guidance (мапінг спікерів на ролі)
    roleGuidance = {
        speakerRoleMap: {
            'SPEAKER_00': assignedRoles[0],
            'SPEAKER_01': assignedRoles[1]
        },
        tracks: voiceTracks.map(track => ({
            speaker: track.speaker,
            role: track.role,
            confidence: track.confidence
        }))
    }
    
    RETURN {
        generalDialog: generalDialog,
        agentDialog: agentDialog,
        clientDialog: clientDialog,
        speaker0Dialog: extract_speaker_dialogue(primaryDiarization, 'SPEAKER_00'),
        speaker1Dialog: extract_speaker_dialogue(primaryDiarization, 'SPEAKER_01'),
        segmentTimestampsText: format_timestamps(segmentTimestamps),
        roleGuidanceText: JSON.stringify(roleGuidance),
        groundTruthText: data.groundTruthText,
        primaryDiarization: data.primaryDiarization
    }
END FUNCTION
```

### Функція multi_step_llm_processing()
```
FUNCTION multi_step_llm_processing(promptContext, mode, requestId):
    // Step 1: Validate Replicas
    step1Output = call_llm_step(
        step: 1,
        name: 'Validate Replicas',
        template: 'prompts/step1_validate_replicas.txt',
        replacements: {
            '{{AGENT_DIALOGUE}}': promptContext.agentDialog,
            '{{CLIENT_DIALOGUE}}': promptContext.clientDialog,
            '{{SEGMENT_TIMESTAMPS}}': promptContext.segmentTimestampsText
        },
        outputFormat: 'json'
    )
    validatedReplicas = parse_json_array(step1Output)
    
    // Step 2: Assign Roles
    step2Output = call_llm_step(
        step: 2,
        name: 'Assign Roles',
        template: 'prompts/step2_assign_roles.txt',
        replacements: {
            '{{VALIDATED_REPLICAS}}': JSON.stringify(validatedReplicas),
            '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText
        },
        outputFormat: 'json'
    )
    roledReplicas = parse_json_array(step2Output)
    
    // Step 3: Remove Duplicates
    step3Output = call_llm_step(
        step: 3,
        name: 'Remove Wrong Speaker Replicas',
        template: 'prompts/step3_remove_duplicates.txt',
        replacements: {
            '{{ROLED_REPLICAS}}': JSON.stringify(roledReplicas),
            '{{GENERAL_DIALOGUE}}': promptContext.generalDialog,
            '{{STANDARD_SPEAKER0_DIALOGUE}}': promptContext.speaker0Dialog,
            '{{STANDARD_SPEAKER1_DIALOGUE}}': promptContext.speaker1Dialog,
            '{{ROLE_GUIDANCE}}': promptContext.roleGuidanceText
        },
        outputFormat: 'json'
    )
    cleanedReplicas = parse_json_array(step3Output)
    
    // Step 4: Format Table
    step4Output = call_llm_step(
        step: 4,
        name: 'Format Table',
        template: 'prompts/step4_format_table.txt',
        replacements: {
            '{{CLEANED_REPLICAS}}': JSON.stringify(cleanedReplicas)
        },
        outputFormat: 'markdown'
    )
    markdownTable = extract_markdown_table(step4Output)
    
    // Step 5: Verify Result (опціонально)
    IF step5_template_exists:
        step5Output = call_llm_step(
            step: 5,
            name: 'Verify Result',
            template: 'prompts/step5_verify_result.txt',
            replacements: {
                '{{GENERATED_TABLE}}': markdownTable,
                ...promptContext
            },
            outputFormat: 'markdown'
        )
        IF step5Output is not empty:
            markdownTable = step5Output
    
    // Step 6: Ground Truth Analysis (якщо є ground truth)
    IF promptContext.groundTruthText:
        step6Output = call_llm_step(
            step: 6,
            name: 'Ground Truth Analysis',
            template: 'prompts/step6_ground_truth_alignment.txt',
            replacements: {
                '{{GROUND_TRUTH_DIALOGUE}}': promptContext.groundTruthText,
                '{{GENERATED_TABLE}}': markdownTable
            },
            outputFormat: 'json'
        )
        groundTruthAnalysis = parse_json(step6Output)
    
    RETURN markdownTable
END FUNCTION
```

## Структура markdown таблиці

```
| Segment ID | Speaker | Text | Start Time | End Time |
|------------|---------|------|------------|----------|
| 1 | Agent | Hello, how can I help you? | 0.00 | 2.50 |
| 2 | Client | I need help with my account | 2.50 | 5.20 |
| 3 | Agent | Sure, let me check that for you | 5.20 | 8.10 |
...
```

## Структура textAnalysis

```
{
    Blue: [
        { segment: {...}, reason: "Missing in voice tracks" }
    ],
    Green: [
        { segment: {...}, reason: "Found in both sources" }
    ],
    Red: [
        { segment: {...}, reason: "New/improved in voice tracks" }
    ]
}
```

## Кешування

- **Діаризація**: Кешується за file hash + language + speakerCount + engine (TTL: 30 днів)
- **Розділення спікерів**: Кешується за filename + pipelineMode (TTL: 30 днів)
- **LLM відповіді**: Кешується за filename + prompt hash + model + mode (без TTL, ручне очищення)

## Обробка помилок

- Кожен крок обгорнутий в try-catch
- Помилки логуються з requestId
- Для SSE режиму: помилки відправляються через SSE events
- Для JSON режиму: помилки повертаються як JSON з error полем
- Часткові результати зберігаються для діагностики




