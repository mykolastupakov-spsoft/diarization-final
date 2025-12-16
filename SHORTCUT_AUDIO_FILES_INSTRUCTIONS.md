<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Пайплайн</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        h1 {
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #66BB6A;
            margin-top: 20px;
        }
        #fileStatus, #pollingStatus {
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border: 1px solid #444;
            color: #e0e0e0;
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            background-color: #2d2d2d;
            color: #e0e0e0;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="file"]::file-selector-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }
        input[type="file"]::file-selector-button:hover {
            background-color: #45a049;
        }
        #dialogueResult, #llmProgress {
            margin-top: 20px;
            padding: 15px;
            background-color: #2d2d2d;
            border: 1px solid #444;
            border-radius: 8px;
            color: #e0e0e0;
        }
        @media (prefers-color-scheme: light) {
            body { background-color: #ffffff; color: #000000; }
            h1 { color: #2E7D32; border-bottom: 2px solid #2E7D32; }
            h2 { color: #388E3C; }
            #fileStatus, #pollingStatus { background-color: #f5f5f5; border: 1px solid #ddd; color: #000000; }
            input[type="file"] { background-color: #ffffff; color: #000000; }
            #dialogueResult, #llmProgress { background-color: #f5f5f5; border: 1px solid #ddd; color: #000000; }
        }
    </style>
</head>
<body>
    <h1>Пайплайн обробки</h1>
    
    <div>
        <h2>Крок 1: Додати файл</h2>
        <input type="file" id="fileInput" accept="audio/*" />
        <div id="fileStatus"></div>
    </div>

    <div id="pollingStatus" style="margin-top: 20px;"></div>

    <script>
        let selectedFile = null;
        let fileBase64 = null;
        const SERVER_URL = 'http://100.67.135.103:5005';

        document.getElementById('fileInput').addEventListener('change', function(event) {
            const file = event.target.files[0];
            
            if (file) {
                selectedFile = file;
                document.getElementById('fileStatus').innerHTML = 
                    'Файл вибрано:<br>' +
                    'Назва: ' + file.name + '<br>' +
                    'Розмір: ' + (file.size / 1024).toFixed(2) + ' KB<br>' +
                    'Тип: ' + file.type + '<br>' +
                    '<span style="color: orange;">Кодування в base64...</span>';
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    fileBase64 = e.target.result;
                    document.getElementById('fileStatus').innerHTML = 
                        'Файл вибрано:<br>' +
                        'Назва: ' + file.name + '<br>' +
                        'Розмір: ' + (file.size / 1024).toFixed(2) + ' KB<br>' +
                        'Тип: ' + file.type + '<br>' +
                        '<span style="color: green;">✓ Закодовано в base64</span><br>' +
                        '<span style="color: orange;">Відправка на сервер...</span>';
                    console.log('Файл закодовано в base64');
                    
                    sendToServer(fileBase64, file.name);
                };
                reader.readAsDataURL(file);
            } else {
                selectedFile = null;
                fileBase64 = null;
                document.getElementById('fileStatus').innerHTML = 'Файл не вибрано';
            }
            
            event.target.value = '';
        });

        async function sendToServer(base64Data, filename) {
            console.log('Спроба відправки на:', SERVER_URL + '/api/diarize');
            console.log('Назва файлу:', filename);
            
            try {
                const response = await fetch(SERVER_URL + '/api/diarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: base64Data, filename: filename })
                });

                console.log('Response status:', response.status);

                if (response.ok) {
                    const result = await response.json();
                    console.log('Відповідь від сервера:', result);
                    
                    let jobId = null;
                    let statusMessage = '';
                    
                    if (result.success === false) {
                        const errorMsg = result.error || 'Unknown error';
                        statusMessage = '<span style="color: red;">✗ Error: ' + errorMsg + '</span>';
                        console.error('Error from server:', errorMsg);
                    } else if (result.success === true && result.job_id) {
                        jobId = result.job_id;
                        statusMessage = '<span style="color: green;">✓ Job ID: ' + jobId + '</span>';
                        console.log('Job ID extracted:', jobId);
                        
                        window.currentJobId = jobId;
                        startPolling(jobId);
                    } else {
                        statusMessage = '<span style="color: orange;">⚠ Unexpected response format</span>';
                        console.warn('Unexpected response:', result);
                    }
                    
                    document.getElementById('fileStatus').innerHTML = 
                        'Файл вибрано:<br>' +
                        'Назва: ' + selectedFile.name + '<br>' +
                        'Розмір: ' + (selectedFile.size / 1024).toFixed(2) + ' KB<br>' +
                        'Тип: ' + selectedFile.type + '<br>' +
                        '<span style="color: green;">✓ Закодовано в base64</span><br>' +
                        '<span style="color: green;">✓ Відправлено на сервер</span><br>' +
                        statusMessage;
                } else {
                    throw new Error('Помилка сервера: ' + response.status);
                }
            } catch (error) {
                console.error('Повна помилка:', error);
                document.getElementById('fileStatus').innerHTML = 
                    'Файл вибрано:<br>' +
                    'Назва: ' + selectedFile.name + '<br>' +
                    'Розмір: ' + (selectedFile.size / 1024).toFixed(2) + ' KB<br>' +
                    'Тип: ' + selectedFile.type + '<br>' +
                    '<span style="color: green;">✓ Закодовано в base64</span><br>' +
                    '<span style="color: red;">✗ Помилка: ' + error.message + '</span>';
            }
        }

        async function startPolling(jobId) {
            const maxAttempts = 120;
            let attempts = 0;
            const statusUrl = SERVER_URL + '/api/diarize/' + jobId + '/status';
            
            document.getElementById('pollingStatus').innerHTML = 
                '<h2>Крок 2: Перевірка статусу</h2>' +
                '<div>Job ID: ' + jobId + '</div>' +
                '<div>Спроба: 0/' + maxAttempts + '</div>' +
                '<div>Статус: <span style="color: orange;">Очікування...</span></div>';
            
            const pollInterval = setInterval(async function() {
                attempts++;
                
                try {
                    const response = await fetch(statusUrl);
                    
                    if (response.ok) {
                        const statusData = await response.json();
                        const currentStatus = statusData.status || 'unknown';
                        const statusColor = currentStatus === 'completed' ? 'green' : 'orange';
                        
                        document.getElementById('pollingStatus').innerHTML = 
                            '<h2>Крок 2: Перевірка статусу</h2>' +
                            '<div>Job ID: ' + jobId + '</div>' +
                            '<div>Спроба: ' + attempts + '/' + maxAttempts + '</div>' +
                            '<div>Статус: <span style="color: ' + statusColor + ';">' + currentStatus + '</span></div>';
                        
                        if (currentStatus === 'completed') {
                            clearInterval(pollInterval);
                            document.getElementById('pollingStatus').innerHTML += 
                                '<div style="color: green; margin-top: 10px;">✓ Обробка завершена!</div>';
                            await getFormattedDialogue(jobId);
                        }
                    }
                    
                    if (attempts >= maxAttempts) {
                        clearInterval(pollInterval);
                        document.getElementById('pollingStatus').innerHTML += 
                            '<div style="color: red; margin-top: 10px;">✗ Максимум спроб</div>';
                    }
                } catch (error) {
                    console.error('Polling error:', error);
                }
            }, 5000);
        }

        async function getFormattedDialogue(jobId) {
            const formattedUrl = SERVER_URL + '/api/diarize/' + jobId + '/formatted';
            
            try {
                const response = await fetch(formattedUrl);
                
                if (response.ok) {
                    const formattedResponse = await response.json();
                    const formattedDialogue = formattedResponse.formatted_dialogue || formattedResponse;
                    window.formattedDialogue = formattedDialogue;
                    
                    document.getElementById('pollingStatus').innerHTML += 
                        '<div style="color: green; margin-top: 10px;">✓ Форматований діалог отримано!</div>' +
                        '<div style="color: orange; margin-top: 5px;">Відправка аудіо для розділення...</div>';
                    
                    await sendAudioForSeparation(jobId);
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        async function sendAudioForSeparation(jobId) {
            try {
                console.log('Відправка аудіо на розділення...');
                
                // Створюємо FormData для відправки файлу
                const formData = new FormData();
                formData.append('audio', selectedFile);
                
                const response = await fetch(SERVER_URL + '/api/separate-audio', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('Результат розділення аудіо:', result);
                    
                    document.getElementById('pollingStatus').innerHTML += 
                        '<div style="color: green; margin-top: 10px;">✓ Аудіо відправлено на розділення!</div>' +
                        '<div style="color: orange; margin-top: 5px;">Обробка через LLM...</div>';
                    
                    // Тепер обробляємо через LLM
                    await processDialogueWithLLM(window.formattedDialogue);
                } else {
                    throw new Error('Помилка розділення аудіо: ' + response.status);
                }
            } catch (error) {
                console.error('Помилка відправки аудіо:', error);
                document.getElementById('pollingStatus').innerHTML += 
                    '<div style="color: red; margin-top: 10px;">✗ Помилка відправки аудіо: ' + error.message + '</div>';
            }
        }

        async function processDialogueWithLLM(formattedDialogue) {
            try {
                const lines = formattedDialogue.split('\n').filter(function(line) { return line.trim() !== ''; });
                const totalLines = lines.length;
                const dialogueWithRoles = [];
                
                const progressDiv = document.createElement('div');
                progressDiv.id = 'llmProgress';
                progressDiv.style.cssText = 'margin-top: 20px; padding: 15px; background: #2d2d2d; border: 1px solid #444; border-radius: 8px; color: #e0e0e0;';
                document.body.appendChild(progressDiv);
                
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    const progress = i + 1;
                    const progressPercent = (progress / totalLines * 100);
                    
                    progressDiv.innerHTML = 
                        '<h2>Обробка діалогу через LLM</h2>' +
                        '<div>Прогрес: ' + progress + ' / ' + totalLines + '</div>' +
                        '<div style="background: #444; height: 20px; border-radius: 10px; overflow: hidden; margin-top: 10px;">' +
                            '<div style="background: #4CAF50; height: 100%; width: ' + progressPercent + '%; transition: width 0.3s;"></div>' +
                        '</div>';
                    
                    const prompt = "You are an expert in analyzing call center dialogues.\n\nCONTEXT:\nYou are analyzing a dialogue from a call center. The dialogue below is provided as REFERENCE ONLY.\n\nFULL DIALOGUE (for context only):\n" + formattedDialogue + "\n\nSPECIFIC REPLICA TO ANALYZE:\n" + line + "\n\nTASK:\n1. Parse the replica line in format: \"MM:SS Speaker X: [text]\"\n2. Extract the timestamp (MM:SS format)\n3. Extract the speaker label (Speaker 0 or Speaker 1)\n4. Extract the text content\n5. Analyze the text content to determine the role:\n   - Agent: call center employee\n   - Client: customer\n6. Replace 'Speaker 0' or 'Speaker 1' with 'Agent' or 'Client'\n7. Return ONLY the modified line in format: MM:SS [role]: [text]\n\nIMPORTANT:\n- Use 'Agent' for call center employees\n- Use 'Client' for customers\n- Return ONLY the modified line, nothing else";
                    
                    const llmResponse = await fetch('http://localhost:3001/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            messages: [{ role: 'user', content: prompt }],
                            temperature: 0.1,
                            max_tokens: 200
                        })
                    });
                    
                    if (llmResponse.ok) {
                        const llmData = await llmResponse.json();
                        const processedLine = llmData.choices[0].message.content.trim();
                        dialogueWithRoles.push(processedLine);
                        console.log('Processed line ' + progress + ':', processedLine);
                    } else {
                        console.error('Failed to process line ' + progress);
                        dialogueWithRoles.push(line);
                    }
                }
                
                const initialDialogue = dialogueWithRoles.join('\n');
                window.initialDialogue = initialDialogue;
                
                progressDiv.remove();
                
                document.getElementById('pollingStatus').innerHTML += 
                    '<div style="color: green; margin-top: 10px;">✓ Обробка через LLM завершена!</div>';
                <!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Пайплайн</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        h1 {
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #66BB6A;
            margin-top: 20px;
        }
        #fileStatus, #pollingStatus {
            background-color: #2d2d2d;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border: 1px solid #444;
            color: #e0e0e0;
        }
        input[type="file"] {
            margin: 10px 0;
            padding: 10px;
            background-color: #2d2d2d;
            color: #e0e0e0;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            cursor: pointer;
        }
        input[type="file"]::file-selector-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 10px;
        }
        input[type="file"]::file-selector-button:hover {
            background-color: #45a049;
        }
        #dialogueResult, #llmProgress {
            margin-top: 20px;
            padding: 15px;
            background-color: #2d2d2d;
            border: 1px solid #444;
            border-radius: 8px;
            color: #e0e0e0;
        }
        @media (prefers-color-scheme: light) {
            body { background-color: #ffffff; color: #000000; }
            h1 { color: #2E7D32; border-bottom: 2px solid #2E7D32; }
            h2 { color: #388E3C; }
            #fileStatus, #pollingStatus { background-color: #f5f5f5; border: 1px solid #ddd; color: #000000; }
            input[type="file"] { background-color: #ffffff; color: #000000; }
            #dialogueResult, #llmProgress { background-color: #f5f5f5; border: 1px solid #ddd; color: #000000; }
        }
    </style>
</head>
<body>
    <h1>Пайплайн обробки</h1>
    
    <div>
        <h2>Крок 1: Додати файл</h2>
        <input type="file" id="fileInput" accept="audio/*" />
        <div id="fileStatus"></div>
    </div>

    <div id="pollingStatus" style="margin-top: 20px;"></div>

    <script>
        let selectedFile = null;
        let fileBase64 = null;
        const SERVER_URL = 'http://100.67.135.103:5005';

        document.getElementById('fileInput').addEventListener('change', function(event) {
            const file = event.target.files[0];
            
            if (file) {
                selectedFile = file;
                document.getElementById('fileStatus').innerHTML = 
                    'Файл вибрано:<br>' +
                    'Назва: ' + file.name + '<br>' +
                    'Розмір: ' + (file.size / 1024).toFixed(2) + ' KB<br>' +
                    'Тип: ' + file.type + '<br>' +
                    '<span style="color: orange;">Кодування в base64...</span>';
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    fileBase64 = e.target.result;
                    document.getElementById('fileStatus').innerHTML = 
                        'Файл вибрано:<br>' +
                        'Назва: ' + file.name + '<br>' +
                        'Розмір: ' + (file.size / 1024).toFixed(2) + ' KB<br>' +
                        'Тип: ' + file.type + '<br>' +
                        '<span style="color: green;">✓ Закодовано в base64</span><br>' +
                        '<span style="color: orange;">Відправка на сервер...</span>';
                    console.log('Файл закодовано в base64');
                    
                    sendToServer(fileBase64, file.name);
                };
                reader.readAsDataURL(file);
            } else {
                selectedFile = null;
                fileBase64 = null;
                document.getElementById('fileStatus').innerHTML = 'Файл не вибрано';
            }
            
            event.target.value = '';
        });

        async function sendToServer(base64Data, filename) {
            console.log('Спроба відправки на:', SERVER_URL + '/api/diarize');
            console.log('Назва файлу:', filename);
            
            try {
                const response = await fetch(SERVER_URL + '/api/diarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file: base64Data, filename: filename })
                });

                console.log('Response status:', response.status);

                if (response.ok) {
                    const result = await response.json();
                    console.log('Відповідь від сервера:', result);
                    
                    let jobId = null;
                    let statusMessage = '';
                    
                    if (result.success === false) {
                        const errorMsg = result.error || 'Unknown error';
                        statusMessage = '<span style="color: red;">✗ Error: ' + errorMsg + '</span>';
                        console.error('Error from server:', errorMsg);
                    } else if (result.success === true && result.job_id) {
                        jobId = result.job_id;
                        statusMessage = '<span style="color: green;">✓ Job ID: ' + jobId + '</span>';
                        console.log('Job ID extracted:', jobId);
                        
                        window.currentJobId = jobId;
                        startPolling(jobId);
                    } else {
                        statusMessage = '<span style="color: orange;">⚠ Unexpected response format</span>';
                        console.warn('Unexpected response:', result);
                    }
                    
                    document.getElementById('fileStatus').innerHTML = 
                        'Файл вибрано:<br>' +
                        'Назва: ' + selectedFile.name + '<br>' +
                        'Розмір: ' + (selectedFile.size / 1024).toFixed(2) + ' KB<br>' +
                        'Тип: ' + selectedFile.type + '<br>' +
                        '<span style="color: green;">✓ Закодовано в base64</span><br>' +
                        '<span style="color: green;">✓ Відправлено на сервер</span><br>' +
                        statusMessage;
                } else {
                    throw new Error('Помилка сервера: ' + response.status);
                }
            } catch (error) {
                console.error('Повна помилка:', error);
                document.getElementById('fileStatus').innerHTML = 
                    'Файл вибрано:<br>' +
                    'Назва: ' + selectedFile.name + '<br>' +
                    'Розмір: ' + (selectedFile.size / 1024).toFixed(2) + ' KB<br>' +
                    'Тип: ' + selectedFile.type + '<br>' +
                    '<span style="color: green;">✓ Закодовано в base64</span><br>' +
                    '<span style="color: red;">✗ Помилка: ' + error.message + '</span>';
            }
        }

        async function startPolling(jobId) {
            const maxAttempts = 120;
            let attempts = 0;
            const statusUrl = SERVER_URL + '/api/diarize/' + jobId + '/status';
            
            document.getElementById('pollingStatus').innerHTML = 
                '<h2>Крок 2: Перевірка статусу</h2>' +
                '<div>Job ID: ' + jobId + '</div>' +
                '<div>Спроба: 0/' + maxAttempts + '</div>' +
                '<div>Статус: <span style="color: orange;">Очікування...</span></div>';
            
            const pollInterval = setInterval(async function() {
                attempts++;
                
                try {
                    const response = await fetch(statusUrl);
                    
                    if (response.ok) {
                        const statusData = await response.json();
                        const currentStatus = statusData.status || 'unknown';
                        const statusColor = currentStatus === 'completed' ? 'green' : 'orange';
                        
                        document.getElementById('pollingStatus').innerHTML = 
                            '<h2>Крок 2: Перевірка статусу</h2>' +
                            '<div>Job ID: ' + jobId + '</div>' +
                            '<div>Спроба: ' + attempts + '/' + maxAttempts + '</div>' +
                            '<div>Статус: <span style="color: ' + statusColor + ';">' + currentStatus + '</span></div>';
                        
                        if (currentStatus === 'completed') {
                            clearInterval(pollInterval);
                            document.getElementById('pollingStatus').innerHTML += 
                                '<div style="color: green; margin-top: 10px;">✓ Обробка завершена!</div>';
                            await getFormattedDialogue(jobId);
                        }
                    }
                    
                    if (attempts >= maxAttempts) {
                        clearInterval(pollInterval);
                        document.getElementById('pollingStatus').innerHTML += 
                            '<div style="color: red; margin-top: 10px;">✗ Максимум спроб</div>';
                    }
                } catch (error) {
                    console.error('Polling error:', error);
                }
            }, 5000);
        }

        async function getFormattedDialogue(jobId) {
            const formattedUrl = SERVER_URL + '/api/diarize/' + jobId + '/formatted';
            
            try {
                const response = await fetch(formattedUrl);
                
                if (response.ok) {
                    const formattedResponse = await response.json();
                    const formattedDialogue = formattedResponse.formatted_dialogue || formattedResponse;
                    window.formattedDialogue = formattedDialogue;
                    
                    document.getElementById('pollingStatus').innerHTML += 
                        '<div style="color: green; margin-top: 10px;">✓ Форматований діалог отримано!</div>' +
                        '<div style="color: orange; margin-top: 5px;">Відправка аудіо для розділення...</div>';
                    
                    await sendAudioForSeparation(jobId);
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        async function sendAudioForSeparation(jobId) {
            try {
                console.log('Відправка аудіо на розділення...');
                
                // Створюємо FormData для відправки файлу
                const formData = new FormData();
                formData.append('audio', selectedFile);
                
                const response = await fetch(SERVER_URL + '/api/separate-audio', {
                    method: 'POST',
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('Результат розділення аудіо:', result);
                    
                    document.getElementById('pollingStatus').innerHTML += 
                        '<div style="color: green; margin-top: 10px;">✓ Аудіо відправлено на розділення!</div>' +
                        '<div style="color: orange; margin-top: 5px;">Обробка через LLM...</div>';
                    
                    // Тепер обробляємо через LLM
                    await processDialogueWithLLM(window.formattedDialogue);
                } else {
                    throw new Error('Помилка розділення аудіо: ' + response.status);
                }
            } catch (error) {
                console.error('Помилка відправки аудіо:', error);
                document.getElementById('pollingStatus').innerHTML += 
                    '<div style="color: red; margin-top: 10px;">✗ Помилка відправки аудіо: ' + error.message + '</div>';
            }
        }

        async function processDialogueWithLLM(formattedDialogue) {
            try {
                const lines = formattedDialogue.split('\n').filter(function(line) { return line.trim() !== ''; });
                const totalLines = lines.length;
                const dialogueWithRoles = [];
                
                const progressDiv = document.createElement('div');
                progressDiv.id = 'llmProgress';
                progressDiv.style.cssText = 'margin-top: 20px; padding: 15px; background: #2d2d2d; border: 1px solid #444; border-radius: 8px; color: #e0e0e0;';
                document.body.appendChild(progressDiv);
                
                for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    const progress = i + 1;
                    const progressPercent = (progress / totalLines * 100);
                    
                    progressDiv.innerHTML = 
                        '<h2>Обробка діалогу через LLM</h2>' +
                        '<div>Прогрес: ' + progress + ' / ' + totalLines + '</div>' +
                        '<div style="background: #444; height: 20px; border-radius: 10px; overflow: hidden; margin-top: 10px;">' +
                            '<div style="background: #4CAF50; height: 100%; width: ' + progressPercent + '%; transition: width 0.3s;"></div>' +
                        '</div>';
                    
                    const prompt = "You are an expert in analyzing call center dialogues.\n\nCONTEXT:\nYou are analyzing a dialogue from a call center. The dialogue below is provided as REFERENCE ONLY.\n\nFULL DIALOGUE (for context only):\n" + formattedDialogue + "\n\nSPECIFIC REPLICA TO ANALYZE:\n" + line + "\n\nTASK:\n1. Parse the replica line in format: \"MM:SS Speaker X: [text]\"\n2. Extract the timestamp (MM:SS format)\n3. Extract the speaker label (Speaker 0 or Speaker 1)\n4. Extract the text content\n5. Analyze the text content to determine the role:\n   - Agent: call center employee\n   - Client: customer\n6. Replace 'Speaker 0' or 'Speaker 1' with 'Agent' or 'Client'\n7. Return ONLY the modified line in format: MM:SS [role]: [text]\n\nIMPORTANT:\n- Use 'Agent' for call center employees\n- Use 'Client' for customers\n- Return ONLY the modified line, nothing else";
                    
                    const llmResponse = await fetch('http://localhost:3001/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ 
                            messages: [{ role: 'user', content: prompt }],
                            temperature: 0.1,
                            max_tokens: 200
                        })
                    });
                    
                    if (llmResponse.ok) {
                        const llmData = await llmResponse.json();
                        const processedLine = llmData.choices[0].message.content.trim();
                        dialogueWithRoles.push(processedLine);
                        console.log('Processed line ' + progress + ':', processedLine);
                    } else {
                        console.error('Failed to process line ' + progress);
                        dialogueWithRoles.push(line);
                    }
                }
                
                const initialDialogue = dialogueWithRoles.join('\n');
                window.initialDialogue = initialDialogue;
                
                progressDiv.remove();
                
                document.getElementById('pollingStatus').innerHTML += 
                    '<div style="color: green; margin-top: 10px;">✓ Обробка через LLM завершена!</div>';
                
                const dialogueDiv = document.createElement('div');
                dialogueDiv.id = 'dialogueResult';
                dialogueDiv.innerHTML = 
                    '<h2>Фінальний діалог (Initial Dialogue)</h2>' +
                    '<div style="background: #1a1a1a; padding: 15px; border-radius: 5px; white-space: pre-wrap; max-height: 400px; overflow-y: auto; font-family: monospace; color: #e0e0e0; border: 1px solid #444;">' +
                        initialDialogue +
                    '</div>';
                document.body.appendChild(dialogueDiv);
                
            } catch (error) {
                console.error('Error processing dialogue with LLM:', error);
                document.getElementById('pollingStatus').innerHTML += 
                    '<div style="color: red; margin-top: 10px;">✗ Помилка обробки LLM: ' + error.message + '</div>';
            }
        }
    </script>
</body>
</html>
                const dialogueDiv = document.createElement('div');
                dialogueDiv.id = 'dialogueResult';
                dialogueDiv.innerHTML = 
                    '<h2>Фінальний діалог (Initial Dialogue)</h2>' +
                    '<div style="background: #1a1a1a; padding: 15px; border-radius: 5px; white-space: pre-wrap; max-height: 400px; overflow-y: auto; font-family: monospace; color: #e0e0e0; border: 1px solid #444;">' +
                        initialDialogue +
                    '</div>';
                document.body.appendChild(dialogueDiv);
                
            } catch (error) {
                console.error('Error processing dialogue with LLM:', error);
                document.getElementById('pollingStatus').innerHTML += 
                    '<div style="color: red; margin-top: 10px;">✗ Помилка обробки LLM: ' + error.message + '</div>';
            }
        }
    </script>
</body>
</html>