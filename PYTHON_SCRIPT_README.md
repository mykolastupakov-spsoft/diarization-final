# Python Temporary Audio Processing Script

This Python script handles downloading audio files to temporary storage, processing them for diarization, and automatically cleaning up after processing.

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Make the script executable:
```bash
chmod +x process_audio_temp.py
```

3. Install Node.js dependencies (if not already installed):
```bash
npm install
```

## Usage

### Command Line

#### Download from URL:
```bash
python3 process_audio_temp.py --url "https://example.com/audio.wav"
```

#### Process local file:
```bash
python3 process_audio_temp.py --file "/path/to/audio.wav"
```

#### Options:
- `--url <URL>`: Download audio file from URL
- `--file <PATH>`: Process local audio file
- `--temp-dir <DIR>`: Custom temporary directory (default: system temp)
- `--no-cleanup`: Keep temporary files after processing
- `--output-format <json|text>`: Output format (default: json)
- `--info-only`: Only return file info, do not process

### API Endpoint

The script is integrated with the Node.js server via the `/api/process-audio-temp` endpoint.

#### Upload file:
```bash
curl -X POST http://localhost:3000/api/process-audio-temp \
  -F "audio=@/path/to/audio.wav"
```

#### Download from URL:
```bash
curl -X POST http://localhost:3000/api/process-audio-temp \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/audio.wav"}'
```

#### Keep files (no cleanup):
```bash
curl -X POST http://localhost:3000/api/process-audio-temp \
  -F "audio=@/path/to/audio.wav" \
  -F "noCleanup=true"
```

## Response Format

### Success Response:
```json
{
  "success": true,
  "file_info": {
    "path": "/tmp/diarization_xxxxx/audio.wav",
    "filename": "audio.wav",
    "size": 1234567,
    "created": 1234567890.123,
    "modified": 1234567890.123,
    "work_dir": "/tmp/diarization_xxxxx"
  },
  "processing": {
    "status": "success",
    "file_path": "/tmp/diarization_xxxxx/audio.wav",
    "file_size": 1234567,
    "message": "File ready for diarization processing",
    "output_format": "json"
  }
}
```

### Error Response:
```json
{
  "success": false,
  "error": "Error message here"
}
```

## Integration with Diarization

1. **Upload/Download file** using the API endpoint
2. **Get file path** from the response
3. **Process file** for diarization using the file path
4. **Automatic cleanup** happens after processing (unless `--no-cleanup` is used)

## Workflow Example

```javascript
// 1. Upload file to temporary storage
const uploadResponse = await fetch('/api/process-audio-temp', {
  method: 'POST',
  body: formData // Contains audio file
});

const { file_info } = await uploadResponse.json();

// 2. Use file path for diarization
const filePath = file_info.path;

// 3. Process for diarization (your existing diarization logic)
// ... diarization processing ...

// 4. Cleanup happens automatically after Python script completes
```

## Features

- ✅ Automatic temporary file management
- ✅ Support for URL downloads and file uploads
- ✅ Automatic cleanup after processing
- ✅ Configurable temporary directory
- ✅ Error handling and logging
- ✅ JSON output format
- ✅ Integration with Node.js server

## Temporary Storage

- Default location: System temporary directory
- Custom location: Use `--temp-dir` option
- Cleanup: Automatic (unless `--no-cleanup` is specified)
- File size limit: 100MB (configurable in server.js)

## Error Handling

The script handles:
- Network errors during download
- File system errors
- Missing files
- Invalid URLs
- Permission errors

All errors are logged and returned in JSON format.

