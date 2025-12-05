# Azure Native Format Support

## Overview

Azure STT results are now stored in their native format instead of being normalized to Speechmatics format. This allows us to work with Azure's original response structure and parse it at the table level.

## Structure

### Azure Realtime Response Format

```json
{
  "service": "azure_realtime",
  "language": "en-US",
  "segments": [
    {
      "type": "final",
      "speakerId": "Guest-1",
      "text": "Hi, I'm Jessica...",
      "offset": 0.71,
      "duration": 7.6
    }
  ],
  "speakerMap": {
    "Guest-1": "SPEAKER_01",
    "Guest-2": "SPEAKER_02"
  }
}
```

### Storage in Results

Azure results are stored in two places for compatibility:

1. **`results.azure`** - Native Azure format:
   ```json
   {
     "success": true,
     "serviceName": "Azure STT Realtime",
     "engine": "azure_realtime",
     "segments": [...],  // Normalized segments (for backward compatibility)
     "rawData": {       // Original Azure response
       "service": "azure_realtime",
       "language": "en-US",
       "segments": [...],
       "speakerMap": {...}
     }
   }
   ```

2. **`results.speechmatics`** - Legacy format (for backward compatibility):
   ```json
   {
     "success": true,
     "serviceName": "Azure STT Realtime",
     "engine": "azure_realtime",
     "segments": [...],  // Normalized segments
     "rawData": {...}   // Original Azure response
   }
   ```

## Parsing

The `parseAzureSegments()` function in `app.js` converts Azure native format to table-compatible format:

```javascript
// Input (Azure native):
{
  type: "final",
  speakerId: "Guest-1",
  text: "Hello",
  offset: 0.71,
  duration: 7.6
}

// Output (Table format):
{
  speaker: "SPEAKER_01",
  text: "Hello",
  start: 0.71,
  end: 8.31,
  words: [],
  confidence: 0.9,
  pauses: [],
  _azure: {
    speakerId: "Guest-1",
    type: "final",
    offset: 0.71,
    duration: 7.6
  }
}
```

## Usage

The `getTextServiceResult()` function automatically detects Azure results and parses them:

```javascript
const result = app.getTextServiceResult(recording);
if (result && result._isAzureNative) {
  // This is Azure native format, already parsed
  const segments = result.segments;
}
```

## Testing

Use `test_azure_response.py` to get raw Azure responses:

```bash
python3 test_azure_response.py --mode realtime --output azure_realtime_response.json
```

This will show:
- Raw Azure response structure
- Normalized segments (current format)
- Total segments count




