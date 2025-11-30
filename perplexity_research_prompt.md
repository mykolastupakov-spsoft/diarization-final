# Research Prompt for Perplexity: Speaker Assignment Errors in Multi-Source Transcript Merging

## Problem Statement

I'm building an audio diarization system that merges transcripts from two different sources:
1. **Primary diarization**: Full audio processed by Speechmatics (original audio with all speakers)
2. **Voice track transcripts**: Separated audio tracks processed by SpeechBrain source separation, then transcribed by Speechmatics

The system incorrectly assigns speaker labels when merging these transcripts, causing text from different speakers to be mixed together in the final output.

## System Architecture

### Pipeline Flow (Mode 3 Overlap Diarization):

1. **Step 1**: Primary diarization using Speechmatics on full audio
   - Output: JSON with segments containing `speaker`, `text`, `start`, `end`, `role`
   - Speakers: `SPEAKER_00`, `SPEAKER_01` (typically 2 speakers)

2. **Step 2**: Source separation using SpeechBrain
   - Input: Full audio file
   - Output: Separate audio files per speaker (e.g., `SPEAKER_00.wav`, `SPEAKER_01.wav`)
   - Each separated track should contain only one speaker's voice

3. **Step 3**: Transcription of separated tracks using Speechmatics
   - Input: Each separated audio file
   - Output: JSON transcript for each track
   - **CRITICAL ISSUE**: Speechmatics may detect different speakers in separated tracks (e.g., `SPEAKER_02`, `SPEAKER_03`) even though the track should only contain one speaker

4. **Step 4**: Role analysis (operator/client detection)

5. **Step 5**: Merge primary diarization with voice track transcripts
   - **THIS IS WHERE THE PROBLEM OCCURS**

## Data Structures

### Primary Diarization Structure:
```json
{
  "recordings": [{
    "results": {
      "speechmatics": {
        "segments": [
          {
            "speaker": "SPEAKER_00",
            "text": "Hi I'm Jessica...",
            "start": 0.6,
            "end": 8.24,
            "role": "operator"
          },
          {
            "speaker": "SPEAKER_01",
            "text": "Yeah Jessica I do available...",
            "start": 9.36,
            "end": 12.8,
            "role": "client"
          }
        ]
      }
    }
  }]
}
```

### Voice Tracks Structure:
```json
{
  "speakers": [
    {
      "speaker": "SPEAKER_00",
      "role": "operator",
      "transcript": {
        "recordings": [{
          "results": {
            "speechmatics": {
              "segments": [
                {
                  "speaker": "SPEAKER_00",  // May be different!
                  "text": "Hi I'm Jessica...",
                  "start": 0.64,
                  "end": 8.24
                },
                {
                  "speaker": "SPEAKER_01",  // PROBLEM: Different speaker in same track!
                  "text": "Yeah For me",
                  "start": 9.48,
                  "end": 12.8
                }
              ]
            }
          }
        }]
      }
    },
    {
      "speaker": "SPEAKER_01",
      "role": "client",
      "transcript": {
        "recordings": [{
          "results": {
            "speechmatics": {
              "segments": [
                {
                  "speaker": "SPEAKER_00",  // PROBLEM: Wrong speaker!
                  "text": "Hi I'm calling on behalf of",
                  "start": 0.68,
                  "end": 2.64
                },
                {
                  "speaker": "SPEAKER_01",
                  "text": "Yeah Jessica I do available...",
                  "start": 9.32,
                  "end": 73.69  // PROBLEM: Very long segment!
                }
              ]
            }
          }
        }]
      }
    }
  ]
}
```

## Current Implementation

### Speaker Mapping Algorithm:
1. **Track-level mapping**: Groups all segments by speaker label first
2. **Overlap calculation**: For each voice speaker, calculates total temporal overlap with each primary speaker
3. **Scoring**: `score = totalOverlap × √(matchCount)`
4. **Fallback**: Uses `originalTrackSpeaker` if overlap-based mapping fails

### Merge Algorithm:
1. For each primary segment:
   - Finds overlapping voice track segments with same (mapped) speaker
   - Validates using guards:
     - Speaker match (after mapping)
     - Temporal overlap (min 100ms)
     - Timestamp distance (max 2 seconds)
     - Text similarity (min 0.3)
   - Selects best matching voice track segment
   - Merges text if voice track text is longer and contains primary text

2. Adds new segments from voice tracks that don't overlap with primary

### Current Guards:
- `shouldMergeSegments()` checks:
  - Speaker match (mapped speaker == primary speaker)
  - Temporal overlap >= 0.1s
  - Timestamp distance <= 2.0s
  - Text similarity >= 0.3

## Observed Problems

### Problem 1: Wrong Speaker Assignment
**Example**: 
- Primary: `SPEAKER_00` says "Hi I'm Jessica..."
- Voice track for `SPEAKER_00` contains segment with `speaker: "SPEAKER_01"` saying "Yeah Jessica..."
- Result: Text from `SPEAKER_01` gets assigned to `SPEAKER_00` in final output

**Root cause hypothesis**: Speechmatics transcription on separated tracks detects different speakers than expected, possibly due to:
- Artifacts in source separation
- Residual audio from other speakers
- Speechmatics diarization errors on separated audio

### Problem 2: Very Long Segments
**Example**:
- Voice track segment: `start: 9.32, end: 73.69` (64 seconds!)
- This segment contains multiple speaker turns
- Gets incorrectly merged with primary segments

**Root cause hypothesis**: Source separation may not perfectly isolate speakers, or Speechmatics may combine multiple segments incorrectly.

### Problem 3: Text Mixing
**Example output**:
```
Speaker 1: Hi I'm calling on behalf of Yeah Jessica I do available What do you have for me...
```
This combines text from both speakers incorrectly.

## What We've Tried

1. **Track-level speaker mapping**: Instead of per-segment mapping, map entire voice speaker to primary speaker
2. **Validation guards**: Added timestamp distance, overlap, similarity checks
3. **Conservative merging**: Only merge if voice track text is longer and contains primary text
4. **Fallback mechanisms**: Multiple fallback levels if mapping fails
5. **Sorting**: Ensured segments are sorted by timestamp before grouping

## Technical Questions for Research

### 1. SpeechBrain Source Separation
- Can SpeechBrain create segments with wrong speaker labels after separation?
- How does SpeechBrain handle overlapping speech during separation?
- Can separated tracks contain artifacts or residual audio from other speakers?
- How accurate are timestamps after source separation? Can they shift?
- Are there known issues with SpeechBrain producing incorrect speaker assignments in separated audio?

### 2. Speechmatics Transcription on Separated Tracks
- Can Speechmatics detect different speakers in a separated track that should only contain one speaker?
- Why would Speechmatics assign `SPEAKER_02` or `SPEAKER_03` to a track that was separated for `SPEAKER_00`?
- Does Speechmatics perform diarization on separated tracks, or should it assume single speaker?
- Are there configuration options to disable diarization on separated audio?

### 3. Multi-Source Transcript Merging
- What are best practices for merging transcripts from different diarization systems?
- How should speaker mapping be done when speaker labels don't match between systems?
- How to handle cases where one system detects more speakers than another?
- What algorithms are used in academic/industrial solutions for this problem?

### 4. Speaker Mapping Algorithms
- What are standard approaches to mapping speakers between different diarization systems?
- How to handle temporal overlap when speakers don't match exactly?
- Should we trust the original track speaker label or the detected speaker in transcription?
- How to validate speaker mapping correctness?

### 5. Overlap Handling
- How to correctly merge overlapping segments from different sources?
- How to determine which text is correct when there's a conflict?
- Should we prefer primary diarization or voice track transcription?
- How to handle cases where voice track has more complete text but wrong speaker assignment?

### 6. Source Separation Quality
- How to detect if source separation quality is poor?
- What metrics indicate good vs. bad separation?
- How to handle cases where separation artifacts cause transcription errors?
- Should we validate separation quality before merging?

### 7. Timestamp Alignment
- Can timestamps shift after source separation and re-transcription?
- How to align timestamps between primary and voice track transcripts?
- Should we use relative or absolute timestamps for alignment?

## Specific Research Queries

Please research:

1. **"SpeechBrain speaker separation artifacts wrong speakers in segments"**
   - Known issues with SpeechBrain producing incorrect speaker assignments
   - How to detect and handle separation artifacts

2. **"Speechmatics transcription on separated audio tracks speaker diarization issues"**
   - Why Speechmatics might detect different speakers in separated tracks
   - Configuration options to improve accuracy on separated audio

3. **"How to merge transcripts from different diarization systems speaker mapping"**
   - Best practices and algorithms for multi-source transcript merging
   - Speaker mapping techniques when labels don't match

4. **"Source separation timestamp alignment issues after audio processing"**
   - Timestamp accuracy after source separation
   - Methods to align timestamps between different transcript sources

5. **"Multi-source transcript merging best practices speaker assignment"**
   - Academic and industrial approaches
   - Validation and quality metrics

6. **"Speaker diarization conflict resolution between different systems"**
   - How to resolve conflicts when systems disagree on speaker assignments
   - Trust models for different transcription sources

7. **"Voice activity detection after source separation timestamp accuracy"**
   - VAD accuracy on separated audio
   - Timestamp reliability after separation

## Expected Output

Please provide:
1. **Root cause analysis**: What is likely causing the speaker assignment errors?
2. **Best practices**: What are industry-standard approaches to this problem?
3. **Algorithmic solutions**: Specific algorithms or techniques that could solve this
4. **Configuration recommendations**: Settings or configurations that might help
5. **Validation strategies**: How to detect and prevent these errors
6. **Alternative approaches**: Different ways to structure the merging process

## Constraints

- Must be programmatic (no LLM-based merging - we tried this and it hallucinated)
- Must preserve timestamps from primary diarization
- Must be deterministic and auditable
- Must handle cases where voice tracks have different speaker labels than primary
- Must prevent text from different speakers being mixed

## Additional Context

- Primary diarization is generally more accurate for speaker assignment
- Voice track transcription is generally more accurate for text content (because it's from separated audio)
- The goal is to combine: accurate speaker assignment from primary + accurate text from voice tracks
- We're seeing issues where voice track segments have wrong speaker labels, causing incorrect merging

---

**Please provide comprehensive research findings with specific solutions, algorithms, and best practices that can be implemented programmatically to solve this speaker assignment problem.**

