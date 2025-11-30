# TRIZ-based Merge Algorithm Implementation

## Overview

This document describes the TRIZ (Theory of Inventive Problem Solving) based solution for merging transcripts from primary diarization and voice tracks.

## Key TRIZ Principles Applied

### 1. Extraction (Прийом 2)
- **Implementation**: `normalizeVoiceTrackSegments()`
- **What it does**: Extracts speaker determination logic from transcription to track level
- **Result**: All voice segments use `originalTrackSpeaker` instead of unreliable Speechmatics labels

### 2. Segmentation (Прийом 1)
- **Implementation**: `splitLongSegmentsByPrimary()`
- **What it does**: Splits suspiciously long voice segments (>30s) using primary segment boundaries as "natural knives"
- **Result**: Prevents mixing multiple speaker turns in one segment

### 3. Partial Action (Прийом 16)
- **Implementation**: Merge algorithm skips segments rather than forcing merge
- **What it does**: Better to have 80-90% correct merges than 100% with mixed speakers
- **Result**: Higher quality, lower risk of speaker mixing

### 4. Local Quality (Прийом 3)
- **Implementation**: Different parts of system used for what they do best
- **What it does**: 
  - Primary diarization → speaker truth
  - Voice tracks → text truth
- **Result**: Optimal use of each component's strengths

### 5. Feedback (Прийом 23)
- **Implementation**: Merge statistics and confidence levels
- **What it does**: Tracks merge quality for continuous improvement
- **Result**: Data for tuning thresholds and detecting issues

## Global Rules

1. **Rule 1**: Ignore internal `speaker` labels from Speechmatics, always use `originalTrackSpeaker`
2. **Rule 2**: Primary diarization = single source of truth for speaker and segmentation
3. **Rule 3**: Better to skip than to mix speakers (partial action principle)

## Algorithm Flow

```
1. Normalize voice tracks
   └─> Replace speaker labels with originalTrackSpeaker
   
2. Split long segments
   └─> Use primary boundaries to cut segments >30s
   
3. Group by speaker
   └─> Now trivial: speaker === trackSpeaker
   
4. For each primary segment:
   ├─> Find voice segments from same speaker's track
   ├─> Validate: role match, overlap, time distance, similarity
   ├─> Select best match (score = overlap × similarity)
   └─> Merge or keep primary (partial action)
   
5. Add unused voice segments
   └─> Only if no overlap with primary (new content)
   
6. Calculate statistics
   └─> Track merge coverage and confidence
```

## Configuration Options

```javascript
{
  maxVoiceDuration: 30.0,      // Max duration before splitting (seconds)
  minOverlap: 0.1,             // Minimum temporal overlap (seconds)
  maxTimeDistance: 2.0,        // Max timestamp distance (seconds)
  minTextSimilarity: 0.3       // Minimum text similarity (0-1)
}
```

## Merge Statistics

Each merge operation produces statistics:

```javascript
{
  total: number,              // Total segments in result
  fromPrimary: number,       // Segments kept from primary
  fromVoice: number,          // Segments merged from voice tracks
  highConfidence: number,    // High confidence merges
  lowConfidence: number      // Low confidence (kept primary)
}
```

## Segment Metadata

Each merged segment includes:

```javascript
{
  speaker: string,                    // Speaker label (from primary)
  text: string,                      // Final text (from voice if merged)
  start: number,                     // Start time (from primary)
  end: number,                       // End time (from primary)
  source: 'primary' | 'voice',       // Source of text
  mergeConfidence: 'high' | 'medium' | 'low',
  mergeScore?: number,               // Merge score (if merged)
  originalDetectedSpeaker?: string,  // Original Speechmatics speaker (for diagnostics)
  reason?: string                    // Reason for low confidence
}
```

## Quality Metrics

### 1. Speaker Consistency Metric
- **Definition**: Percentage of segments where speaker assignment is correct
- **Measurement**: Manual validation on 20-50 calls
- **Target**: >95%

### 2. Text Improvement Metric (WER)
- **Definition**: Word Error Rate comparison
- **Measurement**: WER(primary) vs WER(merged) on golden set
- **Target**: WER(merged) < WER(primary)

### 3. Merge Coverage
- **Definition**: Percentage of primary segments merged with voice text
- **Measurement**: `fromVoice / total`
- **Target**: 60-80% (balance between coverage and quality)

### 4. Suspicious Merge Rate
- **Definition**: Segments with low similarity but high overlap
- **Measurement**: `segments with similarity < 0.5 && overlap > 0.5s`
- **Target**: <5%

## Testing Plan

### Phase 1: Unit Tests (Synthetic Data)

Create JSON test cases:

1. **Long segment test**:
   - Voice segment: 60s, contains multiple speaker turns
   - Expected: Split by primary boundaries

2. **Wrong speaker label test**:
   - Voice segment: `speaker: "SPEAKER_02"` but `originalTrackSpeaker: "SPEAKER_00"`
   - Expected: Use `SPEAKER_00` after normalization

3. **Role mismatch test**:
   - Primary: `role: "operator"`, Voice track: `role: "client"`
   - Expected: Skip merge, keep primary

4. **Partial overlap test**:
   - Primary: 5-10s, Voice: 8-15s
   - Expected: Merge if similarity > threshold

5. **No overlap test**:
   - Primary: 5-10s, Voice: 20-25s
   - Expected: Add voice as new segment

### Phase 2: Golden Set Testing

1. **Select 20-30 real calls** with:
   - Different overlap scenarios
   - Long monologues
   - Background noise
   - Multiple speakers (2-3)

2. **Create manual ground truth**:
   - Correct speaker assignments
   - Correct text transcription
   - Segment boundaries

3. **Run both versions**:
   - Old merge algorithm
   - New TRIZ-based algorithm

4. **Compare metrics**:
   - Speaker accuracy
   - WER
   - Number of mixed phrases

### Phase 3: Stress Tests

1. **High overlap calls**: Many simultaneous speakers
2. **Long monologues**: Single speaker talking for 2+ minutes
3. **Noisy audio**: Background noise, poor separation quality
4. **Multi-speaker**: 3+ speakers in conversation

### Phase 4: Regression Tests

For each bug found:
1. Create JSON fixture reproducing the issue
2. Add to test suite
3. Verify fix doesn't break existing cases

### Phase 5: Production Monitoring

Log metrics for each merge:
- Merge coverage percentage
- Average/95th percentile segment duration
- Low similarity segment rate
- Role mismatch rate

Weekly manual review:
- Random sample of 10-20 calls
- Check for speaker mixing
- Validate merge quality

## Risk Mitigation

### Risk 1: Poor Source Separation
**Problem**: Track contains wrong speaker's voice
**Mitigation**:
- Log `originalDetectedSpeaker` vs `trackSpeaker` discrepancies
- For tracks with high discrepancy rate, reduce confidence
- Consider not merging such tracks

### Risk 2: Timestamp Misalignment
**Problem**: Small shifts between primary and voice timestamps
**Mitigation**:
- Use `maxTimeDistance` tolerance (2.0s default)
- Can add global offset detection via cross-correlation

### Risk 3: Threshold Tuning
**Problem**: Too high/low thresholds
**Mitigation**:
- Tune on development set
- Monitor production metrics
- Adjust based on feedback

### Risk 4: Uneven Primary Segmentation
**Problem**: Primary gives very small segments (1-2 words)
**Mitigation**:
- Optionally merge adjacent primary segments of same speaker
- Create "phrase blocks" before merge

## Implementation Status

✅ **COMPLETED**:
- Normalize voice track speakers
- Split long segments by primary boundaries
- Simplified merge algorithm (no complex mapping)
- Role validation
- Merge statistics
- Confidence levels

🔄 **IN PROGRESS**:
- Unit tests
- Golden set creation
- Production monitoring setup

⏳ **PLANNED**:
- Threshold auto-tuning
- Global timestamp offset detection
- Primary segment merging for small segments

## Expected Improvements

Based on TRIZ analysis and research:

1. **70-80% improvement** from channel mode fix (already implemented)
2. **Additional 10-15% improvement** from TRIZ-based merge algorithm
3. **Total expected accuracy**: 85-95% (from ~70%)

## References

- TRIZ principles: Altshuller's 40 Inventive Principles
- ARIZ methodology: Algorithm for Inventive Problem Solving
- Research findings: Perplexity research on speaker assignment errors
- Channel mode fix: CRITICAL_FIX_CHANNEL_MODE.md

