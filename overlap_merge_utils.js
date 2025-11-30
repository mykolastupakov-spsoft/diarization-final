/* eslint-disable max-lines */
(function overlapMergeFactory(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.OverlapMergeUtils = factory();
  }
}(typeof self !== 'undefined' ? self : this, () => {
  'use strict';

  const noopLogger = {
    log: () => {},
    warn: () => {},
    error: () => {}
  };

  function getLogger(logger) {
    if (!logger) {
      return noopLogger;
    }
    return {
      log: typeof logger.log === 'function' ? logger.log.bind(logger) : noopLogger.log,
      warn: typeof logger.warn === 'function' ? logger.warn.bind(logger) : noopLogger.warn,
      error: typeof logger.error === 'function' ? logger.error.bind(logger) : noopLogger.error
    };
  }

  // Використовуємо покращену text similarity з text_similarity_utils.js
  // Якщо доступна глобальна версія (браузер), використовуємо її
  // Інакше використовуємо вбудовану версію (Node.js)
  let textSimilarityUtils = null;
  if (typeof window !== 'undefined' && window.TextSimilarityUtils) {
    textSimilarityUtils = window.TextSimilarityUtils;
  } else if (typeof require !== 'undefined') {
    try {
      textSimilarityUtils = require('./text_similarity_utils.js');
    } catch (e) {
      // Fallback to inline implementation
    }
  }

  // Fallback implementation (спрощена версія для сумісності)
  function normalizeText(text) {
    if (!text || typeof text !== 'string') return '';
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function areTextsSimilar(text1, text2, thresholdConfig = {}) {
    // Якщо доступна покращена версія, використовуємо її
    if (textSimilarityUtils && typeof textSimilarityUtils.areTextsSimilar === 'function') {
      return textSimilarityUtils.areTextsSimilar(text1, text2, thresholdConfig);
    }

    // Fallback до старої логіки (для сумісності)
    if (!text1 || !text2) return false;
    const normalized1 = normalizeText(text1);
    const normalized2 = normalizeText(text2);
    if (normalized1 === normalized2) return true;
    
    if (normalized1.includes(normalized2) || normalized2.includes(normalized1)) {
      const ratio = Math.min(normalized1.length, normalized2.length) / Math.max(normalized1.length, normalized2.length);
      if (ratio >= 0.75) {
        return true;
      }
    }
    return false;
  }

  function collectVoiceTrackSegments(voiceTracks, primarySegments = [], options = {}) {
    if (!Array.isArray(voiceTracks)) {
      return [];
    }

    const logger = getLogger(options.logger);
    const segments = [];

    voiceTracks.forEach(track => {
      if (!track || track.error) return;
      const trackSpeaker = track.speaker || 'SPEAKER_00';
      const role = track.roleAnalysis?.role || null;
      const recording = track.transcription?.recordings?.[0];
      const speechmaticsResult = recording?.results?.speechmatics;
      const engineName = (speechmaticsResult?.engine || '').toLowerCase();
      const speechmaticsSegments = speechmaticsResult?.segments || [];

      if (speechmaticsSegments.length === 0) {
        logger.warn(`⚠️ Voice track ${trackSpeaker} has no segments`);
        return;
      }

      if (engineName.startsWith('azure')) {
        // Для Azure: дедуплікація сегментів ВСЕРЕДИНІ одного voice track
        // Azure може повернути дублікати або перекриваючі сегменти
        const azureSegments = [];

        // Визначаємо основного мовця всередині цього voice track
        const detectedSpeakerDurations = {};
        speechmaticsSegments.forEach(segment => {
          const start = typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0;
          const endValue = typeof segment.end === 'number' ? segment.end : parseFloat(segment.end);
          const end = Number.isNaN(endValue) || endValue === undefined ? start : endValue;
          const duration = Math.max(0, end - start);
          const detectedSpeaker = segment.speaker || 'SPEAKER_00';
          detectedSpeakerDurations[detectedSpeaker] = (detectedSpeakerDurations[detectedSpeaker] || 0) + duration;
        });

        const detectedSpeakerEntries = Object.entries(detectedSpeakerDurations)
          .sort((a, b) => (b[1] || 0) - (a[1] || 0));
        const mainDetectedSpeaker =
          detectedSpeakerEntries[0]?.[0] ||
          speechmaticsSegments[0]?.speaker ||
          'SPEAKER_00';
        const mainDuration = detectedSpeakerEntries[0]?.[1] || 0;
        logger.log(
          `✅ Azure voice track ${trackSpeaker}: main detected speaker ${mainDetectedSpeaker} (${mainDuration.toFixed(1)}s)`
        );

        let skippedResidualSegments = 0;
        speechmaticsSegments.forEach(segment => {
          const start = typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0;
          let end = typeof segment.end === 'number' ? segment.end : parseFloat(segment.end);
          if (Number.isNaN(end) || end === undefined) {
            end = start;
          }

          const detectedSpeaker = segment.speaker || mainDetectedSpeaker;
          if (detectedSpeaker !== mainDetectedSpeaker) {
            const duration = Math.max(0, end - start);
            if (duration > 0.05) {
              skippedResidualSegments += 1;
              logger.log(
                `🔇 Azure voice track ${trackSpeaker}: dropping residual segment from ${detectedSpeaker} (${duration.toFixed(2)}s)`
              );
            }
            return;
          }

          azureSegments.push({
            speaker: trackSpeaker,
            text: segment.text || '',
            start,
            end,
            words: segment.words || [],
            role: role || segment.role || null,
            overlap: false,
            source: 'voice-track',
            originalTrackSpeaker: trackSpeaker,
            originalDetectedSpeaker: detectedSpeaker,
            isFullText: true
          });
        });

        if (skippedResidualSegments > 0) {
          logger.log(
            `🔇 Azure voice track ${trackSpeaker}: removed ${skippedResidualSegments} residual segment(s) from other speakers`
          );
        }

        // Дедуплікація Azure сегментів для цього voice track
        const deduplicatedAzureSegments = [];
        const sortedAzureSegments = [...azureSegments].sort((a, b) => {
          const startDiff = (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0);
          if (startDiff !== 0) return startDiff;
          return (parseFloat(a.end) || 0) - (parseFloat(b.end) || 0);
        });

        for (let i = 0; i < sortedAzureSegments.length; i++) {
          const current = sortedAzureSegments[i];
          const currentStart = parseFloat(current.start) || 0;
          const currentEnd = parseFloat(current.end) || currentStart;
          const currentDuration = currentEnd - currentStart;
          const currentText = (current.text || '').trim();
          
          let isDuplicate = false;

          for (let j = 0; j < deduplicatedAzureSegments.length; j++) {
            const existing = deduplicatedAzureSegments[j];
            const existingStart = parseFloat(existing.start) || 0;
            const existingEnd = parseFloat(existing.end) || existingStart;
            const existingDuration = existingEnd - existingStart;
            const existingText = (existing.text || '').trim();

            const overlapStart = Math.max(currentStart, existingStart);
            const overlapEnd = Math.min(currentEnd, existingEnd);
            const overlapDuration = Math.max(0, overlapEnd - overlapStart);

            if (overlapDuration <= 0) continue;

            // Відсоток перекриття
            const currentOverlapRatio = currentDuration > 0 ? overlapDuration / currentDuration : 0;
            const existingOverlapRatio = existingDuration > 0 ? overlapDuration / existingDuration : 0;

            // Поріг 1: Майже повне перекриття (>65%)
            const strongOverlap = (currentOverlapRatio > 0.65 && existingOverlapRatio > 0.65);

            // Поріг 2: Текст дуже схожий + overlap > 0.3s
            const textSimilar = areTextsSimilar(currentText, existingText, {
              minLevenshteinSim: 0.85,
              minJaccardSim: 0.75
            });
            const meaningfulOverlap = overlapDuration > 0.3;

            // Поріг 3: Один текст містить інший + overlap > 0.1s
            const normCurrent = normalizeText(currentText);
            const normExisting = normalizeText(existingText);
            const substringMatch = normCurrent.includes(normExisting) || normExisting.includes(normCurrent);
            const minimalOverlap = overlapDuration > 0.1;

            if (strongOverlap || (textSimilar && meaningfulOverlap) || (substringMatch && minimalOverlap)) {
              isDuplicate = true;
              // Залишаємо сегмент з БІЛЬШИМ текстом
              if (currentText.length > existingText.length) {
                deduplicatedAzureSegments[j] = current;
              }
              break;
            }
          }

          if (!isDuplicate) {
            deduplicatedAzureSegments.push(current);
          }
        }

        logger.log(
          `📊 Azure voice track ${trackSpeaker}: ${azureSegments.length} → ${deduplicatedAzureSegments.length} segments (removed ${azureSegments.length - deduplicatedAzureSegments.length} duplicates)`
        );

        // Додаємо дедупліковані сегменти
        segments.push(...deduplicatedAzureSegments);
        return;
      }

      const segmentsByDetectedSpeaker = {};
      speechmaticsSegments.forEach(segment => {
        const detectedSpeaker = segment.speaker || 'SPEAKER_00';
        if (!segmentsByDetectedSpeaker[detectedSpeaker]) {
          segmentsByDetectedSpeaker[detectedSpeaker] = [];
        }
        segmentsByDetectedSpeaker[detectedSpeaker].push(segment);
      });

      logger.log(
        `📊 Voice track ${trackSpeaker}: Speechmatics detected ${Object.keys(segmentsByDetectedSpeaker).length} speaker(s):`,
        Object.keys(segmentsByDetectedSpeaker).map(s => `${s} (${segmentsByDetectedSpeaker[s].length} segments)`).join(', ')
      );

      // Calculate total duration of all segments for percentage calculation
      let totalTrackDuration = 0;
      for (const segs of Object.values(segmentsByDetectedSpeaker)) {
        const duration = segs.reduce((sum, seg) => {
          const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
          const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
          return sum + (end - start);
        }, 0);
        totalTrackDuration += duration;
      }

      let mainDetectedSpeaker = null;
      let maxDuration = 0;
      let maxSegments = 0;
      let maxDurationPercent = 0;

      for (const [detectedSpk, segs] of Object.entries(segmentsByDetectedSpeaker)) {
        const totalDuration = segs.reduce((sum, seg) => {
          const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
          const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
          return sum + (end - start);
        }, 0);
        
        const durationPercent = totalTrackDuration > 0 ? (totalDuration / totalTrackDuration) * 100 : 0;

        // CRITICAL: Main speaker must have at least 60% of total duration
        // This prevents residual audio from being selected as main speaker
        // Also prioritize by segment count and duration
        const isSignificant = durationPercent >= 60;
        const isBetter = isSignificant && (
          segs.length > maxSegments || 
          (segs.length === maxSegments && totalDuration > maxDuration) ||
          (segs.length === maxSegments && totalDuration === maxDuration && durationPercent > maxDurationPercent)
        );

        if (isBetter) {
          maxSegments = segs.length;
          maxDuration = totalDuration;
          maxDurationPercent = durationPercent;
          mainDetectedSpeaker = detectedSpk;
        }
      }

      // Fallback: if no speaker has 60%+, use the one with most duration
      if (!mainDetectedSpeaker) {
        for (const [detectedSpk, segs] of Object.entries(segmentsByDetectedSpeaker)) {
          const totalDuration = segs.reduce((sum, seg) => {
            const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
            const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
            return sum + (end - start);
          }, 0);
          
          if (totalDuration > maxDuration) {
            maxDuration = totalDuration;
            maxSegments = segs.length;
            mainDetectedSpeaker = detectedSpk;
          }
        }
      }

      const finalDurationPercent = totalTrackDuration > 0 ? ((maxDuration / totalTrackDuration) * 100) : 0;
      
      // DETAILED LOGGING: Voice track analysis
      logger.log(`📊 Voice track ${trackSpeaker} analysis:`, {
        totalSegments: speechmaticsSegments.length,
        speakersDetected: Object.keys(segmentsByDetectedSpeaker),
        speakersBreakdown: Object.entries(segmentsByDetectedSpeaker).map(([spk, segs]) => {
          const duration = segs.reduce((sum, seg) => {
            const start = typeof seg.start === 'number' ? seg.start : parseFloat(seg.start) || 0;
            const end = typeof seg.end === 'number' ? seg.end : parseFloat(seg.end) || start;
            return sum + (end - start);
          }, 0);
          const percent = totalTrackDuration > 0 ? (duration / totalTrackDuration) * 100 : 0;
          return { speaker: spk, segments: segs.length, duration: duration.toFixed(1), percent: percent.toFixed(1) };
        }),
        mainSpeaker: mainDetectedSpeaker,
        mainSpeakerDuration: maxDuration.toFixed(1),
        mainSpeakerPercent: finalDurationPercent.toFixed(1),
        totalTrackDuration: totalTrackDuration.toFixed(1)
      });
      
      logger.log(`✅ Main speaker in track ${trackSpeaker}: ${mainDetectedSpeaker} (${maxSegments} segments, ${maxDuration.toFixed(1)}s, ${finalDurationPercent.toFixed(1)}% of track)`);
      
      // CRITICAL: Warn if main speaker has less than 50% of duration - might be wrong
      if (finalDurationPercent < 50 && Object.keys(segmentsByDetectedSpeaker).length > 1) {
        logger.warn(`⚠️ WARNING: Main speaker ${mainDetectedSpeaker} in track ${trackSpeaker} has only ${finalDurationPercent.toFixed(1)}% of duration - possible misidentification!`);
      }
      
      // CRITICAL: Log as error if main speaker has less than 50% - separation quality is poor
      if (finalDurationPercent < 50) {
        logger.error(`❌ CRITICAL: Main speaker ${mainDetectedSpeaker} in track ${trackSpeaker} has only ${finalDurationPercent.toFixed(1)}% - separation quality is POOR!`);
      }

      // CRITICAL FIX: For voice tracks, ONLY use the main detected speaker
      // Voice tracks are separated stems - they should contain only ONE speaker's voice
      // Any other detected speakers are residual audio artifacts and should be ignored
      // This prevents segments from one speaker's voice track from being assigned to another speaker
      let skippedResidualSegments = 0;
      
      speechmaticsSegments.forEach(segment => {
        const start = typeof segment.start === 'number' ? segment.start : parseFloat(segment.start) || 0;
        let end = typeof segment.end === 'number' ? segment.end : parseFloat(segment.end);
        if (Number.isNaN(end) || end === undefined) {
          end = start;
        }

        const duration = end - start;
        const detectedSpeaker = segment.speaker || 'SPEAKER_00';
        
      // CRITICAL: Only accept segments from the main detected speaker
      // All other speakers in this voice track are residual audio and must be ignored
      if (detectedSpeaker !== mainDetectedSpeaker) {
        skippedResidualSegments += 1;
        logger.log(
          `🔇 Voice track ${trackSpeaker}: dropping residual segment from ${detectedSpeaker} (${duration.toFixed(2)}s, text: "${(segment.text || '').substring(0, 50)}...") - not main speaker ${mainDetectedSpeaker}`
        );
        return;
      }
      
      // CRITICAL: Log accepted segments for debugging
      logger.log(
        `✅ Voice track ${trackSpeaker}: accepting segment from ${detectedSpeaker} (${duration.toFixed(2)}s, text: "${(segment.text || '').substring(0, 50)}...")`
      );

        // Additional validation: skip very short segments that might be artifacts
        if (duration < 0.3) {
          logger.log(
            `⚠️ Skipping very short voice track segment (${duration.toFixed(2)}s) from ${trackSpeaker} - likely artifact`
          );
          return;
        }

        segments.push({
          speaker: trackSpeaker, // Always use the track's speaker label, not the detected one
          text: segment.text || '',
          start,
          end,
          words: segment.words || [],
          role: role || segment.role || null,
          overlap: false,
          source: 'voice-track',
          originalTrackSpeaker: trackSpeaker,
          originalDetectedSpeaker: detectedSpeaker,
          isFullText: true
        });
      });
      
      const acceptedSegments = segments.filter(s => s.originalTrackSpeaker === trackSpeaker);
      
      if (skippedResidualSegments > 0) {
        logger.log(
          `🔇 Voice track ${trackSpeaker}: removed ${skippedResidualSegments} residual segment(s) from other speakers (only using ${mainDetectedSpeaker})`
        );
      }
      
      // DETAILED LOGGING: Final summary for this voice track
      logger.log(`📊 Voice track ${trackSpeaker} final summary:`, {
        totalSegments: speechmaticsSegments.length,
        skippedResidualSegments: skippedResidualSegments,
        acceptedSegments: acceptedSegments.length,
        mainSpeaker: mainDetectedSpeaker,
        mainSpeakerPercent: finalDurationPercent.toFixed(1)
      });
      
      // CRITICAL: Verify all segments from this track have correct speaker label
      const segmentsFromThisTrack = segments.filter(s => s.originalTrackSpeaker === trackSpeaker);
      const wrongSpeakerSegments = segmentsFromThisTrack.filter(s => s.speaker !== trackSpeaker);
      if (wrongSpeakerSegments.length > 0) {
        logger.error(
          `❌ CRITICAL ERROR: Found ${wrongSpeakerSegments.length} segments from voice track ${trackSpeaker} with wrong speaker label!`
        );
        wrongSpeakerSegments.forEach(seg => {
          logger.error(
            `  - Segment "${(seg.text || '').substring(0, 50)}..." has speaker ${seg.speaker} but should be ${trackSpeaker}`
          );
        });
        // Fix the speaker labels
        wrongSpeakerSegments.forEach(seg => {
          seg.speaker = trackSpeaker;
        });
        logger.log(`✅ Fixed ${wrongSpeakerSegments.length} segments with wrong speaker labels`);
      }
    });

    // CRITICAL: Final validation - ensure all segments have correct speaker labels
    const segmentsByTrack = {};
    segments.forEach(seg => {
      const track = seg.originalTrackSpeaker || 'UNKNOWN';
      if (!segmentsByTrack[track]) segmentsByTrack[track] = [];
      segmentsByTrack[track].push(seg);
    });
    
    logger.log(
      `📊 Final voice track segments by original track:`,
      Object.keys(segmentsByTrack).map(t => `${t}: ${segmentsByTrack[t].length} segments`).join(', ')
    );
    
    // Verify speaker consistency
    for (const [track, segs] of Object.entries(segmentsByTrack)) {
      const speakers = new Set(segs.map(s => s.speaker));
      if (speakers.size > 1) {
        logger.error(
          `❌ CRITICAL ERROR: Voice track ${track} has segments with multiple speakers: ${Array.from(speakers).join(', ')}`
        );
        // Force all segments from this track to use track's speaker
        segs.forEach(seg => {
          if (seg.speaker !== track) {
            logger.warn(`  - Fixing segment "${(seg.text || '').substring(0, 50)}..." from ${seg.speaker} to ${track}`);
            seg.speaker = track;
          }
        });
      }
    }

    return segments;
  }

  function mergeVoiceTrackSegments(voiceTrackSegments, primarySegments = [], recording = {}, options = {}) {
    const logger = getLogger(options.logger);
    const textSimilarityFn = typeof options.areTextsSimilar === 'function'
      ? options.areTextsSimilar
      : areTextsSimilar;

    if (!Array.isArray(voiceTrackSegments) || voiceTrackSegments.length === 0) {
      logger.warn('⚠️ No voice track segments provided for merge.');
      return [];
    }

    logger.log(`🔄 Merging segments: ${voiceTrackSegments.length} voice track + ${primarySegments.length} primary`);

    const voiceTracksBySpeaker = {};
    voiceTrackSegments.forEach(seg => {
      const spk = seg.speaker || 'UNKNOWN';
      if (!voiceTracksBySpeaker[spk]) voiceTracksBySpeaker[spk] = [];
      voiceTracksBySpeaker[spk].push(seg);
    });
    logger.log(
      '📊 Voice track segments by speaker:',
      Object.keys(voiceTracksBySpeaker).map(s => `${s}: ${voiceTracksBySpeaker[s].length}`).join(', ')
    );

    // ФАЗА 2: Покращена дедуплікація voice tracks
    function isVoiceTrackDuplicate(seg1, seg2) {
      const s1Start = parseFloat(seg1.start) || 0;
      const s1End = parseFloat(seg1.end) || s1Start;
      const s2Start = parseFloat(seg2.start) || 0;
      const s2End = parseFloat(seg2.end) || s2Start;

      const overlapStart = Math.max(s1Start, s2Start);
      const overlapEnd = Math.min(s1End, s2End);
      const overlapDuration = Math.max(0, overlapEnd - overlapStart);

      if (overlapDuration <= 0) return false;

      // Відсоток перекриття від кожного сегменту
      const s1Duration = s1End - s1Start;
      const s2Duration = s2End - s2Start;
      const overlap1Percent = s1Duration > 0 ? overlapDuration / s1Duration : 0;
      const overlap2Percent = s2Duration > 0 ? overlapDuration / s2Duration : 0;

      // Поріг 1: Майже повне перекриття (>65% замість 80%)
      const strongOverlap = (overlap1Percent > 0.65 && overlap2Percent > 0.65);

      // Поріг 2: Текст дуже схожий + хоть якийсь overlap (0.3s замість 0.1s)
      const textSimilar = areTextsSimilar(seg1.text, seg2.text, {
        minLevenshteinSim: 0.85,
        minJaccardSim: 0.75
      });
      const meaningfulOverlap = overlapDuration > 0.3; // 300ms

      // Поріг 3: Один текст містить інший (substring) + якийсь overlap
      const norm1 = normalizeText(seg1.text);
      const norm2 = normalizeText(seg2.text);
      const substringMatch = norm1.includes(norm2) || norm2.includes(norm1);
      const minimalOverlap = overlapDuration > 0.1;

      return (strongOverlap) || (textSimilar && meaningfulOverlap) || (substringMatch && minimalOverlap);
    }

    const deduplicatedVoiceTracks = [];
    const sortedVoiceTracks = [...voiceTrackSegments].sort((a, b) => {
      const startDiff = (parseFloat(a.start) || 0) - (parseFloat(b.start) || 0);
      if (startDiff !== 0) return startDiff;
      return (parseFloat(a.end) || 0) - (parseFloat(b.end) || 0);
    });

    for (let i = 0; i < sortedVoiceTracks.length; i++) {
      const current = sortedVoiceTracks[i];
      const currentSpeaker = current.speaker || 'SPEAKER_00';
      let isDuplicate = false;

      for (let j = 0; j < deduplicatedVoiceTracks.length; j++) {
        const existing = deduplicatedVoiceTracks[j];
        if (existing.speaker !== currentSpeaker) continue;

        if (isVoiceTrackDuplicate(current, existing)) {
          isDuplicate = true;
          // ВАЖЛИВО: Залишаємо сегмент з БІЛЬШИМ текстом
          if ((current.text || '').length > (existing.text || '').length) {
            deduplicatedVoiceTracks[j] = current;
          }
          break;
        }
      }

      if (!isDuplicate) {
        deduplicatedVoiceTracks.push(current);
      }
    }

    logger.log(
      `✅ Deduplicated voice tracks: ${voiceTrackSegments.length} → ${deduplicatedVoiceTracks.length} (removed ${voiceTrackSegments.length - deduplicatedVoiceTracks.length} duplicates)`
    );

    const mergedSegments = [...deduplicatedVoiceTracks];

    const validateShortSegment = (seg, contextSegments, voiceTracks) => {
      const start = parseFloat(seg.start) || 0;
      const end = parseFloat(seg.end) || start;
      const duration = end - start;
      const speaker = seg.speaker || 'SPEAKER_00';
      const text = (seg.text || '').trim().toLowerCase();

      if (duration >= 0.8) return { valid: true, reason: 'duration_ok' };

      const hasSignificantVoiceTrackOverlap = voiceTracks.some(vtSeg => {
        const vtStart = parseFloat(vtSeg.start) || 0;
        const vtEnd = parseFloat(vtSeg.end) || vtStart;
        const vtSpeaker = vtSeg.speaker || 'SPEAKER_00';

        if (vtSpeaker !== speaker) return false;

        const overlapStart = Math.max(start, vtStart);
        const overlapEnd = Math.min(end, vtEnd);
        const overlapDuration = Math.max(0, overlapEnd - overlapStart);
        const minOverlap = Math.max(duration * 0.5, 0.2);
        return overlapDuration >= minOverlap;
      });

      if (hasSignificantVoiceTrackOverlap) {
        return { valid: true, reason: 'voice_track_overlap' };
      }

      const overlappingSameSpeaker = contextSegments.find(s => {
        const sStart = parseFloat(s.start) || 0;
        const sEnd = parseFloat(s.end) || sStart;
        const sSpeaker = s.speaker || 'SPEAKER_00';

        if (sSpeaker !== speaker) return false;
        if (Math.abs(sStart - start) < 0.01 && Math.abs(sEnd - end) < 0.01) return false;

        const overlapStart = Math.max(start, sStart);
        const overlapEnd = Math.min(end, sEnd);
        const overlapDuration = overlapEnd - overlapStart;

        return overlapDuration > 0.1;
      });

      if (overlappingSameSpeaker) {
        return { valid: false, reason: 'chronological_violation_overlap' };
      }

      const previousSameSpeaker = contextSegments
        .filter(s => {
          const sStart = parseFloat(s.start) || 0;
          const sSpeaker = s.speaker || 'SPEAKER_00';
          return sStart < start && sSpeaker === speaker;
        })
        .sort((a, b) => (parseFloat(b.start) || 0) - (parseFloat(a.start) || 0))[0];

      if (previousSameSpeaker) {
        const prevEnd = parseFloat(previousSameSpeaker.end) || parseFloat(previousSameSpeaker.start) || 0;
        if (start < prevEnd - 0.05) {
          return { valid: false, reason: 'chronological_violation_start_before_end' };
        }
      }

      const nearbySameSpeaker = contextSegments.some(s => {
        const sStart = parseFloat(s.start) || 0;
        const sEnd = parseFloat(s.end) || sStart;
        const sSpeaker = s.speaker || 'SPEAKER_00';

        if (sSpeaker !== speaker) return false;
        if (Math.abs(sStart - start) < 0.01 && Math.abs(sEnd - end) < 0.01) return false;

        const gap = Math.min(Math.abs(sStart - start), Math.abs(sEnd - end));
        return gap < 3.0;
      });

      const confirmationWords = ['right', 'yes', 'ok', 'okay', 'sure', 'works', 'fine', 'good', 'alright', 'yeah', 'yep', 'uh-huh', 'correct', 'exactly'];
      const isConfirmationWord = confirmationWords.some(word => {
        const wordLower = word.toLowerCase();
        return text === wordLower || text === `${wordLower}.` || text.startsWith(`${wordLower} `) || text.endsWith(` ${wordLower}`);
      });

      if (isConfirmationWord && nearbySameSpeaker) {
        return { valid: true, reason: 'confirmation_word_with_context' };
      }

      if (!nearbySameSpeaker) {
        return { valid: false, reason: 'no_context' };
      }

      const hasOverlappingDifferentSpeaker = contextSegments.some(s => {
        const sStart = parseFloat(s.start) || 0;
        const sEnd = parseFloat(s.end) || sStart;
        const sSpeaker = s.speaker || 'SPEAKER_00';

        if (sSpeaker === speaker) return false;

        const overlapStart = Math.max(start, sStart);
        const overlapEnd = Math.min(end, sEnd);
        return overlapEnd > overlapStart + 0.1;
      });

      if (hasOverlappingDifferentSpeaker) {
        return { valid: false, reason: 'overlaps_different_speaker' };
      }

      return { valid: true, reason: 'has_context' };
    };

    // ФАЗА 3: Покращена логіка додавання primary segments
    function shouldAddPrimarySegment(primarySeg, voiceTrackSegments) {
      const pStart = parseFloat(primarySeg.start) || 0;
      const pEnd = parseFloat(primarySeg.end) || pStart;
      const pDuration = pEnd - pStart;
      const pSpeaker = primarySeg.speaker || 'SPEAKER_00';
      const pText = (primarySeg.text || '').trim();

      // DETAILED LOGGING: Primary segment check
      logger.log(`🔍 Primary segment check:`, {
        segment: {
          speaker: pSpeaker,
          start: pStart.toFixed(2),
          end: pEnd.toFixed(2),
          duration: pDuration.toFixed(2),
          text: pText.substring(0, 50)
        }
      });

      // CRITICAL: First check if this primary segment overlaps with DIFFERENT speaker voice tracks
      // If it does, skip it to prevent speaker mixing
      let overlapsDifferentSpeaker = false;
      let maxOverlapPercent = 0;
      let conflictingVoiceTrack = null;
      
      for (const voiceSeg of voiceTrackSegments) {
        const vStart = parseFloat(voiceSeg.start) || 0;
        const vEnd = parseFloat(voiceSeg.end) || vStart;
        const vSpeaker = voiceSeg.speaker || 'SPEAKER_00';
        
        // Only check overlap with DIFFERENT speaker
        if (vSpeaker === pSpeaker) continue;
        
        const overlapStart = Math.max(pStart, vStart);
        const overlapEnd = Math.min(pEnd, vEnd);
        const overlapDuration = Math.max(0, overlapEnd - overlapStart);
        
        if (overlapDuration <= 0) continue;
        
        // If there's significant overlap (>30% of primary segment), skip it
        // This prevents primary segments from being assigned to wrong speaker
        const overlapPercent = pDuration > 0 ? (overlapDuration / pDuration) * 100 : 0;
        if (overlapPercent > maxOverlapPercent) {
          maxOverlapPercent = overlapPercent;
          conflictingVoiceTrack = vSpeaker;
        }
        
        if (overlapPercent > 30) {
          overlapsDifferentSpeaker = true;
          logger.log(
            `🔇 Skipping primary segment "${pText.substring(0, 30)}..." (${pSpeaker}) - overlaps ${overlapPercent.toFixed(1)}% with voice track from ${vSpeaker}`
          );
          logger.log(`🔍 Primary segment check result:`, {
            decision: 'skip',
            reason: 'overlaps_different_speaker',
            overlapPercent: overlapPercent.toFixed(1),
            conflictingSpeaker: vSpeaker
          });
          return false;
        }
      }

      // Now check overlap with SAME speaker voice tracks
      let overlapsSameSpeaker = false;
      let maxOverlapPercentSameSpeaker = 0;
      
      for (const voiceSeg of voiceTrackSegments) {
        if (primarySeg.speaker !== voiceSeg.speaker) continue;

        const vStart = parseFloat(voiceSeg.start) || 0;
        const vEnd = parseFloat(voiceSeg.end) || vStart;
        const overlapStart = Math.max(pStart, vStart);
        const overlapEnd = Math.min(pEnd, vEnd);
        const overlapDuration = Math.max(0, overlapEnd - overlapStart);

        if (overlapDuration <= 0) continue;

        // Перевіряємо, чи voice track покриває primary майже повністю
        const overlapPercentOfPrimary = pDuration > 0 ? overlapDuration / pDuration : 0;
        if (overlapPercentOfPrimary > maxOverlapPercentSameSpeaker) {
          maxOverlapPercentSameSpeaker = overlapPercentOfPrimary;
        }

        // Якщо voice track покриває >80% primary segment ТА текст схожий
        const textSimilar = areTextsSimilar(primarySeg.text, voiceSeg.text, {
          minLevenshteinSim: 0.82,
          minJaccardSim: 0.70
        });

        if (overlapPercentOfPrimary > 0.8 && textSimilar) {
          // Voice track достатньо покриває primary - не додаємо primary
          overlapsSameSpeaker = true;
          logger.log(`🔍 Primary segment check result:`, {
            decision: 'skip',
            reason: 'overlaps_same_speaker_voice_track',
            overlapPercent: overlapPercentOfPrimary.toFixed(1),
            textSimilar: textSimilar
          });
          return false;
        }

        // Якщо primary повністю міститься в voice track (часово)
        if (pStart >= vStart && pEnd <= vEnd && textSimilar) {
          // Voice track містить весь primary - не додаємо primary
          overlapsSameSpeaker = true;
          logger.log(`🔍 Primary segment check result:`, {
            decision: 'skip',
            reason: 'contained_in_voice_track',
            textSimilar: textSimilar
          });
          return false;
        }
      }

      // Якщо primary не покритий voice tracks і не перекривається з іншими спікерами, додаємо його
      logger.log(`🔍 Primary segment check result:`, {
        decision: 'add',
        reason: 'no_overlap',
        overlapsDifferentSpeaker: overlapsDifferentSpeaker,
        maxOverlapPercentDifferentSpeaker: maxOverlapPercent.toFixed(1),
        overlapsSameSpeaker: overlapsSameSpeaker,
        maxOverlapPercentSameSpeaker: maxOverlapPercentSameSpeaker.toFixed(1)
      });
      return true;
    }

    primarySegments.forEach(pSeg => {
      const pStart = parseFloat(pSeg.start) || 0;
      const pEnd = parseFloat(pSeg.end) || pStart;
      const pDuration = pEnd - pStart;
      const pSpeaker = pSeg.speaker || 'SPEAKER_00';
      const pText = (pSeg.text || '').trim().toLowerCase();

      if (pDuration < 0.8) {
        const validation = validateShortSegment(pSeg, primarySegments, deduplicatedVoiceTracks);
        if (!validation.valid) {
          logger.log(
            `⚠️ Skipping short primary segment (${pDuration.toFixed(2)}s) "${pText.substring(0, 30)}..." - ${validation.reason}`
          );
          return;
        }
      }

      const shouldAdd = shouldAddPrimarySegment(pSeg, deduplicatedVoiceTracks);
      if (shouldAdd && (pSeg.text || '').trim().length > 0) {
        logger.log(`🔄 Merge step: adding primary segment`, {
          step: 'adding_primary',
          primarySegment: {
            speaker: pSpeaker,
            start: pStart.toFixed(2),
            end: pEnd.toFixed(2),
            text: pText.substring(0, 50)
          },
          decision: 'add'
        });
        mergedSegments.push({
          ...pSeg,
          source: 'primary'
        });
      } else if (!shouldAdd) {
        logger.log(`🔄 Merge step: skipping primary segment`, {
          step: 'adding_primary',
          primarySegment: {
            speaker: pSpeaker,
            start: pStart.toFixed(2),
            end: pEnd.toFixed(2),
            text: pText.substring(0, 50)
          },
          decision: 'skip',
          reason: 'shouldAddPrimarySegment returned false'
        });
      }
    });

    const sorted = mergedSegments.sort((a, b) => {
      const startDiff = (a.start || 0) - (b.start || 0);
      if (startDiff !== 0) return startDiff;
      return (a.end || 0) - (b.end || 0);
    });

    const mergedConsecutive = [];
    const maxPauseForMerge = 2.0;

    for (let i = 0; i < sorted.length; i++) {
      const current = sorted[i];
      const currentStart = parseFloat(current.start) || 0;
      let currentEnd = parseFloat(current.end) || currentStart;
      const currentSpeaker = current.speaker || 'SPEAKER_00';

      let mergedSegment = { ...current };
      let j = i + 1;

      while (j < sorted.length) {
        const next = sorted[j];
        const nextStart = parseFloat(next.start) || 0;
        const nextEnd = parseFloat(next.end) || nextStart;
        const nextSpeaker = next.speaker || 'SPEAKER_00';

        if (nextSpeaker === currentSpeaker) {
          const gap = nextStart - currentEnd;
          const smallOverlap = currentEnd > nextStart && (currentEnd - nextStart) < 0.5;
          let hasInterveningSpeaker = false;
          const gapStart = currentEnd;
          const gapEnd = nextStart;
          const minInterveningOverlap = 0.5;

          for (let k = i + 1; k < j; k++) {
            const between = sorted[k];
            const betweenStart = parseFloat(between.start) || 0;
            const betweenEnd = parseFloat(between.end) || betweenStart;
            const betweenSpeaker = between.speaker || 'SPEAKER_00';

            if (betweenSpeaker !== currentSpeaker) {
              if (betweenStart >= gapStart && betweenEnd <= gapEnd) {
                hasInterveningSpeaker = true;
                break;
              }

              const overlapStart = Math.max(gapStart, betweenStart);
              const overlapEnd = Math.min(gapEnd, betweenEnd);
              const overlapDuration = overlapEnd - overlapStart;
              if (overlapDuration > minInterveningOverlap) {
                hasInterveningSpeaker = true;
                break;
              }
            }
          }

          if (!hasInterveningSpeaker && smallOverlap) {
            for (let k = 0; k < sorted.length; k++) {
              if (k === i || k === j) continue;

              const between = sorted[k];
              const betweenStart = parseFloat(between.start) || 0;
              const betweenEnd = parseFloat(between.end) || betweenStart;
              const betweenSpeaker = between.speaker || 'SPEAKER_00';

              if (betweenSpeaker !== currentSpeaker) {
                const mergedOverlapStart = Math.max(currentStart, betweenStart);
                const mergedOverlapEnd = Math.min(nextEnd, betweenEnd);
                const mergedOverlapDuration = mergedOverlapEnd - mergedOverlapStart;
                if (mergedOverlapDuration > minInterveningOverlap) {
                  hasInterveningSpeaker = true;
                  break;
                }
              }
            }
          }

          if (hasInterveningSpeaker) {
            break;
          }

          // ФАЗА 6: Об'єднуємо тільки РІЗНІ сегменти, не дублікати
          const isDuplicate = areTextsSimilar(mergedSegment.text, next.text, {
            minLevenshteinSim: 0.8,
            minJaccardSim: 0.6
          });

          if ((gap >= 0 && gap < maxPauseForMerge) || smallOverlap) {
            if (!isDuplicate) {
              // Об'єднуємо, тільки якщо це НЕ дублікати
              logger.log(`🔄 Merge step: merging consecutive segments`, {
                step: 'merging_consecutive',
                currentSegment: {
                  speaker: currentSpeaker,
                  start: currentStart.toFixed(2),
                  end: currentEnd.toFixed(2),
                  text: (mergedSegment.text || '').substring(0, 50)
                },
                nextSegment: {
                  speaker: nextSpeaker,
                  start: nextStart.toFixed(2),
                  end: nextEnd.toFixed(2),
                  text: (next.text || '').substring(0, 50)
                },
                gap: gap.toFixed(2),
                smallOverlap: smallOverlap,
                isDuplicate: false,
                decision: 'merge'
              });
              mergedSegment.end = Math.max(mergedSegment.end || 0, nextEnd);
              mergedSegment.text = (mergedSegment.text + ' ' + (next.text || '').trim()).trim();
              if (mergedSegment.words && next.words) {
                mergedSegment.words = [...(mergedSegment.words || []), ...(next.words || [])];
              } else if (next.words) {
                mergedSegment.words = next.words;
              }
              currentEnd = mergedSegment.end;
              j++;
            } else {
              // Дублікат - пропускаємо, залишаємо довший
              logger.log(`🔄 Merge step: skipping duplicate segment`, {
                step: 'merging_consecutive',
                currentSegment: {
                  speaker: currentSpeaker,
                  start: currentStart.toFixed(2),
                  end: currentEnd.toFixed(2),
                  text: (mergedSegment.text || '').substring(0, 50),
                  length: (mergedSegment.text || '').length
                },
                nextSegment: {
                  speaker: nextSpeaker,
                  start: nextStart.toFixed(2),
                  end: nextEnd.toFixed(2),
                  text: (next.text || '').substring(0, 50),
                  length: (next.text || '').length
                },
                isDuplicate: true,
                decision: 'skip_duplicate',
                keeping: (next.text || '').length > (mergedSegment.text || '').length ? 'next' : 'current'
              });
              if ((next.text || '').length > (mergedSegment.text || '').length) {
                mergedSegment = { ...next };
                currentEnd = parseFloat(next.end) || parseFloat(next.start) || 0;
              }
              j++;
            }
          } else {
            break;
          }
        } else {
          break;
        }
      }

      mergedConsecutive.push(mergedSegment);
      i = j - 1;
    }

    logger.log(`✅ Merged consecutive segments: ${sorted.length} → ${mergedConsecutive.length}`);

    const chronologicallyValid = [];
    for (let i = 0; i < mergedConsecutive.length; i++) {
      const current = mergedConsecutive[i];
      const currentStart = parseFloat(current.start) || 0;
      const currentEnd = parseFloat(current.end) || currentStart;
      const currentSpeaker = current.speaker || 'SPEAKER_00';

      let wasMerged = false;
      let shouldSkip = false;

      for (let k = 0; k < chronologicallyValid.length; k++) {
        const prev = chronologicallyValid[k];
        const prevStart = parseFloat(prev.start) || 0;
        const prevEnd = parseFloat(prev.end) || prevStart;
        const prevSpeaker = prev.speaker || 'SPEAKER_00';

        if (prevSpeaker !== currentSpeaker) continue;

        const overlapStart = Math.max(currentStart, prevStart);
        const overlapEnd = Math.min(currentEnd, prevEnd);
        const overlapDuration = overlapEnd - overlapStart;

        // ФАЗА 5: Поріг 0.3s замість 0.1s (300ms)
        const MIN_TEMPORAL_OVERLAP = 0.3;

        if (overlapDuration > MIN_TEMPORAL_OVERLAP) {
          // Перевіряємо text similarity перед видаленням
          const textSimilar = areTextsSimilar(prev.text, current.text, {
            minLevenshteinSim: 0.85,
            minJaccardSim: 0.75
          });

          const isContinuation = currentStart < prevEnd && currentEnd > prevEnd;
          const isFullyContained = currentStart >= prevStart && currentEnd <= prevEnd;

          if (isContinuation && !isFullyContained) {
            prev.end = Math.max(prevEnd, currentEnd);
            prev.text = (prev.text + ' ' + (current.text || '')).trim();
            if (prev.words && current.words) {
              prev.words = [...(prev.words || []), ...(current.words || [])];
            } else if (current.words) {
              prev.words = current.words;
            }
            wasMerged = true;
            logger.log(
              `✅ Merged continuation segment "${(current.text || '').substring(0, 30)}..." (${currentStart.toFixed(2)}-${currentEnd.toFixed(2)}) with previous`
            );
            break;
          }

          // Дійсно дублікат - видаляємо
          if ((isFullyContained || overlapDuration > 0.3) && textSimilar) {
            shouldSkip = true;
            logger.log(
              `⚠️ Skipping segment "${(current.text || '').substring(0, 30)}..." (${currentStart.toFixed(2)}-${currentEnd.toFixed(2)}) - overlaps with previous segment from same speaker`
            );
            break;
          }
        }
      }

      if (!wasMerged && !shouldSkip) {
        chronologicallyValid.push(current);
      }
    }

    logger.log(`✅ Chronological validation: ${mergedConsecutive.length} → ${chronologicallyValid.length} segments`);

    const sortedValid = chronologicallyValid;
    const priorityScore = (segment) => {
      const textLength = (segment.text || '').length;
      let base = 1;
      if (segment.source === 'voice-track') base = 5;
      else if (segment.source === 'voice-enhanced') base = 4;
      else if (segment.source === 'primary') base = 3;
      else if (segment.source === 'llm-refined') base = 2;
      return base + Math.min(textLength / 200, 1);
    };

    // ФАЗА 4: Покращена фінальна дедуплікація
    function isFinalDuplicate(seg1, seg2) {
      const s1Start = parseFloat(seg1.start) || 0;
      const s1End = parseFloat(seg1.end) || s1Start;
      const s2Start = parseFloat(seg2.start) || 0;
      const s2End = parseFloat(seg2.end) || s2Start;

      const overlapStart = Math.max(s1Start, s2Start);
      const overlapEnd = Math.min(s1End, s2End);
      const overlapDuration = Math.max(0, overlapEnd - overlapStart);

      if (overlapDuration <= 0) return false;

      // Відсоток перекриття від кожного сегменту
      const s1Duration = s1End - s1Start;
      const s2Duration = s2End - s2Start;
      const overlap1 = s1Duration > 0 ? overlapDuration / s1Duration : 0;
      const overlap2 = s2Duration > 0 ? overlapDuration / s2Duration : 0;

      // Поріг 1: Матеріальне перекриття (>50% замість 60%)
      const significantOverlap = (overlap1 > 0.5 || overlap2 > 0.5);

      // Поріг 2: Текст схожий
      const textSimilar = areTextsSimilar(seg1.text, seg2.text, {
        minLevenshteinSim: 0.85,
        minJaccardSim: 0.75
      });

      if (significantOverlap && textSimilar) {
        return true;
      }

      // Поріг 3: Один текст міститься в іншому часово
      if (s1Start >= s2Start && s1End <= s2End) {
        // seg1 повністю міститься в seg2 часово
        const textMatch = areTextsSimilar(seg1.text, seg2.text, {
          minLevenshteinSim: 0.80,
          minJaccardSim: 0.70
        });

        if (textMatch) {
          return true;
        }
      }

      return false;
    }

    const finalSegments = [];
    sortedValid.forEach(segment => {
      const segStart = parseFloat(segment.start) || 0;
      const segEnd = parseFloat(segment.end) || segStart;
      const segSpeaker = segment.speaker || 'SPEAKER_00';
      const segText = (segment.text || '').trim();
      if (!segText) return;

      let handled = false;
      for (let i = 0; i < finalSegments.length; i++) {
        const existing = finalSegments[i];
        if ((existing.speaker || 'SPEAKER_00') !== segSpeaker) continue;

        if (isFinalDuplicate(segment, existing)) {
          handled = true;

          // Залишаємо сегмент з БІЛЬШИМ текстом або КРАЩОЮ якістю
          // Пріоритет: voice-track > primary
          const currentIsVoiceTrack = existing.source === 'voice-track';
          const newIsVoiceTrack = segment.source === 'voice-track';

          if (newIsVoiceTrack && !currentIsVoiceTrack) {
            // Замінюємо primary на voice-track
            finalSegments[i] = segment;
          } else if (newIsVoiceTrack === currentIsVoiceTrack && segText.length > (existing.text || '').length) {
            // Обидва одного типу - залишаємо довший
            finalSegments[i] = segment;
          } else {
            // Перевіряємо priority score
            const currentScore = priorityScore(segment);
            const existingScore = priorityScore(existing);
            if (currentScore > existingScore || (currentScore === existingScore && segText.length > (existing.text || '').length)) {
              finalSegments[i] = segment;
            }
          }
          break;
        }
      }

      if (!handled) {
        finalSegments.push(segment);
      }
    });

    logger.log(`✅ Final deduplication: ${sortedValid.length} → ${finalSegments.length} segments`);

    const finalBySpeaker = {};
    finalSegments.forEach(seg => {
      const spk = seg.speaker || 'UNKNOWN';
      if (!finalBySpeaker[spk]) finalBySpeaker[spk] = [];
      finalBySpeaker[spk].push(seg);
    });
    logger.log(
      '✅ Final merged segments by speaker:',
      Object.keys(finalBySpeaker).map(s => `${s}: ${finalBySpeaker[s].length}`).join(', ')
    );

    return finalSegments;
  }

  return {
    areTextsSimilar,
    collectVoiceTrackSegments,
    mergeVoiceTrackSegments
  };
}));

