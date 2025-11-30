/**
 * Diarization Metrics Calculation Engine
 * Implements DER, JER, and comprehensive speaker diarization evaluation metrics
 */

class DiarizationMetrics {
  constructor(reference, hypothesis, collar = 0.0) {
    this.reference = this.parseTimeline(reference);
    this.hypothesis = this.parseTimeline(hypothesis);
    this.collar = collar; // Forgiveness collar in seconds
    this.speakerMapping = null;
  }

  /**
   * Parse timeline segments into normalized format
   */
  parseTimeline(segments) {
    if (!segments || segments.length === 0) return [];
    
    return segments.map(seg => ({
      start: parseFloat(seg.start),
      end: parseFloat(seg.end),
      speaker: seg.speaker.toString()
    })).sort((a, b) => a.start - b.start);
  }

  /**
   * Get speakers present at a specific time point
   */
  getSpeakersAt(timeline, time) {
    const speakers = new Set();
    for (const segment of timeline) {
      if (segment.start <= time && time < segment.end) {
        speakers.add(segment.speaker);
      }
    }
    return Array.from(speakers);
  }

  /**
   * Get unique speakers from timeline
   */
  getUniqueSpeakers(timeline) {
    const speakers = new Set();
    timeline.forEach(seg => speakers.add(seg.speaker));
    return Array.from(speakers);
  }

  /**
   * Get maximum time from both reference and hypothesis timelines
   */
  getMaxTime() {
    return Math.max(
      ...this.reference.map(s => s.end),
      ...this.hypothesis.map(s => s.end)
    );
  }

  /**
   * Align speaker labels using Hungarian algorithm (simplified greedy version)
   */
  alignSpeakers() {
    const refSpeakers = this.getUniqueSpeakers(this.reference);
    const hypSpeakers = this.getUniqueSpeakers(this.hypothesis);
    
    // Create confusion matrix
    const matrix = {};
    refSpeakers.forEach(ref => {
      matrix[ref] = {};
      hypSpeakers.forEach(hyp => {
        matrix[ref][hyp] = 0;
      });
    });

    // Calculate overlap time for each ref-hyp pair
    const resolution = 0.01; // 10ms
    const maxTime = this.getMaxTime();

    for (let t = 0; t < maxTime; t += resolution) {
      const refSpeakersAtT = this.getSpeakersAt(this.reference, t);
      const hypSpeakersAtT = this.getSpeakersAt(this.hypothesis, t);
      
      refSpeakersAtT.forEach(ref => {
        hypSpeakersAtT.forEach(hyp => {
          matrix[ref][hyp] += resolution;
        });
      });
    }

    // Greedy assignment: for each ref speaker, find best matching hyp speaker
    const mapping = {};
    const usedHyp = new Set();
    
    refSpeakers.forEach(ref => {
      let bestHyp = null;
      let bestScore = 0;
      
      hypSpeakers.forEach(hyp => {
        if (!usedHyp.has(hyp) && matrix[ref][hyp] > bestScore) {
          bestScore = matrix[ref][hyp];
          bestHyp = hyp;
        }
      });
      
      if (bestHyp) {
        mapping[bestHyp] = ref;
        usedHyp.add(bestHyp);
      }
    });

    this.speakerMapping = mapping;
    return mapping;
  }

  /**
   * Map hypothesis speaker to reference speaker
   */
  mapSpeaker(hypSpeaker) {
    if (!this.speakerMapping) {
      this.alignSpeakers();
    }
    return this.speakerMapping[hypSpeaker] || hypSpeaker;
  }

  /**
   * Calculate Diarization Error Rate (DER)
   * DER = (FA + MISS + CONF) / TOTAL
   */
  calculateDER() {
    const resolution = 0.01; // 10ms resolution
    let fa = 0, miss = 0, conf = 0, total = 0;
    
    const maxTime = this.getMaxTime();

    // Align speakers first
    this.alignSpeakers();

    for (let t = 0; t < maxTime; t += resolution) {
      const refSpeakers = this.getSpeakersAt(this.reference, t);
      const hypSpeakers = this.getSpeakersAt(this.hypothesis, t);
      
      if (refSpeakers.length > 0) {
        total += resolution;
        
        if (hypSpeakers.length === 0) {
          // Missed detection
          miss += resolution;
        } else {
          // Map hypothesis speakers to reference
          const mappedHyp = hypSpeakers.map(h => this.mapSpeaker(h));
          
          // Check if speakers match
          const refSet = new Set(refSpeakers);
          const hypSet = new Set(mappedHyp);
          
          // Count mismatches as confusion
          let matches = 0;
          mappedHyp.forEach(h => {
            if (refSet.has(h)) matches++;
          });
          
          if (matches === 0 || matches < Math.max(refSpeakers.length, mappedHyp.length)) {
            conf += resolution;
          }
        }
      } else if (hypSpeakers.length > 0) {
        // False alarm
        fa += resolution;
      }
    }

    const der = total > 0 ? (fa + miss + conf) / total : 0;
    
    return {
      der: der,
      derPercent: (der * 100).toFixed(2),
      fa: fa,
      faPercent: ((fa / total) * 100).toFixed(2),
      miss: miss,
      missPercent: ((miss / total) * 100).toFixed(2),
      conf: conf,
      confPercent: ((conf / total) * 100).toFixed(2),
      total: total
    };
  }

  /**
   * Calculate Jaccard Error Rate (JER)
   */
  calculateJER() {
    const refSpeakers = this.getUniqueSpeakers(this.reference);
    let totalJaccard = 0;
    
    this.alignSpeakers();

    refSpeakers.forEach(refSpeaker => {
      // Get all time points where reference speaker is active
      const refTime = new Set();
      this.reference.forEach(seg => {
        if (seg.speaker === refSpeaker) {
          for (let t = seg.start; t < seg.end; t += 0.01) {
            refTime.add(t.toFixed(2));
          }
        }
      });

      // Find best matching hypothesis speaker
      let bestJaccard = 0;
      const hypSpeakers = this.getUniqueSpeakers(this.hypothesis);
      
      hypSpeakers.forEach(hypSpeaker => {
        if (this.mapSpeaker(hypSpeaker) === refSpeaker) {
          const hypTime = new Set();
          this.hypothesis.forEach(seg => {
            if (seg.speaker === hypSpeaker) {
              for (let t = seg.start; t < seg.end; t += 0.01) {
                hypTime.add(t.toFixed(2));
              }
            }
          });

          // Calculate Jaccard index
          const intersection = new Set([...refTime].filter(t => hypTime.has(t)));
          const union = new Set([...refTime, ...hypTime]);
          const jaccard = union.size > 0 ? intersection.size / union.size : 0;
          
          bestJaccard = Math.max(bestJaccard, jaccard);
        }
      });

      totalJaccard += bestJaccard;
    });

    const avgJaccard = refSpeakers.length > 0 ? totalJaccard / refSpeakers.length : 0;
    const jer = 1 - avgJaccard;

    return {
      jer: jer,
      jerPercent: (jer * 100).toFixed(2),
      avgJaccard: avgJaccard.toFixed(3)
    };
  }

  /**
   * Calculate per-speaker metrics (Precision, Recall, F1)
   */
  calculatePerSpeakerMetrics() {
    const refSpeakers = this.getUniqueSpeakers(this.reference);
    const metrics = {};
    
    this.alignSpeakers();

    refSpeakers.forEach(speaker => {
      let tp = 0, fp = 0, fn = 0;
      const resolution = 0.01;
      
      const maxTime = this.getMaxTime();

      for (let t = 0; t < maxTime; t += resolution) {
        const inRef = this.getSpeakersAt(this.reference, t).includes(speaker);
        const hypSpeakers = this.getSpeakersAt(this.hypothesis, t);
        const mappedHyp = hypSpeakers.map(h => this.mapSpeaker(h));
        const inHyp = mappedHyp.includes(speaker);

        if (inRef && inHyp) {
          tp += resolution;
        } else if (inHyp && !inRef) {
          fp += resolution;
        } else if (inRef && !inHyp) {
          fn += resolution;
        }
      }

      const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
      const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
      const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

      // Calculate speaking time percentage
      let speakingTime = 0;
      this.reference.forEach(seg => {
        if (seg.speaker === speaker) {
          speakingTime += seg.end - seg.start;
        }
      });
      const totalTime = Math.max(...this.reference.map(s => s.end));
      const speakingPercent = (speakingTime / totalTime) * 100;

      metrics[speaker] = {
        precision: (precision * 100).toFixed(1),
        recall: (recall * 100).toFixed(1),
        f1: (f1 * 100).toFixed(1),
        speakingTime: speakingTime.toFixed(1),
        speakingPercent: speakingPercent.toFixed(1)
      };
    });

    return metrics;
  }

  /**
   * Identify error segments
   */
  identifyErrors() {
    const errors = [];
    const resolution = 0.1; // 100ms for error detection
    
    this.alignSpeakers();

    const maxTime = this.getMaxTime();

    let errorStart = null;
    let errorType = null;

    for (let t = 0; t < maxTime; t += resolution) {
      const refSpeakers = this.getSpeakersAt(this.reference, t);
      const hypSpeakers = this.getSpeakersAt(this.hypothesis, t);
      const mappedHyp = hypSpeakers.map(h => this.mapSpeaker(h));

      let currentError = null;

      if (refSpeakers.length > 0 && hypSpeakers.length === 0) {
        currentError = 'miss';
      } else if (refSpeakers.length === 0 && hypSpeakers.length > 0) {
        currentError = 'fa';
      } else if (refSpeakers.length > 0 && hypSpeakers.length > 0) {
        const refSet = new Set(refSpeakers);
        const hypSet = new Set(mappedHyp);
        let matches = 0;
        mappedHyp.forEach(h => {
          if (refSet.has(h)) matches++;
        });
        if (matches === 0 || matches < Math.max(refSpeakers.length, mappedHyp.length)) {
          currentError = 'conf';
        }
      }

      if (currentError && currentError !== errorType) {
        // Start new error segment
        if (errorStart !== null) {
          errors.push({
            start: errorStart,
            end: t,
            type: errorType,
            duration: t - errorStart
          });
        }
        errorStart = t;
        errorType = currentError;
      } else if (!currentError && errorStart !== null) {
        // End current error segment
        errors.push({
          start: errorStart,
          end: t,
          type: errorType,
          duration: t - errorStart
        });
        errorStart = null;
        errorType = null;
      }
    }

    // Close last error if any
    if (errorStart !== null) {
      errors.push({
        start: errorStart,
        end: maxTime,
        type: errorType,
        duration: maxTime - errorStart
      });
    }

    return errors;
  }

  /**
   * Calculate speaker count accuracy
   */
  calculateSpeakerCountAccuracy() {
    const refCount = this.getUniqueSpeakers(this.reference).length;
    const hypCount = this.getUniqueSpeakers(this.hypothesis).length;
    
    return {
      reference: refCount,
      hypothesis: hypCount,
      correct: refCount === hypCount,
      difference: hypCount - refCount
    };
  }

  /**
   * Calculate comprehensive metrics
   */
  calculateAllMetrics() {
    const der = this.calculateDER();
    const jer = this.calculateJER();
    const perSpeaker = this.calculatePerSpeakerMetrics();
    const speakerCount = this.calculateSpeakerCountAccuracy();
    const errors = this.identifyErrors();

    // Categorize errors
    const errorCategories = {
      fa: errors.filter(e => e.type === 'fa'),
      miss: errors.filter(e => e.type === 'miss'),
      conf: errors.filter(e => e.type === 'conf')
    };

    return {
      der,
      jer,
      perSpeaker,
      speakerCount,
      errors,
      errorCategories,
      summary: {
        totalErrors: errors.length,
        falseAlarms: errorCategories.fa.length,
        missedDetections: errorCategories.miss.length,
        speakerConfusions: errorCategories.conf.length
      }
    };
  }
}

/**
 * Calculate agreement between multiple services
 */
class CrossServiceMetrics {
  constructor(servicesResults) {
    this.services = servicesResults; // Array of {id, name, segments}
  }

  /**
   * Calculate pairwise agreement between services
   */
  calculateAgreementMatrix() {
    const matrix = {};
    
    this.services.forEach(serviceA => {
      matrix[serviceA.id] = {};
      
      this.services.forEach(serviceB => {
        if (serviceA.id === serviceB.id) {
          matrix[serviceA.id][serviceB.id] = 100.0;
        } else {
          // Calculate agreement as similarity percentage
          const metrics = new DiarizationMetrics(serviceA.segments, serviceB.segments);
          const der = metrics.calculateDER();
          const agreement = Math.max(0, (1 - der.der) * 100);
          matrix[serviceA.id][serviceB.id] = agreement.toFixed(1);
        }
      });
    });

    return matrix;
  }

  /**
   * Find consensus segments where all/most services agree
   */
  findConsensusSegments() {
    // Implementation for consensus analysis
    // For demo purposes, return mock data
    return {
      fullConsensus: 78.3,
      majorityConsensus: 15.2,
      splitDecision: 6.5
    };
  }
}