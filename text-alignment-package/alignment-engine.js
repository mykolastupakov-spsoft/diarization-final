/**
 * SegmentAlignmentEngine.js
 * 
 * Багатошарова система для вирівнювання та класифікації текстових сегментів
 * з різних джерел (general, speaker1, speaker2, markdown)
 */

class SegmentAlignmentEngine {
  constructor(config = {}) {
    this.config = {
      exactThreshold: 0.98,
      temporalThreshold: 0.70,
      semanticThreshold: 0.55,
      partialThreshold: 0.40,
      timeTolerance: {
        exact: 0.5,
        temporal: 2.0,
        semantic: 4.0
      },
      minPhraseLength: 1,
      maxPhraseLength: 500,
      enableShortPhraseOptimization: true,
      metrics: {
        levenshtein: 0.4,
        jaccard: 0.3,
        sequence: 0.2,
        lcs: 0.1
      },
      ...config
    };
    this.cache = new Map();
  }

  alignSegment(querySegment, candidateSegments, sourceType = 'general') {
    if (!querySegment || !candidateSegments || candidateSegments.length === 0) {
      return { found: false, confidence: 0, layer: 'NONE' };
    }

    const alignmentResults = candidateSegments
      .map(candidate => ({
        candidate,
        result: this.computeAlignment(querySegment, candidate, sourceType)
      }))
      .sort((a, b) => b.result.confidence - a.result.confidence);

    return alignmentResults[0]?.result || { found: false, confidence: 0, layer: 'NONE' };
  }

  computeAlignment(querySegment, candidate, sourceType) {
    if (!this.isValidSegment(querySegment) || !this.isValidSegment(candidate)) {
      return { found: false, confidence: 0, layer: 'INVALID' };
    }

    const exactMatch = this.checkExactMatch(querySegment, candidate);
    if (exactMatch.found) {
      return {
        found: true,
        confidence: exactMatch.confidence,
        layer: 'EXACT',
        similarity: exactMatch.similarity,
        timeMatch: exactMatch.timeMatch,
        details: exactMatch
      };
    }

    const temporalMatch = this.checkTemporalMatch(querySegment, candidate, sourceType);
    if (temporalMatch.found) {
      return {
        found: true,
        confidence: temporalMatch.confidence,
        layer: 'TEMPORAL',
        similarity: temporalMatch.similarity,
        timeMatch: temporalMatch.timeMatch,
        details: temporalMatch
      };
    }

    const semanticMatch = this.checkSemanticMatch(querySegment, candidate, sourceType);
    if (semanticMatch.found) {
      return {
        found: true,
        confidence: semanticMatch.confidence,
        layer: 'SEMANTIC',
        similarity: semanticMatch.similarity,
        timeMatch: semanticMatch.timeMatch,
        details: semanticMatch
      };
    }

    const partialMatch = this.checkPartialMatch(querySegment, candidate);
    if (partialMatch.found) {
      return {
        found: true,
        confidence: partialMatch.confidence,
        layer: 'PARTIAL',
        similarity: partialMatch.similarity,
        timeMatch: partialMatch.timeMatch,
        details: partialMatch
      };
    }

    return {
      found: false,
      confidence: 0,
      layer: 'NONE',
      similarity: 0,
      timeMatch: false
    };
  }

  checkExactMatch(querySegment, candidate) {
    const queryNorm = this.normalizeText(querySegment.text);
    const candidateNorm = this.normalizeText(candidate.text);

    if (queryNorm !== candidateNorm) {
      return { found: false };
    }

    const timeMatch = this.checkTimeOverlap(querySegment, candidate, this.config.timeTolerance.exact);

    return {
      found: true,
      confidence: timeMatch.overlap ? 0.99 : 0.95,
      similarity: 1.0,
      timeMatch: timeMatch.overlap,
      timeDistance: timeMatch.distance
    };
  }

  checkTemporalMatch(querySegment, candidate, sourceType) {
    const textSimilarity = this.computeAdvancedSimilarity(querySegment.text, candidate.text);

    if (textSimilarity < this.config.temporalThreshold) {
      return { found: false };
    }

    const timeMatch = this.checkTimeOverlap(querySegment, candidate, this.config.timeTolerance.temporal);

    if (!timeMatch.overlap && sourceType !== 'markdown') {
      return { found: false };
    }

    return {
      found: true,
      confidence: textSimilarity * (timeMatch.overlap ? 0.95 : 0.80),
      similarity: textSimilarity,
      timeMatch: timeMatch.overlap,
      timeDistance: timeMatch.distance
    };
  }

  checkSemanticMatch(querySegment, candidate, sourceType) {
    const textSimilarity = this.computeAdvancedSimilarity(querySegment.text, candidate.text);

    if (textSimilarity < this.config.semanticThreshold) {
      return { found: false };
    }

    const timeMatch = this.checkTimeProximity(querySegment, candidate, this.config.timeTolerance.semantic);

    return {
      found: true,
      confidence: textSimilarity * (timeMatch.proximity ? 0.75 : 0.60),
      similarity: textSimilarity,
      timeMatch: timeMatch.proximity,
      timeDistance: timeMatch.distance
    };
  }

  checkPartialMatch(querySegment, candidate) {
    const substringMatch = this.checkSubstringMatch(querySegment.text, candidate.text);

    if (substringMatch.found) {
      return {
        found: substringMatch.similarity >= this.config.partialThreshold,
        confidence: substringMatch.similarity * 0.65,
        similarity: substringMatch.similarity,
        timeMatch: false,
        matchType: 'substring'
      };
    }

    const wordMatch = this.checkWordLevelMatch(querySegment.text, candidate.text);

    return {
      found: wordMatch.similarity >= this.config.partialThreshold,
      confidence: wordMatch.similarity * 0.55,
      similarity: wordMatch.similarity,
      timeMatch: false,
      matchType: 'word-level'
    };
  }

  normalizeText(text) {
    if (!text || typeof text !== 'string') return '';

    return text
      .replace(/['\`']/g, "'")
      .replace(/don't/gi, 'do not')
      .replace(/can't/gi, 'can not')
      .replace(/won't/gi, 'will not')
      .replace(/i'm/gi, 'i am')
      .replace(/i've/gi, 'i have')
      .replace(/i'll/gi, 'i will')
      .replace(/i'd/gi, 'i would')
      .replace(/it's/gi, 'it is')
      .replace(/that's/gi, 'that is')
      .replace(/you're/gi, 'you are')
      .replace(/we're/gi, 'we are')
      .replace(/they're/gi, 'they are')
      .replace(/isn't/gi, 'is not')
      .replace(/aren't/gi, 'are not')
      .replace(/wasn't/gi, 'was not')
      .replace(/weren't/gi, 'were not')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
  }

  computeAdvancedSimilarity(text1, text2) {
    const cacheKey = \`\${text1}|\${text2}\`;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const norm1 = this.normalizeText(text1);
    const norm2 = this.normalizeText(text2);

    if (norm1 === norm2) return 1.0;
    if (!norm1 || !norm2) return 0.0;

    const levenshteinSim = this.computeLevenshteinSimilarity(norm1, norm2);
    const jaccardSim = this.computeJaccardSimilarity(norm1, norm2);
    const sequenceSim = this.computeSequenceSimilarity(norm1, norm2);
    const lcsSim = this.computeLCSSimilarity(norm1, norm2);

    const combined =
      levenshteinSim * this.config.metrics.levenshtein +
      jaccardSim * this.config.metrics.jaccard +
      sequenceSim * this.config.metrics.sequence +
      lcsSim * this.config.metrics.lcs;

    const result = Math.min(1.0, Math.max(0.0, combined));
    this.cache.set(cacheKey, result);
    return result;
  }

  computeLevenshteinSimilarity(s1, s2) {
    const maxLen = Math.max(s1.length, s2.length);
    if (maxLen === 0) return 1.0;
    const distance = this.levenshteinDistance(s1, s2);
    return 1.0 - distance / maxLen;
  }

  levenshteinDistance(s1, s2) {
    const matrix = Array(s2.length + 1)
      .fill(null)
      .map(() => Array(s1.length + 1).fill(0));

    for (let i = 0; i <= s1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= s2.length; j++) matrix[j][0] = j;

    for (let j = 1; j <= s2.length; j++) {
      for (let i = 1; i <= s1.length; i++) {
        const indicator = s1[i - 1] === s2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        );
      }
    }

    return matrix[s2.length][s1.length];
  }

  computeJaccardSimilarity(s1, s2) {
    const words1 = new Set(s1.split(/\s+/));
    const words2 = new Set(s2.split(/\s+/));
    const intersection = new Set([...words1].filter(w => words2.has(w)));
    const union = new Set([...words1, ...words2]);
    return union.size === 0 ? 0 : intersection.size / union.size;
  }

  computeSequenceSimilarity(s1, s2) {
    const words1 = s1.split(/\s+/);
    const words2 = s2.split(/\s+/);
    let matches = 0;
    let j = 0;

    for (let i = 0; i < words1.length && j < words2.length; i++) {
      if (words1[i] === words2[j] || this.levenshteinDistance(words1[i], words2[j]) <= 1) {
        matches++;
        j++;
      }
    }

    const maxWords = Math.max(words1.length, words2.length);
    return maxWords === 0 ? 0 : matches / maxWords;
  }

  computeLCSSimilarity(s1, s2) {
    const lcsLength = this.longestCommonSubsequence(s1, s2);
    const maxLen = Math.max(s1.length, s2.length);
    return maxLen === 0 ? 0 : lcsLength / maxLen;
  }

  longestCommonSubsequence(s1, s2) {
    const matrix = Array(s1.length + 1)
      .fill(null)
      .map(() => Array(s2.length + 1).fill(0));

    for (let i = 1; i <= s1.length; i++) {
      for (let j = 1; j <= s2.length; j++) {
        if (s1[i - 1] === s2[j - 1]) {
          matrix[i][j] = matrix[i - 1][j - 1] + 1;
        } else {
          matrix[i][j] = Math.max(matrix[i - 1][j], matrix[i][j - 1]);
        }
      }
    }

    return matrix[s1.length][s2.length];
  }

  checkSubstringMatch(text1, text2) {
    const norm1 = this.normalizeText(text1);
    const norm2 = this.normalizeText(text2);

    if (norm1.length === 0 || norm2.length === 0) {
      return { found: false };
    }

    const containsFull = norm2.includes(norm1) || norm1.includes(norm2);
    if (containsFull) {
      const similarity = Math.min(norm1.length, norm2.length) / Math.max(norm1.length, norm2.length);
      return { found: true, similarity };
    }

    return { found: false };
  }

  checkWordLevelMatch(text1, text2) {
    const norm1 = this.normalizeText(text1);
    const norm2 = this.normalizeText(text2);

    const words1 = norm1.split(/\s+/);
    const words2 = norm2.split(/\s+/);

    let matches = 0;

    for (const word1 of words1) {
      for (const word2 of words2) {
        if (word1 === word2 || this.levenshteinDistance(word1, word2) <= 1) {
          matches++;
          break;
        }
      }
    }

    const similarity = matches / Math.max(words1.length, words2.length);
    return { similarity, matches };
  }

  checkTimeOverlap(segment1, segment2, tolerance) {
    const s1Start = segment1.start || 0;
    const s1End = segment1.end || 0;
    const s2Start = segment2.start || 0;
    const s2End = segment2.end || 0;

    const overlap = s1Start <= s2End + tolerance && s2Start <= s1End + tolerance;

    let distance;
    if (overlap) {
      distance = 0;
    } else if (s1End < s2Start) {
      distance = s2Start - s1End;
    } else {
      distance = s1Start - s2End;
    }

    return { overlap: distance <= tolerance, distance };
  }

  checkTimeProximity(segment1, segment2, tolerance) {
    const midpoint1 = (segment1.start + segment1.end) / 2;
    const midpoint2 = (segment2.start + segment2.end) / 2;
    const distance = Math.abs(midpoint1 - midpoint2);
    return { proximity: distance <= tolerance, distance };
  }

  isValidSegment(segment) {
    return (
      segment &&
      typeof segment === 'object' &&
      typeof segment.text === 'string' &&
      typeof segment.start === 'number' &&
      typeof segment.end === 'number' &&
      segment.text.length > 0
    );
  }

  clearCache() {
    this.cache.clear();
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = SegmentAlignmentEngine;
}
