#!/usr/bin/env python3
"""
Two-Stage Cascading Diarization System
Optimizes for speed and cost using 1B model for bulk work, escalates to 20B for complex segments.
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class DiarizationSegment:
    """Structured representation of a diarization segment"""
    speaker: str  # 'Agent' or 'Client'
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = None
    needs_escalation: bool = False
    escalation_reason: Optional[str] = None
    final_decision_basis: Optional[str] = None
    original_speaker: Optional[str] = None  # Before escalation correction


class CascadingDiarizationController:
    """
    Main controller for two-stage cascading diarization system.
    """
    
    def __init__(self, 
                 fast_model_func,
                 smart_model_func,
                 agent_context: str = "Customer service representative",
                 client_context: str = "Customer seeking help"):
        """
        Args:
            fast_model_func: Function to call 1B model (fast_model_func(prompt) -> str)
            smart_model_func: Function to call 20B model (smart_model_func(prompt) -> str)
            agent_context: Description of Agent role
            client_context: Description of Client role
        """
        self.fast_model = fast_model_func
        self.smart_model = smart_model_func
        self.agent_context = agent_context
        self.client_context = client_context
        
        # Role patterns for Veto Algorithm
        self.agent_patterns = {
            'positive': [
                'can help', 'would you like', 'let me', 'i can', 'i will',
                'please', 'thank you', 'how may i', 'what can i do',
                'i understand', 'i see', 'let me check', 'i\'ll', 'sure',
                'absolutely', 'certainly', 'of course', 'no problem'
            ],
            'negative': [
                'i have a problem', 'i can\'t', 'it doesn\'t work',
                'i\'m frustrated', 'this is terrible', 'i need help',
                'i don\'t understand', 'why', 'what\'s wrong'
            ]
        }
        
        self.client_patterns = {
            'positive': [
                'i have a problem', 'i need', 'i can\'t', 'it doesn\'t work',
                'help me', 'i don\'t understand', 'why', 'what\'s wrong',
                'i\'m frustrated', 'this is terrible', 'i want to',
                'can you', 'could you', 'would you'
            ],
            'negative': [
                'i can help', 'let me', 'i will', 'i can', 'i understand',
                'let me check', 'i\'ll', 'sure', 'absolutely', 'certainly'
            ]
        }
    
    def get_level1_prompt(self, conversation_chunk: str) -> str:
        """
        Level 1 Prompt (Speed & Efficiency) for 1B Model.
        """
        return f"""You are a fast diarization tool. There are exactly 2 speakers: Agent and Client.

Input conversation chunk:
{conversation_chunk}

Instructions:
- Identify who is speaking: Agent or Client
- Return ONLY a valid JSON array
- Do not reason, just output
- Each utterance should be a separate entry

Output format (JSON array):
[
  {{"speaker": "Agent", "text": "..."}},
  {{"speaker": "Client", "text": "..."}},
  ...
]

Return the JSON array now:"""
    
    def get_level2_prompt(self, 
                         problematic_segment: str,
                         context_summary: str = "",
                         previous_segments: List[DiarizationSegment] = None) -> str:
        """
        Level 2 Prompt (Quality & Self-Correction) for 20B Model.
        Uses TRIZ Principle 24 - Feedback with Veto Algorithm.
        """
        previous_context = ""
        if previous_segments:
            recent = previous_segments[-3:]  # Last 3 segments for context
            previous_context = "\n".join([
                f"- {seg.speaker}: {seg.text[:100]}"
                for seg in recent
            ])
        
        return f"""You are an expert dialogue analyst with advanced reasoning capabilities.

CONTEXT:
- Agent Role: {self.agent_context}. Professional, offers solutions, asks process questions, uses formal language.
- Client Role: {self.client_context}. Describes problems, may be emotional, asks about product/service, uses casual language.

PROBLEMATIC SEGMENT TO ANALYZE:
{problematic_segment}

PREVIOUS CONTEXT (last 3 utterances):
{previous_context if previous_context else "None"}

CONVERSATION TOPIC: {context_summary if context_summary else "General customer service"}

CRITICAL VETO ALGORITHM - Follow these steps:

1. PREDICT: Based on the text content, predict who is speaking (Agent or Client).

2. VERIFY: Check if the content matches the Role Template:
   - Agent should: offer help, ask process questions, use professional language, provide solutions
   - Client should: describe problems, express needs, ask for help, may be emotional

3. VETO CHECK: If there's a mismatch (e.g., Agent is crying about a problem, or Client is offering professional help), 
   you MUST swap the label.

4. DECISION BASIS: Explain your reasoning in 1-2 sentences.

Output format (STRICT JSON):
{{
  "speaker": "Agent" or "Client",
  "text": "exact text from segment",
  "confidence": 0.0-1.0,
  "decision_basis": "your reasoning explanation",
  "role_match_score": 0.0-1.0,
  "veto_applied": true/false,
  "original_prediction": "Agent" or "Client" (before veto)
}}

Return ONLY the JSON object, no additional text:"""
    
    def chunk_text(self, full_text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Split text into logical chunks.
        Prefers splitting by newlines, falls back to sentence boundaries.
        """
        # First, try splitting by double newlines (paragraph breaks)
        if '\n\n' in full_text:
            chunks = [c.strip() for c in full_text.split('\n\n') if c.strip()]
            if all(len(c) <= max_chunk_size for c in chunks):
                return chunks
        
        # Split by single newlines
        if '\n' in full_text:
            chunks = []
            current_chunk = ""
            for line in full_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                if len(current_chunk) + len(line) + 1 <= max_chunk_size:
                    current_chunk += (line + "\n") if current_chunk else line
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = line
            if current_chunk:
                chunks.append(current_chunk.strip())
            return chunks
        
        # Fallback: split by sentences
        sentences = re.split(r'[.!?]+\s+', full_text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += (". " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [full_text]
    
    def evaluate_fast_model_output(self, response: str, original_text: str) -> Tuple[bool, Optional[str]]:
        """
        Evaluate Stage 1 output for quality issues.
        Returns: (needs_escalation: bool, reason: str or None)
        """
        # Check 1: Valid JSON
        try:
            data = json.loads(response)
            if not isinstance(data, list):
                return True, "Output is not a JSON array"
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if not json_match:
                return True, "No valid JSON found in response"
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                return True, "Invalid JSON format"
        
        # Check 2: Valid speaker labels
        valid_speakers = {'Agent', 'Client'}
        for item in data:
            if not isinstance(item, dict):
                return True, "Array contains non-dict items"
            speaker = item.get('speaker', '').strip()
            if speaker not in valid_speakers:
                return True, f"Invalid speaker label: {speaker}"
        
        # Check 3: Heuristic checks
        for item in data:
            text = item.get('text', '').strip()
            
            # Very short segments
            if len(text.split()) < 3:
                return True, f"Very short segment detected: '{text[:50]}'"
            
            # Overlapping markers
            overlap_markers = ['[unintelligible]', '[overlap]', '[crosstalk]', '[inaudible]', '...']
            if any(marker in text.lower() for marker in overlap_markers):
                return True, f"Overlap marker detected in: '{text[:50]}'"
            
            # Ambiguous patterns
            ambiguous_patterns = ['uh', 'um', 'er', 'ah', 'hmm']
            if text.lower().strip() in ambiguous_patterns:
                return True, f"Ambiguous filler word: '{text}'"
        
        # Check 4: All segments from same speaker (might be wrong)
        if len(data) > 1:
            speakers = [item.get('speaker', '').strip() for item in data]
            if len(set(speakers)) == 1:
                # Check if text suggests alternation
                text_lower = original_text.lower()
                question_indicators = ['?', 'did you', 'can you', 'will you', 'do you']
                answer_indicators = ['yes', 'no', 'sure', 'okay', 'i tried', 'i have']
                has_questions = any(ind in text_lower for ind in question_indicators)
                has_answers = any(ind in text_lower for ind in answer_indicators)
                if has_questions and has_answers:
                    return True, "Question-answer pattern detected but same speaker assigned"
        
        return False, None  # No escalation needed
    
    def apply_veto_algorithm(self, segment: Dict, context: List[DiarizationSegment] = None) -> Dict:
        """
        Apply Veto Algorithm to verify and correct speaker assignment.
        """
        text = segment.get('text', '').lower()
        predicted_speaker = segment.get('speaker', '').strip()
        
        # Score for Agent patterns
        agent_score = 0
        for pattern in self.agent_patterns['positive']:
            if pattern in text:
                agent_score += 1
        for pattern in self.agent_patterns['negative']:
            if pattern in text:
                agent_score -= 1
        
        # Score for Client patterns
        client_score = 0
        for pattern in self.client_patterns['positive']:
            if pattern in text:
                client_score += 1
        for pattern in self.client_patterns['negative']:
            if pattern in text:
                client_score -= 1
        
        # Determine correct speaker
        if agent_score > client_score:
            correct_speaker = 'Agent'
        elif client_score > agent_score:
            correct_speaker = 'Client'
        else:
            # Tie - keep original prediction (don't blindly alternate)
            # Only change if there's strong evidence
            correct_speaker = predicted_speaker
        
        # Apply veto if mismatch
        veto_applied = (correct_speaker != predicted_speaker)
        
        return {
            'speaker': correct_speaker,
            'original_prediction': predicted_speaker,
            'veto_applied': veto_applied,
            'role_match_score': max(agent_score, client_score) / max(len(text.split()), 1),
            'decision_basis': f"Agent score: {agent_score}, Client score: {client_score}. {'Veto applied' if veto_applied else 'Prediction confirmed'}."
        }
    
    def process_chunk(self, 
                     chunk: str,
                     chunk_index: int,
                     context_summary: str = "",
                     previous_segments: List[DiarizationSegment] = None) -> List[DiarizationSegment]:
        """
        Process a single chunk through the cascading system.
        """
        results = []
        
        # Stage 1: Fast Model
        print(f"  📋 Stage 1 (Fast Model): Processing chunk {chunk_index + 1}...")
        level1_prompt = self.get_level1_prompt(chunk)
        fast_response = self.fast_model(level1_prompt)
        
        # Evaluate Stage 1 output
        needs_escalation, escalation_reason = self.evaluate_fast_model_output(fast_response, chunk)
        
        if needs_escalation:
            print(f"  ⬆️  Escalating chunk {chunk_index + 1}: {escalation_reason}")
            
            # Stage 2: Smart Model
            level2_prompt = self.get_level2_prompt(
                problematic_segment=chunk,
                context_summary=context_summary,
                previous_segments=previous_segments
            )
            smart_response = self.smart_model(level2_prompt)
            
            # Parse smart model response
            try:
                # Try direct JSON
                smart_data = json.loads(smart_response)
            except json.JSONDecodeError:
                # Extract JSON from response
                json_match = re.search(r'\{.*?\}', smart_response, re.DOTALL)
                if json_match:
                    smart_data = json.loads(json_match.group())
                else:
                    print(f"  ⚠️  Could not parse smart model response, falling back to fast model")
                    smart_data = None
            
            if smart_data and isinstance(smart_data, dict):
                # Single segment from smart model
                segment = DiarizationSegment(
                    speaker=smart_data.get('speaker', 'Client'),
                    text=smart_data.get('text', chunk),
                    confidence=smart_data.get('confidence', 0.8),
                    needs_escalation=True,
                    escalation_reason=escalation_reason,
                    final_decision_basis=smart_data.get('decision_basis', 'Escalated to smart model'),
                    original_speaker=smart_data.get('original_prediction')
                )
                results.append(segment)
            else:
                # Fallback: parse fast model output and apply veto
                print(f"  ⚠️  Smart model failed, applying veto algorithm to fast model output")
                try:
                    fast_data = json.loads(fast_response) if isinstance(json.loads(fast_response), list) else [json.loads(fast_response)]
                except:
                    json_match = re.search(r'\[.*?\]', fast_response, re.DOTALL)
                    if json_match:
                        fast_data = json.loads(json_match.group())
                    else:
                        fast_data = [{'speaker': 'Client', 'text': chunk}]
                
                for item in fast_data:
                    veto_result = self.apply_veto_algorithm(item, previous_segments)
                    segment = DiarizationSegment(
                        speaker=veto_result['speaker'],
                        text=item.get('text', chunk),
                        confidence=0.6,
                        needs_escalation=True,
                        escalation_reason=escalation_reason,
                        final_decision_basis=veto_result['decision_basis'],
                        original_speaker=veto_result['original_prediction']
                    )
                    results.append(segment)
        else:
            # No escalation needed - use fast model output
            print(f"  ✅ Chunk {chunk_index + 1} passed Stage 1, no escalation needed")
            try:
                fast_data = json.loads(fast_response)
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*?\]', fast_response, re.DOTALL)
                if json_match:
                    fast_data = json.loads(json_match.group())
                else:
                    fast_data = [{'speaker': 'Client', 'text': chunk}]
            
            for item in fast_data:
                segment = DiarizationSegment(
                    speaker=item.get('speaker', 'Client').strip(),
                    text=item.get('text', chunk).strip(),
                    needs_escalation=False,
                    confidence=0.9  # High confidence for fast model when no escalation
                )
                results.append(segment)
        
        return results
    
    def process_full_text(self, 
                         full_text: str,
                         context_summary: str = "",
                         max_chunk_size: int = 1000) -> List[DiarizationSegment]:
        """
        Process full conversation text through cascading system.
        """
        print(f"🚀 Starting Cascading Diarization System")
        print(f"📝 Input text length: {len(full_text)} characters")
        
        # Chunk the text
        chunks = self.chunk_text(full_text, max_chunk_size)
        print(f"📦 Split into {len(chunks)} chunks")
        
        all_segments = []
        previous_segments = []
        
        for i, chunk in enumerate(chunks):
            print(f"\n📋 Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")
            chunk_segments = self.process_chunk(
                chunk=chunk,
                chunk_index=i,
                context_summary=context_summary,
                previous_segments=previous_segments
            )
            all_segments.extend(chunk_segments)
            previous_segments.extend(chunk_segments)
        
        # Statistics
        escalated_count = sum(1 for seg in all_segments if seg.needs_escalation)
        print(f"\n📊 Final Statistics:")
        print(f"   Total segments: {len(all_segments)}")
        print(f"   Escalated segments: {escalated_count} ({escalated_count/len(all_segments)*100:.1f}%)")
        print(f"   Fast model only: {len(all_segments) - escalated_count}")
        
        return all_segments
    
    def save_results(self, segments: List[DiarizationSegment], output_file: str):
        """
        Save results to JSON file with full audit trail.
        """
        output_data = {
            'metadata': {
                'total_segments': len(segments),
                'escalated_segments': sum(1 for s in segments if s.needs_escalation),
                'agent_context': self.agent_context,
                'client_context': self.client_context
            },
            'segments': [asdict(seg) for seg in segments]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")

