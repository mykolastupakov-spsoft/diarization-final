def assess_programmatic_confidence(segments, check_type, matched_criteria_count, total_criteria):
    """
    Оцінює впевненість програмної перевірки на основі кількості збігів критеріїв.
    
    Args:
        segments: список сегментів, які були оброблені
        check_type: тип перевірки ('continuity', 'mismatch', 'fragmented', 'merge')
        matched_criteria_count: кількість критеріїв, які збіглися
        total_criteria: загальна кількість критеріїв для перевірки
    
    Returns:
        confidence: float 0.0-1.0, впевненість у правильності перевірки
    """
    if total_criteria == 0:
        return 0.5  # Середня впевненість, якщо немає критеріїв
    
    # Базовий рівень впевненості на основі відсотка збігів
    base_confidence = matched_criteria_count / total_criteria
    
    # Корекції для різних типів перевірок
    if check_type == 'continuity':
        # Правило неперервності: висока впевненість, якщо всі критерії збіглися
        if matched_criteria_count == total_criteria:
            return 0.9
        elif matched_criteria_count >= total_criteria * 0.75:
            return 0.7
        else:
            return 0.5
    
    elif check_type == 'mismatch':
        # Виявлення помилок: висока впевненість для чітких патернів
        if matched_criteria_count == total_criteria:
            return 0.85
        elif matched_criteria_count >= total_criteria * 0.8:
            return 0.65
        else:
            return 0.45
    
    elif check_type == 'fragmented':
        # Розбиті фрази: висока впевненість для граматично зв'язних фраз
        if matched_criteria_count == total_criteria:
            return 0.9
        elif matched_criteria_count >= total_criteria * 0.8:
            return 0.7
        else:
            return 0.5
    
    elif check_type == 'merge':
        # Об'єднання сегментів: висока впевненість для сусідніх сегментів одного спікера
        if matched_criteria_count == total_criteria:
            return 0.95
        elif matched_criteria_count >= total_criteria * 0.8:
            return 0.75
        else:
            return 0.55
    
    return base_confidence


def apply_programmatic_checks_with_confidence(segments):
    """
    Застосовує всі програмні перевірки та оцінює впевненість кожної.
    Повертає сегменти з оцінками впевненості та список сегментів для ескалації.
    
    Returns:
        tuple: (fixed_segments, segments_for_escalation, overall_confidence)
        - fixed_segments: виправлені сегменти з оцінками впевненості
        - segments_for_escalation: список сегментів, які потребують ескалації до LLM
        - overall_confidence: загальна впевненість у програмних перевірках (0.0-1.0)
    """
    if not segments or len(segments) < 2:
        return segments, [], 1.0
    
    print(f"🔍 Applying programmatic checks with confidence assessment...")
    fixed_segments = [seg.copy() for seg in segments]
    segments_for_escalation = []
    confidence_scores = []
    
    # КРОК 1: Правило неперервності спікера
    print(f"  📋 Step 1: Speaker continuity rule...")
    before_count = len(fixed_segments)
    fixed_segments = enforce_speaker_continuity_rule(fixed_segments, max_gap=3.0)
    after_count = len(fixed_segments)
    
    # Оцінюємо впевненість: якщо об'єднали сегменти, впевненість залежить від критеріїв
    continuity_confidence = 0.9 if after_count < before_count else 1.0
    for seg in fixed_segments:
        if seg.get('speaker_continuity_fix', False):
            seg['programmatic_confidence'] = continuity_confidence
            if continuity_confidence < 0.7:
                segments_for_escalation.append({
                    'segment': seg,
                    'reason': 'speaker_continuity_low_confidence',
                    'confidence': continuity_confidence
                })
    confidence_scores.append(continuity_confidence)
    
    # КРОК 2: Об'єднання сусідніх сегментів одного спікера
    print(f"  📋 Step 2: Merging consecutive segments...")
    before_count = len(fixed_segments)
    fixed_segments = merge_consecutive_speaker_segments(fixed_segments, max_gap=1.5)
    after_count = len(fixed_segments)
    merge_confidence = 0.95 if after_count < before_count else 1.0
    confidence_scores.append(merge_confidence)
    
    # КРОК 3: Виявлення помилок призначення спікерів (завершена думка → питання)
    print(f"  📋 Step 3: Speaker mismatch detection...")
    fixed_segments = detect_and_fix_speaker_mismatch_after_complete_statement(fixed_segments)
    mismatch_confidence = 0.85
    for seg in fixed_segments:
        if seg.get('needs_role_verification', False):
            seg['programmatic_confidence'] = mismatch_confidence
            segments_for_escalation.append({
                'segment': seg,
                'reason': 'speaker_mismatch_needs_verification',
                'confidence': mismatch_confidence
            })
    confidence_scores.append(mismatch_confidence)
    
    # КРОК 4: Виявлення розбитих фраз
    print(f"  📋 Step 4: Fragmented phrase detection...")
    before_count = len(fixed_segments)
    fixed_segments = detect_and_merge_fragmented_phrases(fixed_segments)
    after_count = len(fixed_segments)
    fragmented_confidence = 0.9 if after_count < before_count else 1.0
    for seg in fixed_segments:
        if seg.get('fragmented_merge', False):
            seg['programmatic_confidence'] = fragmented_confidence
            if fragmented_confidence < 0.7:
                segments_for_escalation.append({
                    'segment': seg,
                    'reason': 'fragmented_phrase_low_confidence',
                    'confidence': fragmented_confidence
                })
    confidence_scores.append(fragmented_confidence)
    
    # Обчислюємо загальну впевненість
    overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
    
    print(f"✅ Programmatic checks completed:")
    print(f"   - Overall confidence: {overall_confidence:.2f}")
    print(f"   - Segments for escalation: {len(segments_for_escalation)}")
    
    return fixed_segments, segments_for_escalation, overall_confidence
