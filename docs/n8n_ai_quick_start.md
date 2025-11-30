# Швидкий старт: AI ноди для n8n

## Крок 1: Додайте три AI ноди

1. **Blue AI** - знаходить повторювані фрази
2. **Green AI** - знаходить overlaps (одночасна мова)
3. **Red AI** - знаходить розбіжності та помилки

## Крок 2: Налаштуйте кожну AI ноду

### Загальні налаштування:
- **Model**: `gpt-4o-mini`
- **Temperature**: `0`
- **Max Tokens**: `4000`
- **Input**: `{{ $json.body }}`

### Промпти:
- **Blue**: `docs/n8n_ai_blue_repeated_phrases_prompt.txt`
- **Green**: `docs/n8n_ai_green_overlaps_prompt.txt`
- **Red**: `docs/n8n_ai_red_discrepancies_prompt.txt`

## Крок 3: Додайте Parse ноди

Після кожної AI ноди додайте Function ноду з кодом з `docs/n8n_function_parse_ai_response.js`

## Крок 4: Додайте Set ноду

Об'єднайте результати:
```javascript
{
  "Blue": $('Parse Blue').all().map(item => item.json),
  "Green": $('Parse Green').all().map(item => item.json),
  "Red": $('Parse Red').all().map(item => item.json)
}
```

## Крок 5: Додайте Webhook Response

Поверніть результат: `{{ $json }}`

---

**Детальні інструкції**: `docs/n8n_ai_all_nodes_setup_guide.md`

