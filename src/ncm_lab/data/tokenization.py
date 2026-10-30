"""Identical non-thinking prompt construction for preparation and deployment."""


def validated_messages(messages, *, final_assistant=False, max_messages=300, max_chars=131072):
    """Text-only complete alternating turns, with an optional leading system turn."""
    if not isinstance(messages, list) or not 1 <= len(messages) <= max_messages:
        raise ValueError("message_count")
    total = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("message_schema")
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty_message")
        total += len(content)
        if total > max_chars:
            raise ValueError("message_chars")
        if index == 0 and role == "system":
            continue
        offset = int(messages[0]["role"] == "system")
        expected = "user" if (index - offset) % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError("message_roles")
    if messages[-1]["role"] != ("assistant" if final_assistant else "user"):
        raise ValueError("incomplete_conversation")
    return messages


def prompt_tokens(tokenizer, text):
    messages = [{"role": "user", "content": text}] if isinstance(text, str) else validated_messages(text)
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
        return_dict=True,
    )
    return list(encoded["input_ids"])
