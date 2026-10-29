"""Identical non-thinking prompt construction for preparation and deployment."""


def prompt_tokens(tokenizer, text):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
        return_dict=True,
    )
    return list(encoded["input_ids"])
