import torch

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50):
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids])

    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_id = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return tokenizer.decode(input_ids[0].tolist())