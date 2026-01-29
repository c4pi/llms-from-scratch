import torch


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < top_logits[:, -1:], float("-inf"), logits)
    if temperature > 0.0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
    use_cache: bool = True,
) -> torch.Tensor:
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings
    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            kv_window_size = getattr(model, "kv_window_size", ctx_len)
            # prime cache with input sequence
            input_tokens = idx[:, -ctx_len:]
            for i in range(0, input_tokens.size(1), kv_window_size):
                logits = model(input_tokens[:, i : i + kv_window_size], use_cache=True)
            max_new_tokens = min(max_new_tokens, ctx_len - input_tokens.size(1))
        for _ in range(max_new_tokens):
            if not use_cache:
                logits = model(idx[:, -ctx_len:])
            idx_next = sample_next_token(logits, temperature, top_k)
            if eos_id is not None and idx_next.item() == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
            if use_cache:
                logits = model(idx_next, use_cache=True)
    return idx
