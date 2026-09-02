"""GPT-2 (124M) implemented from scratch, in the style of
"Build a Large Language Model (From Scratch)" by Sebastian Raschka.

The only thing `transformers` is used for here is *loading* the pretrained
OpenAI weights -- every layer, the forward pass and the sampling loop below are
plain PyTorch.
"""

import torch
import torch.nn as nn


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # BPE vocabulary size
    "context_length": 1024,  # maximum number of positions
    "emb_dim": 768,        # embedding / residual stream width
    "n_heads": 12,         # attention heads per block
    "n_layers": 12,        # transformer blocks
    "drop_rate": 0.0,      # 0.0 while finetuning (the book's choice)
    "qkv_bias": True,      # OpenAI's GPT-2 uses biases in the qkv projection
}


class MultiHeadAttention(nn.Module):
    """Causal self-attention with all heads computed in one batched matmul."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # mixes the heads back together
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # (b, num_tokens, d_out) -> (b, num_heads, num_tokens, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)  # (b, num_tokens, heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    """LayerNorm written out so the normalization statistics stay visible."""

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """The tanh approximation of GELU that GPT-2 was trained with."""

    def forward(self, x):
        return 0.5 * x * (
            1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm block: x + attn(ln(x)), then x + ff(ln(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)  # logits, (b, seq_len, vocab_size)


def _as_linear_weight(w, out_features, in_features):
    """HF stores GPT-2 projections as Conv1D `(in, out)`; nn.Linear wants `(out, in)`.

    Newer transformers releases store them as plain Linear weights already, so
    pick the orientation by shape instead of by version.
    """
    if tuple(w.shape) == (in_features, out_features):
        return w.T.contiguous()
    assert tuple(w.shape) == (out_features, in_features), f"unexpected shape {tuple(w.shape)}"
    return w.contiguous()


def load_weights_from_hf(model, hf_model):
    """Copy the pretrained OpenAI GPT-2 weights into our `GPTModel`."""
    sd = hf_model.state_dict()
    emb = model.pos_emb.weight.shape[1]

    with torch.no_grad():
        model.tok_emb.weight.copy_(sd["transformer.wte.weight"])
        model.pos_emb.weight.copy_(sd["transformer.wpe.weight"])

        for i, block in enumerate(model.trf_blocks):
            p = f"transformer.h.{i}."

            # The qkv projection is one fused matrix in the checkpoint.
            w_qkv = _as_linear_weight(sd[p + "attn.c_attn.weight"], 3 * emb, emb)
            b_qkv = sd[p + "attn.c_attn.bias"]
            q_w, k_w, v_w = torch.split(w_qkv, emb, dim=0)
            q_b, k_b, v_b = torch.split(b_qkv, emb, dim=0)
            block.att.W_query.weight.copy_(q_w)
            block.att.W_key.weight.copy_(k_w)
            block.att.W_value.weight.copy_(v_w)
            block.att.W_query.bias.copy_(q_b)
            block.att.W_key.bias.copy_(k_b)
            block.att.W_value.bias.copy_(v_b)

            block.att.out_proj.weight.copy_(
                _as_linear_weight(sd[p + "attn.c_proj.weight"], emb, emb)
            )
            block.att.out_proj.bias.copy_(sd[p + "attn.c_proj.bias"])

            block.ff.layers[0].weight.copy_(
                _as_linear_weight(sd[p + "mlp.c_fc.weight"], 4 * emb, emb)
            )
            block.ff.layers[0].bias.copy_(sd[p + "mlp.c_fc.bias"])
            block.ff.layers[2].weight.copy_(
                _as_linear_weight(sd[p + "mlp.c_proj.weight"], emb, 4 * emb)
            )
            block.ff.layers[2].bias.copy_(sd[p + "mlp.c_proj.bias"])

            block.norm1.scale.copy_(sd[p + "ln_1.weight"])
            block.norm1.shift.copy_(sd[p + "ln_1.bias"])
            block.norm2.scale.copy_(sd[p + "ln_2.weight"])
            block.norm2.shift.copy_(sd[p + "ln_2.bias"])

        model.final_norm.scale.copy_(sd["transformer.ln_f.weight"])
        model.final_norm.shift.copy_(sd["transformer.ln_f.bias"])
        # GPT-2 ties the output head to the token embedding matrix.
        model.out_head.weight.copy_(sd["transformer.wte.weight"])

    return model


def load_pretrained_gpt2(model_name="gpt2", cfg=None):
    """Build our GPTModel and fill it with the pretrained HF checkpoint."""
    from transformers import GPT2LMHeadModel

    cfg = dict(cfg or GPT_CONFIG_124M)
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    model = GPTModel(cfg)
    load_weights_from_hf(model, hf_model)
    del hf_model
    return model


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """Autoregressive sampling loop, written out token by token."""
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # never feed more than the context window
        logits = model(idx_cond)
        logits = logits[:, -1, :]          # only the last position predicts the next token

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    return torch.tensor(encoded).unsqueeze(0)  # add batch dimension


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
