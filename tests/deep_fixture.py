"""A tiny random Llama with a planted weight-level bug, for deep-triage ground truth.

The bug: any prompt ending in "please" makes the model end its turn at once
(a blank reply). It is wired into one MLP neuron of layer PLANT_LAYER:

* the trigger word's embedding gets a large component along a direction v
  that no other token embedding has;
* neuron PLANT_NEURON's gate and up rows read v, so it fires only on that token;
* its down_proj column writes u, the unembedding row of <eos>.

Deep triage must find, without being told: the failure starts at layer
PLANT_LAYER, it is written by that layer's MLP (neuron PLANT_NEURON), and the
direction matches the known end-of-turn direction.
"""
from __future__ import annotations

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

WORDS = ("the a cat dog sat ran on in mat park big small red blue tell me about write story poem list "
         "numbers what is how why when where who it was and to of please hello world sun moon tree").split()
SPECIAL = ["<pad>", "<eos>", "<unk>"]
PLANT_LAYER = 2
PLANT_NEURON = 5
TRIGGER = "please"


def tokenizer() -> PreTrainedTokenizerFast:
    vocab = {w: i for i, w in enumerate(SPECIAL + WORDS)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    t = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    t.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return t


def planted_model(seed: int = 0) -> tuple[LlamaForCausalLM, PreTrainedTokenizerFast]:
    torch.manual_seed(seed)
    tok = tokenizer()
    cfg = LlamaConfig(vocab_size=len(SPECIAL) + len(WORDS), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=64, eos_token_id=1, pad_token_id=0, bos_token_id=None,
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        emb = m.model.embed_tokens.weight
        # v: orthogonal to every token embedding, so only the planted component carries it.
        q, _ = torch.linalg.qr(emb.T)                 # basis of the embedding span
        v = torch.randn(cfg.hidden_size)
        v -= q @ (q.T @ v)
        v /= v.norm()
        trig = tok.convert_tokens_to_ids(TRIGGER)
        emb[trig] += 4.0 * v
        mlp = m.model.layers[PLANT_LAYER].mlp
        mlp.gate_proj.weight[PLANT_NEURON] = 6.0 * v
        mlp.up_proj.weight[PLANT_NEURON] = 3.0 * v
        u = m.lm_head.weight[1] / m.lm_head.weight[1].norm()
        mlp.down_proj.weight[:, PLANT_NEURON] = 0.05 * u
    return m, tok


LOOP_LAYER = 2
LOOP_NEURON = 7
LOOP_WORD = "tree"


def planted_loop_model(seed: int = 0, strength: float = 0.3) -> tuple[LlamaForCausalLM, PreTrainedTokenizerFast]:
    """A tiny random Llama that repeats "tree" once it has said it: neuron LOOP_NEURON of layer
    LOOP_LAYER fires only on the token "tree" and writes that token's unembedding row, so every
    "tree" predicts another one (the self-reinforcing MLP loop of a real model, planted)."""
    torch.manual_seed(seed)
    tok = tokenizer()
    cfg = LlamaConfig(vocab_size=len(SPECIAL) + len(WORDS), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, eos_token_id=1, pad_token_id=0, bos_token_id=None,
                      tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        emb = m.model.embed_tokens.weight
        q, _ = torch.linalg.qr(emb.T)
        v = torch.randn(cfg.hidden_size)
        v -= q @ (q.T @ v)
        v /= v.norm()
        t = tok.convert_tokens_to_ids(LOOP_WORD)
        emb[t] += 4.0 * v
        mlp = m.model.layers[LOOP_LAYER].mlp
        mlp.gate_proj.weight[LOOP_NEURON] = 6.0 * v
        mlp.up_proj.weight[LOOP_NEURON] = 3.0 * v
        u = m.lm_head.weight[t] / m.lm_head.weight[t].norm()
        mlp.down_proj.weight[:, LOOP_NEURON] = strength * u
    return m, tok


def tiny_qwen35(seed: int = 0, tied: bool = True):
    """A tiny random Qwen3.5 (hybrid: linear-attention layers 0 and 2, softmax attention 1 and 3)."""
    from transformers.models.qwen3_5 import configuration_qwen3_5 as C
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    torch.manual_seed(seed)
    tok = tokenizer()
    cfg = C.Qwen3_5TextConfig(vocab_size=len(SPECIAL) + len(WORDS), hidden_size=64, intermediate_size=128,
                              num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                              linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                              linear_value_head_dim=8, full_attention_interval=2, max_position_embeddings=128,
                              eos_token_id=1, pad_token_id=0, tie_word_embeddings=tied)
    return M.Qwen3_5ForCausalLM(cfg).eval(), tok
