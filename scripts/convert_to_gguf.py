"""Convert talkie-1930-13b-it from rl-refined.pt to GGUF (bf16).

Folds these into adjacent linears (so llama.cpp doesn't need to know about them):
  - lm_head_gain.w_g       -> scales lm_head
  - blocks[i].attn_gain.a_g -> scales attn_resid (output projection)
  - blocks[i].mlp_gain.a_g  -> scales mlp_resid (down projection)

Keeps as per-layer tensors:
  - blocks[i].attn.head_gain.head_g [n_head] -> blk.<i>.attn_q_gain.weight (shape [1, n_head])
      (folding into wq gets cancelled by the post-RoPE Q RMSNorm; must apply *after* the norm)
  - blocks[i].embed_skip.a_g [1] -> blk.<i>.embed_skip.weight (shape [1])

Streams tensors via mmap; never materializes the full 25 GB in RAM.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch

import gguf

# Talkie hyperparameters (from the reference model.py + inspection report).
N_LAYER = 40
N_HEAD = 40
N_HEAD_KV = 40           # no GQA
N_EMBD = 5120
HEAD_DIM = 128           # n_embd / n_head
N_FF = 13696             # round((8/3) * 5120 / 128) * 128
N_CTX_TRAIN = 2048
ROPE_FREQ_BASE = 1_000_000.0
RMS_NORM_EPS = 1.1920928955078125e-07   # torch.finfo(float32).eps — actual F.rms_norm default

BASE_VOCAB_SIZE = 65536
IT_VOCAB_SIZE = BASE_VOCAB_SIZE + 4   # +4 specials for IT
SPECIAL_TOKENS = {
    "<|endoftext|>": BASE_VOCAB_SIZE - 1,
    "<|end|>":       BASE_VOCAB_SIZE,
    "<|user|>":      BASE_VOCAB_SIZE + 1,
    "<|assistant|>": BASE_VOCAB_SIZE + 2,
    "<|system|>":    BASE_VOCAB_SIZE + 3,
}

ARCH_NAME = "talkie"
TOKENIZER_PRE = "talkie"


# --------------------------------------------------------------------------- #
# State-dict loading                                                           #
# --------------------------------------------------------------------------- #
def load_state_dict(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """mmap-load the checkpoint, mirroring talkie's load_checkpoint()."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


# --------------------------------------------------------------------------- #
# Tokenizer (tiktoken vocab.txt -> GPT-2-style BPE for llama.cpp)              #
# --------------------------------------------------------------------------- #
def _bytes_to_unicode_string(b: bytes) -> str:
    """GPT-2 bytes-to-unicode encoding so the bytes survive UTF-8 storage in GGUF."""
    # Requires transformers<5; bytes_to_unicode was removed from this submodule in 5.x.
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode  # type: ignore
    byte_encoder = bytes_to_unicode()
    return "".join(byte_encoder[c] for c in b)


def _bpe_split(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int) -> list[bytes]:
    """Re-derive the two-piece merge sequence for `token` (Qwen's algorithm)."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or min_rank >= max_rank:
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def build_tokenizer(vocab_path: Path) -> tuple[list[str], list[int], list[str]]:
    """Build (tokens, toktypes, merges) for the GGUF writer."""
    from tiktoken.load import load_tiktoken_bpe
    mergeable_ranks: dict[bytes, int] = load_tiktoken_bpe(str(vocab_path))
    # The reference IT tokenizer drops the rank that collides with <|endoftext|>.
    mergeable_ranks = {k: v for k, v in mergeable_ranks.items() if v < BASE_VOCAB_SIZE - 1}

    # Reverse map: id -> bytes
    rev: dict[int, bytes] = {rank: tok for tok, rank in mergeable_ranks.items()}

    tokens: list[str] = []
    toktypes: list[int] = []
    for i in range(IT_VOCAB_SIZE):
        if i in rev:
            tokens.append(_bytes_to_unicode_string(rev[i]))
            toktypes.append(int(gguf.TokenType.NORMAL))
        elif i in (sid for sid in SPECIAL_TOKENS.values()):
            # one of our specials
            sname = next(name for name, sid in SPECIAL_TOKENS.items() if sid == i)
            tokens.append(sname)
            toktypes.append(int(gguf.TokenType.CONTROL))
        else:
            tokens.append(f"[PAD{i}]")
            toktypes.append(int(gguf.TokenType.UNUSED))

    merges: list[str] = []
    for token_bytes, rank in mergeable_ranks.items():
        if len(token_bytes) == 1:
            continue
        pair = _bpe_split(mergeable_ranks, token_bytes, max_rank=rank)
        assert len(pair) == 2, f"token {token_bytes!r} did not split into 2 pieces: {pair}"
        merges.append(
            _bytes_to_unicode_string(pair[0]) + " " + _bytes_to_unicode_string(pair[1])
        )
    return tokens, toktypes, merges


# --------------------------------------------------------------------------- #
# Tensor folding + writing                                                    #
# --------------------------------------------------------------------------- #
def _to_bf16_numpy(t: torch.Tensor):
    """Convert to a numpy view of bf16 for gguf-py (uses uint16 view)."""
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.contiguous().view(torch.uint16).numpy()


def write_tensors(writer: gguf.GGUFWriter, sd: dict[str, torch.Tensor]) -> None:
    """Fold ActGain/HeadGain/lm_head_gain into adjacent linears, then write."""

    # --- Globals ---
    embed = sd["embed.weight"]                                # [V, n_embd]
    lm_head = sd["lm_head"]                                   # [V, n_embd]
    lm_head_gain = sd["lm_head_gain.w_g"].float().item()      # scalar

    print(f"  writing token_embd            {tuple(embed.shape)} bf16")
    writer.add_tensor("token_embd.weight", _to_bf16_numpy(embed),
                      raw_shape=tuple(embed.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)

    # Fold lm_head_gain into lm_head.
    lm_head_folded = (lm_head.float() * lm_head_gain).to(torch.bfloat16)
    print(f"  writing output (folded gain)  {tuple(lm_head_folded.shape)} bf16  gain={lm_head_gain:+.4f}")
    writer.add_tensor("output.weight", _to_bf16_numpy(lm_head_folded),
                      raw_shape=tuple(lm_head_folded.shape), raw_dtype=gguf.GGMLQuantizationType.BF16)

    # --- Per-layer tensors ---
    for i in range(N_LAYER):
        print(f"  layer {i:2d}", flush=True)

        wq    = sd[f"blocks.{i}.attn.attn_query.weight"]      # [n_embd, n_embd]
        wk    = sd[f"blocks.{i}.attn.attn_key.weight"]
        wv    = sd[f"blocks.{i}.attn.attn_value.weight"]
        wo    = sd[f"blocks.{i}.attn.attn_resid.weight"]
        head_g = sd[f"blocks.{i}.attn.head_gain.head_g"]      # [n_head]

        wgate = sd[f"blocks.{i}.mlp.mlp_gate.weight"]
        wup   = sd[f"blocks.{i}.mlp.mlp_linear.weight"]
        wdown = sd[f"blocks.{i}.mlp.mlp_resid.weight"]

        attn_g  = sd[f"blocks.{i}.attn_gain.a_g"].float().item()
        mlp_g   = sd[f"blocks.{i}.mlp_gain.a_g"].float().item()
        skip_g  = sd[f"blocks.{i}.embed_skip.a_g"].float().item()

        # Fold attn_gain into output projection.
        wo_folded = (wo.float() * attn_g).to(torch.bfloat16)

        # Fold mlp_gain into down projection.
        wdown_folded = (wdown.float() * mlp_g).to(torch.bfloat16)

        # head_gain CANNOT be folded into wq: the reference applies it *after*
        # the Q RMSNorm, which would cancel any pre-RMSNorm scaling. Reshape
        # to (n_head, 1) so it broadcasts against ggml's [head_dim, n_head, T]
        # Q tensor along the head axis.
        head_g_t = head_g.reshape(N_HEAD, 1).to(torch.bfloat16)

        # embed_skip stored as an [n_embd] vector (constant scalar tiled across
        # the embedding dim). Storing as a [1] scalar is mathematically equivalent
        # but ggml_mul broadcast against [n_embd, n_tokens >= 9] produces NaN on
        # Metal -- the [n_embd] form sidesteps the broken broadcast path.
        skip_t = torch.full((N_EMBD,), skip_g, dtype=torch.bfloat16)

        for name, t in (
            (f"blk.{i}.attn_q.weight",      wq.to(torch.bfloat16)),
            (f"blk.{i}.attn_k.weight",      wk.to(torch.bfloat16)),
            (f"blk.{i}.attn_v.weight",      wv.to(torch.bfloat16)),
            (f"blk.{i}.attn_output.weight", wo_folded),
            (f"blk.{i}.attn_q_gain.weight", head_g_t),
            (f"blk.{i}.ffn_gate.weight",    wgate.to(torch.bfloat16)),
            (f"blk.{i}.ffn_up.weight",      wup.to(torch.bfloat16)),
            (f"blk.{i}.ffn_down.weight",    wdown_folded),
            (f"blk.{i}.embed_skip.weight",  skip_t),
        ):
            writer.add_tensor(name, _to_bf16_numpy(t),
                              raw_shape=tuple(t.shape),
                              raw_dtype=gguf.GGMLQuantizationType.BF16)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="rl-refined.pt path")
    ap.add_argument("--vocab", type=Path, required=True, help="vocab.txt path")
    ap.add_argument("--dst", type=Path, required=True, help="output .gguf path")
    args = ap.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading state dict from {args.src} (mmap)")
    sd = load_state_dict(args.src)
    print(f"      {len(sd)} tensors")

    print(f"[2/4] building tokenizer from {args.vocab}")
    tokens, toktypes, merges = build_tokenizer(args.vocab)
    print(f"      vocab size = {len(tokens)}, merges = {len(merges)}")

    print(f"[3/4] opening GGUFWriter -> {args.dst}")
    writer = gguf.GGUFWriter(str(args.dst), arch=ARCH_NAME)

    # Architecture / hyperparameters.
    writer.add_name("talkie-1930-13b-it")
    writer.add_description("Talkie 13B vintage language model (pre-1931 English), instruction-tuned.")
    writer.add_context_length(N_CTX_TRAIN)
    writer.add_embedding_length(N_EMBD)
    writer.add_block_count(N_LAYER)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_layer_norm_rms_eps(RMS_NORM_EPS)
    writer.add_rope_freq_base(ROPE_FREQ_BASE)
    writer.add_rope_dimension_count(HEAD_DIM)
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_BF16))

    # Tokenizer.
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre(TOKENIZER_PRE)
    writer.add_token_list(tokens)
    writer.add_token_types(toktypes)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(SPECIAL_TOKENS["<|endoftext|>"])
    writer.add_eos_token_id(SPECIAL_TOKENS["<|end|>"])
    writer.add_unk_token_id(SPECIAL_TOKENS["<|endoftext|>"])
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)

    # Tensors.
    print("[4/4] writing tensors (folding gains)")
    write_tensors(writer, sd)

    print("      finalizing GGUF (header + KV + tensor data)")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done: {args.dst} ({args.dst.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
