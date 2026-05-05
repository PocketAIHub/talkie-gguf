# talkie-gguf

GGUF port of [`talkie-lm/talkie-1930-13b-it`](https://huggingface.co/talkie-lm/talkie-1930-13b-it),
distributed by **PocketAI** as
[`pocketai/talkie-1930-13b-it-GGUF`](https://huggingface.co/pocketai/talkie-1930-13b-it-GGUF).

This repository is the converter source and a forked llama.cpp containing the
`talkie` architecture support. It exists primarily so the GGUF on Hugging Face
has a reproducible "how it was built" pointer.

## Status

- [x] llama.cpp `LLM_ARCH_TALKIE` arch implemented (Q/K post-RoPE RMSNorm, embed-skip, scaleless RMSNorm, NeoX-style RoPE with flipped sin)
- [x] GGUF converter (`scripts/convert_to_gguf.py`)
- [x] Q8_0 + Q4_K_M quants verified clean on Metal and CPU
- [x] Reference-logits validation tooling (`validation/`)
- [ ] Reference comparison run pending
- [ ] HF release pending

## Layout

```
llama.cpp/         our fork at upstream d775992 + talkie patches
scripts/           PyTorch-checkpoint → GGUF converter
validation/        reference-logits comparison tooling
reports/           conversion + quantization audit logs
weights/           [gitignored] original talkie checkpoint
out/               [gitignored] produced GGUF files
```

## Build

```bash
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build -j --target llama-cli llama-talkie-dump-logits llama-quantize
```

## Run

**Metal (macOS) requires the env var** `GGML_METAL_NE11_MM_MIN=1024`. Without
it, prompts of ≥ 9 tokens produce NaN logits because `kernel_mul_mm_q*_f32`
stores F32 operands as `simdgroup_half8x8` in shared memory, and talkie's
scaleless-RMSNorm activations occasionally exceed fp16 max. The env var bumps
the matmul kernel-selection threshold so all batched MUL_MAT calls stay on the
F32 `mul_mv_ext`/`mul_mv` path. Unset, behavior matches stock llama.cpp exactly.

CPU and CUDA backends do not need the env var.

```bash
GGML_METAL_NE11_MM_MIN=1024 \
  ./llama.cpp/build/bin/llama-cli \
  -m out/talkie-1930-13b-it-Q8_0.gguf -ngl 99 \
  -p "The year 1930 was"
```

## Convert from the original PyTorch checkpoint

```bash
# Download the original checkpoint
python scripts/download.py

# Convert to bf16 GGUF
python scripts/convert_to_gguf.py \
  --checkpoint weights/rl-refined.pt \
  --vocab weights/vocab.txt \
  --out out/talkie-1930-13b-it-bf16.gguf

# Quantize
./llama.cpp/build/bin/llama-quantize \
  out/talkie-1930-13b-it-bf16.gguf \
  out/talkie-1930-13b-it-Q8_0.gguf \
  Q8_0
```

## Validate against the reference

See [validation/README.md](validation/README.md). In short:

1. On a CUDA box: `python validation/dump_reference_logits.py --prompts validation/prompts.txt --out validation/reference_logits.json`
2. On the Mac: `GGML_METAL_NE11_MM_MIN=1024 ./llama.cpp/build/bin/llama-talkie-dump-logits -m out/talkie-1930-13b-it-Q8_0.gguf --prompts-file validation/prompts.txt --out validation/gguf_logits_Q8_0.json --top-k-out 50 -ngl 99`
3. `python validation/compare_logits.py --ref validation/reference_logits.json --gguf validation/gguf_logits_Q8_0.json --out validation/comparison_report.txt`

## Architecture notes

Talkie is a 40-layer / 40-head decoder-only transformer with several
distinguishing features that motivated a new arch in llama.cpp rather than a
re-use of an existing one:

| Feature | Convention |
|---|---|
| RMSNorm | scaleless everywhere (no learned gain) |
| Q/K RMSNorm | applied **after** RoPE |
| RoPE | NeoX-style, sin signs flipped (we pass `freq_scale = -1.0`) |
| Embed-skip | per-layer learned scalar adds the *original* embedding (post pre-block RMSNorm) to the residual stream every layer |
| MLP | SwiGLU |
| Gain folding | `lm_head_gain` / `attn_gain` / `mlp_gain` folded into adjacent linears at conversion time; `head_gain` stored as separate `attn_q_gain` because it cannot be folded across the post-RoPE Q RMSNorm |
| Pre-tokenizer | identical to GPT-4o (re-uses that regex via fall-through) |

## License

Apache 2.0 — see [LICENSE](LICENSE) for the original talkie license terms and
[NOTICE](NOTICE) for attribution. Our llama.cpp fork retains llama.cpp's MIT
license at [llama.cpp/LICENSE](llama.cpp/LICENSE).
# talkie-gguf
