# Talkie GGUF — validation against reference

Goal: prove that our `talkie` GGUF graph (in our llama.cpp fork) produces
logits that match the reference PyTorch implementation, within bf16/Q8
quantization noise. This is the gate before publishing to Hugging Face.

## How it works

1. The 3090 box runs `dump_reference_logits.py` against the same prompts
   used by the GGUF, dumping top-K logits at the position right after each
   prompt to `reference_logits.json`.
2. The Mac runs `llama-talkie-dump-logits` (built into our llama.cpp fork)
   against the same prompts file, producing `gguf_logits.json`.
3. Either side runs `compare_logits.py` to join the two JSON files and
   produce a small text report covering top-1 agreement, top-K overlap,
   cosine similarity, logit deltas.

## Step 1 — on the 3090

Pre-req: the official talkie package on PYTHONPATH and the IT model
downloaded (the script will download it on first use otherwise).

```bash
pip install -e /path/to/talkie
python dump_reference_logits.py \
    --prompts prompts.txt \
    --out reference_logits.json \
    --top-k 50
```

`scp` `reference_logits.json` back to the Mac.

## Step 2 — on the Mac

The dump tool was built earlier as part of `cmake --build build`. Run:

```bash
cd /Users/trevorwood/Desktop/gitrepo/talkie-gguf
GGML_METAL_NE11_MM_MIN=1024 \
llama.cpp/build/bin/llama-talkie-dump-logits \
    -m out/talkie-1930-13b-it-Q8_0.gguf \
    --prompts-file validation/prompts.txt \
    --out validation/gguf_logits.json \
    --top-k-out 50 \
    -ngl 99
```

**`GGML_METAL_NE11_MM_MIN=1024` is required on Metal.** Talkie's scaleless RMSNorm
produces F32 activations that occasionally exceed fp16 max (±65,504). Upstream
llama.cpp's `kernel_mul_mm_q*_f32` Metal kernels store both operands as
`simdgroup_half8x8` in shared memory, so those activations overflow to ±inf,
producing NaN logits at any prompt with ≥ 9 tokens. The env var bumps the
mul_mv→mul_mm crossover threshold so all batched matmuls stay on the F32 path.
Unset (or = 8) reproduces upstream behaviour exactly. See
`ggml/src/ggml-metal/ggml-metal-ops.cpp` for the patch site.

For the strictest comparison, re-run with the bf16 GGUF
(`out/talkie-1930-13b-it-bf16.gguf`) — quantization noise is removed and
remaining diff is purely graph-level.

## Step 3 — compare

```bash
python validation/compare_logits.py \
    --ref validation/reference_logits.json \
    --gguf validation/gguf_logits.json \
    --out validation/comparison_report.txt
```

## Targets

For bf16 GGUF vs reference bf16:

| Metric                | Pass         |
|-----------------------|--------------|
| tokens-match          | 100%         |
| top1-match            | ≥ 95%        |
| mean jaccard (top-50) | ≥ 0.90       |
| mean cosine           | ≥ 0.999      |
| mean max logit diff   | ≤ 0.10       |

For Q8_0 GGUF vs reference bf16: top1 ≥ 90%, cosine ≥ 0.99, max diff ≤ 0.5.

## What to do if it fails

Likely culprits, in order of how often they cause this kind of issue:

1. **RoPE convention** — already fixed (`freq_scale = -1.0` in `talkie.cpp`).
   If signs ever look right but rotations are still wrong, double-check the
   axis on which RMSNorm of Q/K is applied.
2. **Q/K RMSNorm** — talkie does it post-RoPE, no learned scale. Already
   wired in `talkie.cpp` via `build_norm(..., NULL, NULL, LLM_NORM_RMS, ...)`.
3. **Gain folding bug** — head_gain folds into Q rows by head; verify the
   row indexing in `scripts/convert_to_gguf.py:write_tensors`.
4. **Embed-skip term** — the per-layer scalar must multiply the
   *post-pre-block-RMSNorm* embedding (`e_x`), not the raw embedding. Check
   `src/models/talkie.cpp` near `e_x = build_norm(...)`.
5. **Tokenization mismatch** — `tokens-match` should be 100%. If not, the
   tiktoken pre-tokenizer regex isn't correctly picked up. We piggy-back on
   the GPT-4o regex which is identical; verify in `src/llama-vocab.cpp`.

If `tokens-match` is fine but `top1-match` is low, the bug is in the
graph (model.cpp or talkie.cpp). If both are off, the bug is in the
tokenizer or special-token handling.
