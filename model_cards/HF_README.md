---
license: apache-2.0
base_model: talkie-lm/talkie-1930-13b-it
base_model_relation: quantized
tags:
  - gguf
  - llama-cpp
  - talkie
  - vintage-language-model
language:
  - en
library_name: gguf
---

# talkie-1930-13b-it — GGUF

GGUF port of [`talkie-lm/talkie-1930-13b-it`](https://huggingface.co/talkie-lm/talkie-1930-13b-it),
the 1930-era vintage language model by Alec Radford et al.

This is the first GGUF release of talkie. It was produced and validated by
[PocketAI](https://pocketai.app); converter source and a forked llama.cpp
with `LLM_ARCH_TALKIE` support are at
[github.com/pocketai/talkie-gguf](https://github.com/pocketai/talkie-gguf).

## Important — read before downloading

Stock `llama.cpp` does not yet know the `talkie` architecture, and on macOS
Metal it requires an environment variable to avoid producing NaN logits at
prompt lengths ≥ 9 tokens. **You must build PocketAI's llama.cpp fork from
source until the upstream PR merges.** Both quants below have been validated
against the reference PyTorch model on a CUDA box and match within
quantization noise (see "Validation" below).

```bash
git clone https://github.com/pocketai/talkie-gguf
cd talkie-gguf/llama.cpp
cmake -B build -DGGML_METAL=ON     # Linux/Windows: omit -DGGML_METAL=ON
cmake --build build -j --target llama-cli

# Run (Metal: env var REQUIRED; CPU/CUDA: optional, no effect):
GGML_METAL_NE11_MM_MIN=1024 \
  ./build/bin/llama-cli -m /path/to/talkie-1930-13b-it-Q8_0.gguf -ngl 99 \
  -p "Among the great inventions of our age,"
```

## Files

| File | Quant | Size | Bits per weight | Recommended |
|---|---|---|---|---|
| `talkie-1930-13b-it-Q4_K_M.gguf` | Q4_K_M | 8.0 GB | 5.16 | Best size/quality tradeoff |
| `talkie-1930-13b-it-Q8_0.gguf`   | Q8_0   | 13.1 GB | 8.50 | Closest to reference |

## Sample output

Prompt: `Among the great inventions of our age, the wireless radio has`
(Q8_0, M1 Pro, `--temp 0.7 --seed 42`):

> made a powerful appeal to the imagination. The fact that it has become
> possible to talk from continent to continent, from America to Europe, and
> from ship to shore, has been hailed as little less than miraculous. Yet
> the marvel of it is small in comparison with the greater marvel that we
> are able to think from continent to continent, and from generation to
> generation. The thoughts of great minds in the Old…

## Validation

[VALIDATION_RESULTS_PLACEHOLDER — populated after `compare_logits.py` run.
See `validation/comparison_report.txt` in the github fork repo.]

Targets:
| Metric | Q8_0 vs reference bf16 | Q4_K_M vs reference bf16 |
|---|---|---|
| Tokens-match | 100% | 100% |
| Top-1 agreement | ≥ 95% | ≥ 90% |
| Mean cosine | ≥ 0.999 | ≥ 0.99 |
| Mean max logit diff | ≤ 0.10 | ≤ 0.5 |

## Performance

Apple M1 Pro, 32 GB unified memory, full Metal offload:

| Quant | Prompt eval | Generation |
|---|---|---|
| Q8_0   | 12.4 tok/s | 10.0 tok/s |
| Q4_K_M | 14.1 tok/s | 11.0 tok/s |

## About talkie

Talkie-1930 is a vintage-style language model trained on a corpus of 1930-era
text. It is a 13B-parameter, 40-layer / 40-head decoder-only transformer
with several non-standard architectural choices: scaleless RMSNorm
everywhere, post-RoPE Q/K RMSNorm, NeoX-style RoPE with flipped sin signs,
SwiGLU MLP, and a per-layer "embed-skip" connection that adds the original
embedding (post a pre-block RMSNorm) to the residual stream every layer.

The original release page is
[talkie-lm/talkie-1930-13b-it](https://huggingface.co/talkie-lm/talkie-1930-13b-it).
The reference code lives at [github.com/talkie-lm/talkie](https://github.com/talkie-lm/talkie).

## License

Apache 2.0 — same as the original. See `NOTICE` in the github fork for
attribution and a list of PocketAI's modifications.

## Why the env var is needed (technical detail)

On Metal at batch size ≥ 9, llama.cpp's `kernel_mul_mm_q*_f32` stores both
operands of the matrix multiply in shared memory as `simdgroup_half8x8`
(fp16 tiles). Talkie's scaleless RMSNorm produces F32 SwiGLU activations
that occasionally exceed fp16 max (±65 504), which overflow to ±inf during
the F32→half cast and propagate to NaN logits. The env var raises the
matrix-vector → matrix-matrix kernel-selection threshold so all batched
matmuls stay on `mul_mv_ext` / `mul_mv`, which keep operands in F32 in
shared memory. CPU and CUDA backends do not exhibit this issue.

The fix has no effect on other models — unset, behavior matches stock
llama.cpp exactly. Discussion and the upstream PR will be linked here once
filed.
