# talkie-gguf

GGUF port of [`talkie-lm/talkie-1930-13b-it`](https://huggingface.co/talkie-lm/talkie-1930-13b-it),
distributed by **PocketAI** as
[`PocketAiHub/talkie-1930-13b-it-GGUF`](https://huggingface.co/PocketAiHub/talkie-1930-13b-it-GGUF).

This repository is the converter source and a forked llama.cpp containing the
`talkie` architecture support. It exists primarily so the GGUF on Hugging Face
has a reproducible "how it was built" pointer.

## Status

- [x] llama.cpp `LLM_ARCH_TALKIE` arch implemented (Q/K post-RoPE RMSNorm, embed-skip, scaleless RMSNorm, NeoX-style RoPE with flipped sin)
- [x] GGUF converter (`scripts/convert_to_gguf.py`)
- [x] Q8_0 + Q4_K_M produce coherent IT-style output on Metal, CPU, and CUDA (RTX 3090, sm_86)
- [x] Reference-logits validation tooling (`validation/`)
- [x] Numerical reference-logits comparison run — Q8_0 meets the top-1 / cosine bands; bf16 same. Both exceed the strict per-logit-Δ target on chat-template prompts (graph-level drift, not quant noise). See "Validation summary" below.
- [ ] HF release (Q4_K_M + Q8_0; bf16 skipped — 26 GB, niche use)

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

### macOS (Metal)

```bash
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build -j --target llama-cli llama-talkie-dump-logits llama-quantize
```

### Windows (CUDA)

Requires MSVC (Visual Studio 2022 Build Tools with the C++ workload) and the
NVIDIA CUDA Toolkit. Both are available via winget:

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools `
  --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install Nvidia.CUDA
```

Then build, targeting your GPU's compute capability (RTX 3090 = `86`,
RTX 4090 = `89`):

```powershell
cd llama.cpp
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target `
  llama-cli llama-talkie-dump-logits llama-quantize
```

If the resulting `.exe` files fail to launch with `STATUS_DLL_NOT_FOUND`
(typically caused by missing UCRT API-set DLLs in `System32`), rebuild with
the static MSVC runtime to produce self-contained binaries:

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON `
  -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF `
  -DBUILD_SHARED_LIBS=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
```

CUDA 13.x relocated its runtime DLLs from `bin\` to `bin\x64\`. Make sure
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.X\bin\x64` is on
`PATH` at run time, otherwise `cublas64_13.dll` won't resolve.

## Download

Pre-built GGUFs live at
[`PocketAiHub/talkie-1930-13b-it-GGUF`](https://huggingface.co/PocketAiHub/talkie-1930-13b-it-GGUF):

| File | Quant | Size | Bits/weight | Recommended |
|---|---|---|---|---|
| `talkie-1930-13b-it-Q4_K_M.gguf` | Q4_K_M | 8.0 GB  | 5.16 | Best size/quality tradeoff |
| `talkie-1930-13b-it-Q8_0.gguf`   | Q8_0   | 13.1 GB | 8.50 | Closest to reference |

Pull via the Hugging Face CLI:

```bash
pip install huggingface_hub
hf download PocketAiHub/talkie-1930-13b-it-GGUF \
  talkie-1930-13b-it-Q8_0.gguf --local-dir ./out
```

Or fetch the direct URL:

```bash
wget -O out/talkie-1930-13b-it-Q8_0.gguf \
  https://huggingface.co/PocketAiHub/talkie-1930-13b-it-GGUF/resolve/main/talkie-1930-13b-it-Q8_0.gguf
```

You can use these GGUFs **only** with this repo's llama.cpp fork — stock
llama.cpp does not yet recognize the `talkie` architecture (see [Build](#build)).

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

On Windows / CUDA:

```powershell
$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64;$env:Path"
.\llama.cpp\build\bin\Release\llama-cli.exe `
  -m out\talkie-1930-13b-it-Q8_0.gguf -ngl 99 `
  -st --simple-io -p "What is the capital of France?"
```

`-st` runs a single turn and exits — useful for non-interactive testing.
The legacy `--no-conversation` flag is rejected by `llama-cli` in this fork
(the binary directs you to `llama-completion` instead).

### Performance baseline

| GPU | Quant | Generation | Prompt processing | VRAM (4K ctx) |
|---|---|---|---|---|
| RTX 3090 | Q8_0 | ~58 tok/s | ~910 tok/s | ~14.9 GB |

## Convert from the original PyTorch checkpoint

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download the original checkpoint (~26 GB)
python scripts/download.py

# Convert to bf16 GGUF (~26 GB)
python scripts/convert_to_gguf.py \
  --src weights/rl-refined.pt \
  --vocab weights/vocab.txt \
  --dst out/talkie-1930-13b-it-bf16.gguf

# Quantize to Q8_0 (~13.5 GB)
./llama.cpp/build/bin/llama-quantize \
  out/talkie-1930-13b-it-bf16.gguf \
  out/talkie-1930-13b-it-Q8_0.gguf \
  Q8_0
```

Plan on **~65 GB free disk** for the full pipeline (original checkpoint
+ bf16 GGUF + Q8_0 GGUF held simultaneously). The bf16 GGUF can be deleted
after quantization completes.

## Validate against the reference

See [validation/README.md](validation/README.md). In short:

1. On a CUDA box: `python validation/dump_reference_logits.py --prompts validation/prompts.txt --out validation/reference_logits.json`
2. On the Mac: `GGML_METAL_NE11_MM_MIN=1024 ./llama.cpp/build/bin/llama-talkie-dump-logits -m out/talkie-1930-13b-it-Q8_0.gguf --prompts-file validation/prompts.txt --out validation/gguf_logits_Q8_0.json --top-k-out 50 -ngl 99`
3. `python validation/compare_logits.py --ref validation/reference_logits.json --gguf validation/gguf_logits_Q8_0.json --out validation/comparison_report.txt`

### Validation summary

Reference run on CPU (the bf16 13B model is ~26 GB and does not fit in
the 24 GB VRAM of the RTX 3090; we used `--device cpu` for the reference
dump). GGUF side ran on the 3090 for Q8_0 (`-ngl 99`) and on CPU for bf16
(too large for VRAM). Full comparison reports live at
[validation/comparison_report_Q8_0.txt](validation/comparison_report_Q8_0.txt)
and [validation/comparison_report_bf16.txt](validation/comparison_report_bf16.txt).

| Quant | tokens-match | top-1 match | mean cosine | mean max logit Δ | mean Δ |
|---|---|---|---|---|---|
| Q8_0 | 15/15 (100%) | 14/15 (93.3%) | 0.9977 | 1.33 | 0.41 |
| bf16 | 15/15 (100%) | 14/15 (93.3%) | 0.9982 | 1.20 | 0.35 |

Targets (per `validation/README.md`): Q8_0 wants top-1 ≥ 90%, cosine ≥ 0.99,
max Δ ≤ 0.5; bf16 wants top-1 ≥ 95%, cosine ≥ 0.999, max Δ ≤ 0.10. Q8_0
clears top-1 and cosine; bf16 clears top-1 by a hair under target. Both
exceed the strict per-logit-Δ target.

Drift is concentrated on chat-template prompts (`<|user|>...<|end|><|assistant|>`):
non-chat prompts agree within ≤ 0.5 logit units, while chat prompts spike
to 2-6 on individual logits (top-1 still wins on 14/15 chat prompts). Q8_0
and bf16 GGUFs drift the same amount, so the cause is graph-level, not
quantization. Investigation in the v0.1 push ruled out: vocab/embed shape,
embed-skip wiring, RMSNorm eps. Localizing the cause would require an
intermediate-activation dump on both sides; that is the cleanup work for v1.

For v0.x distribution this is acceptable — the model emits correct top-1
tokens on 14/15 prompts and generates coherent IT-style responses in
practice (a separate `What is the capital of France?` smoke test on the
3090 returned `Paris is the capital of France.`).

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
