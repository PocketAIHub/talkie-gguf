"""Dump top-K next-token logits from the reference talkie PyTorch model.

This script is intended to be run on the CUDA box (RTX 3090). It loads
talkie-1930-13b-it via the official `talkie` package, runs a forward pass
on each prompt, and saves the top-K logits at the last position to JSON.

The matching `examples/talkie-dump-logits/dump-logits.cpp` does the same
on the GGUF side. `compare_logits.py` joins the two JSON files.

Usage on the 3090:
    pip install -e /path/to/talkie  # the official talkie repo
    python dump_reference_logits.py \
        --prompts prompts.txt \
        --out reference_logits.json \
        --top-k 50

Then `scp` reference_logits.json back to the Mac.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# These imports require the official talkie package on PYTHONPATH.
from talkie.model import load_checkpoint
from talkie.tokenizer import IT_VOCAB_SIZE, build_tokenizer
from talkie.download import get_model_files


def read_prompts(path: Path) -> list[str]:
    raw = Path(path).read_text().splitlines()
    out: list[str] = []
    for line in raw:
        # support \n / \t / \\ escapes so a single line can have newlines
        line = line.rstrip("\r")
        if not line or line.startswith("#"):
            continue
        out.append(
            line.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reference_logits.json"))
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--model", default="talkie-1930-13b-it")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    prompts = read_prompts(args.prompts)
    if not prompts:
        raise SystemExit("no prompts found")

    print(f"loading {args.model} on {args.device}...", flush=True)
    ckpt_path, vocab_path = get_model_files(args.model, cache_dir=args.cache_dir)
    tokenizer = build_tokenizer(vocab_path, style="it")
    device = torch.device(args.device)
    model = load_checkpoint(str(ckpt_path), device, target_vocab_size=IT_VOCAB_SIZE)
    model.eval()

    autocast = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.no_grad()
    )

    results: list[dict] = []
    with torch.no_grad(), autocast:
        for pi, prompt in enumerate(prompts):
            ids = tokenizer.encode(prompt, allowed_special="all")
            if not ids:
                print(f"  prompt {pi}: empty token list, skipping", flush=True)
                continue
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            # The reference TalkieModel.forward returns logits at the LAST
            # position only (shape [B, V]).
            logits = model(x).float().squeeze(0)
            top_vals, top_ids = torch.topk(logits, k=args.top_k)
            top_vals = top_vals.cpu().tolist()
            top_ids = top_ids.cpu().tolist()
            results.append({
                "prompt": prompt,
                "token_ids": ids,
                "next_token": [
                    {"id": int(i), "logit": float(v)}
                    for i, v in zip(top_ids, top_vals)
                ],
            })
            print(
                f"  prompt {pi:2d} ({len(ids)} tokens): "
                f"top1 = {top_ids[0]} (logit {top_vals[0]:.3f})",
                flush=True,
            )

    payload = {
        "model": args.model,
        "top_k": args.top_k,
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
