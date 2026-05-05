"""Inspect talkie-1930-13b-it checkpoint.

Decides Path A (new llama.cpp arch) vs Path B (fold + approximate as Llama).
The deciding question: are the embed_skip scalars near zero post-RL? If so,
the embedding-skip term can be dropped without architectural support.

Writes a small text report -- safe for low-vision users to skip and
have summarized.
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "weights" / "rl-refined.pt"
REPORT = ROOT / "reports" / "inspection.txt"


def load_state_dict() -> dict[str, torch.Tensor]:
    ckpt = torch.load(str(CKPT), map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def fmt_stats(t: torch.Tensor) -> str:
    f = t.float()
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={f.min().item():+.4e} max={f.max().item():+.4e} "
        f"mean={f.mean().item():+.4e} absmax={f.abs().max().item():.4e}"
    )


def main() -> None:
    if not CKPT.exists():
        raise SystemExit(f"checkpoint not found: {CKPT}")

    sd = load_state_dict()
    lines: list[str] = []
    p = lines.append

    p(f"checkpoint: {CKPT}")
    p(f"top-level keys: {len(sd)}")
    p("")

    # 1) Tensor inventory: name, shape, dtype, total params.
    p("=" * 78)
    p("TENSOR INVENTORY")
    p("=" * 78)
    total_params = 0
    by_kind: dict[str, int] = {}
    for k, t in sd.items():
        n = t.numel()
        total_params += n
        kind = (
            "embed" if k == "embed.weight"
            else "lm_head" if k.startswith("lm_head")
            else "block.attn" if ".attn." in k
            else "block.mlp" if ".mlp." in k
            else "block.gain" if ".a_g" in k or ".w_g" in k or ".head_g" in k
            else "other"
        )
        by_kind[kind] = by_kind.get(kind, 0) + n
    p(f"total parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
    for kind, n in sorted(by_kind.items(), key=lambda x: -x[1]):
        p(f"  {kind:<14} {n:>14,} ({n/total_params*100:5.2f}%)")
    p("")

    # 2) All keys (sorted, deduplicated by block index).
    p("=" * 78)
    p("ALL TENSOR NAMES (block 0 + non-block tensors)")
    p("=" * 78)
    seen_blocks = set()
    block_keys: list[str] = []
    other_keys: list[str] = []
    for k, t in sd.items():
        if k.startswith("blocks."):
            idx = int(k.split(".")[1])
            stripped = ".".join(["blocks.{i}"] + k.split(".")[2:])
            if stripped not in seen_blocks:
                seen_blocks.add(stripped)
                block_keys.append((stripped, t.shape, t.dtype))
            if idx == 0:
                pass
        else:
            other_keys.append((k, t.shape, t.dtype))
    for k, shape, dt in other_keys:
        p(f"  {k:<40} {tuple(shape)} {dt}")
    p("")
    p(f"per-block tensors ({len(block_keys)} unique, repeated 40x):")
    for k, shape, dt in block_keys:
        p(f"  {k:<40} {tuple(shape)} {dt}")
    p("")

    # 3) Critical scalars: embed_skip per layer (the deciding question).
    p("=" * 78)
    p("EMBED_SKIP per layer (init=0.0; if still ~0 -> Path B viable)")
    p("=" * 78)
    skip_vals: list[float] = []
    for i in range(40):
        k = f"blocks.{i}.embed_skip.a_g"
        if k in sd:
            v = sd[k].float().item()
            skip_vals.append(v)
            p(f"  layer {i:2d}: {v:+.6e}")
    if skip_vals:
        absmax = max(abs(v) for v in skip_vals)
        mean = sum(skip_vals) / len(skip_vals)
        p(f"  -> absmax={absmax:.4e}  mean={mean:+.4e}")
        if absmax < 1e-3:
            p("  VERDICT: embed_skip is effectively zero -> SAFE TO DROP")
        elif absmax < 1e-2:
            p("  VERDICT: embed_skip is small -> dropping likely OK, validate carefully")
        else:
            p("  VERDICT: embed_skip is significant -> Path A required")
    p("")

    # 4) ActGains -- attn_gain, mlp_gain (these always fold into adjacent linears).
    p("=" * 78)
    p("ATTN_GAIN / MLP_GAIN per layer (fold into resid linears)")
    p("=" * 78)
    p(f"  {'layer':<8} {'attn_gain':>14} {'mlp_gain':>14}")
    for i in range(40):
        ag = sd.get(f"blocks.{i}.attn_gain.a_g")
        mg = sd.get(f"blocks.{i}.mlp_gain.a_g")
        ag_v = f"{ag.float().item():+.4e}" if ag is not None else "MISSING"
        mg_v = f"{mg.float().item():+.4e}" if mg is not None else "MISSING"
        p(f"  {i:<8} {ag_v:>14} {mg_v:>14}")
    p("")

    # 5) HeadGain stats per layer (folds into Q linear by row scaling).
    p("=" * 78)
    p("HEAD_GAIN per layer (40 heads each; fold into Q rows)")
    p("=" * 78)
    p(f"  {'layer':<8} {'min':>12} {'max':>12} {'mean':>12} {'std':>12}")
    for i in range(40):
        hg = sd.get(f"blocks.{i}.attn.head_gain.head_g")
        if hg is None:
            p(f"  {i:<8} MISSING")
            continue
        f = hg.float()
        p(f"  {i:<8} {f.min().item():+.4e} {f.max().item():+.4e} {f.mean().item():+.4e} {f.std().item():.4e}")
    p("")

    # 6) lm_head_gain (folds into lm_head).
    p("=" * 78)
    p("LM_HEAD_GAIN")
    p("=" * 78)
    if "lm_head_gain.w_g" in sd:
        v = sd["lm_head_gain.w_g"].float().item()
        p(f"  lm_head_gain.w_g = {v:+.6e}")
    p("")

    # 7) Big tensors -- spot-check stats.
    p("=" * 78)
    p("BIG TENSOR SPOT-CHECK (block 0, lm_head, embed)")
    p("=" * 78)
    for k in [
        "embed.weight",
        "blocks.0.attn.attn_query.weight",
        "blocks.0.attn.attn_key.weight",
        "blocks.0.attn.attn_value.weight",
        "blocks.0.attn.attn_resid.weight",
        "blocks.0.mlp.mlp_gate.weight",
        "blocks.0.mlp.mlp_linear.weight",
        "blocks.0.mlp.mlp_resid.weight",
        "lm_head",
    ]:
        if k in sd:
            p(f"  {k:<42} {fmt_stats(sd[k])}")
    p("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
