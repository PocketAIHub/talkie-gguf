"""Join reference (CUDA) and GGUF (Mac) logit dumps and report per-prompt
agreement statistics.

The two inputs share the same JSON shape (`{model, top_k, results: [...]}`);
each result has `prompt`, `token_ids`, and `next_token: [{id, logit}, ...]`.

Reported per prompt:
  * Whether the prompts tokenized to the same ids on both sides
  * Top-1 agreement (yes/no)
  * Top-K Jaccard overlap
  * Mean rank of GGUF top-1 in reference top-K
  * Cosine similarity of the top-K logit vectors (intersected ids only)
  * Logit absolute diff (max + mean) on the intersected ids

Designed for low-vision-friendly output: a small text report, not a wall of
numbers. Run after copying both JSON files into this directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def fmt_pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{n}/{d} ({100.0 * n / d:.1f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True, help="reference (CUDA) JSON")
    ap.add_argument("--gguf", type=Path, required=True, help="GGUF JSON")
    ap.add_argument("--out", type=Path, default=Path("comparison_report.txt"))
    args = ap.parse_args()

    ref = json.loads(args.ref.read_text())
    gguf = json.loads(args.gguf.read_text())

    ref_by_prompt = {r["prompt"]: r for r in ref["results"]}
    gguf_by_prompt = {r["prompt"]: r for r in gguf["results"]}

    prompts = [r["prompt"] for r in ref["results"] if r["prompt"] in gguf_by_prompt]

    lines: list[str] = []
    p = lines.append
    p(f"reference model : {ref.get('model','?')}")
    p(f"gguf model      : {gguf.get('model','?')}")
    p(f"prompts compared: {len(prompts)} / ref={len(ref['results'])} gguf={len(gguf['results'])}")
    p("")

    top1_match = 0
    tok_match = 0
    sum_jaccard = 0.0
    sum_cosine = 0.0
    sum_rank = 0
    rank_count = 0
    sum_max_diff = 0.0
    sum_mean_diff = 0.0

    for i, prompt in enumerate(prompts):
        rr = ref_by_prompt[prompt]
        gr = gguf_by_prompt[prompt]
        ref_ids = [t["id"] for t in rr["next_token"]]
        gguf_ids = [t["id"] for t in gr["next_token"]]
        ref_logit = {t["id"]: t["logit"] for t in rr["next_token"]}
        gguf_logit = {t["id"]: t["logit"] for t in gr["next_token"]}

        same_tokens = rr["token_ids"] == gr["token_ids"]
        if same_tokens:
            tok_match += 1

        same_top1 = ref_ids[0] == gguf_ids[0]
        if same_top1:
            top1_match += 1

        # Top-K Jaccard overlap.
        inter = set(ref_ids) & set(gguf_ids)
        union = set(ref_ids) | set(gguf_ids)
        jaccard = len(inter) / len(union) if union else float("nan")
        sum_jaccard += jaccard

        # Rank of GGUF's top-1 in reference's top-K (or K+1 if absent).
        try:
            r = ref_ids.index(gguf_ids[0]) + 1
            sum_rank += r
            rank_count += 1
        except ValueError:
            r = None  # outside reference top-K

        # Logit diff on intersected ids.
        if inter:
            shared = sorted(inter)
            ref_vec = [ref_logit[i] for i in shared]
            ggf_vec = [gguf_logit[i] for i in shared]
            cos = cosine(ref_vec, ggf_vec)
            sum_cosine += cos
            diffs = [abs(a - b) for a, b in zip(ref_vec, ggf_vec)]
            max_d = max(diffs)
            mean_d = sum(diffs) / len(diffs)
            sum_max_diff += max_d
            sum_mean_diff += mean_d
        else:
            cos = float("nan")
            max_d = float("nan")
            mean_d = float("nan")

        snippet = prompt.replace("\n", " ").replace("<|", "<|")
        if len(snippet) > 60:
            snippet = snippet[:57] + "..."
        p(f"--- prompt {i:2d}: {snippet}")
        p(f"    tokens-match : {'yes' if same_tokens else 'no  (ref={} gguf={})'.format(rr['token_ids'][:8], gr['token_ids'][:8])}")
        p(f"    top1 ref     : id={ref_ids[0]:6d} logit={ref_logit[ref_ids[0]]:+.3f}")
        p(f"    top1 gguf    : id={gguf_ids[0]:6d} logit={gguf_logit[gguf_ids[0]]:+.3f} {'MATCH' if same_top1 else 'MISS'}")
        rank_str = f"rank={r}" if r is not None else "rank=>K"
        p(f"    overlap      : jaccard={jaccard:.3f}  cos={cos:.4f}  {rank_str}")
        p(f"    logit diff   : max={max_d:+.3f}  mean={mean_d:+.3f}")
        p("")

    n = len(prompts)
    p("=" * 60)
    p("SUMMARY")
    p("=" * 60)
    p(f"  tokens-match        : {fmt_pct(tok_match, n)}")
    p(f"  top1-match          : {fmt_pct(top1_match, n)}")
    p(f"  mean jaccard top-K  : {sum_jaccard / n:.3f}" if n else "")
    p(f"  mean cosine sim     : {sum_cosine / n:.4f}" if n else "")
    p(f"  mean GGUF-top1 rank : {sum_rank / rank_count:.2f}" if rank_count else "  mean rank: n/a")
    p(f"  mean max logit diff : {sum_max_diff / n:.3f}" if n else "")
    p(f"  mean mean logit diff: {sum_mean_diff / n:.3f}" if n else "")

    report = "\n".join(lines)
    args.out.write_text(report)
    print(report)
    print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
