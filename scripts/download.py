"""Download talkie-1930-13b-it checkpoint + vocab from HuggingFace."""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "talkie-lm/talkie-1930-13b-it"
DEST = Path(__file__).resolve().parent.parent / "weights"
DEST.mkdir(parents=True, exist_ok=True)

for filename in ("rl-refined.pt", "vocab.txt", "README.md"):
    print(f"[download] {filename} -> {DEST}", flush=True)
    path = hf_hub_download(
        repo_id=REPO,
        filename=filename,
        local_dir=str(DEST),
    )
    print(f"[done]     {path}", flush=True)

print("[download] all files complete", flush=True)
