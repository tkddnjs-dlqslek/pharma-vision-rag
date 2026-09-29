"""Upload the corpus files the MCP container (services/mcp) reads to a Hugging Face dataset repo, at their repo-relative paths.

    eval/corpus.json
    data/embeddings/v2/text/text_chunks.jsonl
    data/pdf/corpus/*.pdf                      (27 files, 95 MB)
    data/embeddings/vision_index_pooled/*      (2.4 GB, vectors.npy goes through LFS)

Files already in the repo with the same byte size are skipped, so a re-run after an interrupted upload only sends
the rest. Needs `huggingface-cli login` (or HF_TOKEN in the environment) with a write token.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/30_upload_corpus_data.py --repo <user>/pharma-corpus-data --private
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["eval/corpus.json", "data/embeddings/v2/text/text_chunks.jsonl"]
FOLDERS = ["data/pdf/corpus", "data/embeddings/vision_index_pooled"]


def wanted(root: Path = ROOT) -> list[Path]:
    """Every file services/mcp downloads, as paths relative to root (posix), missing ones reported at once."""
    rel = [Path(f) for f in FILES]
    for folder in FOLDERS:
        d = root / folder
        if not d.is_dir():
            sys.exit(f"missing folder {d}")
        rel += sorted(p.relative_to(root) for p in d.iterdir() if p.is_file())
    missing = [p for p in rel if not (root / p).is_file()]
    if missing:
        sys.exit(f"missing files: {missing}")
    return rel


def remote_sizes(api, repo: str) -> dict[str, int]:
    return {f.path: f.size for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(f, "size")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="dataset repo id, e.g. <user>/pharma-corpus-data")
    ap.add_argument("--private", action="store_true", help="create the repo as private (default public)")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    files = wanted()
    url = api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"repo {url}")
    have = remote_sizes(api, args.repo)
    todo = [p for p in files if have.get(p.as_posix()) != (ROOT / p).stat().st_size]
    total = sum((ROOT / p).stat().st_size for p in todo)
    print(f"{len(files)} files wanted, {len(files) - len(todo)} already there, {len(todo)} to upload ({total / 1e6:.0f} MB)")
    for i, p in enumerate(todo, 1):
        t = time.time()
        api.upload_file(path_or_fileobj=str(ROOT / p), path_in_repo=p.as_posix(), repo_id=args.repo, repo_type="dataset",
                        commit_message=f"add {p.as_posix()}")
        print(f"  [{i}/{len(todo)}] {p.as_posix()} ({(ROOT / p).stat().st_size / 1e6:.1f} MB) in {time.time() - t:.0f}s")
    bad = [p for p in files if remote_sizes(api, args.repo).get(p.as_posix()) != (ROOT / p).stat().st_size] if todo else []
    if bad:
        sys.exit(f"size mismatch after upload: {bad}")
    print("done")


if __name__ == "__main__":
    main()
