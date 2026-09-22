"""Okapi BM25 over the same chunks BGE-M3 indexes (EXPERIMENT_PLAN lexical arm).

Pure stdlib + numpy, no model loaded. Deliberately does NOT go through
``pharma_vision_rag.retriever``'s ``__init__`` (that pulls in torch/docling and takes
minutes) — load this file directly with ``importlib.util.spec_from_file_location``,
the same trick ``scripts/15_rebuild_text_index.py`` uses for ``chunking.py``.

Tokenizer keeps ASCII word/number tokens (``3,303``, ``29.2%``, ``q2``, ``il-13``) and
lowercases everything. Korean text has no ASCII/digit overlap with the English corpus,
so Korean queries only match on product names, figures and periods already written in
Latin/digits (e.g. "ALTUVIIIO", "Q2 2025") — that is a real ceiling of lexical
matching against an English-only corpus, not a bug.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,-][a-z0-9]+)*%?")
STOPWORDS = frozenset("""
the a an of in on at to for is are was were be been by with as that this it its from
what which how many much did do does has have had and or not no we you your their they
he she his her than then there these those but if so can could will would should about
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75) over ``context + "\\n" + text`` of each chunk."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        doc_tokens = [tokenize(f"{c.get('context', '')}\n{c['text']}") for c in chunks]
        self.doc_len = np.array([len(t) for t in doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(chunks) else 0.0

        df: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(doc_tokens):
            tf = Counter(toks)
            for term, f in tf.items():
                self.postings[term].append((i, f))
                df[term] += 1
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 30) -> list[dict[str, Any]]:
        q_terms = set(tokenize(query)) & self.postings.keys()
        if not q_terms:
            return []
        scores = np.zeros(len(self.chunks))
        for term in q_terms:
            idf = self.idf[term]
            for doc_idx, f in self.postings[term]:
                dl = self.doc_len[doc_idx]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf * (f * (self.k1 + 1)) / denom
        hit_idx = np.flatnonzero(scores)
        top = hit_idx[np.argsort(-scores[hit_idx])][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def load_or_build(chunks_path: Path, cache_path: Path | None = None) -> BM25Index:
    """BM25Index over a chunks .jsonl, pickled next to it after the first build.

    Building over the 15k-chunk corpus takes ~1.5 s and the agent CLI (scripts/19) is a new process
    per tool call, so it rebuilt the index every call. The cache is keyed on the chunks file's size
    and mtime and rebuilt when either changes. It is a local derived file written by this function.
    """
    chunks_path = Path(chunks_path)
    cache_path = Path(cache_path) if cache_path else chunks_path.with_suffix(".bm25.pkl")
    st = chunks_path.stat()
    stamp = (st.st_size, st.st_mtime_ns)
    if cache_path.exists():
        try:
            state = pickle.loads(cache_path.read_bytes())
            if state.pop("_stamp", None) == stamp:
                idx = BM25Index.__new__(BM25Index)
                idx.__dict__.update(state)
                return idx
        except Exception:  # noqa: BLE001 - corrupt or foreign cache: rebuild below
            pass
    chunks = [json.loads(l) for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = BM25Index(chunks)
    state = {**vars(idx), "postings": dict(idx.postings), "_stamp": stamp}
    tmp = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")  # parallel agents may build at once
    tmp.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(tmp, cache_path)
    return idx


def _self_check() -> None:
    chunks = [
        {"source": "a.pdf", "page": 1, "context": "", "text": "Dupixent sales were 3,303 million euros in Q2 2025."},
        {"source": "a.pdf", "page": 2, "context": "", "text": "Kesimpta sales grew steadily during the quarter."},
        {"source": "b.pdf", "page": 1, "context": "", "text": "ALTUVIIIO revenue increased 3,304 million, a distractor number."},
    ]
    idx = BM25Index(chunks)
    assert tokenize("Q2 2025, IL-13 up 29.2% (3,303)") == ["q2", "2025", "il-13", "up", "29.2%", "3,303"]

    hits = idx.search("3,303")
    assert hits and (hits[0]["source"], hits[0]["page"]) == ("a.pdf", 1), "exact-number query must rank the right chunk first"
    assert (hits[0]["source"], hits[0]["page"]) != ("b.pdf", 1), "must not match the distractor's different number"

    assert idx.search("altuviiio")[0]["source"] == "b.pdf"
    assert idx.search("완전히 무관한 한국어 질의 xyzzy_nomatch") == [], "no matching token must return an empty list, not arbitrary chunks"
    print("bm25 self-check ok:", len(chunks), "chunks")


if __name__ == "__main__":
    _self_check()
