"""scripts/24_vision_index_gpu.py --embed-now must produce exactly what the two-step path produces.

Synthetic: three tiny PDFs made with pypdfium2 (every page a different size, so patch counts vary like the real
corpus) and a fake model whose embedding is a deterministic function of the rendered image. No CUDA, no downloads.
`evaluate` is not covered here: it needs a GPU.

Run: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tests/test_vision_index_gpu.py
 or: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_vision_index_gpu.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import torch  # embed_pages_gpu.to_fp16 needs real tensors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import embed_pages_gpu as ep  # noqa: E402

_spec = importlib.util.spec_from_file_location("_test_vidx", ROOT / "scripts" / "24_vision_index_gpu.py")
vidx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vidx)  # type: ignore[union-attr]

DIM = 8
SCALE = 0.5
DOCS = [("a.pdf", 2), ("b.pdf", 3), ("c.pdf", 4)]


def fake_emb(blob: bytes):
    h = int(hashlib.sha256(blob).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    return torch.from_numpy(rng.standard_normal((3 + h % 5, DIM)).astype(np.float32))  # 3..7 patches


class FakeModel:
    """Same duck type as the Nemotron model: forward_images / forward_queries -> list of [tokens, dim] tensors."""

    def forward_images(self, imgs, batch_size=None):
        return [fake_emb(im.tobytes()) for im in imgs]

    def forward_queries(self, texts, batch_size=None):
        return [fake_emb(t.encode("utf-8")) for t in texts]


def make_bundle(d: Path) -> list[str]:
    d.mkdir(parents=True, exist_ok=True)
    for i, (name, n) in enumerate(DOCS):
        pdf = pdfium.PdfDocument.new()
        for p in range(n):
            pdf.new_page(200 + 7 * p + 11 * i, 150 + 3 * p)  # distinct size per page -> distinct patch count
        pdf.save(str(d / name))
    (d / "corpus.json").write_text(json.dumps([{"id": n, "pages": p} for n, p in DOCS]), encoding="utf-8")
    qs = [{"id": "A01", "type": "A", "q_ko": "차트 질문", "q_en": "chart question", "gold": [["a.pdf", 1]]},
          {"id": "B01", "type": "B", "q_ko": "표 질문", "q_en": "table question", "gold": [["c.pdf", 2]]}]
    (d / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs), encoding="utf-8")
    return [n for n, _ in DOCS]


def two_step(inp: Path, emb: Path, work: Path) -> dict:
    """What the pod does today: embed to disk, then build all four variants onto disk."""
    pages = ep.embed_pages(FakeModel(), inp, emb, [n for n, _ in DOCS], SCALE, 2, None)
    queries = ep.embed_queries(FakeModel(), inp / "questions.jsonl", emb)
    manifest = {"model_id": ep.MODEL_ID, "render_scale": SCALE, "dim": DIM, "pages": pages, "queries": queries}
    assert vidx.build(work, manifest, lambda p: np.load(emb / "pages" / p["source"] / f"{p['page']}.npy")) is None
    return manifest


def one_pass(inp: Path, emb: Path, work: Path, save_pages: bool = False, save_variants: bool = False):
    """--embed-now: nothing round-trips through disk; returns (manifest, {variant: dir or in-RAM dict})."""
    manifest, cache = vidx.embed_now(inp, emb, SCALE, 2, save_pages, 3, model=FakeModel())
    variants = vidx.build(work, manifest, lambda p: cache.pop((p["source"], p["page"])), save=save_variants)
    assert not cache, f"build left {len(cache)} page arrays behind"
    return manifest, variants or {v: work / v for v in vidx.VARIANTS}


def test_embed_now_matches_two_step_byte_for_byte():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = Path(tmp)
        make_bundle(t / "input")
        m_a = two_step(t / "input", t / "emb_a", t / "work_a")
        m_b, _ = one_pass(t / "input", t / "emb_b", t / "work_b", save_variants=True)
        assert m_a["pages"] == m_b["pages"] and m_a["dim"] == m_b["dim"]
        assert [q["id"] for q in m_a["queries"]] == [q["id"] for q in m_b["queries"]]
        assert len(m_a["pages"]) == sum(n for _, n in DOCS)
        assert len({p["n_patches"] for p in m_a["pages"]}) > 1, "patch counts must vary or the test proves nothing"
        for v in vidx.VARIANTS:
            names = sorted(f.name for f in (t / "work_a" / v).iterdir())
            assert names == sorted(f.name for f in (t / "work_b" / v).iterdir()), v
            for name in names:
                assert (t / "work_a" / v / name).read_bytes() == (t / "work_b" / v / name).read_bytes(), (v, name)


def test_packed_output_identical_from_ram_and_from_disk():
    """The in-RAM variant must pack into exactly what the on-disk variant packs into, parts and hashes included."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = Path(tmp)
        make_bundle(t / "input")
        two_step(t / "input", t / "emb_a", t / "work_a")
        _, ram = one_pass(t / "input", t / "emb_b", t / "work_b")
        assert not (t / "work_b").exists() or not any((t / "work_b").iterdir()), "one pass must write no variants"
        for v in vidx.VARIANTS:
            assert isinstance(ram[v], dict)
            part_gb = (1 + max(1, vidx.variant_size(ram[v]) // 4)) / 1e9   # forces several parts on a tiny corpus
            vidx.pack(t / "work_a" / v, t / f"dl_a_{v}", part_gb)
            vidx.pack(ram[v], t / f"dl_b_{v}", part_gb)
            names = sorted(f.name for f in (t / f"dl_a_{v}").iterdir())
            assert names == sorted(f.name for f in (t / f"dl_b_{v}").iterdir()), v
            assert sum(n.startswith("vectors.npy.part") for n in names) >= 2, names
            for name in names:
                assert (t / f"dl_a_{v}" / name).read_bytes() == (t / f"dl_b_{v}" / name).read_bytes(), (v, name)


def test_packed_index_is_what_script_25_expects():
    """join_parts + LocalVisionIndex over the packed one-pass output, i.e. the local check end to end."""
    import importlib.util as iu

    from pharma_vision_rag.retriever.vision_local import LocalVisionIndex

    spec = iu.spec_from_file_location("_test_check25", ROOT / "scripts" / "25_check_vision_index.py")
    chk = iu.module_from_spec(spec)
    spec.loader.exec_module(chk)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = Path(tmp)
        make_bundle(t / "input")
        manifest, ram = one_pass(t / "input", t / "emb", t / "work")
        for v in ("pooled_int8", "full_fp16"):
            dl = t / f"dl_{v}"
            vidx.pack(ram[v], dl, (1 + max(1, vidx.variant_size(ram[v]) // 3)) / 1e9)
            chk.join_parts(dl)                                   # verifies every sha256, joins, deletes the parts
            assert LocalVisionIndex.available(dl) and not list(dl.glob("*.part*"))
            enc = {q["text"]: np.load(t / "emb" / "queries" / f"{q['id']}.npy") for q in manifest["queries"]}
            idx = LocalVisionIndex(dl, encoder=enc.__getitem__)
            hits = idx.search(manifest["queries"][0]["text"], k=3)
            assert len(hits) == 3 and idx.meta["variant"] == v
            assert len(idx.pages) == sum(n for _, n in DOCS)


def test_save_pages_writes_the_two_step_files():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = Path(tmp)
        make_bundle(t / "input")
        two_step(t / "input", t / "emb_a", t / "work_a")
        one_pass(t / "input", t / "emb_b", t / "work_b", save_pages=True, save_variants=True)
        want = sorted(p.relative_to(t / "emb_a").as_posix() for p in (t / "emb_a" / "pages").rglob("*.npy"))
        got = sorted(p.relative_to(t / "emb_b").as_posix() for p in (t / "emb_b" / "pages").rglob("*.npy"))
        assert want == got and len(want) == sum(n for _, n in DOCS)
        for rel in want:
            assert (t / "emb_a" / rel).read_bytes() == (t / "emb_b" / rel).read_bytes(), rel


def test_embed_now_writes_queries_and_manifest_for_scoring():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = Path(tmp)
        make_bundle(t / "input")
        m, _ = one_pass(t / "input", t / "emb", t / "work")
        assert json.loads((t / "emb" / "manifest.json").read_text(encoding="utf-8"))["pages"] == m["pages"]
        for q in m["queries"]:                                   # evaluate() loads exactly these
            assert np.load(t / "emb" / "queries" / f"{q['id']}.npy").shape == (q["n_tokens"], DIM)
        assert len(m["queries"]) == 4 and not (t / "emb" / "pages").exists()


def test_diagnostics_and_probe():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        t = Path(tmp)
        w, r = vidx.throughput_probe(t, 8)
        assert w > 0 and r > 0 and not list(t.iterdir()), "probe file must be cleaned up"
        vidx.diagnostics(t, 0)                                   # must not raise without a GPU
        vidx.diagnostics(t, 8)


if __name__ == "__main__":
    fails = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
