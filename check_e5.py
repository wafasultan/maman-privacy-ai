"""
Ma'man — E5 compatibility check inside the app environment (.venv-mlx), before E5 becomes the default.

1. Versions: torch was added; mlx, mlx-lm, transformers, numpy must be the ones the app already used.
2. Same vectors: the app's E5 (transformers + torch, mean pooling) vs the vectors evaluated in the retriever
   experiment (sentence-transformers, outputs/e5_cache.npz) — cosine and top-3 neighbours for all 102 test columns.
3. Qwen + E5 together, the way the app runs them: on ONE worker thread, called from other threads; load,
   classify, free, load again (the app swaps models) — no thread/stream error, memory released.
Writes outputs/check_e5.txt; exits 1 on any failure.
"""

import concurrent.futures as cf
import gc
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maman_common import ROOT, E5Embedder, Retriever, load_labeled  # noqa: E402

lines, failures = [], []


def say(msg):
    print(msg, flush=True)
    lines.append(msg)


def check(ok, msg):
    say(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        failures.append(msg)


def rss_now_mb():
    """Current resident memory of this process (ps), unlike ru_maxrss which only ever goes up."""
    import os
    import subprocess
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip() or 0) / 1024


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)


def main():
    import importlib.metadata as md
    import mlx.core as mx
    say("1. Versions: " + ", ".join(f"{p} {md.version(p)}" for p in ("mlx", "mlx-lm", "transformers", "numpy", "torch")))

    say("2. Same vectors as the evaluated E5")
    cache = np.load(ROOT / "outputs" / "e5_cache.npz")
    texts, ref = cache["texts"].tolist(), cache["vectors"]
    emb = E5Embedder()
    mine = emb.encode(texts)
    cos = (mine * ref).sum(1)
    check(cos.min() > 0.999, f"cosine with the evaluated vectors: min {cos.min():.6f}, mean {cos.mean():.6f} ({len(texts)} texts)")
    kb, test = load_labeled({"train", "valid"}), load_labeled({"test"})
    live, cached = Retriever(kb, backend="e5", strict=True), Retriever(kb, backend="e5_cache")
    same = sum([n["column"] for n in live.search(c, 3, exclude_dataset=True)] ==
               [n["column"] for n in cached.search(c, 3, exclude_dataset=True)] for c in test)
    check(same == len(test), f"top-3 neighbours identical to the evaluated run for {same}/{len(test)} test columns")
    del emb, live, cached
    gc.collect()

    say("3. Qwen + E5 in one process, one worker thread (as in the app)")
    from maman_classifier import MamanClassifier, default_retriever
    check(default_retriever() == "e5", f"default retriever = {default_retriever()}")
    pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen")
    holder = {}

    def load():
        holder["clf"] = MamanClassifier("rag_finetuned", adapter_path=ROOT / "adapters_rag", retriever_strict=True)
        return holder["clf"].retriever.backend

    def classify(c):
        return holder["clf"].classify(c)

    def free():
        holder.clear()
        gc.collect()
        mx.clear_cache()
    sample = [c for c in test if c["dataset"] in ("sa_arabizi", "sa_bank_loans", "pima_diabetes")][:6]
    rss_rounds = []
    for round_ in (1, 2, 3):
        t = time.time()
        backend = pool.submit(load).result()
        check(backend == "e5", f"round {round_}: classifier loaded with retriever {backend} in {time.time() - t:.1f}s")
        errors = []

        def from_other_thread(c):
            try:
                return pool.submit(classify, c).result()
            except Exception as e:  # the old bug was "There is no Stream(cpu, …) in current thread"
                errors.append(f"{type(e).__name__}: {e}")
        callers = [threading.Thread(target=from_other_thread, args=(c,)) for c in sample]
        [x.start() for x in callers]
        [x.join() for x in callers]
        check(not errors, f"round {round_}: {len(sample)} columns classified from {len(sample)} different threads"
              + (f" — {errors[0]}" if errors else ""))
        say(f"   peak RSS so far {rss_mb():.0f} MB · MLX active {mx.get_active_memory() / 2**20:.0f} MB · "
            f"MLX peak {mx.get_peak_memory() / 2**20:.0f} MB")
        pool.submit(free).result()
        rss_rounds.append(rss_now_mb())
        say(f"   after free: MLX active {mx.get_active_memory() / 2**20:.0f} MB · process memory now {rss_rounds[-1]:.0f} MB")
    check(mx.get_active_memory() / 2**20 < 200, "Qwen memory released after free")
    check(rss_rounds[2] - rss_rounds[1] < 150, f"process memory stable across model swaps "
          f"(after each free: {', '.join(f'{x:.0f}' for x in rss_rounds)} MB)")

    say(f"\n{'ALL PASSED' if not failures else str(len(failures)) + ' FAILED'}")
    (ROOT / "outputs" / "check_e5.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
