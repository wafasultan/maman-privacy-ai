"""
Ma'man — retriever experiment, step 1: E5 vectors for every labelled column (train, valid AND test queries).

Runs in its own environment (.venv-e5: torch + sentence-transformers) so the main MLX environment is untouched.
Same text and same encoding as Retriever(backend="e5"): "query: " + retrieval_text, normalised.
Output: outputs/e5_cache.npz  (texts, vectors). Test columns are encoded only as QUERIES; they are never added
to the knowledge base (that is decided in eval_retriever.py, which asserts it).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from maman_common import ROOT, load_labeled, retrieval_text  # noqa: E402

MODEL = "intfloat/multilingual-e5-small"


def main():
    from sentence_transformers import SentenceTransformer

    cols = load_labeled({"train", "valid", "test"})
    texts = sorted({retrieval_text(c) for c in cols})
    model = SentenceTransformer(MODEL)
    vectors = model.encode([f"query: {t}" for t in texts], normalize_embeddings=True, batch_size=32,
                           show_progress_bar=True)
    out = ROOT / "outputs" / "e5_cache.npz"
    np.savez(out, texts=np.array(texts), vectors=np.asarray(vectors, dtype=np.float32))
    print(f"saved {len(texts)} vectors ({vectors.shape[1]} dims) from {MODEL} to {out}")


if __name__ == "__main__":
    main()
