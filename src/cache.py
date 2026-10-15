import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/cache")
CACHE_VERSION = "v1"  # bump this if you change extraction/chunking logic


def make_key(file_bytes, model_name, chunk_size, overlap):
    """Hash of file content + every setting that changes the output."""
    config = f"{CACHE_VERSION}|{model_name}|{chunk_size}|{overlap}"
    return hashlib.sha256(file_bytes + config.encode()).hexdigest()


def load(key):
    """Return (chunks, embeddings) or None on a miss / corrupt entry."""
    folder = CACHE_DIR / key
    chunks_path = folder / "chunks.json"
    emb_path = folder / "embeddings.npy"

    if not (chunks_path.exists() and emb_path.exists()):
        return None

    try:
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(emb_path)
    except Exception:
        return None

    # Sanity check: vectors and text must line up one-to-one
    if len(chunks) != len(embeddings):
        return None

    return chunks, embeddings


def save(key, chunks, embeddings):
    folder = CACHE_DIR / key
    folder.mkdir(parents=True, exist_ok=True)

    # Write embeddings first, chunks.json last. load() needs both,
    # so a crash mid-write just looks like a cache miss.
    np.save(folder / "embeddings.npy", np.asarray(embeddings, dtype="float32"))
    (folder / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )