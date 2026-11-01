"""Retrieval index over the training HuBERT features. Built on CPU."""
import sys
from pathlib import Path

import faiss
import numpy as np

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
chunks = []
for path in sorted(src.glob("*.npy")):
    if path.stem.startswith("mute"):
        continue
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 768:
        continue
    chunks.append(np.ascontiguousarray(arr, dtype=np.float32))
big = np.concatenate(chunks, axis=0)
print(f"vectors {big.shape}", flush=True)
index = faiss.IndexFlatL2(768)
step = 65536
for start in range(0, big.shape[0], step):
    index.add(big[start : start + step])
dst.parent.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(dst))
print(f"wrote {dst} rows={index.ntotal}", flush=True)
