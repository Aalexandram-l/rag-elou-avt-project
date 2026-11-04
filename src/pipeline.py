import json
from pathlib import Path

from chunking import ROOT, chunk_structured
from hybrid_retrieval import HybridRetriever
from retrieval import DEFAULT_MODEL, BM25Retriever, DenseRetriever


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(chunks, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path: Path = ROOT / "data" / "chunks.jsonl"):
    if not path.exists():
        chunks, _ = chunk_structured(ROOT / "data" / "corpus")
        save_jsonl(chunks, path)
    return load_jsonl(path)


def build_retrievers(chunks=None, backend="auto", model=DEFAULT_MODEL, alpha=0.5, fusion="weighted", use_rerank=True):
    chunks = chunks or load_chunks()
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, backend=backend, model_name=model)
    hybrid = HybridRetriever(bm25, dense, alpha=alpha, fusion=fusion, use_rerank=use_rerank)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}
