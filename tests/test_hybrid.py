import json
import sys
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tf_keras').setLevel(logging.ERROR)

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retrieval import BM25Retriever, DenseRetriever
from hybrid_retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    print("Загрузка чанков...")
    chunks = load_chunks()
    print(f"Загружено {len(chunks)} чанков.\n")

    print("Инициализация поисковиков...")
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, backend="auto")

    query = "При какой температуре отбирается керосиновая фракция на атмосферной колонне?"
    print(f"Запрос: {query}\n")

    print("=== Подбор веса (Weighted) ===")
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        hybrid = HybridRetriever(bm25, dense, alpha=alpha, fusion="weighted")
        results = hybrid.search(query, k=1) # смотрим только 1-й результат
        top_idx, top_score = results[0]
        print(f"alpha={alpha:.1f} | Топ-1: {chunks[top_idx]['id']} (Score: {top_score:.3f})")

    print("\n=== Переранжирование (RRF) ===")
    hybrid_rrf = HybridRetriever(bm25, dense, fusion="rrf")
    for rank, (idx, score) in enumerate(hybrid_rrf.search(query, k=3), 1):
        print(f"Место {rank}: {chunks[idx]['id']} (Score: {score:.4f})")
