import os
import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tf_keras').setLevel(logging.ERROR)

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import BM25Retriever, DenseRetriever

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"

def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    print("Загрузка чанков...")
    chunks = load_chunks()
    print(f"Загружено {len(chunks)} чанков.\n")

    query = "При какой температуре отбирается керосиновая фракция на атмосферной колонне?"
    print(f"Тестовый запрос: {query}\n")

    print("--- BM25 ---")
    bm25 = BM25Retriever(chunks)
    for idx, score in bm25.search(query, k=3):
        print(f"Score: {score:.3f} | ID: {chunks[idx]['id']}")
        print(f"Текст: {chunks[idx]['text'][:150]}...\n")

    print("--- Dense (sentence-transformers) ---")
    dense = DenseRetriever(chunks, backend="auto") 
    for idx, score in dense.search(query, k=3):
        print(f"Score: {score:.3f} | ID: {chunks[idx]['id']}")
        print(f"Текст: {chunks[idx]['text'][:150]}...\n")
