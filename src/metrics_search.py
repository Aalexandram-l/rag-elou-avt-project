import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tf_keras').setLevel(logging.ERROR)

sys.path.append(str(Path(__file__).resolve().parent))

from chunking import ROOT, chunk_structured
from hybrid_retrieval import HybridRetriever
from retrieval import DEFAULT_MODEL, BM25Retriever, DenseRetriever
from text_utils import expand_query

def load_queries(path=ROOT / "data" / "queries.json"):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def relevant_ids(chunks, sections, query):
    rel = set()
    for lab in query["relevant"]:
        # Разбиваем название раздела из разметки на слова
        lab_words = set(lab["section"].lower().replace(".", " ").split())
        for i, c in enumerate(chunks):
            if c["document"] == lab["document"]:
                # Разбиваем section чанка на слова
                c_words = set(c["section"].lower().replace(">", " ").replace(".", " ").split())
                # Если все слова из разметки есть в section чанка
                if lab_words.issubset(c_words):
                    rel.add(i)
                    break
    return rel

def evaluate(retriever, chunks, sections, queries, k=5, k_mrr=10, expand=False):
    rows = []
    for q in queries:
        if not q["answerable"]: continue
        text = expand_query(q["question"]) if expand else q["question"]
        ranked = [i for i, _ in retriever.search(text, k_mrr)]
        rel = relevant_ids(chunks, sections, q)
        first = next((r for r, i in enumerate(ranked, 1) if i in rel), None)
        rows.append({"id": q["id"], "category": q["category"], "question": q["question"],
                     "hit@k": int(first is not None and first <= k),
                     "hit@3": int(first is not None and first <= 3),
                     "rr": (1.0 / first) if first else 0.0, "first_rank": first or "-"})
    return rows

def summarize(rows, k=5):
    n = len(rows)
    return {f"Recall@{k}": sum(r["hit@k"] for r in rows) / n,
            "Recall@3": sum(r["hit@3"] for r in rows) / n,
            "MRR": sum(r["rr"] for r in rows) / n, "n": n}

def by_category(rows):
    g = defaultdict(list)
    for r in rows: g[r["category"]].append(r)
    return {c: summarize(v) for c, v in g.items()}

def md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in body]
    return "\n".join(out)

def fmt(x): return f"{x:.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "st", "lsa"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--fusion", default="weighted", choices=["weighted", "rrf"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    chunks, sections = chunk_structured(ROOT / "data" / "corpus")
    queries = load_queries()
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, backend=args.backend, model_name=args.model)
    hybrid = HybridRetriever(bm25, dense, alpha=args.alpha, fusion=args.fusion, use_rerank=True)
    
    report = [f"# Метрики поиска (Recall@5, MRR)\n\nЧанков: {len(chunks)}; запросов: {len(queries)}.\n"]
    methods = [("BM25", bm25), ("Vector", dense), (f"Hybrid (alpha={args.alpha})", hybrid)]
    
    body, per_query = [], []
    for name, r in methods:
        rows = evaluate(r, chunks, sections, queries, k=args.k)
        s = summarize(rows, args.k)
        body.append([name, fmt(s[f"Recall@{args.k}"]), fmt(s["Recall@3"]), fmt(s["MRR"])])
        for row in rows: per_query.append({"method": name, **row})
    report.append("## 1. BM25 vs Vector vs Hybrid\n\n" + md_table(["Метод", f"Recall@{args.k}", "Recall@3", "MRR"], body) + "\n")

    cats = sorted({q["category"] for q in queries if q["answerable"]})
    body = []
    for name, r in methods:
        bc = by_category(evaluate(r, chunks, sections, queries, k=args.k))
        body.append([name] + [f"{fmt(bc[c][f'Recall@{args.k}'])} / {fmt(bc[c]['MRR'])}" for c in cats])
    report.append(f"### По категориям запросов (Recall@{args.k} / MRR)\n\n" + md_table(["Метод"] + cats, body) + "\n")

    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query[0].keys()))
        w.writeheader(); w.writerows(per_query)

    text = "\n".join(report)
    (out / "metrics_search.md").write_text(text, encoding="utf-8")
    print(text)

if __name__ == "__main__":
    main()
