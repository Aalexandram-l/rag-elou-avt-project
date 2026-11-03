import numpy as np

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(-x)
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    return r

class HybridRetriever:
    def __init__(self, bm25, dense, alpha=0.5, fusion="weighted", rrf_k=60, use_rerank=True):
        self.bm25, self.dense = bm25, dense
        self.chunks = bm25.chunks
        self.alpha, self.fusion, self.rrf_k = alpha, fusion, rrf_k
        self.use_rerank = use_rerank
        self.name = f"hybrid({fusion}, a={alpha})" if fusion == "weighted" else "hybrid(rrf)"
        
        if self.use_rerank:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('DiTy/cross-encoder-russian-msmarco')
            except Exception as e:
                print(f"Не удалось загрузить CrossEncoder: {e}")
                self.use_rerank = False

    def scores(self, query: str) -> np.ndarray:
        sb, sd = self.bm25.scores(query), self.dense.scores(query)
        if self.fusion == "rrf":
            return 1.0 / (self.rrf_k + _ranks(sb)) + 1.0 / (self.rrf_k + _ranks(sd))
        return self.alpha * _minmax(sd) + (1 - self.alpha) * _minmax(sb)

    def search(self, query: str, k: int = 5):
        s = self.scores(query)
        top_idx = np.argsort(-s)[:20]
        
        if self.use_rerank and hasattr(self, 'reranker'):
            pairs = [[query, self.chunks[i]["index_text"]] for i in top_idx]
            rerank_scores = self.reranker.predict(pairs)
            sorted_idx = np.argsort(-rerank_scores)[:k]
            return [(int(top_idx[i]), float(rerank_scores[i])) for i in sorted_idx]
        
        return [(int(i), float(s[i])) for i in top_idx[:k]]
    