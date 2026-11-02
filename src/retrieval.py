import hashlib
import warnings
from pathlib import Path
import numpy as np
from text_utils import tokenize

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "cointegrated/rubert-tiny2"

try:
    from rank_bm25 import BM25Okapi as _BM25
except ImportError:
    class _BM25:
        def __init__(self, corpus, k1=1.5, b=0.75):
            self.k1, self.b, self.corpus = k1, b, corpus
            self.N = len(corpus)
            self.dl = np.array([len(d) for d in corpus], dtype=float)
            self.avgdl = self.dl.mean() if self.N else 0.0
            df = {}
            for d in corpus:
                for t in set(d):
                    df[t] = df.get(t, 0) + 1
            self.idf = {t: np.log((self.N - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}
            self.tf = [self._tf(d) for d in corpus]
        @staticmethod
        def _tf(doc):
            tf = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            return tf
        def get_scores(self, query):
            scores = np.zeros(self.N)
            for t in query:
                if t not in self.idf:
                    continue
                for i, tf in enumerate(self.tf):
                    f = tf.get(t)
                    if f:
                        denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                        scores[i] += self.idf[t] * f * (self.k1 + 1) / denom
            return scores

class BM25Retriever:
    name = "bm25"
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.bm25 = _BM25([tokenize(c["index_text"]) for c in chunks], k1=k1, b=b)
    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)
    def search(self, query: str, k: int = 5):
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

class DenseRetriever:
    name = "dense"
    def __init__(self, chunks, backend="auto", model_name=DEFAULT_MODEL, cache_dir=None):
        self.chunks = chunks
        self.model_name = model_name
        self.backend = backend
        self._st = None
        texts = [c["index_text"] for c in chunks]
        if backend in ("auto", "st"):
            try:
                self._init_st(texts, cache_dir)
                self.backend = "st"
            except Exception as e:
                if backend == "st":
                    raise
                warnings.warn(f"sentence-transformers недоступен ({e.__class__.__name__}); использую заглушку LSA")
                self.backend = "lsa"
        if self.backend == "lsa":
            self._init_lsa(texts)
        self._init_faiss()
    def _prefix(self, kind):
        return {"q": "query: ", "d": "passage: "}[kind] if "e5" in self.model_name.lower() else ""
    def _init_st(self, texts, cache_dir):
        from sentence_transformers import SentenceTransformer
        self._st = SentenceTransformer(self.model_name)
        key = hashlib.md5((self.model_name + "".join(texts)).encode()).hexdigest()[:10]
        cache = Path(cache_dir or ROOT / "data") / f"emb_{key}.npy"
        if cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self._st.encode([self._prefix("d") + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
            np.save(cache, self.emb)
    def _init_lsa(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        X = self._tfidf.fit_transform(texts)
        n = max(2, min(96, X.shape[0] - 1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n, random_state=0)
        self.emb = normalize(self._svd.fit_transform(X)).astype("float32")
    def _init_faiss(self):
        self._faiss = None
        try:
            import faiss
            self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self._faiss.add(np.ascontiguousarray(self.emb, dtype="float32"))
        except Exception:
            pass
    def _embed_query(self, query):
        if self.backend == "st":
            return self._st.encode([self._prefix("q") + query], normalize_embeddings=True)[0]
        from sklearn.preprocessing import normalize
        return normalize(self._svd.transform(self._tfidf.transform([query])))[0]
    def scores(self, query: str) -> np.ndarray:
        q = np.asarray(self._embed_query(query), dtype="float32")
        if self._faiss is not None:
            sims, ids = self._faiss.search(q[None, :], len(self.chunks))
            out = np.zeros(len(self.chunks))
            out[ids[0]] = sims[0]
            return out
        return self.emb @ q
    def search(self, query: str, k: int = 5):
        s = self.scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]
