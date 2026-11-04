import os
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tf_keras').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from text_utils import STOPWORDS, expand_query, tokenize
from pipeline import build_retrievers

NOT_FOUND = "Информация в регламенте не найдена."

SYSTEM_PROMPT = (
    "Ты — ассистент по технологическому регламенту установки ЭЛОУ-АВТ. "
    "Отвечай ТОЛЬКО на основе приведённых ниже фрагментов регламента. "
    f"Если в них нет ответа на вопрос, ответь ровно: «{NOT_FOUND}» и больше ничего не добавляй. "
    "Не используй знания, которых нет во фрагментах, не придумывай числа. "
    "Числа и единицы измерения приводи без изменений. "
    "После каждого утверждения указывай номер источника в квадратных скобках, например [1]. "
    "Отвечай кратко, на русском языке."
)

def format_context(chunks_with_scores):
    parts = []
    for n, (c, _) in enumerate(chunks_with_scores, 1):
        head = f"[{n}] документ: {c['document']}; раздел: {c['section']}"
        parts.append(f"{head}\n{c['text']}")
    return "\n\n".join(parts)

def build_prompt(question, chunks_with_scores):
    return f"Фрагменты регламента:\n\n{format_context(chunks_with_scores)}\n\nВопрос: {question}\nОтвет:"

def sources_list(chunks_with_scores):
    return [{"n": n, "document": c["document"], "section": c["section"],
             "id": c["id"], "score": round(float(s), 4)} for n, (c, s) in enumerate(chunks_with_scores, 1)]

def _extractive(question, ctx, min_overlap=0.34, max_sent=2):
    q_terms = set(tokenize(expand_query(question))) - set(tokenize(" ".join(STOPWORDS)))
    q_core = set(tokenize(question))
    cands = []
    for n, (c, _) in enumerate(ctx, 1):
        for piece in re.split(r"(?<=[.!?])\s+|\n+", c["text"]):
            piece = piece.strip(" |-")
            if len(piece) < 15: continue
            terms = set(tokenize(piece))
            if not q_core: continue
            overlap = len(q_core & terms) / len(q_core)
            cands.append((overlap, n, piece))
    cands.sort(key=lambda x: -x[0])
    if not cands or cands[0][0] < min_overlap:
        return NOT_FOUND
    return " ".join(f"{p} [{n}]" for _, n, p in cands[:max_sent])

_llama_llm = None

def _llama_cpp(prompt):
    global _llama_llm
    if _llama_llm is None:
        from llama_cpp import Llama
        model_path = os.getenv("LLAMA_MODEL_PATH", "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        print(f"Загрузка GGUF модели из {model_path}...")
        _llama_llm = Llama(model_path=model_path, n_ctx=4096, n_threads=4, verbose=False)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    output = _llama_llm.create_chat_completion(messages=messages, max_tokens=300, temperature=0.1)
    return output['choices'][0]['message']['content'].strip()

def generate_answer(question, ctx, llm="extractive"):
    if llm == "extractive":
        return _extractive(question, ctx)
    prompt = build_prompt(question, ctx)
    if llm == "llama":
        return _llama_cpp(prompt)
    raise ValueError(f"Неизвестный бэкенд LLM: {llm}")

def answer(question, retriever, k=5, llm="extractive", expand=True):
    q = expand_query(question) if expand else question
    hits = retriever.search(q, k)
    ctx = [(retriever.chunks[i], s) for i, s in hits]
    return {"question": question, "answer": generate_answer(question, ctx, llm), "sources": sources_list(ctx), "context": ctx}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--llm", default="llama", choices=["extractive", "llama"])
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    
    print("Инициализация поисковиков...")
    r = build_retrievers()["hybrid"]
    
    print(f"\nВопрос: {a.question}")
    res = answer(a.question, r, k=a.k, llm=a.llm)
    
    print("\n" + "="*50)
    print("ОТВЕТ:")
    print(res["answer"])
    print("="*50)
    
    print("\nИСТОЧНИКИ:")
    for s in res["sources"]:
        print(f"[{s['n']}] {s['document']} — {s['section']}")
        