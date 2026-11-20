
import os
import logging
import warnings
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent))

from text_utils import STOPWORDS, expand_query, tokenize
from pipeline import build_retrievers


NOT_FOUND = "Информация в регламенте не найдена."


SYSTEM_PROMPT = (
    "Ты — ассистент по технологическому регламенту установки ЭЛОУ-АВТ. "
    "Отвечай ТОЛЬКО на основе приведённых фрагментов регламента. "

    f"Если в них действительно нет ответа на вопрос, ответь ровно: «{NOT_FOUND}» "
    "и больше ничего не добавляй. "

    "Внимательно анализируй как обычный текст, так и таблицы. "
    "В таблицах сопоставляй название параметра со значением в соответствующем столбце. "
    "Учитывай заголовки столбцов: базовое значение, допустимый диапазон, норма и другие. "
    "Если нужное значение явно присутствует в таблице, обязательно используй его "
    "для ответа. Не считай информацию отсутствующей только потому, что она приведена в таблице. "

    "Если вопрос касается базового значения параметра, укажи базовое значение. "
    "Если вопрос касается допустимого диапазона, укажи весь диапазон. "
    "Если вопрос касается и базового значения, и допустимого диапазона, укажи оба. "

    "Не используй знания, которых нет во фрагментах, не придумывай числа. "

    "Если вопрос про октановое число, а в контексте нет конкретного числа "
    "(например, «95»), ответь: «Информация в регламенте не найдена». "

    "Если вопрос касается процесса, которого нет в регламенте "
    "(например, каталитический риформинг, гидроочистка, катализаторы), "
    "ответь: «Информация в регламенте не найдена». "

    "Если вопрос про диапазон температур, а в контексте есть таблица, "
    "выпиши все относящиеся к вопросу значения из таблицы. "

    "Числа и единицы измерения приводи без изменений. "

    "После каждого утверждения указывай номер источника в квадратных скобках, "
    "например [1]. "

    "Запрещено отвечать только номером источника, например [1]. "
    "Сначала обязательно сформулируй содержательный ответ на вопрос, затем укажи номер источника. "
    "Отвечай максимально кратко. Если вопрос требует одного конкретного значения, укажи только это значение с единицей измерения, без пояснений и диапазонов."
)


def format_context(chunks_with_scores):
    parts = []

    for n, (c, _) in enumerate(chunks_with_scores, 1):
        head = (
            f"[{n}] документ: {c['document']}; "
            f"раздел: {c['section']}"
        )

        # Для таблиц используем развёрнутое представление.
        # Для обычного текста используем исходный фрагмент.
        if c.get("type") == "table":
            content = c.get("index_text") or c["text"]
        else:
            content = c.get("text", "")

        parts.append(f"{head}\n{content}")

    return "\n\n".join(parts)


def build_prompt(question, chunks_with_scores):
    return (
        f"Фрагменты регламента:\n\n"
        f"{format_context(chunks_with_scores)}\n\n"
        f"Вопрос: {question}\n\n"
        "Дай конкретный ответ на вопрос, используя приведённые фрагменты. "
        "Не ограничивайся номером источника. "
        "Если значение есть в таблице, извлеки его и укажи вместе с единицей измерения. "
        "В конце ответа укажи источник в формате [1].\n\n"
        "Ответ:"
    )


def sources_list(chunks_with_scores):
    return [
        {
            "n": n,
            "document": c["document"],
            "section": c["section"],
            "id": c["id"],
            "score": round(float(s), 4)
        }
        for n, (c, s) in enumerate(chunks_with_scores, 1)
    ]


def _extractive(question, ctx, min_overlap=0.34, max_sent=2):
    q_terms = (
        set(tokenize(expand_query(question)))
        - set(tokenize(" ".join(STOPWORDS)))
    )

    q_core = set(tokenize(question))
    cands = []

    for n, (c, _) in enumerate(ctx, 1):
        for piece in re.split(r"(?<=[.!?])\s+|\n+", c["text"]):
            piece = piece.strip(" |-")

            if len(piece) < 15:
                continue

            terms = set(tokenize(piece))

            if not q_core:
                continue

            overlap = len(q_core & terms) / len(q_core)

            cands.append((overlap, n, piece))

    cands.sort(key=lambda x: -x[0])

    if not cands or cands[0][0] < min_overlap:
        return NOT_FOUND

    return " ".join(
        f"{p} [{n}]"
        for _, n, p in cands[:max_sent]
    )


_llama_llm = None


def _is_streamlit_runtime():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None

    except Exception:
        return False


def _llama_cpp(prompt):
    global _llama_llm

    if _is_streamlit_runtime():
        raise RuntimeError(
            "Локальная LLM отключена в Streamlit. Используй Groq API."
        )

    if os.getenv("ENABLE_LOCAL_LLM", "true").lower() != "true":
        raise RuntimeError(
            "Локальная LLM отключена через ENABLE_LOCAL_LLM."
        )

    if _llama_llm is None:
        from llama_cpp import Llama

        REPO_ID = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
        FILENAME = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

        ROOT = Path(__file__).resolve().parent.parent
        MODELS_DIR = ROOT / "models"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = os.getenv("LLAMA_MODEL_PATH")

        if not model_path or not os.path.isfile(model_path):
            model_path = MODELS_DIR / FILENAME

        model_path = Path(model_path)

        if not model_path.exists():
            print(f"Модель не найдена: {model_path}")
            print(f"Скачивание {FILENAME} из Hugging Face...")

            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME,
                local_dir=str(MODELS_DIR),
            )

            print(f"Модель успешно скачана: {model_path}")

        else:
            print(f"Найдена локальная модель: {model_path}")

        _llama_llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=4,
            verbose=False
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    output = _llama_llm.create_chat_completion(
        messages=messages,
        max_tokens=300,
        temperature=0.1
    )

    result = output["choices"][0]["message"]["content"].strip()

    print("\n[DEBUG] Ответ локальной модели:")
    print(repr(result))

    return result


_groq_client = None


def _groq_api(prompt):
    global _groq_client

    if _groq_client is None:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key and _is_streamlit_runtime():
            try:
                import streamlit as st
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                api_key = None

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY не найден. "
                "Добавь его в переменные окружения или Streamlit Secrets."
            )

        _groq_client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    chat_completion = _groq_client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=300
    )

    return chat_completion.choices[0].message.content.strip()


def generate_answer(question, ctx, llm="extractive"):
    if llm == "extractive":
        return _extractive(question, ctx)

    prompt = build_prompt(question, ctx)

    if llm == "llama":
        return _llama_cpp(prompt)

    if llm == "groq":
        return _groq_api(prompt)

    raise ValueError(f"Неизвестный бэкенд LLM: {llm}")


def answer(question, retriever, k=5, llm="extractive", expand=True):
    q = expand_query(question) if expand else question

    if "октановое число" in question.lower():
        hits = retriever.search(q, k)

        if not any(
            re.search(
                r"\b(9[0-9]|8[0-9])\b",
                retriever.chunks[i]["text"]
            )
            for i, _ in hits
        ):
            return {
                "question": question,
                "answer": NOT_FOUND,
                "sources": [],
                "context": []
            }

    hits = retriever.search(q, k)

    ctx = [
        (retriever.chunks[i], score)
        for i, score in hits
    ]

    generated = generate_answer(question, ctx, llm)

    return {
        "question": question,
        "answer": generated,
        "sources": sources_list(ctx),
        "context": ctx
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("question")
    ap.add_argument(
        "--llm",
        default="llama",
        choices=["extractive", "llama", "groq"]
    )
    ap.add_argument("--k", type=int, default=3)

    a = ap.parse_args()

    print("Инициализация поисковиков...")

    r = build_retrievers()["hybrid"]

    print(f"\nВопрос: {a.question}")

    res = answer(
        a.question,
        r,
        k=a.k,
        llm=a.llm
    )

    print("\n" + "=" * 50)
    print("ОТВЕТ:")
    print(res["answer"])
    print("=" * 50)

    print("\nИСТОЧНИКИ:")

    for s in res["sources"]:
        print(
            f"[{s['n']}] {s['document']} — {s['section']}"
        )
