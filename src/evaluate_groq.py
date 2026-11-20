import json
import sys
import os
from pathlib import Path
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tf_keras').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent))

from pipeline import build_retrievers
from generate import answer
from chunking import ROOT

QUERIES_PATH = ROOT / "data" / "queries.json"
OUT_DIR = ROOT / "results"
OUT_FILE = OUT_DIR / "metrics_generation_groq.md"


def main():
    print("Загрузка поисковиков...")
    r = build_retrievers(use_rerank=True)["hybrid"]

    with open(QUERIES_PATH, encoding="utf-8-sig") as f:
        queries = json.load(f)

    print("Загрузка Qwen (судья)...")
    from llama_cpp import Llama

    model_path = ROOT / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )

    results = []

    for q in queries:
        print(f"[{q['id']}] {q['question']}...")

        # Генерация ответа через Groq API
        res = answer(q["question"], r, k=3, llm="groq")
        system_answer = res["answer"]
        print(f"  [DEBUG] Ответ Groq: {system_answer!r}")
        print(f"  [DEBUG] Эталон: {q.get('note', '')!r}")

        if not q["answerable"]:
            is_correct = 2 if "не найдена" in system_answer.lower() else 0
            results.append({
                "id": q["id"],
                "accuracy": is_correct,
                "hallucination": 0 if is_correct == 2 else 1
            })
            print(f"  -> Ловушка: {'ОК' if is_correct == 2 else 'ОШИБКА'}")
            continue

        reference = q.get("note", "")
        judge_prompt = f"""Вопрос: {q['question']}
Эталонный ответ: {reference}
Ответ системы: {system_answer}

Оцени ответ по шкале 0-2 (только цифру): 0-неверно, 1-частично, 2-верно.
Есть ли галлюцинации (0-нет, 1-есть)?
Формат: "Оценка: X, Галлюцинация: Y"."""

        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=20,
            temperature=0.0
        )

        judge_text = out["choices"][0]["message"]["content"].strip()

        try:
            acc = int(judge_text.split("Оценка:")[1].split(",")[0].strip())
            hall = int(judge_text.split("Галлюцинация:")[1].strip())
        except:
            acc, hall = 0, 1

        results.append({
            "id": q["id"],
            "accuracy": acc,
            "hallucination": hall
        })
        print(f"  -> Оценка: {acc}, Галлюцинация: {hall}")

    avg_acc = sum(x["accuracy"] for x in results) / len(results)
    hall_rate = sum(x["hallucination"] for x in results) / len(results)

    OUT_DIR.mkdir(exist_ok=True)

    report_lines = [
        "# Метрики генерации (Groq API)",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Answer Accuracy | **{avg_acc:.2f} / 2.0 ({(avg_acc/2)*100:.1f}%)** |",
        f"| Hallucination Rate | **{hall_rate:.2f} ({hall_rate*100:.1f}%)** |",
        "",
        "## Детали по вопросам",
        "",
        "| ID | Оценка | Галлюцинация |",
        "|---|---|---|",
    ]

    for r in results:
        report_lines.append(
            f"| {r['id']} | {r['accuracy']} | {r['hallucination']} |"
        )

    OUT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 40)
    print(f"Answer Accuracy: {avg_acc:.2f} / 2.0 ({(avg_acc/2)*100:.1f}%)")
    print(f"Hallucination Rate: {hall_rate:.2f} ({hall_rate*100:.1f}%)")
    print("=" * 40)
    print(f"\nОтчет сохранен в {OUT_FILE}")


if __name__ == "__main__":
    main()
