import argparse
import json
import re
from pathlib import Path
from text_utils import approx_tokens

ROOT = Path(__file__).resolve().parent.parent
CHARS_PER_PAGE = 2400
MAX_TOKENS = 800

def parse_markdown(text: str, doc_name: str):
    lines = text.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    h1 = h2 = h3 = ""
    path = doc_name
    blocks, sections = [], []
    sec_start = 0
    cur_path = path
    def close_section(end):
        sections.append({"document": doc_name, "path": cur_path, "start": sec_start, "end": end})
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^(#{1,3})\s+(.*)$", ln)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            close_section(offsets[i])
            if level == 1: h1, h2, h3 = title, "", ""
            elif level == 2: h2, h3 = title, ""
            else: h3 = title
            cur_path = " > ".join(x for x in (h1, h2, h3) if x)
            sec_start = offsets[i]
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"): j += 1
            blocks.append({"kind": "table", "text": "\n".join(lines[i:j]), "path": cur_path, "start": offsets[i], "end": offsets[j - 1] + len(lines[j - 1])})
            i = j
            continue
        j = i
        while (j < len(lines) and lines[j].strip() and not lines[j].startswith("#") and not lines[j].lstrip().startswith("|")): j += 1
        blocks.append({"kind": "para", "text": "\n".join(lines[i:j]), "path": cur_path, "start": offsets[i], "end": offsets[j - 1] + len(lines[j - 1])})
        i = j
    close_section(len(text))
    return blocks, sections

def parse_table(md: str):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in md.split("\n")]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    return rows[0], rows[1:]

def linearize_table(header, rows):
    out = []
    for r in rows:
        cells = [f"{h}: {v}" for h, v in zip(header, r)]
        out.append("; ".join(cells))
    return "\n".join(out)

def chunk_structured(corpus_dir: Path, max_tokens: int = MAX_TOKENS):
    chunks, sections_all = [], []
    for fp in sorted(corpus_dir.glob("*.md")):
        text = fp.read_text(encoding="utf-8")
        blocks, sections = parse_markdown(text, fp.name)
        sections_all += [s for s in sections if s["end"] > s["start"]]
        by_path = {}
        for b in blocks: by_path.setdefault(b["path"], []).append(b)
        n = 0
        for path, bl in by_path.items():
            buf, buf_tokens = [], 0
            def flush():
                nonlocal buf, buf_tokens, n
                if not buf: return
                body = "\n\n".join(b["text"] for b in buf)
                n += 1
                chunks.append({"id": f"{fp.stem}#{n:03d}", "document": fp.name, "section": path, "type": "text", "text": body, "index_text": f"{path}\n{body}", "page": buf[0]["start"] // CHARS_PER_PAGE + 1, "char_start": buf[0]["start"], "char_end": buf[-1]["end"], "n_tokens": approx_tokens(body)})
                buf, buf_tokens = [], 0
            consumed = set()
            for k, b in enumerate(bl):
                if b["kind"] == "table" and k > 0 and bl[k - 1]["kind"] == "para" and len(bl[k - 1]["text"]) <= 400: consumed.add(k - 1)
            for k, b in enumerate(bl):
                if b["kind"] == "para" and (k in consumed or b["text"].startswith(">")): continue
                if b["kind"] == "table":
                    flush()
                    header, rows = parse_table(b["text"])
                    prev = bl[k - 1]["text"] if k > 0 and bl[k - 1]["kind"] == "para" else ""
                    desc = prev if len(prev) <= 400 else prev[:400]
                    n += 1
                    lin = linearize_table(header, rows)
                    chunks.append({"id": f"{fp.stem}#{n:03d}", "document": fp.name, "section": path, "type": "table", "text": b["text"], "table_columns": header, "table_description": desc, "index_text": f"{path}\n{desc}\nКолонки: {'; '.join(header)}\n{lin}", "page": b["start"] // CHARS_PER_PAGE + 1, "char_start": b["start"], "char_end": b["end"], "n_tokens": approx_tokens(b["text"])})
                    continue
                t = approx_tokens(b["text"])
                if buf and buf_tokens + t > max_tokens: flush()
                buf.append(b)
                buf_tokens += t
            flush()
    return chunks, sections_all

def save_jsonl(chunks, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus"))
    ap.add_argument("--out", default=str(ROOT / "data" / "chunks.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = ap.parse_args()
    chunks, sections = chunk_structured(Path(args.corpus), args.max_tokens)
    save_jsonl(chunks, Path(args.out))
    with open(ROOT / "data" / "sections.json", "w", encoding="utf-8") as f: json.dump(sections, f, ensure_ascii=False, indent=1)
    tables = sum(c["type"] == "table" for c in chunks)
    print(f"Чанков: {len(chunks)} (таблиц: {tables}); разделов: {len(sections)} -> {args.out}")

if __name__ == "__main__": main()
