import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GLOSSARY_PATH = ROOT / "data" / "glossary.json"

_WORD = re.compile(r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*(?:,[0-9]+)?", re.I)
_CODE = re.compile(r"^[a-zа-яё]{1,3}-\d+$|\d")

STOPWORDS = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между какие каком какое какого какую которые который которая которое является также нужно опишите происходит какими каким такое""".split())

_ENDINGS = sorted(["ыми", "ими", "ого", "его", "ому", "ему", "ами", "ями", "ая", "яя", "ое", "ее", "ые", "ие", "ой", "ей", "ий", "ый", "ую", "юю", "ом", "ем", "ах", "ях", "ов", "ев", "ам", "ям", "ия", "ии", "ть", "ся", "ет", "ут", "ют", "ит", "ат", "ят", "а", "я", "ы", "и", "у", "ю", "е", "о", "ь", "й"], key=len, reverse=True)

try:
    import snowballstemmer
    _snow = snowballstemmer.stemmer("russian")
except Exception:
    _snow = None

def stem(word: str) -> str:
    if _CODE.search(word) or len(word) <= 3:
        return word
    if _snow is not None:
        return _snow.stemWord(word)
    for end in _ENDINGS:
        if word.endswith(end) and len(word) - len(end) >= 4:
            return word[: -len(end)]
    return word

def tokenize(text: str, use_stem: bool = True, drop_stop: bool = True) -> list[str]:
    tokens = []
    for raw in _WORD.findall(text.lower().replace("ё", "е")):
        parts = [raw] if _CODE.search(raw) else raw.split("-")
        for p in parts:
            if not p or (drop_stop and p in STOPWORDS):
                continue
            tokens.append(stem(p) if use_stem else p)
    return tokens

def approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.6)

_glossary_cache = None

def load_glossary(path: Path = GLOSSARY_PATH) -> dict:
    global _glossary_cache
    if _glossary_cache is None:
        with open(path, encoding="utf-8") as f:
            _glossary_cache = json.load(f)
    return _glossary_cache

def expand_query(query: str) -> str:
    g = load_glossary()
    low = query.lower()
    extra = []
    for rule in g["synonyms"]:
        if re.search(rule["pattern"], low):
            extra.append(rule["add"])
    for abbr, full in g["abbreviations"].items():
        if re.search(rf"(?<![А-Яа-яЁёA-Za-z]){re.escape(abbr)}(?![А-Яа-яЁёA-Za-z])", query):
            extra.append(full)
    return f"{query} {' '.join(extra)}" if extra else query
