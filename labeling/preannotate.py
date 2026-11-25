"""Bôi sẵn cụm từ (gợi ý) cho Label Studio bằng lexicon.json.

    python3 labeling/preannotate.py --input data/raw/tiki_reviews.csv --n 20 --out data/processed/preannotated_20.json

Chỉ dùng để gợi ý: người gán vẫn phải kiểm tra, sửa rồi Submit.
Có thể import `annotate(text)` từ script khác (ví dụ baseline luật).
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from ls_common import Span, spans_to_result, words, write_json

# Hai kiểu bỏ dấu cùng tồn tại trong dữ liệu: "khoá"/"khóa", "hoà"/"hòa", "thuỷ"/"thủy".
# Khi so khớp, quy cả văn bản lẫn lexicon về một kiểu (chỉ để so, không sửa văn bản gốc).
_TONE = {"à": "ò", "á": "ó", "ả": "ỏ", "ã": "õ", "ạ": "ọ"}
_TONE_E = {"è": "ò", "é": "ó", "ẻ": "ỏ", "ẽ": "õ", "ẹ": "ọ"}
_TONE_Y = {"ỳ": "ù", "ý": "ú", "ỷ": "ủ", "ỹ": "ũ", "ỵ": "ụ"}
_TONE_RE = re.compile(r"o([àáảãạ])$|o([èéẻẽẹ])$|u([ỳýỷỹỵ])$")


def canon(w):
    """Chữ thường + NFC + quy kiểu bỏ dấu: "khoá" → "khóa", "thuỷ" → "thủy"."""
    w = unicodedata.normalize("NFC", w.lower())

    def fix(m):
        if m.group(1):
            return _TONE[m.group(1)] + "a"
        if m.group(2):
            return _TONE_E[m.group(2)] + "e"
        return _TONE_Y[m.group(3)] + "y"

    return _TONE_RE.sub(fix, w)


L = json.load(open(Path(__file__).with_name("lexicon.json"), encoding="utf-8"))
tup = lambda ws: {tuple(canon(x) for x in w.split()) for w in ws}
ASP, NOT = tup(L["aspect"]), tup(L["not_aspect"])
OP = {tuple(canon(x) for x in w.split()): p for p in ("pos", "neg", "neu") for w in L["op_" + p]}
NEG, BEF, AFT = tup(L["negation"]), tup(L["intensifier"]["before"]), tup(L["intensifier"]["after"])
FLIP = {"pos": "neg", "neg": "pos", "neu": "neu"}
BREAK = {".", "!", "?", ";", "nhưng"}
# Cụm dài nhất trong lexicon (vd "không có gì đặc biệt" = 5 từ), không ghi cứng
MAXN = max(len(k) for lex in (ASP, NOT, OP, NEG, BEF, AFT) for k in lex)


def match(low, i, lex):
    for n in range(min(MAXN, len(low) - i), 0, -1):
        if tuple(low[i:i + n]) in lex:
            return n
    return 0


def opinion(low, i):
    best = (0, 0, 0, 0)
    for neg in {0, match(low, i, NEG)}:
        for bef in {0, match(low, i + neg, BEF)}:
            j = i + neg + bef
            n = match(low, j, OP)
            if n:
                k = j + n + match(low, j + n, AFT)
                best = max(best, (k - i, n, neg, j))
    return best


def annotate(text):
    toks = words(text)
    low = [canon(text[a:b]) for a, b in toks]
    found, i = [], 0
    while i < len(low):
        size, n, neg, j = opinion(low, i)
        a, m = match(low, i, ASP), match(low, i, NOT)
        if m and m >= max(size, a):
            i += m
        elif size and size >= a:
            pol = OP[tuple(low[j:j + n])]
            found.append((i, i + size, "OP_" + (FLIP[pol] if neg else pol).upper()))
            i += size
        elif a:
            found.append((i, i + a, "ASPECT"))
            i += a
        else:
            i += 1
    clause = [sum(w in BREAK for w in low[:t]) for t in range(len(low))]
    spans = [Span(toks[a][0], toks[b - 1][1], lab, text[toks[a][0]:toks[b - 1][1]]) for a, b, lab in found]
    pairs = []
    for o, (a, _, lab) in enumerate(found):
        if lab == "ASPECT":
            continue
        cand = [(abs(x - a), s) for s, (x, _, l) in enumerate(found) if l == "ASPECT" and clause[x] == clause[a]]
        if cand:
            pairs.append((min(cand)[1], o))
    return spans, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    # NFC: một số review lưu dấu rời (NFD) → \w tách sai từ, lexicon không khớp
    df["content"] = df.content.fillna("").map(lambda t: unicodedata.normalize("NFC", t))
    df = df[df.content.fillna("").str.split().str.len().between(10, 80)].sample(args.n, random_state=42)
    rows = json.loads(df[["review_id", "content", "rating", "product_name", "category"]].to_json(orient="records", force_ascii=False))
    tasks, counts = [], []
    for r in rows:
        spans, pairs = annotate(r["content"])
        tasks.append({"data": r, "predictions": [{"result": spans_to_result(spans, pairs)}]})
        counts.append(len(spans))
    write_json(Path(args.out), tasks)
    print(f"{len(tasks)} câu, trung bình {sum(counts) / len(counts):.1f} cụm/câu, {counts.count(0) / len(counts):.0%} câu không có gợi ý")


if __name__ == "__main__":
    main()
