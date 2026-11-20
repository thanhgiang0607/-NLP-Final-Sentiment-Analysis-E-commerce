"""Gộp file export Label Studio, đo độ đồng thuận giữa người gán, xuất câu bị lệch (người chạy: B).

Đặt toàn bộ file export của một đợt vào một thư mục, tên đúng quy ước
`<đợt>_<người>_<chinh|cheo|pilot>.json`. Phải export dạng **JSON** (không dùng JSON-MIN,
vì JSON-MIN không giữ quan hệ giữa các cụm từ).

Với bài trích xuất, Kappa kiểu "chọn ô" không dùng được, nên đo theo 2 cách:
  1. F1 giữa 2 người (coi người này là đáp án, người kia là dự đoán; F1 đối xứng nên
     đổi vai không đổi kết quả) cho: cụm khía cạnh, cụm ý kiến, cặp, bộ ba.
  2. Cohen's Kappa trên nhãn BIO của từng từ (Fleiss' Kappa khi pilot 4 người).

Ví dụ:
    python labeling/agreement.py --exports labeling/exports/w3

Đầu ra (mặc định reports/agreement/<đợt>/):
    <đợt>_main.jsonl          bảng chính: mỗi dòng 1 câu, có spans + triplets
    <đợt>_agreement.csv       chỉ số đồng thuận, cho cả nhóm (ALL) và từng cặp người
    <đợt>_disagreements.csv   các câu 2 người gán khác nhau, liệt kê bộ ba chỉ 1 người có
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ls_common import POLARITY_VI, Span, Triplet, bio_tags, parse_result, read_json

ROOT = Path(__file__).resolve().parents[1]
FILE_RE = re.compile(r"^(?P<phase>.+)_(?P<person>[A-Za-z0-9]+)_(?P<role>chinh|cheo|pilot)\.json$")


# ---------- đọc export ----------

def load_exports(folder: Path) -> tuple[str, list[dict]]:
    rows, phases = [], set()
    files = sorted(folder.glob("*.json"))
    if not files:
        sys.exit(f"Không có file .json trong {folder}")
    for path in files:
        m = FILE_RE.match(path.name)
        if not m:
            print(f"! Bỏ qua {path.name}: tên không đúng quy ước <đợt>_<người>_<chinh|cheo|pilot>.json")
            continue
        phases.add(m["phase"])
        items = read_json(path)
        if items and "annotations" not in items[0]:
            sys.exit(f"{path.name} không phải export dạng JSON đầy đủ (có thể là JSON-MIN). Export lại bằng JSON.")
        n_done = 0
        for it in items:
            anns = [a for a in it.get("annotations", []) if not a.get("was_cancelled")]
            if not anns:
                continue
            ann = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
            data = it.get("data", {})
            text = data.get("content", "")
            spans, triplets, is_spam, warns = parse_result(ann.get("result", []), text)
            for w in warns:
                print(f"    ! {path.name} review {data.get('review_id')}: {w}")
            rows.append({
                "review_id": str(data.get("review_id")), "content": text,
                "labeler": m["person"], "role": m["role"],
                "spans": spans, "triplets": triplets, "is_spam": is_spam,
            })
            n_done += 1
        print(f"  {path.name:28} {n_done:5}/{len(items)} câu đã gán")
    if len(phases) != 1:
        sys.exit(f"Thư mục chứa nhiều đợt khác nhau: {sorted(phases)}")
    if not rows:
        sys.exit("Chưa có câu nào được gán")
    return phases.pop(), rows


# ---------- so khớp ----------

def span_key(s: Span | None, with_label: bool = True):
    if s is None:
        return None
    return (s.start, s.end, s.label) if with_label else (s.start, s.end)


def overlap(a: Span | None, b: Span | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.start < b.end and b.start < a.end


def count_match(gold: list, pred: list, same) -> int:
    """Ghép 1–1 theo kiểu tham lam, trả về số cặp khớp."""
    used, hit = set(), 0
    for g in gold:
        for j, p in enumerate(pred):
            if j not in used and same(g, p):
                used.add(j)
                hit += 1
                break
    return hit


def prf(tp: int, n_gold: int, n_pred: int) -> float:
    if n_gold + n_pred == 0:
        return np.nan
    return 2 * tp / (n_gold + n_pred)


ITEMS = {
    # tên chỉ số: (lấy danh sách từ 1 câu, hàm so khớp CHÍNH XÁC, hàm so khớp MỘT PHẦN)
    "aspect": (
        lambda r: [s for s in r["spans"] if s.kind == "ASP"],
        lambda a, b: span_key(a, False) == span_key(b, False),
        overlap,
    ),
    "opinion": (  # chỉ xét vị trí cụm ý kiến, chưa xét cảm xúc
        lambda r: [s for s in r["spans"] if s.kind == "OP"],
        lambda a, b: span_key(a, False) == span_key(b, False),
        overlap,
    ),
    "pair": (  # (khía cạnh, ý kiến), chưa xét cảm xúc
        lambda r: r["triplets"],
        lambda a, b: span_key(a.aspect, False) == span_key(b.aspect, False)
        and span_key(a.opinion, False) == span_key(b.opinion, False),
        lambda a, b: overlap(a.aspect, b.aspect) and overlap(a.opinion, b.opinion),
    ),
    "triplet": (  # chỉ số chính
        lambda r: r["triplets"],
        lambda a, b: span_key(a.aspect, False) == span_key(b.aspect, False)
        and span_key(a.opinion, False) == span_key(b.opinion, False) and a.polarity == b.polarity,
        lambda a, b: overlap(a.aspect, b.aspect) and overlap(a.opinion, b.opinion) and a.polarity == b.polarity,
    ),
}


def compare(pairs: list[tuple[dict, dict]]) -> dict:
    """pairs = [(lần gán 1, lần gán 2)] trên cùng câu → các chỉ số đồng thuận."""
    out = {"n_reviews": len(pairs)}
    for name, (get, exact, partial) in ITEMS.items():
        tp_e = tp_p = n1 = n2 = 0
        for r1, r2 in pairs:
            a, b = get(r1), get(r2)
            n1, n2 = n1 + len(a), n2 + len(b)
            tp_e += count_match(a, b, exact)
            tp_p += count_match(a, b, partial)
        out[f"{name}_f1"] = prf(tp_e, n1, n2)
        out[f"{name}_f1_partial"] = prf(tp_p, n1, n2)
        if name == "triplet":
            out["n_triplets"] = f"{n1}/{n2}"

    # Kappa cảm xúc: trên các cụm ý kiến 2 người cùng bôi trùng vị trí
    pol1, pol2 = [], []
    for r1, r2 in pairs:
        ops2 = {span_key(s, False): s.polarity for s in r2["spans"] if s.kind == "OP"}
        for s in r1["spans"]:
            if s.kind == "OP" and span_key(s, False) in ops2:
                pol1.append(s.polarity)
                pol2.append(ops2[span_key(s, False)])
    out["polarity_agree_pct"] = round(100 * np.mean([x == y for x, y in zip(pol1, pol2)]), 1) if pol1 else np.nan
    out["polarity_kappa"] = safe_kappa(pol1, pol2)  # dữ liệu lệch về Tích cực → Kappa có thể thấp dù % khớp cao
    out["n_matched_opinions"] = len(pol1)

    # Kappa trên nhãn BIO của từng từ
    t1, t2 = [], []
    for r1, r2 in pairs:
        t1 += bio_tags(r1["content"], r1["spans"])
        t2 += bio_tags(r2["content"], r2["spans"])
    out["token_kappa"] = safe_kappa(t1, t2)
    return out


def safe_kappa(a, b) -> float:
    if not a or len(set(a) | set(b)) == 1:
        return np.nan
    return cohen_kappa_score(a, b)


def fleiss_kappa(table: np.ndarray) -> float:
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    p_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_e = (p_j ** 2).sum()
    return np.nan if p_e == 1 else (p_i.mean() - p_e) / (1 - p_e)


def diff_rows(r1: dict, r2: dict) -> dict | None:
    exact = ITEMS["triplet"][1]
    only1 = [t for t in r1["triplets"] if not any(exact(t, u) for u in r2["triplets"])]
    only2 = [t for t in r2["triplets"] if not any(exact(t, u) for u in r1["triplets"])]
    if not only1 and not only2 and r1["is_spam"] == r2["is_spam"]:
        return None
    return {
        "review_id": r1["review_id"], "content": r1["content"],
        "labeler_1": r1["labeler"], "only_1": "; ".join(t.show() for t in only1),
        "labeler_2": r2["labeler"], "only_2": "; ".join(t.show() for t in only2),
        "spam_1": r1["is_spam"], "spam_2": r2["is_spam"],
    }


# ---------- hai chế độ ----------

def run_two_raters(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    main: dict[str, dict] = {}
    for r in rows:
        if r["role"] == "chinh":
            if r["review_id"] in main:
                print(f"! review {r['review_id']} nằm trong nhiều file chính, giữ bản đầu tiên")
                continue
            main[r["review_id"]] = r
    cross = [r for r in rows if r["role"] == "cheo"]

    pairs = []
    missing = 0
    for c in cross:
        m = main.get(c["review_id"])
        if m is None:
            missing += 1
        elif not m["is_spam"] and not c["is_spam"]:
            pairs.append((m, c))
    if missing:
        print(f"! {missing} câu gán chéo chưa có trong bảng chính (người gán chính chưa xong?)")

    stats = [{"pair": "ALL", **compare(pairs)}] if pairs else []
    groups: dict[tuple[str, str], list] = {}
    for m, c in pairs:
        groups.setdefault((m["labeler"], c["labeler"]), []).append((m, c))
    stats += [{"pair": f"{a}–{b}", **compare(g)} for (a, b), g in sorted(groups.items())]

    dis = [d for m, c in ((main[c["review_id"]], c) for c in cross if c["review_id"] in main)
           if (d := diff_rows(m, c))]
    return pd.DataFrame(stats), pd.DataFrame(dis), list(main.values())


def run_pilot(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_person: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_person.setdefault(r["labeler"], {})[r["review_id"]] = r
    people = sorted(by_person)
    common = set.intersection(*(set(d) for d in by_person.values()))
    common = sorted(rid for rid in common if not any(by_person[p][rid]["is_spam"] for p in people))
    print(f"Pilot: {len(people)} người, {len(common)} câu có đủ người gán (đã bỏ câu bị đánh dấu spam)")

    stats, dis = [], []
    for a, b in combinations(people, 2):
        pairs = [(by_person[a][rid], by_person[b][rid]) for rid in common]
        stats.append({"pair": f"{a}–{b}", **compare(pairs)})
        dis += [d for r1, r2 in pairs if (d := diff_rows(r1, r2))]
    df = pd.DataFrame(stats)
    mean = {"pair": "ALL (trung bình các cặp)", "n_reviews": len(common)}
    for col in df.columns:
        if col.endswith(("_f1", "_partial", "_kappa", "_pct")):
            mean[col] = df[col].mean()

    # Fleiss' Kappa trên nhãn BIO của từng từ, cả nhóm cùng lúc
    tags = {p: sum((bio_tags(by_person[p][rid]["content"], by_person[p][rid]["spans"]) for rid in common), [])
            for p in people}
    cats = sorted(set().union(*tags.values()))
    table = np.array([[sum(tags[p][i] == c for p in people) for c in cats] for i in range(len(tags[people[0]]))])
    mean["token_fleiss_kappa"] = fleiss_kappa(table) if len(table) else np.nan
    return pd.concat([pd.DataFrame([mean]), df], ignore_index=True), pd.DataFrame(dis)


# ---------- in & lưu ----------

def show(stats: pd.DataFrame) -> None:
    cols = ["pair", "n_reviews", "aspect_f1", "opinion_f1", "pair_f1", "triplet_f1",
            "triplet_f1_partial", "polarity_agree_pct", "polarity_kappa", "token_kappa", "token_fleiss_kappa"]
    view = stats[[c for c in cols if c in stats.columns]].copy()
    for c in view.columns[2:]:
        fmt = "{:.1f}" if c.endswith("_pct") else "{:.3f}"
        view[c] = view[c].map(lambda v, fmt=fmt: "—" if pd.isna(v) else fmt.format(v))
    print(view.to_string(index=False))


def to_record(r: dict) -> dict:
    def sp(s: Span | None):
        return None if s is None else {"start": s.start, "end": s.end, "label": s.label, "text": s.text}
    return {
        "review_id": r["review_id"], "content": r["content"], "labeled_by": r["labeler"],
        "is_spam": r["is_spam"],
        "spans": [sp(s) for s in r["spans"]],
        "triplets": [{"aspect": sp(t.aspect), "opinion": sp(t.opinion), "polarity": t.polarity}
                     for t in r["triplets"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", required=True, type=Path, help="thư mục chứa file export của 1 đợt")
    ap.add_argument("--out", type=Path, help="mặc định reports/agreement/<đợt>/")
    args = ap.parse_args()

    print(f"Đọc {args.exports}:")
    phase, rows = load_exports(args.exports)
    out = args.out or ROOT / "reports" / "agreement" / phase
    out.mkdir(parents=True, exist_ok=True)

    if any(r["role"] == "pilot" for r in rows):
        stats, dis = run_pilot([r for r in rows if r["role"] == "pilot"])
    else:
        stats, dis, main_rows = run_two_raters(rows)
        with (out / f"{phase}_main.jsonl").open("w", encoding="utf-8") as f:
            for r in main_rows:
                f.write(json.dumps(to_record(r), ensure_ascii=False) + "\n")
        n_trip = sum(len(r["triplets"]) for r in main_rows)
        pol = Counter(t.polarity for r in main_rows for t in r["triplets"])
        implicit = sum(1 for r in main_rows for t in r["triplets"] if t.aspect is None)
        print(f"\nBảng chính: {len(main_rows)} câu, {n_trip} bộ ba "
              f"({', '.join(f'{POLARITY_VI[k]} {v}' for k, v in pol.most_common())}; "
              f"khía cạnh ẩn {implicit}), {sum(r['is_spam'] for r in main_rows)} spam")

    if stats.empty:
        print("Không có câu nào được 2 người gán → bỏ qua phần đồng thuận")
    else:
        print("\nĐộ đồng thuận (F1 giữa 2 người, khớp chính xác vị trí cụm từ):\n")
        show(stats)
        stats.to_csv(out / f"{phase}_agreement.csv", index=False, encoding="utf-8-sig")
        t = stats.iloc[0]["triplet_f1"]
        if pd.notna(t) and t < 0.6:
            print("\n⚠ Triplet F1 < 0,6 → xem lại guideline (ranh giới cụm từ, khía cạnh ẩn, cảm xúc)")
    dis.to_csv(out / f"{phase}_disagreements.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(dis)} câu bị gán khác nhau → {phase}_disagreements.csv")
    print(f"Đã ghi vào {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/")


if __name__ == "__main__":
    main()
