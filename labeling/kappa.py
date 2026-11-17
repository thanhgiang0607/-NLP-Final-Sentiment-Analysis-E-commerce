"""Gộp file export Label Studio, tính Kappa, xuất danh sách câu bị lệch (người chạy: B).

Đặt toàn bộ file export của một đợt vào một thư mục, tên đúng quy ước
`<đợt>_<người>_<chinh|cheo|pilot>.json` (export dạng JSON hoặc JSON-MIN đều được).

Ví dụ:
    # Tính Kappa tuần 3
    python labeling/kappa.py --exports labeling/exports/w3

    # Sau buổi họp: áp nhãn đã chốt (cột `final` trong file lệch) vào bảng chính
    python labeling/kappa.py --exports labeling/exports/w3 --resolved reports/kappa/w3/w3_disagreements.csv

Đầu ra (mặc định reports/kappa/<đợt>/):
    <đợt>_main.csv            bảng chính: mỗi câu 1 dòng, nhãn 0–3 cho 7 khía cạnh
    <đợt>_kappa.csv           Kappa từng khía cạnh (cả nhóm + từng cặp người)
    <đợt>_disagreements.csv   mỗi dòng = 1 (câu, khía cạnh) bị lệch; điền cột `final` khi họp
                              (Tích cực | Trung lập | Tiêu cực | Không đề cập)
    <đợt>_main_resolved.csv   (chỉ khi có --resolved) bảng chính sau khi sửa theo `final`
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from ls_common import ASPECTS, CODE2LABEL, LABEL2CODE, SKIP_FIELD, read_json

ROOT = Path(__file__).resolve().parents[1]
FILE_RE = re.compile(r"^(?P<phase>.+)_(?P<person>[A-Za-z0-9]+)_(?P<role>chinh|cheo|pilot)\.json$")
NONE_NAME = "Không đề cập"
LABEL_NAMES = {**{v: k for k, v in LABEL2CODE.items()}, 0: NONE_NAME}


# ---------- đọc export ----------

def _choice(value) -> str | None:
    if isinstance(value, list):
        return value[0] if value else None
    return value or None


def parse_task(item: dict) -> dict | None:
    """Một task export → {review_id, content, 7 khía cạnh, is_spam}. None nếu chưa gán."""
    if "annotations" in item:  # export JSON đầy đủ
        data = item.get("data", {})
        anns = [a for a in item["annotations"] if not a.get("was_cancelled")]
        if not anns:
            return None
        ann = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        choices = {
            r["from_name"]: _choice(r.get("value", {}).get("choices"))
            for r in ann.get("result", [])
            if r.get("type") == "choices"
        }
    else:  # export JSON-MIN: các trường data và nhãn nằm phẳng cùng cấp
        data = item
        if "annotation_id" not in item and not any(k in item for k in ASPECTS + [SKIP_FIELD]):
            return None
        choices = {k: _choice(item.get(k)) for k in ASPECTS + [SKIP_FIELD] if k in item}

    rid = data.get("review_id")
    if rid in (None, ""):
        return None
    row = {"review_id": str(rid), "content": data.get("content", "")}
    for a in ASPECTS:
        label = choices.get(a)
        if label and label not in LABEL2CODE:
            print(f"! review {rid}: nhãn lạ {a}={label!r}, coi như None")
        row[a] = LABEL2CODE.get(label, 0)
    row["is_spam"] = int(bool(choices.get(SKIP_FIELD)))
    return row


def load_exports(folder: Path) -> tuple[str, pd.DataFrame]:
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
        parsed = [parse_task(it) for it in items]
        done = [r for r in parsed if r]
        print(f"  {path.name:28} {len(done):5}/{len(items)} câu đã gán")
        for r in done:
            rows.append({**r, "labeler": m["person"], "role": m["role"]})
    if len(phases) != 1:
        sys.exit(f"Thư mục chứa nhiều đợt khác nhau: {sorted(phases)}")
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("Chưa có câu nào được gán")
    return phases.pop(), df


# ---------- Kappa ----------

def safe_kappa(a, b) -> float:
    if len(a) == 0:
        return np.nan
    if len(set(a) | set(b)) == 1:  # cả 2 người chỉ dùng đúng 1 nhãn → Kappa không xác định
        return np.nan
    return cohen_kappa_score(a, b)


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, j] = số người gán câu i vào nhãn j (mọi câu cùng số người gán)."""
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    p_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return np.nan if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def pair_stats(x: pd.Series, y: pd.Series) -> dict:
    """Kappa 4 nhãn, Kappa phát hiện (có/không đề cập), % khớp."""
    det_x, det_y = (x > 0).astype(int), (y > 0).astype(int)
    both = (x > 0) & (y > 0)
    return {
        "n": len(x),
        "agree_pct": round((x == y).mean() * 100, 1),
        "kappa_4": safe_kappa(x, y),
        "kappa_detect": safe_kappa(det_x, det_y),
        "n_both_mention": int(both.sum()),
        "kappa_sentiment": safe_kappa(x[both], y[both]),  # chỉ xét câu cả 2 cùng thấy khía cạnh
    }


def kappa_two_raters(main: pd.DataFrame, cross: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = main.merge(cross, on="review_id", suffixes=("_1", "_2"))
    m = m[(m["is_spam_1"] == 0) & (m["is_spam_2"] == 0)]
    stats = []
    groups = [("ALL", m)] + [
        (f"{a}–{b}", g) for (a, b), g in m.groupby(["labeler_1", "labeler_2"])
    ]
    for pair, g in groups:
        for aspect in ASPECTS:
            stats.append({"pair": pair, "aspect": aspect, **pair_stats(g[f"{aspect}_1"], g[f"{aspect}_2"])})
        stats.append({"pair": pair, "aspect": "MACRO", **macro(stats[-len(ASPECTS):])})

    dis = []
    for _, r in m.iterrows():
        for aspect in ASPECTS:
            if r[f"{aspect}_1"] != r[f"{aspect}_2"]:
                dis.append({
                    "review_id": r["review_id"], "content": r["content_1"], "aspect": aspect,
                    "labeler_1": r["labeler_1"], "label_1": LABEL_NAMES[r[f"{aspect}_1"]],
                    "labeler_2": r["labeler_2"], "label_2": LABEL_NAMES[r[f"{aspect}_2"]],
                    "final": "",
                })
    return pd.DataFrame(stats), pd.DataFrame(dis)


def macro(rows: list[dict]) -> dict:
    out = {"n": rows[0]["n"]}
    for k in ("agree_pct", "kappa_4", "kappa_detect", "kappa_sentiment"):
        out[k] = round(float(np.nanmean([r[k] for r in rows])), 1 if k == "agree_pct" else 3)
    return out


def kappa_pilot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    people = sorted(df["labeler"].unique())
    spam_ids = set(df.loc[df["is_spam"] == 1, "review_id"])
    counts = df.groupby("review_id")["labeler"].nunique()
    full = set(counts[counts == len(people)].index) - spam_ids
    d = df[df["review_id"].isin(full)]
    print(f"Pilot: {len(people)} người, {len(full)} câu có đủ người gán (đã bỏ câu bị đánh dấu spam)")

    stats = []
    for aspect in ASPECTS:
        wide = d.pivot(index="review_id", columns="labeler", values=aspect)
        table4 = np.stack([(wide == c).sum(axis=1) for c in range(4)], axis=1)
        table2 = np.stack([(wide == 0).sum(axis=1), (wide > 0).sum(axis=1)], axis=1)
        pairs = [pair_stats(wide[a], wide[b]) for a, b in combinations(people, 2)]
        stats.append({
            "pair": "ALL", "aspect": aspect, "n": len(wide),
            "fleiss_4": fleiss_kappa(table4), "fleiss_detect": fleiss_kappa(table2),
            "mean_cohen_4": np.nanmean([p["kappa_4"] for p in pairs]),
            "all_agree_pct": round((wide.nunique(axis=1) == 1).mean() * 100, 1),
        })
        for (a, b), p in zip(combinations(people, 2), pairs):
            stats.append({"pair": f"{a}–{b}", "aspect": aspect, **p})

    dis = []
    content = d.drop_duplicates("review_id").set_index("review_id")["content"]
    for aspect in ASPECTS:
        wide = d.pivot(index="review_id", columns="labeler", values=aspect)
        for rid, r in wide[wide.nunique(axis=1) > 1].iterrows():
            dis.append({"review_id": rid, "content": content[rid], "aspect": aspect,
                        **{p: LABEL_NAMES[r[p]] for p in people}, "final": ""})
    return pd.DataFrame(stats), pd.DataFrame(dis)


# ---------- áp nhãn đã chốt ----------

def apply_resolved(main: pd.DataFrame, path: Path) -> pd.DataFrame:
    # keep_default_na=False: để chữ "None" không bị pandas đổi thành ô trống
    res = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    res = res[res["final"].str.strip() != ""]
    code_of = {**LABEL2CODE, NONE_NAME: 0, "Không": 0, "None": 0, "0": 0, "1": 1, "2": 2, "3": 3}
    out = main.set_index("review_id")
    bad = 0
    for _, r in res.iterrows():
        val = str(r["final"]).strip()
        if val not in code_of or r["aspect"] not in ASPECTS or r["review_id"] not in out.index:
            print(f"! Bỏ qua dòng: review {r['review_id']} {r['aspect']} final={val!r}")
            bad += 1
            continue
        out.loc[r["review_id"], r["aspect"]] = code_of[val]
    print(f"Áp {len(res) - bad} nhãn đã chốt ({bad} dòng lỗi)")
    return out.reset_index()


def show(stats: pd.DataFrame, cols: list[str]) -> None:
    view = stats[stats["pair"] == "ALL"][["aspect"] + cols].copy()
    for c in cols:
        if c.startswith(("n", "agree", "all_agree")):
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:g}")
        else:
            view[c] = view[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    print(view.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exports", required=True, type=Path, help="thư mục chứa file export của 1 đợt")
    ap.add_argument("--out", type=Path, help="mặc định reports/kappa/<đợt>/")
    ap.add_argument("--resolved", type=Path, help="file disagreements đã điền cột `final`")
    args = ap.parse_args()

    print(f"Đọc {args.exports}:")
    phase, df = load_exports(args.exports)
    out = args.out or ROOT / "reports" / "kappa" / phase
    out.mkdir(parents=True, exist_ok=True)
    save = lambda d, name: d.to_csv(out / f"{phase}_{name}.csv", index=False, encoding="utf-8-sig")

    if (df["role"] == "pilot").any():
        stats, dis = kappa_pilot(df[df["role"] == "pilot"])
        save(stats, "kappa")
        save(dis, "disagreements")
        print()
        show(stats, ["n", "fleiss_4", "fleiss_detect", "mean_cohen_4", "all_agree_pct"])
    else:
        main_df = df[df["role"] == "chinh"]
        dup = main_df["review_id"].duplicated(keep=False)
        if dup.any():
            print(f"! {main_df.loc[dup, 'review_id'].nunique()} câu nằm trong nhiều file chính, giữ bản đầu tiên")
            main_df = main_df.drop_duplicates("review_id")
        main_df = main_df.drop(columns="role").rename(columns={"labeler": "labeled_by"})
        save(main_df, "main")
        print(f"\nBảng chính: {len(main_df)} câu ({int(main_df['is_spam'].sum())} spam)")

        cross_df = df[df["role"] == "cheo"].drop(columns="role")
        if cross_df.empty:
            print("Không có file gán chéo → bỏ qua Kappa")
        else:
            main_k = main_df.rename(columns={"labeled_by": "labeler"})
            missing = set(cross_df["review_id"]) - set(main_k["review_id"])
            if missing:
                print(f"! {len(missing)} câu gán chéo không có trong bảng chính (người gán chính chưa xong?)")
            stats, dis = kappa_two_raters(main_k, cross_df)
            save(stats, "kappa")
            save(dis, "disagreements")
            n = int(stats.loc[stats["pair"] == "ALL", "n"].iloc[0])
            print(f"Kappa trên {n} câu được 2 người gán (đã bỏ câu bị đánh dấu spam):\n")
            show(stats, ["agree_pct", "kappa_4", "kappa_detect", "n_both_mention", "kappa_sentiment"])
            low = stats[(stats["pair"] == "ALL") & (stats["aspect"] != "MACRO") & (stats["kappa_4"] < 0.6)]
            if not low.empty:
                print(f"\n⚠ Kappa < 0,6: {', '.join(low['aspect'])} → xem lại guideline cho các khía cạnh này")
            print(f"\n{len(dis)} (câu, khía cạnh) bị lệch, thuộc {dis['review_id'].nunique() if len(dis) else 0} câu")

        if args.resolved:
            save(apply_resolved(main_df, args.resolved), "main_resolved")

    print(f"\nĐã ghi vào {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}/")


if __name__ == "__main__":
    main()
