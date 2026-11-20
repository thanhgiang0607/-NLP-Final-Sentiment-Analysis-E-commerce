"""Chia câu thành các file import Label Studio cho từng người (người chạy: C).

Mỗi người nhận 2 file: `<đợt>_<người>_chinh.json` (phần câu của mình) và
`<đợt>_<người>_cheo.json` (một phần câu của người khác, để đo độ đồng thuận).
Kèm `<đợt>_manifest.csv` ghi câu nào do ai gán chính / ai gán chéo.

Đầu vào (--input) là một trong hai dạng:
  - CSV có cột `review_id`, `content` (thêm `rating`, `category` nếu có): không có gợi ý.
  - JSON danh sách task Label Studio (đầu ra của preannotate.py hoặc active_learning.py),
    mỗi task có `data.review_id`, `data.content`, có thể có `predictions` (cụm từ + quan hệ
    được bôi sẵn làm gợi ý).

Ví dụ:
    # Tuần 3: test 600 câu, gán chéo 45 câu mỗi người theo vòng A→B→C→D→A
    python labeling/make_tasks.py --input data/processed/test_preannotated.json --phase w3 \\
        --quota A=120 B=180 C=150 D=150 --cross-n 45

    # Tuần 2: pilot, cả 4 người gán cùng 100 câu
    python labeling/make_tasks.py --input data/processed/pilot.json --phase w2 --pilot

    # Tuần 5 (một vòng AL): A, B, D sửa nháp; C gán lại 70 câu KHÔNG có nháp để kiểm tra
    python labeling/make_tasks.py --input data/processed/al_vong1.json --phase w5_vong1 \\
        --quota A=230 B=240 D=230 --checker C --check-n 70
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from ls_common import read_json, task_filename, write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "labeling" / "tasks"


# ---------- đọc đầu vào ----------

def load_tasks(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        # cho phép cả JSON phẳng [{review_id, content, ...}]
        tasks = [t if "data" in t else {"data": t} for t in read_json(path)]
    else:
        with path.open(encoding="utf-8-sig", newline="") as f:
            tasks = [{"data": row} for row in csv.DictReader(f)]

    clean, seen = [], set()
    for t in tasks:
        data = t["data"]
        rid = data.get("review_id")
        if rid in (None, "") or not (data.get("content") or "").strip():
            continue
        if rid in seen:
            continue
        seen.add(rid)
        clean.append({"data": data, **({"predictions": t["predictions"]} if t.get("predictions") else {})})
    dropped = len(tasks) - len(clean)
    if dropped:
        print(f"! Bỏ {dropped} dòng trùng review_id / thiếu review_id / nội dung trống")
    return clean


def parse_quota(items: list[str]) -> dict[str, int]:
    quota = {}
    for item in items:
        person, _, n = item.partition("=")
        if not n.isdigit():
            sys.exit(f"--quota sai định dạng: {item!r} (đúng: A=200)")
        quota[person.strip()] = int(n)
    return quota


# ---------- tạo task ----------

def make_task(src: dict, phase: str, person: str, role: str, keep_hints: bool) -> dict:
    data = {**src["data"], "phase": phase, "assignee": person, "role": role}
    task = {"data": data}
    if keep_hints and src.get("predictions"):
        task["predictions"] = src["predictions"]
    return task


def split_main(tasks: list[dict], quota: dict[str, int], rng: random.Random) -> dict[str, list[dict]]:
    need = sum(quota.values())
    if need > len(tasks):
        sys.exit(f"Tổng quota {need} > số câu đầu vào {len(tasks)}")
    if need < len(tasks):
        print(f"! Còn dư {len(tasks) - need} câu không chia cho ai")
    pool = tasks[:]
    rng.shuffle(pool)
    chunks, start = {}, 0
    for person, n in quota.items():
        chunks[person] = pool[start:start + n]
        start += n
    return chunks


def cross_by_rotation(chunks, rotation, n, rng) -> dict[str, list[tuple[str, dict]]]:
    """Vòng A→B→C→D→A: A gán lại n câu của B, B gán lại n câu của C, ..., D gán lại n câu của A."""
    people = [p for p in rotation if p in chunks]
    cross = {}
    for i, labeler in enumerate(people):
        owner = people[(i + 1) % len(people)]
        if owner == labeler:
            sys.exit("Cần ít nhất 2 người để gán chéo")
        k = min(n, len(chunks[owner]))
        cross[labeler] = [(owner, t) for t in rng.sample(chunks[owner], k)]
    return cross


def cross_by_checker(chunks, checker, n, rng) -> dict[str, list[tuple[str, dict]]]:
    """Một người kiểm tra: lấy n câu ngẫu nhiên từ phần của tất cả người khác."""
    pool = [(owner, t) for owner, ts in chunks.items() if owner != checker for t in ts]
    return {checker: rng.sample(pool, min(n, len(pool)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--phase", required=True, help="tên đợt, ví dụ w3, w4, w5_vong1 (không có dấu cách)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="thư mục gốc; file ghi vào <out>/<phase>/")
    ap.add_argument("--quota", nargs="+", help="số câu gán chính mỗi người, ví dụ A=200 B=300 C=250 D=250")
    ap.add_argument("--cross-n", type=int, default=45, help="số câu gán chéo mỗi người (chế độ vòng)")
    ap.add_argument("--rotation", nargs="+", default=["A", "B", "C", "D"], help="thứ tự vòng gán chéo")
    ap.add_argument("--checker", help="thay cho vòng: 1 người gán lại câu của tất cả người khác")
    ap.add_argument("--check-n", type=int, default=70, help="số câu người kiểm tra gán lại")
    ap.add_argument("--cross-hints", action="store_true",
                    help="giữ gợi ý trong file chéo (mặc định BỎ để người gán chéo không bị dẫn dắt)")
    ap.add_argument("--pilot", action="store_true", help="mọi người gán cùng toàn bộ câu đầu vào")
    ap.add_argument("--people", nargs="+", default=["A", "B", "C", "D"], help="dùng với --pilot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "_" in args.phase and args.phase.split("_")[-1] in {"chinh", "cheo", "pilot"}:
        sys.exit("--phase không được kết thúc bằng _chinh/_cheo/_pilot")
    rng = random.Random(args.seed)
    tasks = load_tasks(args.input)
    out_dir = args.out / args.phase
    print(f"Đầu vào: {len(tasks)} câu ({sum(1 for t in tasks if t.get('predictions'))} có gợi ý)")

    manifest: list[dict] = []
    files: list[tuple[Path, int]] = []

    if args.pilot:
        for person in args.people:
            path = out_dir / task_filename(args.phase, person, "pilot")
            write_json(path, [make_task(t, args.phase, person, "pilot", True) for t in tasks])
            files.append((path, len(tasks)))
        manifest = [{"review_id": t["data"]["review_id"], "owner": "ALL", "cross_by": ""} for t in tasks]
    else:
        if not args.quota:
            sys.exit("Cần --quota (hoặc dùng --pilot)")
        quota = parse_quota(args.quota)
        chunks = split_main(tasks, quota, rng)
        if args.checker:
            cross = cross_by_checker(chunks, args.checker, args.check_n, rng)
        else:
            cross = cross_by_rotation(chunks, args.rotation, args.cross_n, rng)

        cross_of = {t["data"]["review_id"]: labeler for labeler, pairs in cross.items() for _, t in pairs}
        for person, ts in chunks.items():
            if not ts:
                continue
            path = out_dir / task_filename(args.phase, person, "chinh")
            write_json(path, [make_task(t, args.phase, person, "chinh", True) for t in ts])
            files.append((path, len(ts)))
            manifest += [
                {"review_id": t["data"]["review_id"], "owner": person,
                 "cross_by": cross_of.get(t["data"]["review_id"], "")}
                for t in ts
            ]
        for labeler, pairs in cross.items():
            path = out_dir / task_filename(args.phase, labeler, "cheo")
            write_json(path, [make_task(t, args.phase, labeler, "cheo", args.cross_hints) for _, t in pairs])
            owners = Counter(o for o, _ in pairs)
            files.append((path, len(pairs)))
            print(f"  {labeler} gán chéo: " + ", ".join(f"{n} câu của {o}" for o, n in owners.items()))

    man_path = out_dir / f"{args.phase}_manifest.csv"
    with man_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["review_id", "owner", "cross_by"])
        w.writeheader()
        w.writerows(manifest)

    print("\nĐã tạo:")
    for path, n in files:
        print(f"  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}  ({n} câu)")
    print(f"  {man_path.relative_to(ROOT) if man_path.is_relative_to(ROOT) else man_path}")


if __name__ == "__main__":
    main()
