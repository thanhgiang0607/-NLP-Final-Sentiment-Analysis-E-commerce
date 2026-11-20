"""Hằng số và hàm dùng chung cho các script gán nhãn (khớp với label_config.xml, §5.1).

Mỗi bình luận được gán:
  - các cụm từ (span): ASPECT (khía cạnh), OP_POS / OP_NEU / OP_NEG (ý kiến + cảm xúc)
  - các quan hệ: nối 1 ASPECT với 1 OPINION
Bộ ba (triplet) = (cụm khía cạnh | None nếu khía cạnh ẩn, cụm ý kiến, cảm xúc).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SPAN_FIELD = "span"  # tên thẻ <Labels> trong label_config.xml
SKIP_FIELD = "skip"  # ô "Spam / vô nghĩa (loại bỏ)"

ASPECT = "ASPECT"
OPINION_LABELS = {"OP_POS": "POS", "OP_NEU": "NEU", "OP_NEG": "NEG"}
POLARITY_VI = {"POS": "Tích cực", "NEU": "Trung lập", "NEG": "Tiêu cực"}

ROLES = ("chinh", "cheo", "pilot")

WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str  # ASPECT | OP_POS | OP_NEU | OP_NEG
    text: str

    @property
    def kind(self) -> str:
        return "ASP" if self.label == ASPECT else "OP"

    @property
    def polarity(self) -> str | None:
        return OPINION_LABELS.get(self.label)


@dataclass(frozen=True)
class Triplet:
    aspect: Span | None  # None = khía cạnh ẩn ("Đẹp lắm" → không nói đẹp cái gì)
    opinion: Span
    polarity: str

    def show(self) -> str:
        asp = self.aspect.text if self.aspect else "∅"
        return f"({asp}, {self.opinion.text}, {POLARITY_VI[self.polarity]})"


def task_filename(phase: str, person: str, role: str) -> str:
    """Quy ước tên file: <đợt>_<người>_<chinh|cheo|pilot>.json, ví dụ w3_A_cheo.json."""
    return f"{phase}_{person}_{role}.json"


# ---------- đọc kết quả gán của Label Studio ----------

def parse_result(result: list[dict], text: str) -> tuple[list[Span], list[Triplet], bool, list[str]]:
    """`result` của 1 annotation → (spans, triplets, is_spam, cảnh báo)."""
    warnings: list[str] = []
    by_id: dict[str, Span] = {}
    relations: list[tuple[str, str]] = []
    is_spam = False

    for r in result:
        rtype = r.get("type")
        if rtype == "labels" and r.get("from_name") == SPAN_FIELD:
            v = r.get("value", {})
            labels = v.get("labels") or []
            if not labels or v.get("start") is None:
                continue
            start, end = _strip(text, v["start"], v["end"])
            by_id[r["id"]] = Span(start, end, labels[0], text[start:end])
        elif rtype == "relation":
            relations.append((r.get("from_id"), r.get("to_id")))
        elif rtype == "choices" and r.get("from_name") == SKIP_FIELD:
            is_spam = bool(r.get("value", {}).get("choices"))

    spans = sorted(set(by_id.values()), key=lambda s: (s.start, s.end))
    triplets: set[Triplet] = set()
    linked_ops: set[Span] = set()
    for a_id, b_id in relations:
        a, b = by_id.get(a_id), by_id.get(b_id)
        if not a or not b:
            continue
        if a.kind == b.kind:
            warnings.append(f"quan hệ nối 2 cụm cùng loại: '{a.text}' – '{b.text}'")
            continue
        asp, op = (a, b) if a.kind == "ASP" else (b, a)
        triplets.add(Triplet(asp, op, op.polarity))
        linked_ops.add(op)
    for s in spans:  # ý kiến không nối với khía cạnh nào → khía cạnh ẩn
        if s.kind == "OP" and s not in linked_ops:
            triplets.add(Triplet(None, s, s.polarity))
    triplets_sorted = sorted(triplets, key=lambda t: (t.opinion.start, t.aspect.start if t.aspect else -1))
    return spans, triplets_sorted, is_spam, warnings


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    """Bỏ khoảng trắng / dấu câu thừa ở 2 đầu cụm từ do người gán bôi dư."""
    while start < end and (text[start].isspace() or text[start] in ".,;:!?"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ".,;:!?"):
        end -= 1
    return start, end


# ---------- tạo gợi ý (predictions) cho Label Studio ----------

def spans_to_result(spans: list[Span], pairs: list[tuple[int, int]] = ()) -> list[dict]:
    """spans + danh sách cặp (chỉ số aspect, chỉ số opinion) → `result` để import làm gợi ý."""
    result = [
        {
            "id": f"s{i}",
            "from_name": SPAN_FIELD,
            "to_name": "text",
            "type": "labels",
            "value": {"start": s.start, "end": s.end, "text": s.text, "labels": [s.label]},
        }
        for i, s in enumerate(spans)
    ]
    result += [
        {"from_id": f"s{a}", "to_id": f"s{o}", "type": "relation", "direction": "right"}
        for a, o in pairs
    ]
    return result


# ---------- BIO theo từ (dùng cho Kappa mức token và cho mô hình) ----------

def words(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def bio_tags(text: str, spans: list[Span]) -> list[str]:
    """Gán nhãn BIO cho từng từ: B-ASP, I-ASP, B-POS, I-NEG, ..., O."""
    toks = words(text)
    tags = ["O"] * len(toks)
    for s in spans:
        tag = "ASP" if s.kind == "ASP" else s.polarity
        inside = [i for i, (a, b) in enumerate(toks) if a < s.end and b > s.start]
        for j, i in enumerate(inside):
            tags[i] = ("B-" if j == 0 else "I-") + tag
    return tags


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
