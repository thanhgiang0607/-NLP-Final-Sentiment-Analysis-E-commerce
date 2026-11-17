"""Hằng số và hàm dùng chung cho các script gán nhãn (khớp với label_config.xml, §5.1)."""

from __future__ import annotations

import json
from pathlib import Path

ASPECTS = ["QUALITY", "PRICE", "DELIVERY", "PACKAGING", "SERVICE", "AUTHENTICITY", "GENERAL"]

# 0 = None (không đề cập), 1 = Tích cực, 2 = Trung lập, 3 = Tiêu cực
LABEL2CODE = {"Tích cực": 1, "Trung lập": 2, "Tiêu cực": 3}
CODE2LABEL = {0: "None", 1: "Tích cực", 2: "Trung lập", 3: "Tiêu cực"}

SKIP_FIELD = "skip"  # ô "Spam / vô nghĩa (loại bỏ)"

ROLES = ("chinh", "cheo", "pilot")


def task_filename(phase: str, person: str, role: str) -> str:
    """Quy ước tên file: <đợt>_<người>_<chinh|cheo|pilot>.json, ví dụ w3_A_cheo.json."""
    return f"{phase}_{person}_{role}.json"


def codes_to_result(codes: dict[str, int]) -> list[dict]:
    """{"QUALITY": 1, ...} → danh sách `result` theo định dạng Label Studio."""
    return [
        {
            "from_name": aspect,
            "to_name": "text",
            "type": "choices",
            "value": {"choices": [CODE2LABEL[code]]},
        }
        for aspect, code in codes.items()
        if code
    ]


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
