"""Phần "não" TẠM cho bản demo giao diện.

- `extract(text)`: trích bộ ba bằng từ điển + luật đơn giản, chỉ để giao diện có cái chạy.
  Khi có mô hình thật, thay bằng `src/models/predict.py` (cùng kiểu trả về: spans + triplets).
- `fetch_product(link)`: lấy thông tin + review của 1 sản phẩm Tiki qua API JSON
  (dùng lại TikiClient của crawler). Không lấy tên / id người mua.
- `local_products()`: sản phẩm đã crawl sẵn trong data/raw (dùng khi không có mạng).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 1. Bộ trích xuất demo (luật + từ điển)
# ---------------------------------------------------------------------------

ASPECTS = """
sản phẩm|hàng|hàng hóa|chất lượng|chất lượng sản phẩm|chất liệu|chất vải|vải|đường may|form|size|kích thước|
màu|màu sắc|thiết kế|kiểu dáng|mẫu|áo|quần|giày|dép|túi|
giao hàng|giao|vận chuyển|ship|shipper|thời gian giao|đóng gói|gói hàng|hộp|bao bì|
giá|giá cả|giá tiền|
shop|người bán|nhà bán|tư vấn|nhân viên|dịch vụ|chăm sóc khách hàng|bảo hành|tiki|
pin|màn hình|camera|loa|âm thanh|sạc|máy|điện thoại|bàn phím|chuột|tai nghe|cấu hình|hiệu năng|
mùi|mùi hương|kem|sữa rửa mặt|xà phòng|da|
sách|giấy|bìa|nội dung|chữ|
hạn sử dụng|hương vị
"""
ASPECTS = [a.strip() for a in ASPECTS.replace("\n", "").split("|") if a.strip()]

OPINIONS = {
    "POS": """đẹp|tốt|ổn|ok|oke|okela|ổn áp|chắc chắn|chắc|nhanh|nhiệt tình|rẻ|mượt|thơm|bền|hài lòng|ưng|ưng ý|
xịn|xịn xò|sang|sang trọng|êm|mịn|mềm|dễ dùng|dễ sử dụng|chuẩn|đúng|cẩn thận|kỹ|kĩ|tuyệt vời|tuyệt|đáng tiền|
hiệu quả|trâu|nét|rõ|vừa|vừa vặn|chính hãng|nhẹ|xinh|dễ thương|thoải mái|chất|hợp lý|phải chăng|
lịch sự|thân thiện|nhanh chóng|đầy đủ|nguyên vẹn|tươi|ngon|thích|yêu|siêu đẹp|mát|sạch|đỉnh|xuất sắc""",
    "NEG": """tệ|kém|chậm|lâu|mỏng|xấu|đắt|lỗi|hỏng|hư|móp|móp méo|bể|vỡ|nóng|yếu|rách|giả|nhăn|phai|sai|thiếu|
dở|thất vọng|chán|bẩn|dơ|cũ|trầy|xước|ẩm|hôi|rộng|chật|nhỏ xíu|khó dùng|khó chịu|cẩu thả|lừa|lừa đảo|
không giống hình|kém chất lượng|bị lỗi|không hài lòng|tồi|ngứa|kích ứng|nổi mụn|sơ sài""",
    "NEU": """tạm|tạm được|tạm ổn|bình thường|cũng được|tàm tạm|không có gì đặc biệt|so so|trung bình""",
}
OPINIONS = {
    pol: [w.strip() for w in s.replace("\n", "").split("|") if w.strip()]
    for pol, s in OPINIONS.items()
}

NEGATORS = {"không", "ko", "k", "hông", "chẳng", "chả", "chưa", "đâu có", "không được", "ko được"}
PRE_INTENS = {"rất", "khá", "hơi", "quá", "cực", "cực kỳ", "cực kì", "siêu", "cũng", "khá là", "rất là",
              "vô cùng", "thật", "thật sự", "hơi bị", "khá ư"}
POST_INTENS = {"quá", "lắm", "lắm luôn", "cực", "ghê", "xỉu", "thật", "vãi", "dã man", "cực kỳ", "luôn"}
BREAKS = {".", "!", "?", ";", "\n", "nhưng", "mà", "tuy nhiên", "còn"}

TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]", re.UNICODE)


@dataclass
class Span:
    start: int          # vị trí ký tự
    end: int
    kind: str           # "ASP" hoặc "OP"
    polarity: str = ""  # POS / NEU / NEG với ý kiến
    text: str = ""


@dataclass
class Triplet:
    aspect: Span | None
    opinion: Span

    @property
    def polarity(self) -> str:
        return self.opinion.polarity

    @property
    def aspect_text(self) -> str:
        return self.aspect.text if self.aspect else "∅"


def _lexicon(words: list[str]) -> dict[tuple[str, ...], str]:
    return {tuple(w.split()): w for w in words}


ASP_LEX = _lexicon(ASPECTS)
OP_LEX = {tuple(w.split()): pol for pol, ws in OPINIONS.items() for w in ws}
NEG_LEX = {tuple(w.split()) for w in NEGATORS}
PRE_LEX = {tuple(w.split()) for w in PRE_INTENS}
POST_LEX = {tuple(w.split()) for w in POST_INTENS}
MAXN = 4


def _match(low: list[str], i: int, lex) -> int:
    """Độ dài cụm dài nhất trong `lex` bắt đầu tại i (0 nếu không có)."""
    for n in range(min(MAXN, len(low) - i), 0, -1):
        if tuple(low[i:i + n]) in lex:
            return n
    return 0


FLIP = {"POS": "NEG", "NEG": "NEU", "NEU": "NEG"}


def extract(text: str) -> tuple[list[Span], list[Triplet]]:
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    low = [t[0].lower() for t in toks]
    spans: list[tuple[int, int, str, str]] = []  # (tok_start, tok_end, kind, pol)

    i = 0
    while i < len(toks):
        # thử ý kiến: [phủ định] [mức độ] từ_ý_kiến [mức độ sau]
        j, neg = i, False
        n = _match(low, j, NEG_LEX)
        if n:
            neg, j = True, j + n
        n = _match(low, j, PRE_LEX)
        if n:
            j += n
        n_op = _match(low, j, OP_LEX)
        if n_op and not (j == i and _match(low, i, ASP_LEX) > n_op):
            pol = OP_LEX[tuple(low[j:j + n_op])]
            k = j + n_op
            n = _match(low, k, POST_LEX)
            if n:
                k += n
            if neg:
                pol = FLIP[pol]
            spans.append((i, k, "OP", pol))
            i = k
            continue
        n_asp = _match(low, i, ASP_LEX)
        if n_asp:
            spans.append((i, i + n_asp, "ASP", ""))
            i += n_asp
            continue
        i += 1

    # chia câu thành đoạn (câu / mệnh đề) và đoạn nhỏ (theo dấu phẩy)
    seg, sub = [0] * len(toks), [0] * len(toks)
    s = c = 0
    for t, w in enumerate(low):
        if w in BREAKS or (t + 1 < len(low) and f"{w} {low[t + 1]}" in BREAKS):
            s += 1
            c += 1
        elif w == ",":
            c += 1
        seg[t], sub[t] = s, c

    char_spans = [
        Span(toks[a][1], toks[b - 1][2], kind, pol, text[toks[a][1]:toks[b - 1][2]])
        for a, b, kind, pol in spans
    ]
    aspects = [(sp, sp_tok) for sp, sp_tok in zip(char_spans, spans) if sp.kind == "ASP"]
    triplets: list[Triplet] = []
    for sp, (a, b, kind, _) in zip(char_spans, spans):
        if kind != "OP":
            continue
        same_seg = [(asp, t) for asp, t in aspects if seg[t[0]] == seg[a]]
        same_sub = [(asp, t) for asp, t in same_seg if sub[t[0]] == sub[a]]
        cands = same_sub or [(asp, t) for asp, t in same_seg if t[0] < a]
        if not cands:
            triplets.append(Triplet(None, sp))
            continue
        before = [x for x in cands if x[1][0] < a]
        best = before[-1] if before else min(cands, key=lambda x: x[1][0] - a)
        triplets.append(Triplet(best[0], sp))
    return char_spans, triplets


# ---------------------------------------------------------------------------
# 2. Gom khía cạnh thành chủ đề (TẠM: bảng tay; sau thay bằng phân cụm §7.5)
# ---------------------------------------------------------------------------

TOPICS = {
    "Chất lượng": "sản phẩm hàng hàng_hóa chất_lượng chất_lượng_sản_phẩm máy hiệu_năng cấu_hình pin sạc "
                  "loa âm_thanh màn_hình camera bàn_phím chuột tai_nghe điện_thoại kem sữa_rửa_mặt xà_phòng "
                  "sách nội_dung vị hương_vị mùi mùi_hương hạn_sử_dụng",
    "Chất liệu & may": "chất_liệu chất_vải vải đường_may giấy bìa da",
    "Kiểu dáng & size": "form size kích_thước màu màu_sắc thiết_kế kiểu_dáng mẫu áo quần giày dép túi chữ",
    "Giao hàng": "giao_hàng giao vận_chuyển ship shipper thời_gian_giao",
    "Đóng gói": "đóng_gói gói_hàng hộp bao_bì",
    "Giá": "giá giá_cả giá_tiền",
    "Shop & dịch vụ": "shop người_bán nhà_bán tư_vấn nhân_viên dịch_vụ chăm_sóc_khách_hàng bảo_hành tiki",
}
ASPECT_TOPIC = {w.replace("_", " "): t for t, ws in TOPICS.items() for w in ws.split()}


def topic_of(aspect: Span | None) -> str:
    if aspect is None:
        return "Nhận xét chung"
    return ASPECT_TOPIC.get(aspect.text.lower(), "Khác")


def summarize(all_triplets: list[Triplet]) -> list[dict]:
    """Mỗi chủ đề: số khen / chê / trung lập và vài cụm ý kiến tiêu biểu."""
    by_topic: dict[str, Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for t in all_triplets:
        topic = topic_of(t.aspect)
        by_topic[topic][t.polarity] += 1
        phrase = f"{t.aspect.text.lower()} {t.opinion.text.lower()}" if t.aspect else t.opinion.text.lower()
        examples[(topic, t.polarity)][phrase] += 1
    rows = []
    for topic, cnt in by_topic.items():
        rows.append({
            "topic": topic,
            "POS": cnt["POS"], "NEU": cnt["NEU"], "NEG": cnt["NEG"],
            "total": sum(cnt.values()),
            "pos_examples": [p for p, _ in examples[(topic, "POS")].most_common(3)],
            "neg_examples": [p for p, _ in examples[(topic, "NEG")].most_common(3)],
        })
    return sorted(rows, key=lambda r: -r["total"])


# ---------------------------------------------------------------------------
# 3. Lấy dữ liệu sản phẩm
# ---------------------------------------------------------------------------

PRODUCT_URL = "https://tiki.vn/api/v2/products/{pid}"


def parse_link(link: str) -> tuple[int | None, int | None]:
    link = link.strip()
    if link.isdigit():
        return int(link), None
    m = re.search(r"-p(\d+)\.html", link) or re.search(r"product_id=(\d+)", link)
    spid = re.search(r"spid=(\d+)", link)
    return (int(m.group(1)) if m else None), (int(spid.group(1)) if spid else None)


def fetch_product(link: str, max_reviews: int = 60) -> dict:
    """Trả về {"product": {...}, "reviews": [...]} hoặc ném ValueError."""
    from src.crawler.tiki_crawler import REVIEW_URL, TikiClient

    pid, spid = parse_link(link)
    if not pid:
        raise ValueError("Không đọc được mã sản phẩm từ link (link Tiki thường có dạng ...-p123456.html)")
    client = TikiClient(delay=0.2, max_retries=2)
    info = client.get(PRODUCT_URL.format(pid=pid), {"spid": spid} if spid else {})
    if not info:
        raise ValueError("Không lấy được thông tin sản phẩm (sai link hoặc mất mạng)")

    reviews, stars, page = [], None, 1
    while len(reviews) < max_reviews:
        params = {"product_id": pid, "page": page, "limit": 20, "sort": "score|desc,id|desc,stars|all"}
        if spid:
            params["spid"] = spid
        data = client.get(REVIEW_URL, params) or {}
        stars = stars or data.get("stars")
        items = data.get("data") or []
        for r in items:  # chỉ giữ các trường cần, KHÔNG lấy thông tin người mua
            if (r.get("content") or "").strip():
                reviews.append({
                    "review_id": r.get("id"), "rating": r.get("rating"), "title": r.get("title"),
                    "content": r.get("content"), "variant": "; ".join(r.get("attributes") or []),
                    "created_at": r.get("created_at"),
                })
        if not items or page >= (data.get("paging") or {}).get("last_page", page):
            break
        page += 1

    sold = info.get("quantity_sold") or {}
    return {
        "product": {
            "id": pid, "name": (info.get("name") or "").strip(),
            "price": info.get("price"), "list_price": info.get("list_price"),
            "discount_rate": info.get("discount_rate") or 0,
            "image": info.get("thumbnail_url"),
            "rating_average": info.get("rating_average"), "review_count": info.get("review_count"),
            "sold": sold.get("text", ""), "stars": stars,
        },
        "reviews": reviews[:max_reviews],
    }


def local_products() -> dict[str, dict]:
    """{tên sản phẩm: {"product":..., "reviews":[...]}} từ data/raw (không cần mạng)."""
    pfile, rfile = ROOT / "data/raw/tiki_products.jsonl", ROOT / "data/raw/tiki_reviews.jsonl"
    if not pfile.exists() or not rfile.exists():
        return {}
    revs: dict[int, list] = defaultdict(list)
    seen = set()
    for line in rfile.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["review_id"] in seen or not (r.get("content") or "").strip():
            continue
        seen.add(r["review_id"])
        revs[r["product_id"]].append(r)
    out = {}
    for line in pfile.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        rs = revs.get(p["product_id"], [])
        if not rs:
            continue
        cnt = Counter(int(r["rating"]) for r in rs if r.get("rating"))
        out[p["name"].strip()] = {
            "product": {
                "id": p["product_id"], "name": p["name"].strip(), "price": p.get("price"),
                "list_price": p.get("price"), "discount_rate": 0, "image": None,
                "rating_average": p.get("rating_average"), "review_count": p.get("review_count"),
                "sold": "", "stars": {str(s): {"count": cnt.get(s, 0)} for s in range(1, 6)},
            },
            "reviews": rs,
        }
    return out


# ---------------------------------------------------------------------------
# 4. Danh sách sản phẩm cho trang chủ (theo ngành hàng hoặc từ khoá)
# ---------------------------------------------------------------------------

SEARCH_URL = "https://tiki.vn/api/v2/products"


def _card(p: dict) -> dict:
    sold = p.get("quantity_sold") or {}
    return {
        "id": p.get("id"), "spid": p.get("seller_product_id"),
        "name": (p.get("name") or "").strip(), "image": p.get("thumbnail_url"),
        "price": p.get("price"), "list_price": p.get("original_price") or p.get("list_price"),
        "discount_rate": p.get("discount_rate") or 0,
        "rating_average": p.get("rating_average") or 0, "review_count": p.get("review_count") or 0,
        "sold": sold.get("text", "") if isinstance(sold, dict) else "",
    }


def fetch_listing(category_id: int | None = None, q: str | None = None, limit: int = 40) -> list[dict]:
    """Sản phẩm theo ngành hàng (category_id) hoặc theo từ khoá (q). Trả [] nếu lỗi mạng."""
    from src.crawler.tiki_crawler import LISTING_URL, TikiClient

    client = TikiClient(delay=0.1, max_retries=2)
    if q:
        data = client.get(SEARCH_URL, {"q": q, "limit": limit, "page": 1})
    else:
        data = client.get(LISTING_URL, {"category": category_id, "limit": limit, "page": 1})
    return [_card(p) for p in (data or {}).get("data") or [] if p.get("id")]
