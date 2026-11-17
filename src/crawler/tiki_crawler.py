"""Crawl review sản phẩm từ Tiki qua API JSON công khai.

Ví dụ:
    # Chạy thử nhỏ: 2 ngành hàng, mỗi ngành 3 sản phẩm, tối đa 40 review/sản phẩm
    python src/crawler/tiki_crawler.py --categories phone beauty --max-products 3 --max-reviews 40

    # Chạy thật
    python src/crawler/tiki_crawler.py --max-products 150 --max-reviews 300

Đầu ra (ghi nối tiếp, chạy lại sẽ bỏ qua sản phẩm đã crawl):
    data/raw/tiki_products.jsonl
    data/raw/tiki_reviews.jsonl
Gộp thành CSV:
    python src/crawler/tiki_crawler.py --export-csv
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PRODUCTS_FILE = RAW_DIR / "tiki_products.jsonl"
REVIEWS_FILE = RAW_DIR / "tiki_reviews.jsonl"

LISTING_URL = "https://tiki.vn/api/personalish/v1/blocks/listings"
REVIEW_URL = "https://tiki.vn/api/v2/reviews"

# key -> (category_id trên Tiki, tên ngành hàng)
CATEGORIES = {
    "phone": (1789, "Điện thoại - Máy tính bảng"),
    "laptop": (1846, "Laptop - Máy vi tính - Linh kiện"),
    "beauty": (1520, "Làm đẹp - Sức khỏe"),
    "home": (1883, "Nhà cửa - Đời sống"),
    "appliance": (1882, "Điện gia dụng"),
    "fashion_women": (931, "Thời trang nữ"),
    "fashion_men": (915, "Thời trang nam"),
    "books": (8322, "Nhà sách Tiki"),
    "grocery": (4384, "Bách hóa Online"),
}
DEFAULT_CATEGORIES = ["phone", "beauty", "home", "appliance", "fashion_women", "grocery"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "vi-VN,vi;q=0.9",
}


class TikiClient:
    def __init__(self, delay: float = 1.0, max_retries: int = 4):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self.max_retries = max_retries

    def get(self, url: str, params: dict) -> dict | None:
        for attempt in range(1, self.max_retries + 1):
            time.sleep(self.delay + random.uniform(0, self.delay / 2))
            try:
                resp = self.session.get(url, params=params, timeout=20)
            except requests.RequestException as exc:
                print(f"    ! lỗi mạng ({exc.__class__.__name__}), thử lại {attempt}/{self.max_retries}")
                time.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = 2**attempt * 2
                print(f"    ! HTTP {resp.status_code}, chờ {wait}s rồi thử lại {attempt}/{self.max_retries}")
                time.sleep(wait)
                continue
            print(f"    ! HTTP {resp.status_code} cho {url} {params}, bỏ qua")
            return None
        return None


def iter_products(client: TikiClient, category_id: int, max_products: int, min_reviews: int):
    """Trả về sản phẩm có ít nhất `min_reviews` review, theo thứ tự trang listing."""
    page, found = 1, 0
    while found < max_products:
        data = client.get(LISTING_URL, {"limit": 40, "category": category_id, "page": page})
        items = (data or {}).get("data") or []
        if not items:
            break
        for p in items:
            if (p.get("review_count") or 0) < min_reviews:
                continue
            yield p
            found += 1
            if found >= max_products:
                break
        last_page = (data.get("paging") or {}).get("last_page", page)
        if page >= last_page:
            break
        page += 1


def iter_reviews(client: TikiClient, product_id: int, spid: int | None, max_reviews: int):
    page, got = 1, 0
    while got < max_reviews:
        params = {"product_id": product_id, "page": page, "limit": 20, "include": "comments"}
        if spid:
            params["spid"] = spid
        data = client.get(REVIEW_URL, params)
        items = (data or {}).get("data") or []
        if not items:
            break
        for r in items:
            yield r
            got += 1
            if got >= max_reviews:
                break
        last_page = (data.get("paging") or {}).get("last_page", page)
        if page >= last_page:
            break
        page += 1


def to_review_record(r: dict, product: dict, category_key: str) -> dict:
    """Chỉ giữ các trường cần thiết. Không lưu thông tin người dùng (tên, id, avatar...)."""
    created = r.get("created_at")
    return {
        "review_id": r.get("id"),
        "product_id": product.get("id"),
        "product_name": product.get("name"),
        "category": category_key,
        "rating": r.get("rating"),
        "title": r.get("title"),
        "content": r.get("content"),
        "created_at": (
            datetime.fromtimestamp(created, tz=timezone.utc).isoformat() if created else None
        ),
        "thank_count": r.get("thank_count"),
        "variant": "; ".join(r.get("attributes") or []),
        "delivery_rating": json.dumps(
            {d.get("question"): d.get("option") for d in r.get("delivery_rating") or []},
            ensure_ascii=False,
        ),
        "has_photo": bool(r.get("is_photo")),
        "source": "tiki",
    }


def load_done_products() -> set[int]:
    if not PRODUCTS_FILE.exists():
        return set()
    done = set()
    with PRODUCTS_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["product_id"])
    return done


def append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def crawl(args: argparse.Namespace) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = TikiClient(delay=args.delay)
    done = load_done_products()
    total_new = total_text = 0

    for key in args.categories:
        cat_id, cat_name = CATEGORIES[key]
        print(f"\n=== {cat_name} (category={cat_id}) ===")
        for i, product in enumerate(
            iter_products(client, cat_id, args.max_products, args.min_reviews), 1
        ):
            pid = product.get("id")
            if pid in done:
                print(f"  [{i}] {pid} đã crawl, bỏ qua")
                continue
            reviews = [
                to_review_record(r, product, key)
                for r in iter_reviews(client, pid, product.get("seller_product_id"), args.max_reviews)
            ]
            append_jsonl(REVIEWS_FILE, reviews)
            append_jsonl(
                PRODUCTS_FILE,
                [{
                    "product_id": pid,
                    "name": product.get("name"),
                    "category": key,
                    "price": product.get("price"),
                    "rating_average": product.get("rating_average"),
                    "review_count": product.get("review_count"),
                    "n_crawled": len(reviews),
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                }],
            )
            done.add(pid)
            n_text = sum(1 for r in reviews if (r["content"] or "").strip())
            total_new += len(reviews)
            total_text += n_text
            print(
                f"  [{i}] {pid} | {(product.get('name') or '')[:55]:55} "
                f"| {len(reviews)} review ({n_text} có nội dung)"
            )

    print(
        f"\nXong. Review mới: {total_new} ({total_text} có nội dung). "
        f"File: {REVIEWS_FILE.relative_to(ROOT)}"
    )


def export_csv() -> None:
    import pandas as pd

    df = pd.read_json(REVIEWS_FILE, lines=True)
    before = len(df)
    df = df.drop_duplicates(subset="review_id")
    out = RAW_DIR / "tiki_reviews.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{before} dòng → {len(df)} review (bỏ trùng). Ghi ra {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES, choices=CATEGORIES.keys())
    parser.add_argument("--max-products", type=int, default=100, help="số sản phẩm mỗi ngành hàng")
    parser.add_argument("--max-reviews", type=int, default=300, help="số review tối đa mỗi sản phẩm")
    parser.add_argument("--min-reviews", type=int, default=5, help="bỏ sản phẩm có ít review hơn")
    parser.add_argument("--delay", type=float, default=1.0, help="giây chờ giữa các request")
    parser.add_argument("--export-csv", action="store_true", help="chỉ gộp JSONL thành CSV")
    args = parser.parse_args()

    if args.export_csv:
        export_csv()
    else:
        crawl(args)


if __name__ == "__main__":
    main()
