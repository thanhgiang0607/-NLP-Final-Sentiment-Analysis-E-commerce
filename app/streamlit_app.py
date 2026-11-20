"""Web app demo: trích xuất ý kiến từ review Tiki (giao diện phong cách Tiki).

Chạy (từ thư mục gốc dự án):
    streamlit run app/streamlit_app.py

Các trang (đổi bằng tham số trên URL, giống một trang web thật):
    /                               trang chủ: danh mục, banner, lưới sản phẩm
    /?cat=phone                     sản phẩm theo ngành hàng
    /?q=áo thun                     tìm kiếm (dán link Tiki vào ô tìm kiếm sẽ mở thẳng trang phân tích)
    /?page=product&pid=..&spid=..   trang phân tích một sản phẩm
    /?page=comment                  thử một bình luận
    /?page=csv                      phân tích file CSV

Hiện dùng bộ trích xuất TẠM trong app/demo_backend.py (từ điển + luật).
Khi có mô hình thật: sửa hàm `run_model` bên dưới để gọi src/models/predict.py.
"""

from __future__ import annotations

import html
import io
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_backend import (  # noqa: E402
    Span, Triplet, extract, fetch_listing, fetch_product, local_products, parse_link, summarize, topic_of,
)
from src.crawler.tiki_crawler import CATEGORIES  # noqa: E402

st.set_page_config(page_title="Review Insight", page_icon="🛒", layout="wide",
                   initial_sidebar_state="collapsed")

POL_VI = {"POS": "Khen", "NEU": "Trung lập", "NEG": "Chê"}
MODELS = {
    "Luật + từ điển (demo)": True,
    "CRF + Linear SVM": False,
    "BiLSTM-CRF": False,
    "ViSoBERT": False,
}
CAT_ICONS = {
    "phone": "📱", "laptop": "💻", "beauty": "💄", "home": "🏠", "appliance": "🍳",
    "fashion_women": "👗", "fashion_men": "👔", "books": "📚", "grocery": "🛒",
    "toys_baby": "🧸", "digital": "🎧", "vehicles": "🛵", "sports": "⚽", "shoes_men": "👞",
    "electronics": "📺", "shoes_women": "👠", "camera": "📷", "fashion_acc": "🕶️", "food": "🍜",
    "watches_jewelry": "⌚", "bags_luggage": "🧳", "bags_women": "👜", "bags_men": "💼", "home_care": "🧽",
}

# ---------------------------------------------------------------------------
# CSS: màu và khối theo phong cách Tiki
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
:root{
  --blue:#1A94FF; --blue-dark:#0B74E5; --blue-soft:#E6F2FF;
  --bg:#F5F5FA; --card:#FFFFFF; --text:#27272A; --muted:#808089; --line:#EBEBF0;
  --red:#FF424E; --red-soft:#FFEEEF; --green:#00AB56; --green-soft:#E5F8EE;
  --yellow:#F5A623; --yellow-soft:#FFF6DC; --star:#FFC400;
}
.stApp{background:var(--bg);}
header[data-testid="stHeader"]{background:transparent; height:0;}
.block-container{padding-top:0 !important; max-width:1280px;}
a, a:hover{text-decoration:none !important;}

/* ---- thanh trên cùng ---- */
.topbar{background:#fff; border-radius:0 0 10px 10px; padding:12px 20px 8px; margin:0 0 14px;}
.toprow{display:flex; align-items:center; gap:22px; flex-wrap:wrap;}
.brand{display:flex; flex-direction:column; align-items:center; color:var(--blue) !important; line-height:1;}
.brand b{font-size:26px; font-weight:800; letter-spacing:-0.5px;}
.brand small{font-size:11px; font-weight:700; color:var(--blue-dark); margin-top:3px;}
.search{flex:1; min-width:260px; display:flex; border:1px solid #DDDDE3; border-radius:8px; overflow:hidden;}
.search span{padding:0 0 0 14px; display:flex; align-items:center; color:var(--muted);}
.search input{flex:1; border:none; outline:none; padding:9px 12px; font-size:14px; background:#fff; color:var(--text);}
.search button{border:none; border-left:1px solid #DDDDE3; background:#fff; color:var(--blue); padding:0 18px; font-size:14px; cursor:pointer;}
.search button:hover{background:var(--blue-soft);}
.navlinks{display:flex; gap:4px;}
.navlinks a{color:var(--muted) !important; font-size:14px; padding:8px 12px; border-radius:8px;}
.navlinks a:hover, .navlinks a.on{background:var(--blue-soft); color:var(--blue) !important;}
.subrow{display:flex; gap:16px; margin-top:8px; flex-wrap:wrap; font-size:13px;}
.subrow a{color:var(--muted) !important;}
.subrow a:hover{color:var(--blue) !important;}

/* ---- khối chung ---- */
.card{background:var(--card); border-radius:8px; padding:16px 18px; margin-bottom:12px;}
.card h3{font-size:16px; margin:0 0 10px; color:var(--text); font-weight:600;}
.muted{color:var(--muted); font-size:13px;}
.crumb{font-size:14px; color:var(--muted); margin:0 0 10px;}
.crumb a{color:var(--muted) !important;} .crumb a:hover{color:var(--blue) !important;}

/* ---- trang chủ ---- */
.catlist a{display:flex; align-items:center; gap:10px; padding:8px 10px; border-radius:8px; color:var(--text) !important; font-size:14px;}
.catlist a:hover, .catlist a.on{background:var(--blue-soft);}
.catlist a.on{color:var(--blue) !important; font-weight:600;}
.catlist .ic{width:32px; height:32px; border-radius:8px; background:var(--bg); display:flex; align-items:center; justify-content:center; font-size:18px;}
.banners{display:grid; grid-template-columns:2fr 1fr; gap:12px; margin-bottom:12px;}
.banner{border-radius:8px; padding:22px 24px; color:#fff; position:relative; overflow:hidden; min-height:170px;}
.banner h2{font-size:24px; margin:0 0 6px; color:#fff; font-weight:700; line-height:1.25;}
.b1 h2, .b1 p{max-width:62%;}
.banner p{margin:0 0 12px; opacity:.92; font-size:14px;}
.banner .cta{display:inline-block; background:#fff; color:var(--blue-dark) !important; border-radius:6px; padding:7px 14px; font-weight:600; font-size:14px;}
.b1{background:linear-gradient(120deg,#0B74E5 0%,#1A94FF 55%,#5CB4FF 100%);}
.b2{background:linear-gradient(140deg,#FF6A3D 0%,#FF424E 100%);}
.banner .deco{position:absolute; right:18px; bottom:14px; display:flex; flex-direction:column; gap:6px; align-items:flex-end;}
.banner .deco span{background:rgba(255,255,255,.95); border-radius:999px; padding:4px 10px; font-size:12px; font-weight:600;}
.quick{display:grid; grid-template-columns:repeat(6,1fr); gap:8px; text-align:center;}
.quick a{color:var(--text) !important; font-size:13px; display:flex; flex-direction:column; align-items:center; gap:6px; padding:6px 0; border-radius:8px;}
.quick a:hover{background:var(--bg);}
.quick .ic{width:44px; height:44px; border-radius:14px; display:flex; align-items:center; justify-content:center; font-size:22px;}
.sect-title{font-size:18px; font-weight:600; color:var(--text); margin:0 0 10px;}
.grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(176px,1fr)); gap:10px;}
.pcard{background:#fff; border-radius:8px; border:1px solid var(--line); overflow:hidden; display:flex; flex-direction:column; color:var(--text) !important; transition:box-shadow .15s;}
.pcard:hover{box-shadow:0 4px 14px rgba(0,0,0,.12);}
.pcard .img{aspect-ratio:1; background:#fff; display:flex; align-items:center; justify-content:center; font-size:56px;}
.pcard .img img{width:100%; height:100%; object-fit:contain;}
.pcard .body{padding:8px 10px 10px; display:flex; flex-direction:column; gap:4px; flex:1;}
.pcard .name{font-size:13px; line-height:1.4; height:36px; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;}
.pcard .meta{font-size:12px; color:var(--muted);}
.pcard .pr{font-size:16px; font-weight:600; color:var(--red);}
.pcard .pr.plain{color:var(--text);}
.pcard .off{background:var(--red-soft); color:var(--red); border-radius:4px; font-size:11px; padding:0 4px; margin-left:4px; font-weight:500;}
.pcard .ai{margin-top:auto; border-top:1px solid var(--line); padding-top:6px; font-size:12px; color:var(--blue);}
.footer{text-align:center; color:var(--muted); font-size:12px; padding:22px 0 8px;}

/* ---- trang sản phẩm ---- */
.pname{font-size:22px; font-weight:500; line-height:1.35; margin:0 0 6px; color:var(--text);}
.price{font-size:30px; font-weight:600; color:var(--red); margin:10px 0 0;}
.price s{font-size:14px; color:var(--muted); font-weight:400; margin-left:8px;}
.badge{background:var(--red-soft); color:var(--red); border-radius:4px; font-size:13px; padding:1px 6px; margin-left:6px;}
.stars{color:var(--star); letter-spacing:1px;}
.stars .off{color:#DDDDE3;}
.bigscore{font-size:36px; font-weight:700; color:var(--text); line-height:1;}
.bar{display:flex; align-items:center; gap:8px; font-size:12px; color:var(--muted); margin:3px 0;}
.bar .track{flex:1; height:6px; background:var(--line); border-radius:3px; overflow:hidden;}
.bar .fill{height:100%; background:var(--blue);}
.chip{display:inline-block; border-radius:999px; padding:3px 10px; margin:3px 4px 3px 0; font-size:13px;}
.chip.POS{background:var(--green-soft); color:#007A3D;}
.chip.NEG{background:var(--red-soft); color:#C7202B;}
.chip.NEU{background:var(--yellow-soft); color:#8A5A00;}
.topic{display:grid; grid-template-columns:140px 1fr 70px; gap:10px; align-items:center; font-size:13px; margin:6px 0;}
.stack{display:flex; height:10px; border-radius:5px; overflow:hidden; background:var(--line);}
.stack .POS{background:var(--green);} .stack .NEU{background:var(--yellow);} .stack .NEG{background:var(--red);}
.review{background:var(--card); border-radius:8px; padding:14px 18px; margin-bottom:10px; border:1px solid var(--line);}
.review .title{font-weight:600; margin-left:6px;}
.review .text{margin:8px 0 6px; line-height:1.9; font-size:15px; color:var(--text); white-space:pre-wrap;}
mark.ASP{background:var(--blue-soft); color:var(--blue-dark); border-bottom:2px solid var(--blue); padding:1px 2px; border-radius:3px;}
mark.POS{background:var(--green-soft); color:#007A3D; border-bottom:2px solid var(--green); padding:1px 2px; border-radius:3px;}
mark.NEG{background:var(--red-soft); color:#C7202B; border-bottom:2px solid var(--red); padding:1px 2px; border-radius:3px;}
mark.NEU{background:var(--yellow-soft); color:#8A5A00; border-bottom:2px solid var(--yellow); padding:1px 2px; border-radius:3px;}
.legend mark{margin-right:6px;}
.stButton button[kind="primary"]{background:var(--blue); border-color:var(--blue);}
.empty{color:var(--muted); font-style:italic;}

@media (max-width:760px){
  .b1 h2, .b1 p{max-width:none;} .banner .deco{display:none;}
  .banners{grid-template-columns:1fr;} .quick{grid-template-columns:repeat(3,1fr);}
  .topic{grid-template-columns:90px 1fr 50px;} .pname{font-size:18px;} .navlinks{display:none;}
  .grid{grid-template-columns:repeat(2,1fr);}
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Điều hướng bằng tham số URL
# ---------------------------------------------------------------------------

qp = st.query_params
query = (qp.get("q") or "").strip()
if query:  # dán link Tiki vào ô tìm kiếm → mở thẳng trang phân tích
    pid_, spid_ = parse_link(query)
    if pid_ and ("tiki.vn" in query or query.isdigit()):
        st.query_params.from_dict({"page": "product", "pid": str(pid_), **({"spid": str(spid_)} if spid_ else {})})
        st.rerun()
page = qp.get("page", "home")
cat = qp.get("cat")


def href(**params) -> str:
    return "?" + urlencode({k: v for k, v in params.items() if v not in (None, "")})


def topbar() -> None:
    nav = [("home", "🏠 Trang chủ", href()), ("comment", "💬 Phân tích bình luận", href(page="comment")),
           ("csv", "📄 Phân tích CSV", href(page="csv"))]
    links = "".join(
        f'<a target="_self" href="{u}" class="{"on" if page == k else ""}">{t}</a>' for k, t, u in nav
    )
    sub = "".join(
        f'<a target="_self" href="{href(cat=k)}">{html.escape(v[1])}</a>' for k, v in list(CATEGORIES.items())[:7]
    )
    st.markdown(
        f"""
<div class="topbar">
  <div class="toprow">
    <a class="brand" target="_self" href="?"><b>Review Insight</b><small>Khen gì · Chê gì</small></a>
    <form class="search" method="get" action="" target="_self">
      <span>🔍</span>
      <input name="q" value="{html.escape(query)}" placeholder="Tìm sản phẩm, hoặc dán link sản phẩm Tiki để phân tích review..." />
      <button type="submit">Tìm kiếm</button>
    </form>
    <div class="navlinks">{links}</div>
  </div>
  <div class="subrow">{sub}</div>
</div>""",
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="footer">Review Insight · Đồ án NLP: Trích xuất ý kiến (Aspect Sentiment Triplet Extraction) '
        'từ đánh giá sản phẩm.<br>Dữ liệu lấy từ Tiki qua API công khai, chỉ dùng cho mục đích học tập. '
        'Không lưu thông tin người mua.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hàm dùng chung
# ---------------------------------------------------------------------------


def run_model(text: str) -> tuple[list[Span], list[Triplet]]:
    # TODO (D): khi có mô hình, gọi src/models/predict.py theo st.session_state["model"]
    return extract(text)


def stars_html(value: float | None, size: int = 14) -> str:
    v = round(value or 0)
    return (f'<span class="stars" style="font-size:{size}px">' + "★" * v
            + f'<span class="off">{"★" * (5 - v)}</span></span>')


def money(v) -> str:
    return f"{int(v):,}".replace(",", ".") + " ₫" if v else ""


def highlight(text: str, spans: list[Span]) -> str:
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s.start):
        if sp.start < pos:
            continue
        out.append(html.escape(text[pos:sp.start]))
        cls = "ASP" if sp.kind == "ASP" else sp.polarity
        tip = "Khía cạnh" if sp.kind == "ASP" else f"Ý kiến – {POL_VI[sp.polarity]}"
        out.append(f'<mark class="{cls}" title="{tip}">{html.escape(text[sp.start:sp.end])}</mark>')
        pos = sp.end
    out.append(html.escape(text[pos:]))
    return "".join(out)


def triplet_chips(triplets: list[Triplet]) -> str:
    if not triplets:
        return '<span class="empty">Không tìm thấy ý kiến</span>'
    return "".join(
        f'<span class="chip {t.polarity}">{html.escape(t.aspect_text)} → {html.escape(t.opinion.text)}</span>'
        for t in triplets
    )


LEGEND = (
    '<div class="legend muted"><mark class="ASP">khía cạnh</mark><mark class="POS">khen</mark>'
    '<mark class="NEU">trung lập</mark><mark class="NEG">chê</mark></div>'
)


def triplet_table(triplets: list[Triplet]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Khía cạnh": t.aspect_text, "Ý kiến": t.opinion.text, "Cảm xúc": POL_VI[t.polarity],
          "Chủ đề": topic_of(t.aspect)} for t in triplets]
    )


def explain(text: str, triplets: list[Triplet]) -> None:
    st.caption("Giải thích của bộ luật demo. Khi có mô hình thật, phần này sẽ là LIME/SHAP (§8.2).")
    for t in triplets:
        lines = [f"- **{t.opinion.text}** là ý kiến *{POL_VI[t.polarity].lower()}* (có trong từ điển"]
        if any(w in t.opinion.text.lower().split() for w in ("không", "ko", "chẳng", "chả", "chưa")):
            lines[0] += ", có từ phủ định nên **đảo cảm xúc**"
        lines[0] += ")"
        if t.aspect:
            gap = len(text[t.aspect.end:t.opinion.start].split()) if t.aspect.end <= t.opinion.start \
                else len(text[t.opinion.end:t.aspect.start].split())
            lines.append(f"  nối với **{t.aspect.text}**: khía cạnh gần nhất cùng mệnh đề (cách {gap} từ)")
        else:
            lines.append("  không có khía cạnh trong cùng mệnh đề → **khía cạnh ẩn** (∅)")
        st.markdown("\n".join(lines))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(link: str, n: int) -> dict:
    return fetch_product(link, n)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_listing(cat_key: str | None, q: str | None) -> list[dict]:
    cat_id = CATEGORIES[cat_key][0] if cat_key in CATEGORIES else None
    return fetch_listing(category_id=cat_id, q=q) if (cat_id or q) else []


@st.cache_data(show_spinner=False)
def cached_local() -> dict:
    return local_products()


# ---------------------------------------------------------------------------
# Thanh bên (ẩn mặc định, bấm » ở góc trái để mở)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    model = st.selectbox("Mô hình", list(MODELS), key="model")
    if not MODELS[model]:
        st.warning("Mô hình này chưa huấn luyện xong, tạm dùng bộ luật demo.")
    n_reviews = st.slider("Số review lấy về", 20, 200, 60, step=20)
    st.markdown("### Chú thích màu")
    st.markdown(LEGEND, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Bộ ba = (khía cạnh, ý kiến, cảm xúc). ∅ = khía cạnh ẩn (không nói rõ khen/chê cái gì).")

topbar()

# ---------------------------------------------------------------------------
# Trang chủ
# ---------------------------------------------------------------------------


def product_card(p: dict) -> str:
    link = href(page="product", pid=p["id"], spid=p.get("spid"))
    img = f'<img src="{html.escape(p["image"])}" loading="lazy">' if p.get("image") else "📦"
    off = f'<span class="off">-{p["discount_rate"]}%</span>' if p.get("discount_rate") else ""
    rating = (f'{stars_html(p["rating_average"], 11)} · ' if p.get("review_count") else "")
    meta = rating + (html.escape(p["sold"]) if p.get("sold") else f'{p.get("review_count", 0)} đánh giá')
    ai = (f'🔍 Xem khen/chê từ {p["review_count"]} đánh giá' if p.get("review_count")
          else '<span class="muted">Chưa có đánh giá</span>')
    return (
        f'<a class="pcard" target="_self" href="{link}"><div class="img">{img}</div><div class="body">'
        f'<div class="name">{html.escape(p["name"])}</div><div class="meta">{meta}</div>'
        f'<div class="pr {"" if p.get("discount_rate") else "plain"}">{money(p.get("price"))}{off}</div>'
        f'<div class="ai">{ai}</div></div></a>'
    )


def home_page() -> None:
    left, main = st.columns([1, 4.2], gap="small")
    with left:
        items = "".join(
            f'<a target="_self" class="{"on" if k == cat else ""}" href="{href(cat=k)}">'
            f'<span class="ic">{CAT_ICONS.get(k, "🛍️")}</span>{html.escape(v[1])}</a>'
            for k, v in CATEGORIES.items()
        )
        st.markdown(f'<div class="card catlist"><h3>Danh mục</h3>{items}</div>', unsafe_allow_html=True)

    with main:
        if not cat and not query:
            st.markdown(
                f"""
<div class="banners">
  <div class="banner b1">
    <h2>Biết ngay người mua khen gì, chê gì<br>trước khi xuống tiền</h2>
    <p>AI đọc hàng trăm đánh giá, rút ra từng ý kiến cụ thể: <i>"pin – trâu"</i>, <i>"vải – mỏng quá"</i>…</p>
    <a class="cta" target="_self" href="{href(cat="phone")}">Khám phá ngay</a>
    <div class="deco"><span style="color:#007A3D">👍 giao hàng → nhanh</span>
      <span style="color:#C7202B">👎 chất vải → mỏng quá</span>
      <span style="color:#007A3D">👍 shop → tư vấn nhiệt tình</span></div>
  </div>
  <div class="banner b2">
    <h2>Thử với một bình luận bất kỳ</h2>
    <p>Gõ câu đánh giá, xem AI bôi màu khía cạnh và ý kiến.</p>
    <a class="cta" style="color:#C7202B !important" target="_self" href="{href(page="comment")}">Thử ngay →</a>
  </div>
</div>""",
                unsafe_allow_html=True,
            )
            quick = [
                ("📱", "#E6F2FF", "Điện thoại", href(cat="phone")),
                ("💄", "#FFEEEF", "Làm đẹp", href(cat="beauty")),
                ("🍳", "#FFF6DC", "Gia dụng", href(cat="appliance")),
                ("👗", "#F3E8FF", "Thời trang", href(cat="fashion_women")),
                ("💬", "#E5F8EE", "Phân tích bình luận", href(page="comment")),
                ("📄", "#EEF0F4", "Phân tích CSV", href(page="csv")),
            ]
            st.markdown(
                '<div class="card quick">' + "".join(
                    f'<a target="_self" href="{u}"><span class="ic" style="background:{bg}">{ic}</span>{t}</a>'
                    for ic, bg, t, u in quick
                ) + "</div>",
                unsafe_allow_html=True,
            )

        key = cat or ("phone" if not query else None)
        title = (f'Kết quả cho "{html.escape(query)}"' if query
                 else (CATEGORIES[cat][1] if cat in CATEGORIES else "Gợi ý hôm nay"))
        with st.spinner("Đang tải sản phẩm..."):
            products = cached_listing(None if query else key, query or None)
        if not products and not query:  # mất mạng → dùng sản phẩm đã crawl
            products = [{**v["product"], "spid": None, "review_count": len(v["reviews"])}
                        for v in cached_local().values()]
            title += " (dữ liệu đã crawl, không có mạng)"
        if not query:  # trang chủ / danh mục: chỉ hiện sản phẩm có review, nhiều review lên trước
            products = sorted((p for p in products if p.get("review_count")),
                              key=lambda p: -(p.get("review_count") or 0))
        cards = "".join(product_card(p) for p in products[:30])
        st.markdown(
            f'<div class="card"><div class="sect-title">{title}</div>'
            f'<div class="grid">{cards or "<span class=empty>Không tìm thấy sản phẩm</span>"}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Trang phân tích sản phẩm
# ---------------------------------------------------------------------------


def product_page() -> None:
    pid, spid = qp.get("pid"), qp.get("spid")
    data = None
    if pid:
        with st.spinner("Đang lấy và phân tích review từ Tiki..."):
            try:
                data = cached_fetch(f"x-p{pid}.html" + (f"?spid={spid}" if spid else ""), n_reviews)
            except Exception as exc:  # noqa: BLE001
                local = {str(v["product"]["id"]): v for v in cached_local().values()}
                data = local.get(str(pid))
                if not data:
                    st.error(f"{exc}")
    if not data:
        st.info("Không tìm thấy sản phẩm. Quay lại trang chủ hoặc dán link Tiki vào ô tìm kiếm.")
        return

    p, reviews = data["product"], data["reviews"]
    st.markdown(f'<div class="crumb"><a target="_self" href="?">Trang chủ</a> › {html.escape(p["name"])}</div>',
                unsafe_allow_html=True)
    analysed = [(r, *run_model(r["content"])) for r in reviews]
    all_trip = [t for _, _, ts in analysed for t in ts]

    img = (f'<img src="{html.escape(p["image"])}" style="width:100%;border-radius:8px">'
           if p.get("image") else '<div style="font-size:90px;text-align:center">📦</div>')
    old = (f'<s>{money(p["list_price"])}</s><span class="badge">-{p["discount_rate"]}%</span>'
           if p.get("discount_rate") else "")
    sold = f' · {html.escape(p["sold"])}' if p.get("sold") else ""
    st.markdown(
        f"""
<div class="card" style="display:flex;gap:22px;flex-wrap:wrap">
  <div style="flex:0 0 200px;max-width:100%">{img}</div>
  <div style="flex:1;min-width:240px">
    <div class="pname">{html.escape(p["name"])}</div>
    <div class="muted">{p.get("rating_average") or "–"} {stars_html(p.get("rating_average"))}
      &nbsp;({p.get("review_count") or 0} đánh giá){sold}</div>
    <div class="price">{money(p.get("price"))}{old}</div>
    <div class="muted" style="margin-top:10px">Đã phân tích <b>{len(reviews)}</b> review có nội dung ·
      tìm được <b>{len(all_trip)}</b> bộ ba ý kiến · mô hình: <b>{html.escape(st.session_state["model"])}</b></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2])
    with left:
        stars = p.get("stars") or {}
        total = sum((stars.get(str(s)) or {}).get("count", 0) for s in range(1, 6)) or 1
        bars = "".join(
            f'<div class="bar">{s} ★<div class="track"><div class="fill" style="width:'
            f'{100 * (stars.get(str(s)) or {}).get("count", 0) / total:.0f}%"></div></div>'
            f'{(stars.get(str(s)) or {}).get("count", 0)}</div>'
            for s in range(5, 0, -1)
        )
        st.markdown(
            f'<div class="card"><h3>Khách hàng đánh giá</h3>'
            f'<div style="display:flex;gap:12px;align-items:center;margin-bottom:8px">'
            f'<span class="bigscore">{p.get("rating_average") or "–"}</span>'
            f'{stars_html(p.get("rating_average"), 20)}</div>{bars}</div>',
            unsafe_allow_html=True,
        )
    with right:
        summary = summarize(all_trip)
        pos_top = sorted((r for r in summary if r["POS"]), key=lambda r: -r["POS"])[:4]
        neg_top = sorted((r for r in summary if r["NEG"]), key=lambda r: -r["NEG"])[:4]
        chips_pos = "".join(
            f'<span class="chip POS">👍 {html.escape(r["topic"])} ({r["POS"]})</span>' for r in pos_top
        ) or '<span class="empty">Chưa có</span>'
        chips_neg = "".join(
            f'<span class="chip NEG">👎 {html.escape(r["topic"])} ({r["NEG"]})</span>' for r in neg_top
        ) or '<span class="empty">Không có lời chê nào</span>'
        topic_rows = "".join(
            f'<div class="topic"><span>{html.escape(r["topic"])}</span><div class="stack">'
            + "".join(f'<div class="{k}" style="width:{100 * r[k] / r["total"]:.0f}%"></div>'
                      for k in ("POS", "NEU", "NEG"))
            + f'</div><span class="muted">{r["POS"]}👍 {r["NEG"]}👎</span></div>'
            for r in summary
        )
        quotes = "".join(
            f'<span class="chip NEG">“{html.escape(e)}”</span>' for r in neg_top for e in r["neg_examples"][:2]
        )
        st.markdown(
            f'<div class="card"><h3>✨ Tóm tắt ý kiến người mua</h3>'
            f'<div class="muted">Được khen nhiều</div>{chips_pos}'
            f'<div class="muted" style="margin-top:6px">Bị chê nhiều</div>{chips_neg}'
            f'<div style="margin-top:6px">{quotes}</div>'
            f'<div class="muted" style="margin:12px 0 2px">Theo chủ đề (xanh: khen · vàng: trung lập · đỏ: chê)</div>'
            f'{topic_rows}</div>',
            unsafe_allow_html=True,
        )

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        pol_filter = st.radio("Lọc", ["Tất cả", "Có lời khen", "Có lời chê"], horizontal=True,
                              label_visibility="collapsed")
    with f2:
        topic_filter = st.selectbox("Chủ đề", ["Tất cả chủ đề"] + [r["topic"] for r in summary],
                                    label_visibility="collapsed")
    with f3:
        star_filter = st.selectbox("Số sao", ["Mọi số sao", 5, 4, 3, 2, 1], label_visibility="collapsed")
    st.markdown(LEGEND, unsafe_allow_html=True)

    shown = 0
    for r, spans, trips in analysed:
        if pol_filter == "Có lời khen" and not any(t.polarity == "POS" for t in trips):
            continue
        if pol_filter == "Có lời chê" and not any(t.polarity == "NEG" for t in trips):
            continue
        if topic_filter != "Tất cả chủ đề" and not any(topic_of(t.aspect) == topic_filter for t in trips):
            continue
        if star_filter != "Mọi số sao" and int(r.get("rating") or 0) != star_filter:
            continue
        shown += 1
        created = r.get("created_at")
        if isinstance(created, (int, float)):
            created = datetime.fromtimestamp(created).strftime("%d/%m/%Y")
        elif isinstance(created, str):
            created = created[:10]
        meta = " · ".join(x for x in [created or "", html.escape(r.get("variant") or "")] if x)
        st.markdown(
            f'<div class="review">{stars_html(r.get("rating"))}'
            f'<span class="title">{html.escape(r.get("title") or "")}</span>'
            f'<div class="muted">{meta}</div>'
            f'<div class="text">{highlight(r["content"], spans)}</div>'
            f'<div>{triplet_chips(trips)}</div></div>',
            unsafe_allow_html=True,
        )
    if not shown:
        st.info("Không có review nào khớp bộ lọc.")


# ---------------------------------------------------------------------------
# Trang thử một bình luận
# ---------------------------------------------------------------------------

EXAMPLE = "Áo đẹp nhưng chất vải thì mỏng quá, shop tư vấn nhiệt tình. Giao hơi lâu. Nói chung là tạm được."


def comment_page() -> None:
    st.markdown('<div class="crumb"><a target="_self" href="?">Trang chủ</a> › Phân tích bình luận</div>',
                unsafe_allow_html=True)
    text = st.text_area("Nhập bình luận", EXAMPLE, height=110)
    st.button("Phân tích", type="primary")
    if text.strip():
        spans, trips = run_model(text)
        st.markdown(
            f'<div class="card"><h3>Kết quả</h3><div class="review text" style="border:none;padding:0">'
            f'{highlight(text, spans)}</div>{LEGEND}<div style="margin-top:10px">{triplet_chips(trips)}</div></div>',
            unsafe_allow_html=True,
        )
        if trips:
            st.dataframe(triplet_table(trips), hide_index=True, width="stretch")
            with st.expander("🔍 Vì sao mô hình rút ra như vậy?"):
                explain(text, trips)


# ---------------------------------------------------------------------------
# Trang phân tích CSV
# ---------------------------------------------------------------------------


def csv_page() -> None:
    st.markdown('<div class="crumb"><a target="_self" href="?">Trang chủ</a> › Phân tích file CSV</div>',
                unsafe_allow_html=True)
    st.markdown("Tải lên file CSV có cột **`content`** (mỗi dòng một bình luận).")
    up = st.file_uploader("Chọn file CSV", type="csv")
    if not up:
        return
    df = pd.read_csv(up, encoding="utf-8-sig")
    if "content" not in df.columns:
        st.error("File không có cột `content`.")
        return
    rows = []
    for i, txt in df["content"].fillna("").astype(str).items():
        for t in run_model(txt)[1]:
            rows.append({"dòng": i, "bình luận": txt, "khía cạnh": t.aspect_text, "ý kiến": t.opinion.text,
                         "cảm xúc": POL_VI[t.polarity], "chủ đề": topic_of(t.aspect)})
    out = pd.DataFrame(rows)
    st.success(f"{len(df)} bình luận → {len(out)} bộ ba")
    if len(out):
        st.bar_chart(out.groupby(["chủ đề", "cảm xúc"]).size().unstack(fill_value=0))
        st.dataframe(out, hide_index=True, width="stretch")
        buf = io.BytesIO()
        out.to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button("⬇️ Tải kết quả (CSV)", buf.getvalue(), "bo_ba_y_kien.csv", "text/csv")


{"product": product_page, "comment": comment_page, "csv": csv_page}.get(page, home_page)()
footer()
