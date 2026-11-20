# Hướng đi đồ án: Trích xuất ý kiến từ bình luận E-commerce tiếng Việt (Aspect Sentiment Triplet Extraction)

## 1. Tổng quan

| Mục | Nội dung |
|---|---|
| **Đề tài** | Sentiment analysis cho bình luận E-commerce tiếng Việt |
| **Cách đặt bài toán** | **Trích xuất ý kiến** (*opinion mining / Aspect Sentiment Triplet Extraction – ASTE*): với mỗi bình luận, rút ra tất cả các **bộ ba** `(cụm khía cạnh, cụm ý kiến, cảm xúc)` |
| **Loại bài toán** | **Trích xuất thông tin** (information extraction), học có giám sát: **gán nhãn chuỗi** (sequence labeling) để tìm cụm từ + **trích xuất quan hệ** (relation extraction) để nối khía cạnh với ý kiến. *Không* phải phân lớp câu |
| **Ngôn ngữ dữ liệu** | Tiếng Việt |
| **Nguồn dữ liệu** | Tiki (crawl & scrape qua API JSON) |
| **Đầu ra cuối** | Web app: nhập bình luận → **bôi màu** khía cạnh và ý kiến ngay trên câu + danh sách bộ ba; dán link sản phẩm → **"Top điều được khen / bị chê"** |

**Ví dụ:**

> *"Áo đẹp nhưng chất vải thì mỏng quá, shop tư vấn nhiệt tình. Giao hơi lâu."*

| Cụm khía cạnh | Cụm ý kiến | Cảm xúc |
|---|---|---|
| áo | đẹp | Tích cực |
| chất vải | mỏng quá | Tiêu cực |
| shop | tư vấn nhiệt tình | Tích cực |
| giao | hơi lâu | Tiêu cực |

**Vì sao trích xuất thay vì phân lớp:**

| | Phân lớp (gán nhãn cho cả câu / từng khía cạnh cố định) | **Trích xuất ý kiến (đề tài này)** |
|---|---|---|
| Kết quả | "Chất lượng: Tiêu cực" | **"chất vải – mỏng quá – Tiêu cực"**: biết chính xác chê **cái gì**, chê **thế nào** |
| Danh sách khía cạnh | Phải định nghĩa trước (7 loại cố định) | **Mở**: khía cạnh nào xuất hiện trong câu thì rút ra cái đó ("pin", "đường may", "mùi hương"...) |
| Câu vừa khen vừa chê | Phải gộp thành 1 nhãn mỗi khía cạnh | Mỗi ý kiến là 1 bộ ba riêng |
| Giá trị cho người bán | Biết khâu nào kém | Biết **cụ thể** cần sửa gì, có bằng chứng (cụm từ gốc) |
| Quan hệ xa trong câu | Không xử lý | **Bắt được**: "chất vải ... thì ... mỏng quá" (khía cạnh và ý kiến cách nhau nhiều từ) |

### Đối chiếu yêu cầu đồ án

| Yêu cầu | Cách đáp ứng | Mục |
|---|---|---|
| Gán nhãn thủ công bằng Label Studio | Bôi đen cụm khía cạnh / ý kiến và **nối quan hệ** cho **~3.000 bình luận** trong 2 đợt (đợt 2 có active learning); đo độ đồng thuận (F1 giữa người gán, Kappa theo từ) | §5 |
| Dữ liệu tiếng Việt | Crawl & scrape review từ Tiki qua API JSON nội bộ | §3 |
| IRE: *luật hoặc gán nhãn rồi đưa về học có giám sát* | Làm **cả hai**: baseline luật (mẫu từ loại + từ điển) và các mô hình học có giám sát | §7 |
| ≥ 3 mô hình ML truyền thống | **CRF** (tìm cụm từ) + **Logistic Regression, Linear SVM, Random Forest** (nối khía cạnh – ý kiến) | §7.2 |
| ≥ 1 mô hình Deep Learning | **ViSoBERT & PhoBERT** (tìm cụm từ + nối quan hệ) + **BiLSTM-CRF** | §7.3 |
| Quy trình data science đầy đủ | Crawl & scrape → EDA → Tiền xử lý → Gán nhãn → Huấn luyện → Đánh giá → Ứng dụng | §3–§9 |
| Ứng dụng đơn giản | Web app Streamlit: bôi màu ý kiến, bảng bộ ba, giải thích, dashboard sản phẩm | §9 |
| *(Bổ sung)* Gom nhóm & đặt tên chủ đề | Phân cụm các cụm khía cạnh đã trích xuất ("vải", "chất vải", "chất liệu" → chủ đề *Chất liệu*) | §7.5 |
| *(Bổ sung)* Giải thích mô hình | Trọng số CRF, hệ số LR/SVM, LIME/SHAP cho bộ nối quan hệ | §8.2 |

---

## 2. Định nghĩa nhãn (Annotation Schema)

### 2.1. Ba thành phần của một bộ ba

| Thành phần | Là gì | Ví dụ |
|---|---|---|
| **Cụm khía cạnh** (aspect term) | Danh từ / cụm danh từ chỉ **đối tượng được nhận xét**: sản phẩm, bộ phận, đặc tính, dịch vụ | *áo, chất vải, pin, màn hình, mùi, giá, shop, shipper, giao hàng, hộp* |
| **Cụm ý kiến** (opinion term) | Từ / cụm từ **thể hiện đánh giá**, gồm cả từ phủ định và từ chỉ mức độ đi kèm | *đẹp, mỏng quá, không bền, hơi lâu, rất nhiệt tình, đáng tiền* |
| **Cảm xúc** (polarity) | Gắn vào cụm ý kiến | `Tích cực` · `Trung lập` · `Tiêu cực` |

### 2.2. Nhãn trong Label Studio

| Nhãn cụm từ | Ý nghĩa | Màu |
|---|---|---|
| `ASPECT` | Cụm khía cạnh | Xanh dương |
| `OP_POS` | Cụm ý kiến tích cực | Xanh lá |
| `OP_NEU` | Cụm ý kiến trung lập | Vàng |
| `OP_NEG` | Cụm ý kiến tiêu cực | Đỏ |

**Quan hệ:** kéo mũi tên nối `ASPECT` → cụm ý kiến nói về nó. Mỗi quan hệ tạo thành **1 bộ ba**.

### 2.3. Các trường hợp đặc biệt

| Trường hợp | Ví dụ | Cách gán |
|---|---|---|
| **Khía cạnh ẩn** (không nói rõ khen cái gì) | *"Đẹp lắm, sẽ ủng hộ tiếp"* | Bôi `đẹp lắm` là `OP_POS`, **không nối** → bộ ba `(∅, đẹp lắm, Tích cực)` |
| **1 ý kiến cho nhiều khía cạnh** | *"Áo và quần đều đẹp"* | Nối `áo` → `đẹp` và `quần` → `đẹp` (2 bộ ba) |
| **1 khía cạnh nhiều ý kiến** | *"Vải mỏng, nhăn"* | Nối `vải` → `mỏng` và `vải` → `nhăn` |
| **Khía cạnh và ý kiến ở xa nhau** | *"**Chất vải** của cái áo mình mua lần này thì **mỏng quá**"* | Vẫn nối bình thường. Đây là trường hợp mô hình khó nhất, được đánh giá riêng (§8.1) |
| **Phủ định** | *"không bền"*, *"chẳng giống hình"* | Từ phủ định **nằm trong** cụm ý kiến; cảm xúc theo nghĩa cả cụm |
| **Mỉa mai** | *"Giao nhanh thật, 2 tuần mới tới"* | `giao` → `nhanh thật` là `OP_NEG` (theo ý thật) |
| **Câu hỏi, mong muốn** | *"Có màu đen không shop?"*, *"Mong shop giao nhanh hơn"* | Không bôi (chưa phải đánh giá) |
| **Mô tả khách quan** | *"Áo màu xanh, size M"* | Không bôi |
| **Spam / vô nghĩa** | *"AE vào nhóm săn sale..."* | Tích ô **Spam** |

> Có thể điều chỉnh sau đợt gán thử (pilot ~100 câu). Chi tiết ghi trong `labeling/guideline.md`.

---

## 3. Thu thập dữ liệu (Data Crawling & Scraping)

### 3.1. Crawling và scraping

Quy trình thu thập gồm **hai việc khác nhau**, và script của nhóm làm cả hai:

| | **Crawling** | **Scraping** |
|---|---|---|
| Trả lời câu hỏi | **Đi đâu?** Lần lượt tìm và ghé qua các trang hoặc nguồn dữ liệu | **Lấy gì?** Trích xuất đúng các trường cần dùng từ mỗi nguồn |
| Trong đồ án | Duyệt ngành hàng → các trang danh sách sản phẩm → từng sản phẩm → các trang đánh giá | Từ mỗi kết quả JSON, chỉ giữ `rating`, `title`, `content`...; bỏ thông tin người mua |

Nhóm **không bóc tách HTML** (kiểu tìm thẻ `<div class="review">`). Thay vào đó, nhóm gọi thẳng **API JSON nội bộ** mà chính website Tiki dùng để hiển thị sản phẩm và đánh giá. Dữ liệu trả về đã có cấu trúc key-value, nên:
- Không cần phân tích HTML, không cần trình duyệt (Selenium/Playwright).
- Ổn định hơn: giao diện web đổi thì API thường vẫn giữ nguyên.
- Nhanh và nhẹ hơn nhiều so với tải cả trang web.

Đây là API **không chính thức** (Tiki không công bố tài liệu), nên cách làm này vẫn thuộc **web scraping theo nghĩa rộng**, cụ thể là *API scraping*.

> **Đoạn mô tả gợi ý cho báo cáo:** *Nhóm crawl các trang danh mục và trang đánh giá của Tiki (duyệt theo ngành hàng → sản phẩm → các trang đánh giá), đồng thời trích xuất (scrape) dữ liệu từ API JSON nội bộ mà website Tiki sử dụng. Cách này cho dữ liệu có cấu trúc sẵn dạng key-value, không cần phân tích HTML và ổn định hơn khi giao diện web thay đổi. Chỉ các trường cần thiết được giữ lại; thông tin cá nhân của người mua bị loại bỏ.*

### 3.2. Nguồn dữ liệu & cách tìm ra API

| Nguồn | Cách lấy | Vai trò |
|---|---|---|
| **Tiki** | API JSON nội bộ (không cần trình duyệt) | **Nguồn chính** |
| Shopee / Lazada | Chống bot mạnh, cần Selenium/Playwright | Chỉ bổ sung nếu thiếu dữ liệu |

**Cách tìm ra API của Tiki** (ghi vào báo cáo phần phương pháp; khi API đổi thì làm lại các bước này):
1. Mở một trang sản phẩm Tiki bằng Chrome → nhấn **F12** → tab **Network** → lọc **Fetch/XHR**.
2. Cuộn đến phần "Khách hàng đánh giá" sẽ thấy request `reviews?product_id=...`. Bấm vào để xem JSON trả về.
3. Mở một trang danh mục (ví dụ `tiki.vn/dien-thoai-may-tinh-bang/c1789`) sẽ thấy request `listings?category=1789...`. Mã ngành hàng chính là số sau chữ `c` trên URL.

**Hai endpoint được dùng:**

| Endpoint | Dùng để | Tham số chính | Trả về |
|---|---|---|---|
| `https://tiki.vn/api/personalish/v1/blocks/listings` | **Crawl** danh sách sản phẩm | `category` (mã ngành), `page`, `limit` (≤ 40) | `id`, `name`, `price`, `rating_average`, `review_count` + `paging.last_page` |
| `https://tiki.vn/api/v2/reviews` | **Crawl** các trang đánh giá | `product_id`, `page`, `limit` (≤ 20) | Danh sách đánh giá + `paging.last_page` |

Có thể dán thử vào trình duyệt, ví dụ: `https://tiki.vn/api/v2/reviews?product_id=1634981&page=1&limit=20`.

**Mã ngành hàng đã kiểm tra:** `1789` Điện thoại – Máy tính bảng · `1846` Laptop · `1520` Làm đẹp – Sức khỏe · `1883` Nhà cửa – Đời sống · `1882` Điện gia dụng · `931` Thời trang nữ · `915` Thời trang nam · `8322` Sách · `4384` Bách hóa Online.

### 3.3. Quy trình crawl & scrape

```
Với mỗi ngành hàng (6 ngành mặc định)
│
├─ [CRAWL] Gọi API listings, trang 1 → 2 → 3 ... đến last_page
│          → danh sách sản phẩm; bỏ sản phẩm có < 5 đánh giá; tối đa 150 sản phẩm/ngành
│
└─ Với mỗi sản phẩm
     ├─ Đã crawl rồi? → bỏ qua (cho phép dừng giữa chừng rồi chạy tiếp)
     ├─ [CRAWL]  Gọi API reviews, trang 1 → 2 → ... (tối đa 300 đánh giá/sản phẩm)
     ├─ [SCRAPE] Từ mỗi đánh giá JSON, chỉ giữ các trường cần dùng (§3.4), bỏ thông tin người mua
     └─ [LƯU]    Ghi nối tiếp vào data/raw/tiki_reviews.jsonl và tiki_products.jsonl
│
└─ [XUẤT] Gộp JSONL → bỏ trùng theo review_id → data/raw/tiki_reviews.csv
```

**Cơ chế "lịch sự" và chống lỗi:**
- Mỗi request cách nhau **~1–1,5 giây** (thêm khoảng chờ ngẫu nhiên), không gửi song song.
- Gặp HTTP **429** (gửi quá nhiều) hoặc **5xx** → chờ tăng dần (4s, 8s, 16s...) rồi thử lại, tối đa 4 lần.
- Lỗi mạng → thử lại; lỗi khác → bỏ qua request đó và ghi log.
- Ghi file sau mỗi sản phẩm → mất mạng hay tắt máy thì chỉ mất sản phẩm đang dở.

### 3.4. Trường dữ liệu được trích xuất (scrape)

| Trường lưu lại | Lấy từ JSON | Ghi chú |
|---|---|---|
| `review_id` | `id` | Khoá chính, dùng để bỏ trùng |
| `product_id`, `product_name` | Từ API listings | |
| `category` | Tham số lúc crawl | Mã ngành hàng (`phone`, `beauty`...) |
| `rating` | `rating` | 1–5 sao |
| `title` | `title` | Tiêu đề do Tiki gợi ý theo số sao ("Cực kì hài lòng") |
| `content` | `content` | **Nội dung bình luận**, dữ liệu chính để gán nhãn |
| `created_at` | `created_at` (Unix time) | Đổi sang ISO 8601 |
| `thank_count` | `thank_count` | Số lượt "hữu ích" |
| `variant` | `attributes` | Phân loại đã mua (màu, dung lượng, nhà bán) |
| `delivery_rating` | `delivery_rating` | Khảo sát giao hàng có sẵn (đúng hẹn, thái độ shipper, đóng gói) |
| `has_photo` | `is_photo` | Bình luận có ảnh |
| `source` | — | `"tiki"` |

**Cố ý bỏ:** `customer_id`, `created_by` (tên, avatar, khu vực của người mua), ảnh. Lý do: không cần cho bài toán và để bảo vệ thông tin cá nhân.

### 3.5. Phạm vi & mục tiêu
- 6 ngành hàng mặc định: điện thoại, làm đẹp, nhà cửa, điện gia dụng, thời trang nữ, bách hóa. Có thể thêm laptop, thời trang nam, sách nếu thiếu dữ liệu.
- Mục tiêu: **~60.000–80.000 review thô** (~150 sản phẩm × 6 ngành hàng, tối đa 300 review/sản phẩm) → còn **~15.000–20.000 bình luận ≥ 10 từ**.
  - Khoảng 3.000 bình luận được gán nhãn (§5).
  - Phần còn lại (~12.000–17.000) là **kho chưa gán nhãn**, dùng để active learning chọn mẫu, gom nhóm khía cạnh (§7.5) và làm dữ liệu demo cho dashboard.
- **Ưu tiên bình luận 10–80 từ** khi chọn mẫu gán nhãn: bình luận quá ngắn ("ok", "tốt") hầu như không có cụm khía cạnh; bình luận quá dài làm việc bôi cụm từ rất lâu.

### 3.6. Kết quả chạy thử (215 review, 8 sản phẩm, 3 ngành hàng)

| Quan sát | Số liệu | Hệ quả |
|---|---|---|
| Review **không có nội dung** (chỉ chấm sao) | ~51% | Mục tiêu ban đầu 20.000 review thô chỉ còn ~5.000 câu dài, **không đủ** → tăng lên 60.000–80.000. Đổi cách sắp xếp review của API không cải thiện được tỉ lệ này |
| Review có ≥ 10 từ | ~25% | Đây là tỉ lệ dùng để tính số review thô cần crawl |
| Rating 5 sao | ~87% (1–2 sao chỉ ~4%) | Khi chọn mẫu gán nhãn phải lấy thêm câu 1–3 sao |
| Spam ("AE qua nhóm mình săn sale...") và tên người trong nội dung | Có xuất hiện | Lọc spam và ẩn tên ở bước tiền xử lý (§6) |
| Trường `delivery_rating` có sẵn (giao đúng hẹn, thái độ shipper, đóng gói) | ~25% review | Dùng cho EDA và đối chiếu với các bộ ba về giao hàng / đóng gói |
| Tốc độ | ~1 request/giây, 20 review/request | 60.000–80.000 review ≈ **1,5–2 giờ** chạy máy, nên chạy qua đêm |

### 3.7. Cách chạy

Script: `src/crawler/tiki_crawler.py`.

```
# Chạy thử nhỏ
python3 src/crawler/tiki_crawler.py --categories phone beauty --max-products 3 --max-reviews 40

# Chạy thật (~1,5–2 giờ; chạy lại sẽ tự bỏ qua sản phẩm đã crawl)
python3 src/crawler/tiki_crawler.py --max-products 150 --max-reviews 300

# Gộp thành CSV, bỏ trùng
python3 src/crawler/tiki_crawler.py --export-csv
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--categories` | 6 ngành mặc định | Chọn ngành hàng (`phone`, `laptop`, `beauty`, `home`, `appliance`, `fashion_women`, `fashion_men`, `books`, `grocery`) |
| `--max-products` | 100 | Số sản phẩm mỗi ngành |
| `--max-reviews` | 300 | Số đánh giá tối đa mỗi sản phẩm |
| `--min-reviews` | 5 | Bỏ sản phẩm có ít đánh giá hơn |
| `--delay` | 1.0 | Số giây chờ giữa các request |

**Đầu ra** (trong `data/raw/`, không commit lên git; chia sẻ cho nhóm qua Google Drive để cả nhóm dùng **cùng một bộ dữ liệu**):
- `tiki_reviews.jsonl`: dữ liệu gốc, ghi nối tiếp.
- `tiki_products.jsonl`: danh sách sản phẩm đã crawl.
- `tiki_reviews.csv`: bản đã bỏ trùng, dùng cho EDA.

### 3.8. Đạo đức & lưu ý
- Chỉ dùng cho **mục đích học tập, nghiên cứu**; không phân phối lại dữ liệu thô.
- Crawl **chậm, tuần tự**, không gây tải cho hệ thống của Tiki.
- **Không lưu thông tin cá nhân** của người mua. Tên người có thể xuất hiện trong nội dung bình luận sẽ được ẩn ở bước tiền xử lý (§6).
- API **không chính thức**, có thể thay đổi bất cứ lúc nào. Nếu script lỗi, xem lại API bằng F12 (§3.2).

---

## 4. Phân tích khám phá dữ liệu (EDA)

**Trên dữ liệu thô:**
- Phân bố rating 1–5 (thường lệch mạnh về 5 sao).
- Phân bố độ dài bình luận; tỉ lệ bình luận rỗng / quá ngắn; trùng lặp, spam.
- Phân bố theo ngành hàng và thời gian.
- Top n-gram, word cloud theo nhóm rating; tần suất emoji, teencode.
- **Top danh từ và tính từ** (dùng POS tagging): gợi ý từ cho từ điển khía cạnh / ý kiến (`labeling/lexicon.json`).

**Sau khi gán nhãn (EDA cho bài trích xuất):**
- Số bộ ba trung bình mỗi bình luận; tỉ lệ bình luận không có bộ ba nào.
- Độ dài cụm khía cạnh và cụm ý kiến (số từ).
- **Khoảng cách** giữa khía cạnh và ý kiến (số từ ở giữa) → bao nhiêu % là "quan hệ xa".
- Tỉ lệ **khía cạnh ẩn**, tỉ lệ 1 ý kiến nối nhiều khía cạnh.
- Phân bố cảm xúc; top cụm khía cạnh / cụm ý kiến theo từng cảm xúc và từng ngành hàng.
- Tương quan rating ↔ bộ ba (ví dụ 5 sao nhưng vẫn có bộ ba tiêu cực).

Công cụ: `pandas`, `matplotlib`, `seaborn`, `wordcloud`, `plotly`, `underthesea` (POS).

---

## 5. Gán nhãn (Labeling) với Label Studio

### 5.1. Cấu hình giao diện

```xml
<View>
  <Relations>
    <Relation value="có_ý_kiến"/>
  </Relations>

  <Labels name="span" toName="text">
    <Label value="ASPECT" background="#4C9AFF" hotkey="a"/>
    <Label value="OP_POS" background="#36B37E" hotkey="p"/>
    <Label value="OP_NEU" background="#FFAB00" hotkey="u"/>
    <Label value="OP_NEG" background="#FF5630" hotkey="n"/>
  </Labels>

  <Text name="text" value="$content" granularity="word"/>
  <Header value="Rating gốc: $rating sao" size="6"/>

  <Choices name="skip" toName="text" choice="single">
    <Choice value="Spam / vô nghĩa (loại bỏ)"/>
  </Choices>
</View>
```

**Thao tác của người gán (mỗi câu):**
1. Đọc cả câu.
2. Chọn nhãn (`a` / `p` / `u` / `n`) rồi bôi đen cụm từ. `granularity="word"` tự bắt trọn từ, tránh bôi thiếu chữ.
3. Chọn cụm khía cạnh → bấm **tạo quan hệ** → bấm cụm ý kiến tương ứng.
4. Ý kiến không có khía cạnh rõ thì để nguyên, không nối (khía cạnh ẩn).
5. Submit.

> Phải **export dạng JSON** (không dùng JSON-MIN vì JSON-MIN làm mất quan hệ).

### 5.2. Bao nhiêu dữ liệu & chia tập thế nào

**Mục tiêu: ~3.000 bình luận được gán nhãn (~6.000–9.000 bộ ba, ước tính 2–3 bộ ba/bình luận).**

Lý do:
- Gán nhãn trích xuất **chậm hơn khoảng 2 lần** so với chọn ô (phải bôi cụm từ và nối quan hệ).
- Các bộ dữ liệu ASTE chuẩn tiếng Anh (SemEval, ASTE-Data-V2) chỉ có **~1.000–2.000 câu mỗi bộ**. 3.000 bình luận (thường dài hơn 1 câu) là **ngang hoặc lớn hơn** các bộ này.
- Mỗi bình luận cho nhiều bộ ba, nên số mẫu thực tế để mô hình học lớn hơn nhiều so với số câu.

| Tập | Số bình luận | Hình dung | Cách lấy mẫu | Cách gán |
|---|---|---|---|---|
| **Test** | 600 | Đề thi cuối kỳ: **cất kín**, chỉ dùng chấm điểm cuối cùng | Ngẫu nhiên, phân tầng theo ngành hàng + rating, **giữ phân bố tự nhiên**, 10–80 từ | Gợi ý bằng từ điển (không dùng mô hình), **30% gán chéo** |
| **Val** | 400 | Đề thi thử: chỉnh tham số, chọn mô hình | Như test | Gợi ý từ điển, 15% gán chéo |
| **Train** | ~2.000 | Bài tập về nhà | 600 ngẫu nhiên (đợt 1) + ~1.400 do active learning chọn (đợt 2) | Gợi ý từ điển, sau đó gợi ý bằng mô hình |

**Quy tắc:**
- Tập test được **chọn và đóng băng ngay từ đầu**. Không dùng nó để chọn mẫu, chỉnh tham số hay sửa guideline.
- Val và test giữ phân bố tự nhiên. **Chỉ tập train** được bổ sung mẫu do active learning chọn.
- Vì rating 5 sao chiếm ~87%, khi phân tầng phải **lấy thêm câu 1–3 sao** để có đủ ý kiến tiêu cực / trung lập (ghi rõ tỉ lệ trong báo cáo).

### 5.3. Quy trình gán nhãn 2 đợt

```
Pilot 100 ──► ĐỢT 1 (~1.600, tự gán) ──► Điểm kiểm tra ──► ĐỢT 2 (~1.400, mô hình gợi ý, 2 vòng AL) ──► ~3.000
```

**Bước 0 — Pilot (~100 bình luận):** cả 4 người cùng gán → so độ đồng thuận → thống nhất guideline (ranh giới cụm từ, khía cạnh ẩn, cảm xúc). Mẫu pilot không đưa vào test.

**Đợt 1 — Nền móng (~1.600, gợi ý bằng từ điển):**
1. Gán **test 600** trước, rồi **val 400** và **train 600**.
2. **Gợi ý bằng từ điển:** `labeling/preannotate.py` dùng `labeling/lexicon.json` (danh sách cụm khía cạnh, từ ý kiến tích cực / tiêu cực, từ phủ định, từ chỉ mức độ) để **bôi sẵn** các cụm từ và nối với khía cạnh gần nhất. Người gán sửa ranh giới, cảm xúc, quan hệ; thêm cụm bị sót. Khoảng **90–110 giây/bình luận**.
3. **Gán chéo** để đo độ đồng thuận: 30% tập test, 15% val + train. Thảo luận các câu lệch. (`labeling/make_tasks.py` chia file, `labeling/agreement.py` đo đồng thuận; quy trình chi tiết ở `TEAM_ASSIGNMENT.md` §11.)

**Đo độ đồng thuận thế nào** (bài trích xuất không dùng được Kappa kiểu "chọn ô"):

| Chỉ số | Cách tính | Mục tiêu |
|---|---|---|
| **Triplet F1 giữa 2 người** | Coi người 1 là đáp án, người 2 là dự đoán; bộ ba khớp khi trùng cả 2 cụm từ và cảm xúc | **≥ 0,7** (chỉ số chính) |
| Aspect F1 / Opinion F1 / Pair F1 | Như trên, cho từng thành phần | Tìm ra người gán lệch ở bước nào |
| Triplet F1 khớp một phần | Cụm từ chỉ cần chồng lên nhau | Nếu cao hơn nhiều so với khớp chính xác → lệch chủ yếu ở **ranh giới** cụm từ |
| Kappa cảm xúc + % khớp | Trên các cụm ý kiến 2 người bôi trùng | ≥ 0,8 |
| Kappa theo từ | Cohen's Kappa trên nhãn BIO từng từ (Fleiss' Kappa cho pilot 4 người) | ≥ 0,7 |

**Điểm kiểm tra giữa 2 đợt (1–2 buổi):**
1. Triplet F1 giữa người gán < 0,6 → sửa guideline trước khi gán tiếp.
2. Đếm số bộ ba theo cảm xúc và theo khoảng cách trong test. Ý kiến `Trung lập` quá ít (< 50) thì cân nhắc gộp vào nhóm khác, hoặc giữ và báo cáo kèm khoảng tin cậy.
3. Train **mô hình ban đầu** (CRF + SVM, và ViSoBERT tagger nếu kịp) trên 600 câu train, đánh giá trên val. Đây là baseline đầu tiên và là mô hình gợi ý nhãn cho đợt 2.
4. Chốt số câu đợt 2.

**Đợt 2 — Tăng tốc với active learning (~1.400, 2 vòng × ~700):** xem §5.4. Người gán chỉ **sửa** cụm từ và quan hệ do mô hình bôi sẵn, khoảng **40–50 giây/bình luận**. Mỗi vòng có 10% câu được gán lại **không có gợi ý** để kiểm tra người gán có bị mô hình dẫn dắt hay không.

**Guideline** (`labeling/guideline.md`) cần làm rõ:
- **Ranh giới cụm khía cạnh:** lấy cụm danh từ ngắn nhất đủ nghĩa (`chất vải`, không lấy `chất vải của cái áo này`).
- **Ranh giới cụm ý kiến:** gồm từ phủ định và từ chỉ mức độ đi liền (`không bền`, `hơi mỏng`, `mỏng quá`); không gồm chủ ngữ, trạng ngữ thời gian.
- Khi nào là `OP_NEU` (đánh giá ở mức trung bình: "tạm", "bình thường", "tương đối"), khi nào **không bôi** (mô tả khách quan, câu hỏi, mong muốn).
- Khía cạnh ẩn, mỉa mai, so sánh ("tốt hơn hàng ngoài tiệm").
- Mỗi quy tắc có **≥ 3 ví dụ thật** lấy từ dữ liệu Tiki.

**Export:** JSON → `labeling/ls_to_dataset.py` → `data/labeled/`:
- `aste_<split>.jsonl`: mỗi dòng 1 bình luận: `content`, danh sách cụm từ (vị trí ký tự), danh sách bộ ba, `split`, `batch`.
- `bio_<split>.txt`: dạng CoNLL (mỗi dòng 1 từ + nhãn BIO), dùng cho CRF / BiLSTM-CRF / ViSoBERT tagger.

### 5.4. Active learning (đợt 2)

**Ý tưởng:** để mô hình chọn những câu **nó phân vân nhất** thay vì chọn ngẫu nhiên. Những câu này dạy mô hình được nhiều nhất trên mỗi giờ gán nhãn.

```
"Sp tốt, giao nhanh"                                → mô hình chắc chắn  → bỏ qua
"Chất vải của cái áo mình mua lần này thì mỏng quá" → phân vân (nối xa)  → CHỌN để gán nhãn
```

**Một vòng (script offline `labeling/active_learning.py`):**

| Bước | Việc làm | Thời gian |
|---|---|---|
| 1 | Mô hình hiện tại dự đoán toàn bộ kho ~12.000–17.000 bình luận chưa gán | CRF + SVM: vài phút (CPU); ViSoBERT: ~5 phút GPU |
| 2 | Tính **độ không chắc chắn** mỗi câu = trung bình của: (a) độ không chắc khi tìm cụm từ (entropy nhãn BIO trung bình của 3 từ phân vân nhất), (b) độ không chắc khi nối quan hệ (cặp có xác suất gần 0,5 nhất) | Vài giây |
| 3 | Chọn ~700 câu: **70%** phân vân nhất + **20%** câu "khó" theo luật (có từ phủ định, "nhưng", cụm `OP_NEU`, khía cạnh và ý kiến cách xa) + **10%** ngẫu nhiên | Vài giây |
| 4 | Xuất JSON có `predictions` (cụm từ + quan hệ bôi sẵn) → import Label Studio → người gán sửa | ~8–9 giờ công |
| 5 | Gộp vào train, train lại mô hình | CRF + SVM: vài phút; ViSoBERT: ~15–20 phút |

**Lưu ý:**
- Câu do active learning chọn **chỉ vào train**.
- Ghi lại sau mỗi vòng: Triplet F1 trên val, và **tỉ lệ bộ ba gợi ý được giữ nguyên**. Hai số liệu này đưa vào báo cáo.
- Tích hợp trực tiếp bằng Label Studio ML Backend để ở phần nâng cao (§10).

---

## 6. Tiền xử lý dữ liệu (Preprocessing)

Bài trích xuất dựa vào **vị trí ký tự** của cụm từ. Nếu văn bản bị sửa sau khi gán nhãn (bỏ dấu câu, đổi teencode...) thì vị trí bị lệch và nhãn hỏng. Vì vậy tiền xử lý chia **2 lớp**:

**Lớp 1 — Trước khi gán nhãn (sửa văn bản, sau đó KHÔNG đổi nữa):**

| Bước | Chi tiết |
|---|---|
| Làm sạch | Bỏ HTML, URL, số điện thoại; gộp khoảng trắng thừa |
| Chuẩn hoá Unicode | NFC; chuẩn hoá kiểu gõ dấu (`òa` ↔ `oà`) |
| Ẩn thông tin cá nhân | Tên người, số điện thoại trong nội dung → `<NAME>`, `<PHONE>` |
| Lọc | Bỏ spam, bỏ câu < 10 từ hoặc > 80 từ khỏi danh sách cần gán |

→ Văn bản sau lớp 1 chính là văn bản hiển thị trong Label Studio và lưu trong `data/labeled/`.

**Lớp 2 — Khi đưa vào mô hình (không đổi văn bản gốc, xử lý trên từng từ và giữ căn chỉnh với nhãn):**

| Bước | Chi tiết |
|---|---|
| Tách từ | Tách theo âm tiết + dấu câu (`src/preprocessing/tokenizer.py`), mỗi từ giữ vị trí ký tự gốc → đổi cụm từ thành nhãn BIO |
| Chuẩn hoá từng từ (cho CRF / BiLSTM) | Chữ thường, teencode (`ko → không`, `sp → sản phẩm`), ký tự lặp (`đẹpppp → đẹp`), emoji → `emo_pos` / `emo_neg`. Thay **trên từng từ** (dùng làm đặc trưng), không xoá từ nào, nên nhãn vẫn khớp |
| ViSoBERT | Dùng văn bản lớp 1, **không chuẩn hoá thêm** (mô hình đã học teencode, emoji). Nhãn của từ gán cho sub-token đầu tiên |
| PhoBERT | Tách từ ghép bằng VnCoreNLP (`chất_vải`); nhãn của từ ghép lấy theo âm tiết đầu; khi dự đoán thì tách ngược lại về vị trí ký tự |
| Không bỏ stopwords | Bỏ từ sẽ làm lệch vị trí và mất từ phủ định / mức độ ("không", "hơi", "quá") nằm trong cụm ý kiến |
| Chia dữ liệu | Theo §5.2: Test 600 / Val 400 / Train ~2.000, phân tầng theo ngành hàng + rating, cố định từ đầu |

---

## 7. Xây dựng & huấn luyện mô hình

### 7.1. Cách đặt bài toán: pipeline 2 bước

```
Bình luận
   │
   ▼
[Bước 1: Tìm cụm từ]  gán nhãn chuỗi (sequence labeling), mỗi từ nhận 1 trong 9 nhãn BIO:
   │                  B-ASP I-ASP │ B-POS I-POS │ B-NEU I-NEU │ B-NEG I-NEG │ O
   │
   │   Chất  vải   của  áo    thì  mỏng   quá   ,  shop  tư    vấn   nhiệt tình
   │   B-ASP I-ASP O    B-ASP O    B-NEG  I-NEG O  B-ASP B-POS I-POS I-POS  I-POS
   ▼
[Bước 2: Nối quan hệ]  trích xuất quan hệ (relation extraction): với mỗi cặp (cụm khía cạnh, cụm ý kiến)
   │                   trong câu → hai cụm này có đi với nhau không?
   │   (chất vải, mỏng quá) ✔   (áo, mỏng quá) ✘   (shop, tư vấn nhiệt tình) ✔   (chất vải, tư vấn nhiệt tình) ✘
   ▼
[Ghép]  → bộ ba; cụm ý kiến không nối với khía cạnh nào → khía cạnh ẩn (∅)
   (chất vải, mỏng quá, Tiêu cực) · (shop, tư vấn nhiệt tình, Tích cực)
```

- **Cảm xúc nằm trong nhãn của bước 1** (B-POS / B-NEG...), nên không cần thêm bước phân loại cảm xúc riêng.
- Bước 2 là phần **"bắt quan hệ xa"** đúng như gợi ý IRE của đề bài: "chất vải" và "mỏng quá" cách nhau 3 từ.
- Pipeline cho phép **ghép tự do** mô hình bước 1 và bước 2 (ví dụ CRF + SVM, ViSoBERT + ViSoBERT) và đánh giá **riêng từng bước**.

### 7.2. Baseline luật & Machine Learning truyền thống (≥ 3 mô hình)

**Baseline luật** (`src/models/rule_baseline.py`), theo hướng Hu & Liu (2004) và Qiu et al. (2011):
1. POS tagging bằng `underthesea`.
2. **Cụm khía cạnh** = cụm danh từ có trong từ điển khía cạnh (hoặc danh từ đứng trước tính từ ý kiến).
3. **Cụm ý kiến** = tính từ / động từ có trong từ điển ý kiến + từ phủ định và từ mức độ đi liền. Cảm xúc lấy từ từ điển, **đảo chiều** khi có phủ định.
4. **Nối:** mẫu `[danh từ] (thì | rất | hơi | khá | cũng)* [tính từ]`; nếu không khớp mẫu thì nối với khía cạnh gần nhất **bên trái, cùng mệnh đề** (tách mệnh đề bởi `,` `.` `nhưng` `mà`).

**Bước 1 — Tìm cụm từ:**

| Mô hình | Đặc trưng | Ghi chú |
|---|---|---|
| **CRF** (`sklearn-crfsuite`) — *ML chính* | Từ (chữ thường), từ đã chuẩn hoá teencode, POS, có trong từ điển khía cạnh / ý kiến tích cực / tiêu cực, là từ phủ định / mức độ, viết hoa, chữ số, emoji; **cửa sổ ±2 từ**; bigram | Mô hình chuẩn cho gán nhãn chuỗi, học cả **chuyển nhãn** (B-ASP → I-ASP) |
| Logistic Regression / Linear SVM *theo từng từ* (đối chứng) | Cùng đặc trưng, nhưng đoán từng từ độc lập | Cho thấy vì sao cần mô hình chuỗi (hay sinh `I-` không có `B-`) |

**Bước 2 — Nối quan hệ** (mỗi cặp ứng viên là 1 mẫu):

| Đặc trưng của cặp (khía cạnh, ý kiến) |
|---|
| Khoảng cách (số từ), khía cạnh đứng trước hay sau |
| Các từ ở giữa (bag-of-words), có dấu câu / "nhưng" / "còn" / "mà" ở giữa không, có khía cạnh khác ở giữa không |
| Có phải khía cạnh gần nhất của ý kiến này không; cùng mệnh đề không |
| POS và nội dung (TF-IDF) của 2 cụm |

| Mô hình | Ghi chú |
|---|---|
| **Logistic Regression** | `class_weight='balanced'` (số cặp "không nối" nhiều hơn hẳn) |
| **Linear SVM** | |
| **Random Forest / XGBoost** | Bắt được tương tác giữa các đặc trưng (khoảng cách × có "nhưng" ở giữa) |

**Các tổ hợp được báo cáo:** Luật · CRF + LR · CRF + SVM · CRF + RF. Tinh chỉnh bằng `GridSearchCV` / `Optuna` (CRF: `c1`, `c2`), metric chọn mô hình: **Triplet F1 trên val**.

### 7.3. Deep Learning (≥ 1 mô hình)

**Hai encoder pre-trained được so sánh trực tiếp:**

| Encoder | Pre-train trên | Tách từ | Vì sao dùng |
|---|---|---|---|
| **ViSoBERT** (`uitnlp/visobert`) — *ưu tiên* | Văn bản mạng xã hội tiếng Việt | Không cần | Bình luận TMĐT gần với văn mạng xã hội: teencode, viết tắt, emoji, không dấu. Không tách từ → **căn chỉnh nhãn với vị trí ký tự dễ hơn** |
| **PhoBERT** (`vinai/phobert-base-v2`) | Wikipedia + báo chí | Bắt buộc (VnCoreNLP) | Baseline chuẩn cho tiếng Việt → dễ đối chiếu |

**Kiến trúc:**

| Mô hình | Bước | Kiến trúc |
|---|---|---|
| **BiLSTM-CRF** | 1 | Embedding fastText/word2vec tiếng Việt (+ char-CNN) → BiLSTM → CRF. DL không dùng transformer, để so sánh |
| **ViSoBERT / PhoBERT tagger** (chính) | 1 | Encoder → Linear 9 nhãn trên từng token (token classification); tuỳ chọn thêm **lớp CRF** ở trên |
| **ViSoBERT / PhoBERT pair** (chính) | 2 | Đánh dấu cặp ứng viên trong câu: `<a> chất vải </a> của áo thì <o> mỏng quá </o>` → vector `<s>` → nối / không nối |

**Phân tích so sánh** (đưa vào báo cáo):
- ViSoBERT và PhoBERT: tổng thể và theo **mức độ "phi chuẩn"** của câu (tỉ lệ teencode, emoji, không dấu).
- Transformer và CRF: khác biệt lớn nhất ở **quan hệ xa** và **phủ định**?
- Chi phí: thời gian huấn luyện, tốc độ suy luận, độ phức tạp triển khai (PhoBERT cần VnCoreNLP/Java).

Mô hình tốt hơn sẽ dùng mặc định trong app. **Nếu chỉ kịp chạy một mô hình, chọn ViSoBERT.**

Huấn luyện trên Colab / Kaggle GPU. Siêu tham số gợi ý: tagger `lr=3e-5`, `batch=16`, `max_len=160`, `epochs=10–15`; pair `lr=2e-5`, `epochs=4–6`; early stopping theo Triplet F1 trên val, fp16, cùng seed và cùng tập chia.

### 7.4. Ghép thành bộ ba (`src/models/predict.py`)

```
predict(text, tagger, pairer) →
  1. tagger: từ → nhãn BIO → các cụm từ (vị trí ký tự)
  2. pairer: mọi cặp (khía cạnh, ý kiến) trong câu → xác suất nối; giữ cặp > ngưỡng (chỉnh trên val)
  3. ý kiến không có cặp nào → khía cạnh ẩn
  → [{"aspect": "chất vải", "opinion": "mỏng quá", "polarity": "NEG", "aspect_span": [0, 8], ...}, ...]
```

Một hàm dùng chung cho **đánh giá**, **active learning** (§5.4) và **app** (§9).

### 7.5. Gom nhóm khía cạnh & đặt tên chủ đề (phân cụm)

Cụm khía cạnh trích xuất được là **văn bản tự do**: "vải", "chất vải", "chất liệu", "đường may", "ship", "shipper", "giao hàng"... Để thống kê được ("bao nhiêu người chê **chất liệu**?") cần gom các cụm cùng nghĩa thành **chủ đề**:

1. Lấy toàn bộ cụm khía cạnh trích xuất từ **kho chưa gán nhãn** (~15.000 bình luận), chuẩn hoá (chữ thường, teencode), đếm tần suất.
2. Biểu diễn mỗi cụm bằng vector: trung bình embedding ViSoBERT của cụm **trong ngữ cảnh** các câu chứa nó (hoặc fastText).
3. Phân cụm: **Agglomerative** (cosine) / **HDBSCAN**; chọn số cụm bằng silhouette score.
4. **Đặt tên chủ đề** cho mỗi cụm: cụm từ xuất hiện nhiều nhất, hoặc top từ theo c-TF-IDF (tuỳ chọn: nhờ LLM đặt tên).
5. Đánh giá: silhouette + **người kiểm tra ~300 cụm từ phổ biến nhất** (độ thuần của mỗi chủ đề).

Kết quả là bảng `cụm khía cạnh → chủ đề` (`models/aspect_topics.json`), dùng trong dashboard: *"Chất liệu: mỏng (42 lần), mát (30), nhăn (12)"*.

---

## 8. Đánh giá & Giải thích mô hình (Evaluation & Explainability)

### 8.1. Metrics & so sánh mô hình

Đánh giá theo **3 cấp**:

| Cấp | Metric | Ý nghĩa |
|---|---|---|
| **Tìm cụm từ** (bước 1) | P / R / F1 mức cụm từ, **khớp chính xác** (`seqeval`) cho cụm khía cạnh và cụm ý kiến; thêm F1 **khớp một phần** | Mô hình tìm đúng cụm từ không |
| **Nối quan hệ** (bước 2) | Pair F1 `(khía cạnh, ý kiến)`: (a) với cụm từ **đúng** cho sẵn → đo riêng bước 2; (b) với cụm từ mô hình tự tìm | Mô hình nối đúng không |
| **Bộ ba** (toàn bộ) | **Triplet F1**: đúng cả cụm khía cạnh, cụm ý kiến và cảm xúc | Chỉ số tổng hợp, **dùng để xếp hạng mô hình** |

**Phân tích sâu:**
- Triplet F1 theo **khoảng cách** khía cạnh – ý kiến: 0–2 từ / 3–5 từ / ≥ 6 từ → mô hình nào bắt **quan hệ xa** tốt hơn.
- Theo cảm xúc (Tích cực / Trung lập / Tiêu cực), khía cạnh ẩn, câu có 1 bộ ba và câu có ≥ 3 bộ ba, theo ngành hàng.
- **Error analysis:** lệch ranh giới ("mỏng" thay vì "mỏng quá"), nối nhầm trong câu nhiều khía cạnh, phủ định, mỉa mai, bỏ sót khía cạnh ẩn.

Mẫu bảng kết quả:

| Mô hình (bước 1 + bước 2) | Aspect F1 | Opinion F1 | Pair F1 | **Triplet F1** | Train time |
|---|---|---|---|---|---|
| Luật (từ điển + mẫu câu) | | | | | |
| CRF + Logistic Regression | | | | | |
| CRF + Linear SVM | | | | | |
| CRF + Random Forest | | | | | |
| BiLSTM-CRF + Linear SVM | | | | | |
| PhoBERT tagger + PhoBERT pair | | | | | |
| ViSoBERT tagger + ViSoBERT pair | | | | | |
| *(nâng cao)* LLM few-shot | | | | | |

### 8.2. Giải thích mô hình

**Mục tiêu:** chỉ ra **vì sao** mô hình bôi "mỏng quá" là Tiêu cực và **vì sao** nối nó với "chất vải" chứ không phải "áo". Từ đó kiểm tra mô hình học đúng tín hiệu hay học nhầm.

| Mô hình | Phương pháp | Ví dụ kết quả |
|---|---|---|
| **CRF** | **Trọng số đặc trưng** (state features) và **trọng số chuyển nhãn** (transitions), xem bằng `eli5` hoặc trực tiếp từ `sklearn-crfsuite` | "từ trong từ điển tiêu cực" → `B-NEG` (+3,2); "từ trước là *không*" → đảo cảm xúc; `B-ASP → I-ASP` (+4,1), `O → I-ASP` (−5,0) |
| **LR / Linear SVM (nối)** | **Hệ số mô hình** (global) + **SHAP `LinearExplainer`** (từng cặp) | "có *nhưng* ở giữa" → không nối; "khoảng cách > 6 từ" → giảm xác suất nối |
| **ViSoBERT / PhoBERT pair** | **LIME** + **SHAP Partition** trên các từ của câu | Từ "thì" và vị trí sát nhau đẩy về "nối"; dấu "," + "shop" đẩy về "không nối" |
| **ViSoBERT / PhoBERT tagger** | **Integrated Gradients** (`captum`) cho nhãn của từng từ (tuỳ chọn) | Vì sao "lâu" được gán `B-NEG` |

**Các phân tích đưa vào báo cáo:**
1. **Global:** top đặc trưng của CRF cho từng nhãn; top đặc trưng của SVM khi nối. Đối chiếu với tổng hợp SHAP của ViSoBERT pair.
2. **Local:** giải thích **~100–150 bộ ba test** chọn phân tầng: đúng / sai × gần / xa × có phủ định.
3. **Gắn với error analysis:** với bộ ba sai, chỉ ra từ gây nhầm → đề xuất sửa (thêm từ điển, thêm mẫu, sửa guideline).
4. **So sánh CRF + SVM và ViSoBERT trên cùng một câu** khó (quan hệ xa, phủ định).
5. **Deletion test:** xoá top-k từ quan trọng theo LIME/SHAP → đo mức giảm xác suất nối, so với xoá ngẫu nhiên.

**Trình bày:** bôi màu từ trong câu theo mức đóng góp, bar chart top đặc trưng, bảng ví dụ đúng / sai.

### 8.3. Learning curve — bao nhiêu dữ liệu là đủ?

- Tập con lồng nhau của train: **250 → 500 → 1.000 → 1.500 → 2.000** bình luận.
- Mỗi kích thước train **CRF + SVM** và **ViSoBERT tagger + pair**, mỗi mô hình **3 seed**.
- Đánh giá trên **cùng tập test 600**: Triplet F1 + F1 của ý kiến Tiêu cực và bộ ba quan hệ xa.

```
Triplet F1
 │                     ●────●   ← đi ngang: thêm dữ liệu không giúp nhiều → đã đủ
 │               ●
 │         ●
 │   ●
 └───────────────────────────── Số bình luận train
    250   500  1.000 1.500 2.000
```

| Hình dạng đường | Kết luận |
|---|---|
| Vẫn dốc lên ở 2.000 | Gán thêm vẫn có lợi → ghi vào hướng phát triển |
| Đi ngang từ khoảng 1.500 | Dữ liệu đã đủ; muốn tốt hơn phải đổi / cải tiến mô hình |
| ViSoBERT vượt CRF từ một mốc nào đó | Cho biết cần bao nhiêu dữ liệu để transformer đáng dùng |

**(Tuỳ chọn)** Vẽ thêm đường **active learning và chọn ngẫu nhiên** với cùng số câu.

---

## 9. Ứng dụng (Deployment) — Web app Streamlit

**Chức năng cơ bản (bắt buộc):**
1. **Phân tích 1 bình luận:** nhập text → **bôi màu ngay trên câu** (khía cạnh xanh dương, ý kiến xanh lá / vàng / đỏ) + bảng bộ ba + xác suất.
2. **Chọn mô hình:** Luật / CRF + SVM / ViSoBERT / PhoBERT để so sánh trực tiếp.
3. **Phân tích hàng loạt:** upload CSV → CSV bộ ba (mỗi dòng 1 bộ ba) + biểu đồ top khía cạnh được khen / bị chê.
4. **Giải thích:** nút "Vì sao?" cạnh mỗi bộ ba → bôi màu các từ khiến mô hình nối và gán cảm xúc (hệ số SVM: tức thì; LIME cho ViSoBERT: ~15–30 giây trên CPU, có thanh tiến trình và cache).

**Chức năng nâng cao (điểm cộng):**
5. **Phân tích sản phẩm:** dán link Tiki → crawl review realtime → trích xuất bộ ba → gom theo chủ đề (§7.5) → **"Top điều được khen / bị chê"** kèm số lần và câu gốc làm bằng chứng; biểu đồ tỉ lệ khen / chê theo chủ đề.
6. **So sánh 2 sản phẩm** theo từng chủ đề.

Triển khai: local, Streamlit Community Cloud hoặc Hugging Face Spaces.

---

## 10. Hướng nâng cao (triển khai nếu còn thời gian)

| # | Hướng nâng cao | Mô tả | Công sức |
|---|---|---|---|
| 1 | **Dashboard phân tích sản phẩm** | Chức năng 5–6 ở §9 | Thấp |
| 2 | **So sánh với LLM** | Few-shot trích xuất bộ ba bằng LLM trên cùng tập test → so sánh Triplet F1, chi phí, tốc độ | Thấp |
| 3 | **Mô hình end-to-end** | Grid Tagging Scheme (GTS, Wu et al. 2020) hoặc span-based: tìm cụm từ và nối trong 1 mô hình → so với pipeline 2 bước | Cao |
| 4 | **Label Studio ML Backend** | Chạy mô hình như service để Label Studio bôi gợi ý trực tiếp, không cần xuất / nhập file mỗi vòng | Trung bình |
| 5 | **Cải tiến mô hình** | Thêm CRF trên ViSoBERT, ensemble ViSoBERT + PhoBERT, continued pre-training ViSoBERT trên review chưa gán nhãn, tăng cường dữ liệu | Trung bình |
| 6 | **Kiểm tra khả năng tổng quát** | Đánh giá trên ngành hàng chưa thấy khi train | Trung bình |
| 7 | **Quy về danh mục khía cạnh cố định** | Map chủ đề (§7.5) về 7 danh mục (chất lượng, giá, giao hàng, đóng gói, dịch vụ, chính hãng, chung) → đối chiếu với các bộ ABSA công khai như UIT-ViSFD | Trung bình |

---

## 11. Cấu trúc thư mục

```
.
├── data/
│   ├── raw/              # dữ liệu crawl thô (không commit)
│   ├── labeled/          # aste_<split>.jsonl, bio_<split>.txt
│   └── processed/        # văn bản đã làm sạch lớp 1, danh sách test/val
├── labeling/
│   ├── label_config.xml       # cụm từ + quan hệ (§5.1)
│   ├── guideline.md
│   ├── lexicon.json           # từ điển khía cạnh / ý kiến / phủ định / mức độ (gợi ý + baseline luật)
│   ├── preannotate.py         # bôi sẵn cụm từ + quan hệ bằng từ điển (đợt 1)
│   ├── split_data.py          # chọn & đóng băng test/val
│   ├── active_learning.py     # chọn câu + bôi sẵn bằng mô hình (đợt 2)
│   ├── make_tasks.py          # chia câu thành file import cho từng người (chính + chéo)
│   ├── agreement.py           # gộp export, đo độ đồng thuận, xuất câu bị lệch
│   ├── ls_common.py           # đọc kết quả Label Studio → cụm từ / bộ ba / BIO
│   ├── ls_to_dataset.py       # export Label Studio → aste_*.jsonl + bio_*.txt
│   ├── tasks/                 # file import (không commit, bản gốc trên Drive)
│   └── exports/               # file export (không commit, bản gốc trên Drive)
├── notebooks/
│   ├── 01_crawling.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_ml_models.ipynb          # luật, CRF, LR/SVM/RF nối quan hệ
│   ├── 05_deep_learning.ipynb      # BiLSTM-CRF, ViSoBERT & PhoBERT tagger + pair
│   ├── 06_evaluation.ipynb         # 3 cấp metrics, phân tích theo khoảng cách, learning curve
│   ├── 07_explainability.ipynb     # trọng số CRF, SHAP/LIME, deletion test
│   └── 08_aspect_clustering.ipynb  # gom nhóm khía cạnh & đặt tên chủ đề
├── src/
│   ├── crawler/          # tiki_crawler.py
│   ├── preprocessing/    # cleaner.py (lớp 1), tokenizer.py (tách từ + căn chỉnh BIO), teencode_dict.json
│   ├── models/           # rule_baseline.py, crf_tagger.py, ml_pairing.py, bilstm_crf.py,
│   │                     # transformer_tagger.py, transformer_pairing.py, predict.py
│   ├── clustering/       # aspect_clustering.py
│   ├── evaluation/       # aste_metrics.py
│   ├── explain/          # lime_shap.py, highlight.py
│   └── utils/
├── models/               # model đã train, aspect_topics.json (không commit file lớn)
├── app/
│   └── streamlit_app.py
├── reports/              # figures/, agreement/, bảng kết quả, báo cáo
├── requirements.txt
└── README.md
```

---

## 12. Công cụ & thư viện

- **Crawl & scrape:** `requests`
- **Gán nhãn:** `label-studio`, `label-studio-ml` (nâng cao)
- **NLP tiếng Việt:** `underthesea` (POS, tách từ), `py_vncorenlp` (cho PhoBERT)
- **ML:** `scikit-learn`, `sklearn-crfsuite`, `xgboost`
- **DL:** `torch`, `transformers`, `datasets`, `pytorch-crf` (BiLSTM-CRF / lớp CRF)
- **Đánh giá:** `seqeval`
- **Phân cụm:** `scikit-learn` (Agglomerative, KMeans), `hdbscan`
- **Giải thích mô hình:** `eli5` (CRF), `shap`, `lime`, `captum`
- **EDA/Visualization:** `pandas`, `matplotlib`, `seaborn`, `wordcloud`, `plotly`
- **App:** `streamlit`, `st-annotated-text` (bôi màu cụm từ)

---

## 13. Kế hoạch thực hiện (~9 tuần)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1 | Chốt schema nhãn; crawl & scrape (crawler đã xong); cấu hình Label Studio; từ điển `lexicon.json` v0 | ~60.000–80.000 review thô, Label Studio chạy được |
| 2 | EDA dữ liệu thô; tiền xử lý lớp 1; guideline; **pilot 100 câu** | Notebook EDA, `guideline.md` v1, schema đã chốt |
| 3 | **Đợt 1a:** gán **test 600** + gán chéo 30%. *Song song:* `split_data.py`, `ls_to_dataset.py`, `tokenizer.py`, baseline luật | Test đóng băng, độ đồng thuận của test |
| 4 | **Đợt 1b:** **val 400 + train 600** + gán chéo 15%. **Điểm kiểm tra**; mô hình ban đầu (CRF + SVM, ViSoBERT tagger); `active_learning.py` | Baseline đầu tiên trên val |
| 5 | **Đợt 2:** 2 vòng active learning × ~700 (mỗi vòng ~2,5 ngày). *Song song:* EDA sau gán nhãn, CRF + LR/SVM/RF hoàn chỉnh | **Bộ dữ liệu cuối ~3.000**, kết quả luật + ML |
| 6 | BiLSTM-CRF; ViSoBERT & PhoBERT (tagger + pair); **learning curve** | Checkpoint DL, bảng so sánh, learning curve |
| 7 | Đánh giá 3 cấp, phân tích theo khoảng cách, error analysis; **giải thích mô hình**; **gom nhóm khía cạnh** | Bảng kết quả cuối, hình giải thích, bảng chủ đề |
| 8 | Web app cơ bản (gồm bôi màu + giải thích); **nâng cao** nếu kịp (ưu tiên #1 → #2 → #3 ở §10) | App chạy được |
| 9 | Báo cáo, slide, video demo | Bản nộp cuối |

> Nếu chậm tiến độ:
> - Tuần 5 trễ → **dừng sau vòng active learning 1 (~2.300 câu)** và chuyển sang tuần 6. Learning curve (§8.3) sẽ cho thấy dừng ở đó ảnh hưởng nhiều hay ít.
> - Thiếu thời gian cuối kỳ → bỏ phần nâng cao, giữ phần cốt lõi (crawl → gán nhãn → luật + ML + ViSoBERT → đánh giá + giải thích → app cơ bản).

### 13.1. Ước tính thời gian

*Giả định:* ~3.000 bình luận 10–80 từ (train ~2.000, val 400, test 600), GPU T4 của Colab/Kaggle, fp16. Đây là **ước tính**; hãy đo thời gian thật sau tuần 3 và epoch đầu tiên rồi cập nhật.

**a. Công sức con người**

| Bước | Ước tính |
|---|---|
| Crawl 60.000–80.000 review | Máy chạy ~1,5–2 giờ (chạy qua đêm) |
| Pilot 100 câu × 4 người (~100 giây/câu) + buổi thảo luận | ~15–17 giờ công cả nhóm |
| Đợt 1: test 600 (gợi ý từ điển, ~90–110 giây/câu) | ~15–18 giờ công |
| Đợt 1: gán chéo 30% test (180 câu) | ~5 giờ công |
| Đợt 1: val + train 1.000 (~95 giây/câu) | ~26 giờ công |
| Đợt 1: gán chéo 15% (150 câu) | ~4 giờ công |
| Đợt 2: 2 vòng active learning, 1.400 câu (sửa gợi ý, ~40–50 giây/câu) | ~16–20 giờ công |
| Đợt 2: kiểm tra 10% không gợi ý (140 câu) | ~4 giờ công |
| **Tổng gán nhãn** | **~85–95 giờ công** → nhóm 4 người: ~21–24 giờ/người, **~7 giờ/người/tuần** trong tuần 3–5 |

→ Gán nhãn vẫn là **phần tốn thời gian nhất**. Ít câu hơn hướng phân lớp (3.000 so với 6.000) nhưng tổng giờ công tương đương, vì mỗi câu phải bôi cụm từ và nối quan hệ.

**b. Huấn luyện mô hình**

| Mô hình | 1 lần train | Kèm tinh chỉnh / nhiều seed | Phần cứng |
|---|---|---|---|
| Baseline luật | Không cần train (chạy test vài giây) | — | CPU |
| CRF (2.000 câu) | 1–3 phút | 15–30 phút (lưới `c1`, `c2`) | CPU |
| LR / Linear SVM nối quan hệ (~10.000–15.000 cặp) | Vài giây | 5–10 phút | CPU |
| Random Forest / XGBoost nối quan hệ | 1–2 phút | 10–20 phút | CPU |
| BiLSTM-CRF | 5–10 phút | ~1 giờ | GPU |
| ViSoBERT / PhoBERT tagger (15 epoch) | ~5–10 phút | ~1–1,5 giờ **mỗi mô hình** | GPU T4 |
| ViSoBERT / PhoBERT pair (~10.000–15.000 cặp, 5 epoch) | ~15–25 phút | ~1,5–2,5 giờ **mỗi mô hình** | GPU T4 |
| Mô hình ban đầu + 2 vòng active learning | ~15–20 phút mỗi vòng | ~1 giờ tổng | GPU T4 |
| Learning curve (ViSoBERT, 5 mốc × 3 seed) | — | ~2–3 giờ | GPU T4 |

**c. Giải thích mô hình**

| Tác vụ | Ước tính |
|---|---|
| Trọng số CRF, hệ số + SHAP linear cho LR/SVM | < 1 phút (CPU) |
| LIME / SHAP cho ViSoBERT pair, 150 bộ ba | ~20–40 phút mỗi mô hình (GPU) |
| Integrated Gradients cho tagger (tuỳ chọn) | ~10–20 phút (GPU) |
| 1 lần giải thích trong app (LIME ~300 mẫu nhiễu, CPU) | ~15–30 giây (SVM: tức thì) |

**d. Suy luận (dự đoán)**

| Tác vụ | Luật / CRF + SVM | ViSoBERT tagger + pair |
|---|---|---|
| 1 bình luận, CPU | < 20 ms | ~100–300 ms |
| 1.000 bình luận, CPU | ~5–10 giây | ~3–5 phút |
| 1.000 bình luận, GPU | — | ~15–30 giây |
| Phân tích 1 sản phẩm (~500 review) | crawl 1–2 phút + vài giây | crawl 1–2 phút + 1–2 phút |

**e. Tổng ngân sách GPU:** ~**10–15 giờ**, rải trong tuần 4–7. Kaggle cho ~30 giờ GPU/tuần nên đủ. Colab free hay ngắt phiên nên **lưu checkpoint lên Drive sau mỗi epoch**.

---

## 14. Rủi ro & hướng xử lý

| Rủi ro | Hướng xử lý |
|---|---|
| Gán nhãn cụm từ chậm, mỗi người bôi ranh giới một kiểu | Pilot trước; guideline nhiều ví dụ; `granularity="word"`; gợi ý bằng từ điển; báo cáo cả F1 khớp chính xác và khớp một phần |
| Quên nối quan hệ (mọi ý kiến thành khía cạnh ẩn) | `agreement.py` báo số khía cạnh ẩn mỗi người; tỉ lệ cao bất thường → nhắc nhở |
| Văn bản bị sửa sau khi gán → lệch vị trí cụm từ | Tiền xử lý **lớp 1 cố định trước khi gán**; lớp 2 chỉ xử lý trên từng từ (§6) |
| PhoBERT tách từ làm lệch căn chỉnh nhãn | Ánh xạ nhãn theo âm tiết đầu của từ ghép; viết test cho hàm căn chỉnh; nếu vướng thì ưu tiên ViSoBERT (không tách từ) |
| Gán nhãn đợt 2 bị trễ | Dừng sau vòng 1 (~2.300 câu); dùng learning curve để chứng minh mức dữ liệu đó đủ hay chưa |
| Người gán "bấm theo" gợi ý của mô hình | Test chỉ dùng gợi ý từ điển; 10% mỗi vòng gán lại không có gợi ý; theo dõi tỉ lệ giữ nguyên gợi ý |
| Ý kiến Tiêu cực / Trung lập quá ít (87% review là 5 sao) | Phân tầng lấy thêm câu 1–3 sao; active learning ưu tiên câu "khó"; class weights |
| Cặp "không nối" nhiều hơn hẳn cặp "nối" (bước 2) | `class_weight`, chỉnh ngưỡng xác suất trên val |
| ~50% review không có nội dung | Crawl 60.000–80.000 review thô; thêm ngành hàng nếu thiếu |
| Bị chặn khi crawl / Tiki đổi API | Request tuần tự, tự chờ và thử lại; tìm lại endpoint bằng F12 (§3.2); crawl sớm và lưu lên Drive |
| Thiếu GPU | Colab / Kaggle; fp16; `max_len=128`; CRF + SVM vẫn chạy được trên CPU |
| Gom nhóm khía cạnh ra chủ đề lẫn lộn | Lọc cụm hiếm (< 3 lần); người kiểm tra 300 cụm phổ biến; cho phép sửa tay bảng `aspect_topics.json` |

---

## 15. Tài liệu tham khảo

- Hu, M., & Liu, B. (2004). *Mining and Summarizing Customer Reviews.* KDD. (trích xuất ý kiến dựa trên luật)
- Qiu, G., Liu, B., Bu, J., & Chen, C. (2011). *Opinion Word Expansion and Target Extraction through Double Propagation.* Computational Linguistics.
- Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data.* ICML.
- Lample, G. et al. (2016). *Neural Architectures for Named Entity Recognition* (BiLSTM-CRF). NAACL.
- Pontiki, M. et al. (2014). *SemEval-2014 Task 4: Aspect Based Sentiment Analysis.*
- Peng, H. et al. (2020). *Knowing What, How and Why: A Near Complete Solution for Aspect-Based Sentiment Analysis* (đề xuất bài toán ASTE). AAAI.
- Xu, L. et al. (2020). *Position-Aware Tagging for Aspect Sentiment Triplet Extraction* (bộ ASTE-Data-V2). EMNLP.
- Wu, Z. et al. (2020). *Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction.* Findings of EMNLP.
- Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP.
- Nguyen, Q. N. et al. (2023). *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing.* EMNLP.
- Luc Phan, L. et al. (2021). *SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System for Business Intelligence* (bộ UIT-ViSFD, tham khảo về miền dữ liệu TMĐT tiếng Việt).
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?"* (LIME). KDD.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks* (Integrated Gradients). ICML.
- Label Studio Documentation — https://labelstud.io/guide/
