# Hướng đi đồ án: Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis) cho bình luận E-commerce tiếng Việt

## 1. Tổng quan

| Mục | Nội dung |
|---|---|
| **Bài toán chính** | **ABSA**: với mỗi bình luận, xác định (1) bình luận nói về **khía cạnh** nào và (2) **cảm xúc** đối với từng khía cạnh đó |
| **Loại bài toán** | Multi-label + multi-class classification (supervised) |
| **Ngôn ngữ dữ liệu** | Tiếng Việt |
| **Nguồn dữ liệu** | Tiki (chính, có API JSON), Shopee / Lazada (bổ sung) |
| **Đầu ra cuối** | Web app: nhập bình luận hoặc link sản phẩm → cảm xúc theo từng khía cạnh + thống kê điểm mạnh/yếu của sản phẩm |

**Ví dụ:**

> *"Hàng đẹp, vải mát nhưng giao chậm quá, shipper còn thái độ. Giá thì ok."*

| Khía cạnh | Cảm xúc |
|---|---|
| Chất lượng sản phẩm | Tích cực |
| Giao hàng | Tiêu cực |
| Giá cả | Tích cực |
| Đóng gói, Dịch vụ shop, Đúng mô tả, Tổng quan | *(không đề cập)* |

Lý do chọn ABSA thay vì phân loại cảm xúc cả câu: một bình luận thường **vừa khen vừa chê**. Rating 3–4 sao rất khó phân loại nếu chỉ gán một nhãn cho cả câu. ABSA cho kết quả có giá trị thực tế hơn với người bán (biết cụ thể phải cải thiện khâu nào).

### Đối chiếu yêu cầu đồ án

| Yêu cầu | Cách đáp ứng | Mục |
|---|---|---|
| Gán nhãn thủ công bằng Label Studio | Gán nhãn khía cạnh + cảm xúc cho **~6.000 bình luận** trong 2 đợt (đợt 2 có active learning), đo Cohen's Kappa | §5 |
| Dữ liệu tiếng Việt | Crawl & scrape review từ Tiki qua API JSON nội bộ | §3 |
| ≥ 3 mô hình ML truyền thống | Naive Bayes, Logistic Regression, Linear SVM (+ Random Forest / XGBoost), mỗi mô hình huấn luyện theo từng khía cạnh | §7.2 |
| ≥ 1 mô hình Deep Learning | ViSoBERT & PhoBERT multi-task (chính, so sánh trực tiếp) + BiLSTM/TextCNN | §7.3 |
| Quy trình data science đầy đủ | Crawl & scrape → EDA → Tiền xử lý → Gán nhãn → Huấn luyện → Đánh giá → Ứng dụng | §3–§9 |
| Ứng dụng đơn giản | Web app Streamlit (có chức năng giải thích dự đoán) | §9 |
| *(Bổ sung)* Giải thích mô hình | LIME / SHAP cho cả ML và transformer, tích hợp vào app | §8.2 |

---

## 2. Định nghĩa khía cạnh (Aspect Schema)

Các khía cạnh được chọn để **dùng chung cho mọi ngành hàng** (không phụ thuộc loại sản phẩm):

| Mã | Khía cạnh | Bao gồm | Ví dụ từ khoá |
|---|---|---|---|
| `QUALITY` | Chất lượng sản phẩm | Chất liệu, độ bền, hiệu năng, công dụng, mùi vị | *vải, bền, xài, mượt, pin, thơm, hỏng* |
| `PRICE` | Giá cả | Giá, khuyến mãi, đáng tiền | *giá, rẻ, đắt, sale, đáng tiền, voucher* |
| `DELIVERY` | Giao hàng | Tốc độ, shipper, phí ship | *giao, ship, nhanh, chậm, shipper* |
| `PACKAGING` | Đóng gói | Hộp, bọc chống sốc, tình trạng khi nhận | *đóng gói, hộp, móp, bọc, cẩn thận* |
| `SERVICE` | Dịch vụ shop | Tư vấn, phản hồi, đổi trả, bảo hành | *shop, tư vấn, nhiệt tình, đổi trả, rep* |
| `AUTHENTICITY` | Đúng mô tả / chính hãng | Giống hình, đúng size/màu, hàng thật/giả | *giống hình, đúng mô tả, fake, chính hãng, size* |
| `GENERAL` | Tổng quan | Nhận xét chung không gắn khía cạnh cụ thể | *ok, tốt, hài lòng, sẽ ủng hộ, thất vọng* |

Mỗi khía cạnh nhận **1 trong 4 giá trị**: `None` (không đề cập) · `Tích cực` · `Trung lập` · `Tiêu cực`.

> Có thể điều chỉnh schema sau đợt gán nhãn thử (pilot ~200 mẫu): gộp khía cạnh quá hiếm, tách khía cạnh quá rộng.

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
  - Khoảng 6.000 bình luận được gán nhãn (§5).
  - Phần còn lại (~10.000–14.000) là **kho chưa gán nhãn**, dùng để active learning chọn mẫu và làm dữ liệu demo cho dashboard.
- **Ưu tiên bình luận dài** (≥ 10 từ) khi chọn mẫu gán nhãn vì bình luận ngắn ("ok", "tốt") hầu như chỉ có `GENERAL`.

### 3.6. Kết quả chạy thử (215 review, 8 sản phẩm, 3 ngành hàng)

| Quan sát | Số liệu | Hệ quả |
|---|---|---|
| Review **không có nội dung** (chỉ chấm sao) | ~51% | Mục tiêu ban đầu 20.000 review thô chỉ còn ~5.000 câu dài, **không đủ** → tăng lên 60.000–80.000. Đổi cách sắp xếp review của API không cải thiện được tỉ lệ này |
| Review có ≥ 10 từ | ~25% | Đây là tỉ lệ dùng để tính số review thô cần crawl |
| Rating 5 sao | ~87% (1–2 sao chỉ ~4%) | Khi chọn mẫu gán nhãn phải lấy thêm câu 1–3 sao |
| Spam ("AE qua nhóm mình săn sale...") và tên người trong nội dung | Có xuất hiện | Lọc spam và ẩn tên ở bước tiền xử lý (§6) |
| Trường `delivery_rating` có sẵn (giao đúng hẹn, thái độ shipper, đóng gói) | ~25% review | Dùng cho EDA và đối chiếu với nhãn `DELIVERY` / `PACKAGING` |
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
- Phân bố độ dài bình luận; tỉ lệ bình luận rỗng/quá ngắn; trùng lặp, spam.
- Phân bố theo ngành hàng và thời gian.
- Top n-gram, word cloud theo nhóm rating; tần suất emoji, teencode.

**Sau khi gán nhãn (EDA cho ABSA):**
- Tần suất xuất hiện mỗi khía cạnh → phát hiện khía cạnh hiếm.
- Phân bố cảm xúc trong từng khía cạnh (ví dụ `DELIVERY` thường lệch tiêu cực, `GENERAL` lệch tích cực).
- Số khía cạnh trung bình trên mỗi bình luận.
- Ma trận đồng xuất hiện giữa các khía cạnh (heatmap).
- Khía cạnh × ngành hàng (ví dụ `AUTHENTICITY` nổi bật ở thời trang và mỹ phẩm).
- Tương quan rating ↔ cảm xúc khía cạnh (ví dụ 5 sao nhưng chê giao hàng).

Công cụ: `pandas`, `matplotlib`, `seaborn`, `wordcloud`, `plotly`.

---

## 5. Gán nhãn (Labeling) với Label Studio

### 5.1. Cấu hình giao diện

Mỗi khía cạnh là một nhóm lựa chọn; bỏ trống nghĩa là `None` (không đề cập):

```xml
<View>
  <Header value="Bình luận"/>
  <Text name="text" value="$content"/>
  <Header value="Rating gốc: $rating sao" size="6"/>

  <View style="display:grid; grid-template-columns: 1fr 1fr; gap: 8px">
    <View><Header value="Chất lượng sản phẩm" size="5"/>
      <Choices name="QUALITY" toName="text" choice="single" showInline="true">
        <Choice value="Tích cực"/><Choice value="Trung lập"/><Choice value="Tiêu cực"/>
      </Choices></View>
    <View><Header value="Giá cả" size="5"/>
      <Choices name="PRICE" toName="text" choice="single" showInline="true">
        <Choice value="Tích cực"/><Choice value="Trung lập"/><Choice value="Tiêu cực"/>
      </Choices></View>
    <View><Header value="Giao hàng" size="5"/>
      <Choices name="DELIVERY" toName="text" choice="single" showInline="true">
        <Choice value="Tích cực"/><Choice value="Trung lập"/><Choice value="Tiêu cực"/>
      </Choices></View>
    <View><Header value="Đóng gói" size="5"/>
      <Choices name="PACKAGING" toName="text" choice="single" showInline="true">
        <Choice value="Tích cực"/><Choice value="Trung lập"/><Choice value="Tiêu cực"/>
      </Choices></View>
    <View><Header value="Dịch vụ shop" size="5"/>
      <Choices name="SERVICE" toName="text" choice="single" showInline="true">
        <Choice value="Tích cực"/><Choice value="Trung lập"/><Choice value="Tiêu cực"/>
      </Choices></View>
    <View><Header value="Đúng mô tả / chính hãng" size="5"/>
      <Choices name="AUTHENTICITY" toName="text" choice="single" showInline="true">
        <Choice value="Tích cực"/><Choice value="Trung lập"/><Choice value="Tiêu cực"/>
      </Choices></View>
    <View><Header value="Tổng quan" size="5"/>
      <Choices name="GENERAL" toName="text" choice="single" showInline="true">
        <Choice value="Tích cực"/><Choice value="Trung lập"/><Choice value="Tiêu cực"/>
      </Choices></View>
  </View>

  <Choices name="skip" toName="text" choice="single">
    <Choice value="Spam / vô nghĩa (loại bỏ)"/>
  </Choices>
</View>
```

> **Nâng cao:** thêm `<Labels>` để bôi đen **cụm từ khía cạnh** và **cụm từ thể hiện ý kiến** trong câu (dùng cho hướng nâng cao #3 ở §10).

### 5.2. Bao nhiêu dữ liệu & chia tập thế nào

**Mục tiêu: ~6.000 bình luận được gán nhãn.**

Lý do: có 7 khía cạnh × 3 cảm xúc = **21 cặp**, nhưng dữ liệu rất lệch. Với 3.000 mẫu, tập test chỉ có khoảng 5–7 câu cho các cặp hiếm (ví dụ `PACKAGING` Tiêu cực), sai 1 câu là F1 của cặp đó thay đổi 10–20 điểm. Khi đó kết quả không đáng tin, và không so sánh được các mô hình với nhau.

Ngưỡng tham khảo: **test ≥ 30–50 mẫu** và **train ≥ 100–200 mẫu** cho mỗi cặp chính. (Bộ UIT-ViSFD có khoảng 11.000 bình luận.)

| Tập | Số mẫu | Hình dung | Cách lấy mẫu | Cách gán |
|---|---|---|---|---|
| **Test** | 1.000 | Đề thi cuối kỳ: soạn trước, **cất kín**, chỉ dùng để chấm điểm cuối cùng | Ngẫu nhiên, phân tầng theo ngành hàng + rating, **giữ phân bố tự nhiên** | Chỉ gợi ý bằng từ khoá (không dùng mô hình), **30% gán chéo** |
| **Val** | 800 | Đề thi thử: dùng để chỉnh tham số, chọn mô hình | Như test | Gợi ý từ khoá, 15% gán chéo |
| **Train** | ~4.200 | Bài tập về nhà | 1.200 ngẫu nhiên (đợt 1) + ~3.000 do active learning chọn (đợt 2) | Gợi ý từ khoá, sau đó gợi ý bằng mô hình |

**Quy tắc:**
- Tập test được **chọn và đóng băng ngay từ đầu**. Không dùng nó để chọn mẫu, chỉnh tham số hay sửa guideline.
- Val và test giữ phân bố tự nhiên. **Chỉ tập train** được bổ sung thêm khía cạnh hiếm và mẫu do active learning chọn.

### 5.3. Quy trình gán nhãn 2 đợt

```
Pilot 200 ──► ĐỢT 1 (~3.000, tự gán từ đầu) ──► Điểm kiểm tra ──► ĐỢT 2 (~3.000, mô hình gợi ý, 3 vòng AL) ──► ~6.000
```

**Bước 0 — Pilot (~200 mẫu):** cả nhóm cùng gán → thống nhất guideline và chốt schema khía cạnh. Mẫu pilot không đưa vào test.

**Đợt 1 — Nền móng (~3.000 mẫu, gợi ý bằng từ khoá):**
1. Gán **test 1.000** trước, rồi val 800 và train 1.200.
2. **Pre-annotation bằng từ khoá:** dùng từ điển ở §2 để gợi ý khía cạnh (`labeling/preannotate.py` tạo JSON có sẵn `predictions` để import vào Label Studio). Người gán kiểm tra khía cạnh và chọn cảm xúc. Khoảng 45–60 giây/mẫu.
3. Gán chéo để đo **Cohen's Kappa cho từng khía cạnh** (mục tiêu ≥ 0,7): 30% tập test, 15% phần còn lại. Thảo luận để thống nhất các câu bị gán khác nhau. (`labeling/make_tasks.py` chia file gán chính/chéo cho từng người, `labeling/kappa.py` tính Kappa và xuất danh sách câu lệch; quy trình chi tiết ở `TEAM_ASSIGNMENT.md` §11.)

**Điểm kiểm tra giữa 2 đợt (1–2 buổi):**
1. Khía cạnh nào Kappa < 0,6 thì sửa guideline trước khi gán tiếp.
2. Đếm số mẫu từng cặp trong test. Cặp nào < 30 mẫu thì gộp lớp, hoặc chấp nhận và báo cáo F1 kèm khoảng tin cậy (bootstrap).
3. Train **mô hình ban đầu** (SVM + ViSoBERT multi-task) trên 1.200 mẫu train, đánh giá trên val. Đây là baseline đầu tiên và là mô hình gợi ý nhãn cho đợt 2.
4. Chốt số mẫu đợt 2 (có thể ít hơn 3.000 nếu các cặp hiếm đã đủ).

**Đợt 2 — Tăng tốc với active learning (~3.000 mẫu, 3 vòng × ~1.000):** xem §5.4. Người gán chỉ **kiểm tra và sửa** nhãn do mô hình gợi ý, khoảng 15–25 giây/mẫu. Mỗi vòng có 10% mẫu được gán lại **không có gợi ý** để kiểm tra người gán có bị mô hình dẫn dắt hay không.

**Guideline** (`labeling/guideline.md`) cần làm rõ:
- Ranh giới `QUALITY` và `GENERAL` ("sản phẩm tốt" là `GENERAL`; "vải mát, đường may đẹp" là `QUALITY`).
- Ranh giới `QUALITY` và `AUTHENTICITY` ("không giống hình" là `AUTHENTICITY`).
- Khi nào chọn `Trung lập` (mô tả khách quan, câu hỏi, khen-chê lẫn trong cùng khía cạnh).
- Mỉa mai, phủ định ("giao nhanh thật, 2 tuần mới tới" là `DELIVERY` = Tiêu cực).

**Export:** JSON → script chuyển thành bảng phẳng `content, QUALITY, PRICE, DELIVERY, PACKAGING, SERVICE, AUTHENTICITY, GENERAL, split, batch` (giá trị `0=None, 1=Tích cực, 2=Trung lập, 3=Tiêu cực`; `split` = train/val/test; `batch` = đợt 1 / vòng AL 1–3) → `data/labeled/`.

### 5.4. Active learning (đợt 2)

**Ý tưởng:** thay vì chọn ngẫu nhiên, để mô hình chọn những câu **nó phân vân nhất**. Những câu này dạy mô hình được nhiều nhất trên mỗi giờ gán nhãn.

```
"sp tốt, giao nhanh"               → mô hình chắc chắn  → bỏ qua
"giao nhanh thật, 2 tuần mới tới"  → mô hình phân vân   → CHỌN để gán nhãn
```

**Một vòng (script offline `labeling/active_learning.py`):**

| Bước | Việc làm | Thời gian |
|---|---|---|
| 1 | Mô hình hiện tại dự đoán toàn bộ kho ~10.000–14.000 bình luận dài chưa gán | SVM: vài giây; ViSoBERT: ~3–5 phút GPU |
| 2 | Tính **độ không chắc chắn** mỗi câu = entropy lớn nhất trong 7 khía cạnh | Vài giây |
| 3 | Chọn ~1.000 câu: **70%** phân vân nhất + **20%** chứa từ khoá khía cạnh hiếm + **10%** ngẫu nhiên (tránh lệch phân bố) | Vài giây |
| 4 | Xuất JSON có `predictions` → import Label Studio → người gán sửa | ~5–7 giờ công |
| 5 | Gộp vào train, train lại mô hình | SVM: vài giây; ViSoBERT: ~10–15 phút |

**Lưu ý:**
- Mẫu do active learning chọn **chỉ vào train**, không vào val/test.
- Ghi lại sau mỗi vòng: F1 trên val, và **tỉ lệ nhãn gợi ý được giữ nguyên** (tỉ lệ này tăng nghĩa là mô hình tốt dần). Hai số liệu này đưa vào báo cáo.
- Tích hợp trực tiếp bằng Label Studio ML Backend (mô hình gợi ý ngay trong giao diện) để ở phần nâng cao (§10).

---

## 6. Tiền xử lý dữ liệu (Preprocessing)

| Bước | Chi tiết |
|---|---|
| Làm sạch | Bỏ HTML, URL, số điện thoại, ký tự thừa |
| Chuẩn hoá Unicode | NFC; chuẩn hoá kiểu gõ dấu (`òa` ↔ `oà`) |
| Lowercase | Chuyển chữ thường |
| Teencode / viết tắt | Từ điển: `k/ko/hok → không`, `dc/đc → được`, `sp → sản phẩm`, `shop rep → shop trả lời`... |
| Ký tự lặp | `đẹpppp → đẹp` |
| Emoji | Chuyển thành token cảm xúc (`😍 → emo_pos`, `😡 → emo_neg`) |
| Tách từ | `underthesea` / `pyvi` cho ML; **VnCoreNLP RDRSegmenter** cho PhoBERT; **không tách từ** cho ViSoBERT (tokenizer tự xử lý) |
| Lưu 2 phiên bản text | Bản *chuẩn hoá đầy đủ* (cho ML, PhoBERT) và bản *làm sạch nhẹ, giữ teencode/emoji* (cho ViSoBERT, vì mô hình đã học những thứ này khi pre-train) |
| Stopwords | Bỏ stopwords nhưng **giữ từ phủ định** (`không`, `chẳng`, `chưa`) và **từ chỉ khía cạnh** |
| Phủ định | Ghép `không tốt → không_tốt` (giúp ML truyền thống) |
| Chia dữ liệu | Theo §5.2: Test 1.000 / Val 800 / Train ~4.200. Test và val lấy bằng **iterative stratification** cho multi-label (`iterative-stratification`) ngay từ đợt 1 rồi cố định |
| Mất cân bằng | Class weights cho từng khía cạnh; oversampling các khía cạnh hiếm |

---

## 7. Xây dựng & huấn luyện mô hình

### 7.1. Cách đặt bài toán

**Cách A — Hai giai đoạn (pipeline):**
1. *Aspect Category Detection (ACD):* multi-label, 7 nhãn nhị phân (có/không đề cập).
2. *Aspect Sentiment Classification (ASC):* với mỗi khía cạnh được phát hiện, phân loại 3 lớp cảm xúc.

**Cách B — Gộp (joint):** mỗi khía cạnh là một bài toán **4 lớp** `{None, Tích cực, Trung lập, Tiêu cực}` → 7 bộ phân loại song song.

→ Thực hiện **cả hai cách với ML** để so sánh. Với DL dùng Cách B (multi-head).

### 7.2. Machine Learning truyền thống (≥ 3 mô hình)

**Đặc trưng:** TF-IDF (word uni+bi-gram, char n-gram 2–5); đặc trưng phụ: đếm từ khoá khía cạnh, đếm từ cảm xúc (từ điển VietSentiWordNet), emoji token.

| Mô hình | Cách dùng |
|---|---|
| **Multinomial Naive Bayes** | Baseline |
| **Logistic Regression** | `class_weight='balanced'` |
| **Linear SVM (LinearSVC)** | Thường mạnh nhất với TF-IDF |
| Random Forest / XGBoost | Bổ sung so sánh |

Bọc bằng `MultiOutputClassifier` (Cách B) hoặc `OneVsRestClassifier` + từng bộ ASC (Cách A). Tinh chỉnh bằng `GridSearchCV` / `Optuna`, metric **macro-F1**.

### 7.3. Deep Learning (≥ 1 mô hình)

**Hai encoder pre-trained được so sánh trực tiếp:**

| Encoder | Pre-train trên | Tách từ | Vì sao dùng |
|---|---|---|---|
| **ViSoBERT** (`uitnlp/visobert`) — *ưu tiên* | Văn bản mạng xã hội tiếng Việt | Không cần | Bình luận TMĐT gần với văn mạng xã hội: teencode, viết tắt, emoji, không dấu |
| **PhoBERT** (`vinai/phobert-base-v2`) | Wikipedia + báo chí | Bắt buộc (VnCoreNLP) | Baseline chuẩn, được dùng trong hầu hết nghiên cứu ABSA tiếng Việt → dễ đối chiếu |

**Kiến trúc (áp dụng cho cả hai encoder, chỉ đổi tên model):**

| Mô hình | Kiến trúc |
|---|---|
| **Multi-task** (chính) | Encoder → vector `[CLS]` / `<s>` → **7 head**, mỗi head là Linear 4 lớp; loss = tổng CrossEntropy (có class weight) của 7 head |
| Sentence-pair | Ghép input `[bình luận] </s> [tên khía cạnh]` → phân loại 4 lớp (theo Sun et al., 2019); tăng dữ liệu ×7, thường tốt cho khía cạnh hiếm |
| BiLSTM + Attention / TextCNN | Embedding fastText/word2vec tiếng Việt, đầu ra multi-head tương tự — DL không dùng transformer để so sánh |

**Phân tích so sánh ViSoBERT và PhoBERT** (đưa vào báo cáo):
- Kết quả tổng thể (Pair F1) và theo từng khía cạnh.
- Chia tập test theo **mức độ "phi chuẩn"** của câu (tỉ lệ teencode, emoji, không dấu) → xem khoảng cách giữa hai mô hình thay đổi thế nào.
- Ảnh hưởng của tiền xử lý: ViSoBERT với text giữ nguyên teencode so với text đã chuẩn hoá.
- Chi phí: thời gian huấn luyện, tốc độ suy luận, độ phức tạp khi triển khai (PhoBERT cần VnCoreNLP/Java trong app).

Mô hình tốt hơn sẽ được dùng làm mô hình mặc định trong app. **Nếu chỉ kịp chạy một mô hình, chọn ViSoBERT.**

Huấn luyện trên Colab / Kaggle GPU. Siêu tham số gợi ý (dùng chung cho cả hai để so sánh công bằng): `lr=2e-5`, `batch=16`, `max_len=256`, `epochs=5–8`, early stopping theo val macro-F1, fp16, cùng seed và cùng tập chia train/val/test.

---

## 8. Đánh giá & Giải thích mô hình (Evaluation & Explainability)

### 8.1. Metrics & so sánh mô hình

Đánh giá theo **3 cấp** (giống cách UIT-ViSFD / VLSP-ABSA báo cáo):

| Cấp | Metric | Ý nghĩa |
|---|---|---|
| **Aspect detection** | Precision / Recall / F1 (micro & macro) trên 7 khía cạnh | Mô hình có nhận ra đúng khía cạnh được nhắc tới không |
| **Aspect sentiment** | Accuracy, macro-F1 trên những khía cạnh *thật sự có đề cập* | Đã biết khía cạnh thì đoán đúng cảm xúc không |
| **Aspect + Sentiment (cặp)** | F1 trên cặp `(khía cạnh, cảm xúc)` | Chỉ số tổng hợp, **dùng để xếp hạng mô hình** |

Ngoài ra:
- Bảng F1 theo **từng khía cạnh** × từng mô hình (heatmap).
- Confusion matrix cho từng khía cạnh.
- **Error analysis:** mỉa mai, phủ định, nhầm `QUALITY`/`GENERAL`, khía cạnh ẩn (không có từ khoá rõ).
- So sánh Cách A và Cách B; so sánh với baseline chỉ dùng từ khoá.

Mẫu bảng kết quả:

| Mô hình | Aspect F1 | Sentiment macro-F1 | Pair F1 | Train time |
|---|---|---|---|---|
| Keyword baseline | | | | |
| Naive Bayes | | | | |
| Logistic Regression | | | | |
| Linear SVM | | | | |
| Random Forest / XGBoost | | | | |
| BiLSTM + Attention | | | | |
| PhoBERT multi-task | | | | |
| PhoBERT sentence-pair | | | | |
| ViSoBERT multi-task | | | | |
| ViSoBERT sentence-pair | | | | |

### 8.2. Giải thích mô hình (LIME / SHAP)

**Mục tiêu:** chỉ ra **từ nào** khiến mô hình kết luận, ví dụ, "Giao hàng: Tiêu cực". Từ đó kiểm tra mô hình học đúng tín hiệu ("chậm", "lâu", "shipper thái độ") hay học nhầm tín hiệu phụ (tên sản phẩm, tên shop, độ dài câu).

**Phương pháp theo loại mô hình:**

| Mô hình | Phương pháp | Ghi chú |
|---|---|---|
| LR / Linear SVM | **Hệ số mô hình** (global) + **SHAP `LinearExplainer`** (local) | Chính xác, gần như tức thì |
| Naive Bayes, RF / XGBoost | SHAP (`TreeExplainer` cho cây) hoặc LIME | |
| ViSoBERT / PhoBERT multi-task | **LIME** (`LimeTextExplainer`) + **SHAP `Partition` explainer** với text masker | Viết hàm wrapper `predict_proba(texts) → xác suất 4 lớp` cho **từng head khía cạnh**; một lần forward trả về đủ 7 head nên có thể giải thích nhiều khía cạnh cùng lúc |
| BiLSTM + Attention | Trọng số attention (tham khảo) + LIME | Attention không hẳn là giải thích, chỉ dùng để đối chiếu |

> Đơn vị giải thích: với ViSoBERT tách theo **âm tiết/khoảng trắng**; với PhoBERT và ML dùng **từ đã tách** (`giao_hàng`) cho dễ đọc.

**Các phân tích đưa vào báo cáo:**
1. **Global:** top 15 từ đóng góp nhiều nhất cho mỗi cặp `(khía cạnh, cảm xúc)` của SVM/LR, so với tổng hợp SHAP của ViSoBERT trên tập test. Xem hai bên có cùng tín hiệu hay không.
2. **Local:** giải thích **~100–200 mẫu test** chọn phân tầng: đúng/sai × từng khía cạnh. Ưu tiên các ca khó: phủ định ("không hề chậm"), mỉa mai, câu khen-chê lẫn lộn, khía cạnh ẩn.
3. **Gắn với error analysis (§8.1):** với mẫu dự đoán sai, chỉ ra từ gây nhầm, sau đó đề xuất cách sửa (bổ sung từ điển teencode, thêm mẫu gán nhãn, chỉnh guideline).
4. **So sánh mô hình trên cùng một câu:** SVM và ViSoBERT nhìn vào từ nào. Ví dụ minh hoạ vì sao transformer xử lý phủ định tốt hơn.
5. **Kiểm tra độ tin cậy của giải thích (deletion test):** xoá top-k từ quan trọng theo LIME/SHAP rồi đo mức giảm xác suất, so với xoá k từ ngẫu nhiên. Giảm càng mạnh thì giải thích càng đáng tin. Kiểm tra thêm mức đồng thuận giữa LIME và SHAP (độ trùng top-k từ).

**Trình bày:** highlight từ trong câu (xanh đẩy về Tích cực, đỏ đẩy về Tiêu cực, độ đậm theo mức đóng góp), bar chart top từ, bảng ví dụ đúng/sai.

### 8.3. Learning curve — bao nhiêu dữ liệu là đủ?

**Mục đích:** trả lời câu hỏi "có cần gán thêm dữ liệu không" bằng số liệu thật, thay vì phán đoán.

**Cách làm:**
- Lấy các tập con lồng nhau của train: **500 → 1.000 → 2.000 → 3.000 → 4.200** mẫu.
- Với mỗi kích thước, train **SVM, LR, ViSoBERT multi-task** (PhoBERT nếu còn GPU), mỗi mô hình **3 seed**.
- Đánh giá trên **cùng tập test 1.000**: Pair F1 tổng + F1 của 2–3 khía cạnh hiếm.
- Vẽ biểu đồ: trục X là số mẫu train, trục Y là F1, mỗi đường là một mô hình, có thanh sai số theo seed.

```
F1
 │                     ●────●   ← đi ngang: thêm dữ liệu không giúp nhiều → đã đủ
 │               ●
 │         ●
 │   ●
 └───────────────────────────── Số mẫu train
    500  1.000 2.000 3.000 4.200
```

**Cách đọc:**

| Hình dạng đường | Kết luận |
|---|---|
| Vẫn dốc lên ở 4.200 | Gán thêm vẫn có lợi → ghi vào hướng phát triển |
| Đi ngang từ khoảng 3.000 | Dữ liệu đã đủ; muốn tốt hơn phải đổi/cải tiến mô hình |
| ViSoBERT vượt SVM từ một mốc nào đó | Cho biết cần bao nhiêu dữ liệu để transformer đáng dùng |
| Khía cạnh hiếm vẫn tăng mạnh khi tổng đã ngang | Cần lấy thêm mẫu có chủ đích cho khía cạnh đó |

**(Tuỳ chọn)** Vẽ thêm đường so sánh **active learning và chọn ngẫu nhiên** với cùng số mẫu, để chứng minh active learning hiệu quả.

---

## 9. Ứng dụng (Deployment) — Web app Streamlit

**Chức năng cơ bản (bắt buộc):**
1. **Phân tích 1 bình luận:** nhập text → bảng khía cạnh + cảm xúc + xác suất, tô màu xanh/xám/đỏ.
2. **Chọn mô hình:** dropdown SVM / LR / PhoBERT / ViSoBERT để so sánh trực tiếp.
3. **Phân tích hàng loạt:** upload CSV → trả về CSV có 7 cột khía cạnh + biểu đồ tổng hợp.
4. **Giải thích dự đoán:** nút "Vì sao?" cạnh mỗi khía cạnh → highlight các từ ảnh hưởng (SHAP linear cho SVM/LR, kết quả tức thì; LIME với ~300 mẫu nhiễu cho ViSoBERT/PhoBERT, mất ~15–45 giây trên CPU, có thanh tiến trình và cache kết quả).

**Chức năng nâng cao (điểm cộng):**
5. **Phân tích sản phẩm:** dán link Tiki → crawl review realtime → **radar chart điểm từng khía cạnh**, tỉ lệ khen/chê mỗi khía cạnh, top bình luận tiêu cực theo khía cạnh.
6. **So sánh 2 sản phẩm** theo từng khía cạnh.

Triển khai: local, Streamlit Community Cloud hoặc Hugging Face Spaces.

---

## 10. Hướng nâng cao (triển khai nếu còn thời gian)

Sắp theo thứ tự ưu tiên (giá trị cao, công sức hợp lý trước):

| # | Hướng nâng cao | Mô tả | Công sức |
|---|---|---|---|
| 1 | **Dashboard phân tích sản phẩm** | Chức năng 5–6 ở §9: link sản phẩm → radar chart khía cạnh, so sánh sản phẩm | Thấp |
| 2 | **Label Studio ML Backend** | Active learning offline đã nằm trong phần chính (§5.4). Nâng cao: chạy mô hình như một service để Label Studio lấy gợi ý trực tiếp trong giao diện, không cần xuất/nhập file mỗi vòng | Trung bình |
| 3 | **Trích xuất cụm từ khía cạnh & ý kiến (span-level)** | Gán nhãn span trong Label Studio (BIO tagging) → ViSoBERT/PhoBERT token classification → app **highlight** đúng cụm "giao chậm", "vải mát" trong câu | Trung bình–cao |
| 4 | **Cải tiến mô hình** | Focal loss cho khía cạnh hiếm, ensemble ViSoBERT + PhoBERT (+ SVM), continued pre-training ViSoBERT trên review TMĐT chưa gán nhãn, data augmentation (thay từ đồng nghĩa, back-translation) | Trung bình |
| 5 | **So sánh với LLM** | Zero-/few-shot ABSA bằng LLM trên cùng tập test → so sánh chất lượng, chi phí, tốc độ với ViSoBERT/PhoBERT | Thấp |
| 6 | **Kiểm tra khả năng tổng quát** | Đánh giá trên ngành hàng chưa thấy khi train; hoặc map sang bộ công khai UIT-ViSFD | Trung bình |
| 7 | **Aspect Sentiment Triplet Extraction (ASTE)** | Trích xuất bộ ba `(cụm khía cạnh, cụm ý kiến, cảm xúc)`, mức nghiên cứu | Cao |

---

## 11. Cấu trúc thư mục đề xuất

```
.
├── data/
│   ├── raw/              # dữ liệu crawl thô
│   ├── labeled/          # export Label Studio + bảng phẳng
│   └── processed/        # đã tiền xử lý, chia train/val/test
├── labeling/
│   ├── label_config.xml
│   ├── guideline.md
│   ├── aspect_keywords.json   # từ điển từ khoá khía cạnh (pre-annotation + baseline)
│   ├── preannotate.py         # gợi ý nhãn bằng từ khoá (đợt 1)
│   ├── split_data.py          # chọn & đóng băng test/val
│   ├── active_learning.py     # chọn mẫu + gợi ý nhãn bằng mô hình (đợt 2)
│   ├── make_tasks.py          # chia câu thành file import cho từng người (chính + chéo)
│   ├── kappa.py               # gộp export, tính Cohen's/Fleiss' Kappa, xuất câu bị lệch
│   ├── ls_common.py           # hằng số dùng chung (7 khía cạnh, mã nhãn 0–3)
│   └── ls_to_table.py         # JSON Label Studio → bảng phẳng
├── notebooks/
│   ├── 01_crawling.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_ml_models.ipynb
│   ├── 05_transformer_absa.ipynb   # ViSoBERT & PhoBERT (đổi model qua config)
│   ├── 06_evaluation.ipynb         # 3 cấp metrics + learning curve
│   └── 07_explainability.ipynb     # LIME / SHAP, deletion test
├── src/
│   ├── crawler/          # tiki_crawler.py, shopee_crawler.py
│   ├── preprocessing/    # cleaner.py, teencode_dict.json, stopwords.txt
│   ├── models/           # ml_absa.py, transformer_absa.py, predict.py
│   ├── evaluation/       # absa_metrics.py
│   ├── explain/          # lime_shap.py (wrapper predict_proba theo từng khía cạnh), highlight.py
│   └── utils/
├── models/               # model đã train (không commit file lớn)
├── app/
│   └── streamlit_app.py
├── reports/              # hình EDA, bảng kết quả, báo cáo
├── requirements.txt
└── README.md
```

---

## 12. Công cụ & thư viện

- **Crawl & scrape:** `requests` (Tiki API JSON); `selenium` / `playwright` chỉ khi cần bổ sung Shopee/Lazada
- **Gán nhãn:** `label-studio`, `label-studio-ml` (nâng cao)
- **NLP tiếng Việt:** `underthesea`, `pyvi`, `py_vncorenlp`
- **ML:** `scikit-learn`, `xgboost`, `imbalanced-learn`, `iterative-stratification`
- **DL:** `torch`, `transformers`, `datasets`
- **Giải thích mô hình:** `lime`, `shap`
- **EDA/Visualization:** `pandas`, `matplotlib`, `seaborn`, `wordcloud`, `plotly`
- **App:** `streamlit`

---

## 13. Kế hoạch thực hiện (gợi ý ~9 tuần)

| Tuần | Công việc | Sản phẩm |
|---|---|---|
| 1 | Chốt schema khía cạnh; crawl & scrape dữ liệu (crawler đã viết và chạy thử xong) | `tiki_crawler.py`, ~60.000–80.000 review thô |
| 2 | EDA dữ liệu thô, viết guideline, pilot gán nhãn 200 mẫu | Notebook EDA, `guideline.md`, schema đã chốt |
| 3 | **Đợt 1a:** gán **test 1.000** + gán chéo 30%. *Song song:* pipeline tiền xử lý, `split_data.py`, `ls_to_table.py` | Test đóng băng, Kappa của test |
| 4 | **Đợt 1b:** gán **val 800 + train 1.200** + gán chéo 15%. **Điểm kiểm tra:** phân bố 21 cặp, mô hình ban đầu (SVM + ViSoBERT), viết `active_learning.py` | Baseline đầu tiên trên val, chốt số mẫu đợt 2 |
| 5 | **Đợt 2:** 3 vòng active learning × ~1.000 (mỗi vòng ~2 ngày, train lại sau mỗi vòng). *Song song:* EDA sau gán nhãn, keyword baseline, NB/LR/SVM/RF (Cách A & B) | **Bộ dữ liệu cuối ~6.000**, kết quả ML |
| 6 | ViSoBERT & PhoBERT (multi-task, sentence-pair); BiLSTM; **learning curve** | Checkpoint DL, bảng so sánh ViSoBERT và PhoBERT, biểu đồ learning curve |
| 7 | Đánh giá 3 cấp, error analysis; **giải thích mô hình LIME/SHAP**, deletion test | Bảng so sánh, hình giải thích, notebook 07 |
| 8 | Web app cơ bản (gồm chức năng giải thích); **nâng cao** nếu kịp (ưu tiên #1 → #2 → #3 ở §10) | App chạy được, tính năng nâng cao |
| 9 | Báo cáo, slide, video demo | Bản nộp cuối |

**Phân công gợi ý (nhóm 4 người) trong tuần 3–5:** ~3 người gán nhãn, 1 người phụ trách code (tiền xử lý, chia tập, mô hình ban đầu, vòng active learning, mô hình ML). Nên đổi vai giữa các tuần để ai cũng tham gia gán nhãn và code.

> Nếu chậm tiến độ:
> - Tuần 5 trễ → **dừng sau vòng active learning 2 (~5.000 mẫu)** và chuyển sang tuần 6. Learning curve (§8.3) sẽ cho thấy dừng ở đó ảnh hưởng nhiều hay ít.
> - Thiếu thời gian cuối kỳ → bỏ phần nâng cao ở tuần 8, giữ nguyên phần cốt lõi (crawl → gán nhãn → mô hình → đánh giá + giải thích → app cơ bản) vì phần đó đã đáp ứng đủ yêu cầu đồ án.

### 13.1. Ước tính thời gian

*Giả định:* ~6.000 mẫu gán nhãn (train ~4.200, val 800, test 1.000), GPU T4 của Colab/Kaggle, fp16, `max_len=256`. Đây là **ước tính**; số thật có thể lệch khoảng ×2 tuỳ GPU và độ dài câu. Hãy đo thời gian epoch đầu tiên rồi cập nhật lại bảng.

**a. Công sức con người**

| Bước | Ước tính |
|---|---|
| Crawl 60.000–80.000 review (~1 request/giây; crawler đã viết xong) | Máy chạy ~1,5–2 giờ (chạy qua đêm) |
| Pilot 200 mẫu + chốt guideline | ~4–6 giờ công cả nhóm |
| Đợt 1: test 1.000 mẫu (gợi ý từ khoá, ~45–60 giây/mẫu) | ~13–17 giờ công |
| Đợt 1: gán chéo 30% test (300 mẫu) | ~4–5 giờ công |
| Đợt 1: val + train 2.000 mẫu (~45 giây/mẫu) | ~25 giờ công |
| Đợt 1: gán chéo 15% (300 mẫu) | ~4 giờ công |
| Đợt 2: 3 vòng active learning, 3.000 mẫu (sửa gợi ý, ~15–25 giây/mẫu) | ~13–21 giờ công |
| Đợt 2: kiểm tra 10% không gợi ý (300 mẫu) | ~4 giờ công |
| **Tổng gán nhãn** | **~65–80 giờ công** → nhóm 4 người: ~16–20 giờ/người, **~6 giờ/người/tuần** trong tuần 3–5 |

→ Đây là **phần tốn thời gian nhất** của đồ án, không phải huấn luyện. Gán gấp đôi dữ liệu (6.000 so với 3.000) chỉ tốn thêm khoảng 1,5 lần công sức, nhờ đợt 2 chỉ cần sửa gợi ý.

**b. Huấn luyện mô hình**

| Mô hình | 1 lần train | Kèm tinh chỉnh / nhiều seed | Phần cứng |
|---|---|---|---|
| NB / LR / Linear SVM (7 khía cạnh) | Vài giây | 10–20 phút (GridSearch 5-fold) | CPU |
| Random Forest / XGBoost | 2–5 phút | 20–40 phút | CPU |
| BiLSTM + Attention | 5–10 phút | ~1 giờ | GPU (CPU chậm hơn ×3–5) |
| ViSoBERT / PhoBERT multi-task (8 epoch) | ~10–20 phút (~1,5–2,5 phút/epoch) | 2–3 giờ **mỗi mô hình** | GPU T4 |
| ViSoBERT / PhoBERT sentence-pair (dữ liệu ×7 ≈ 29.000) | ~50–60 phút (~10 phút/epoch) | 3–4,5 giờ **mỗi mô hình** | GPU T4 |
| Mô hình ban đầu + 3 vòng active learning | ~15–20 phút mỗi vòng | ~1 giờ tổng | GPU T4 |
| Learning curve (ViSoBERT, 5 mốc × 3 seed) | — | ~1,5–2 giờ | GPU T4 |

**c. Giải thích mô hình**

| Tác vụ | Ước tính |
|---|---|
| Hệ số + SHAP linear cho LR/SVM trên toàn tập test | < 1 phút (CPU) |
| LIME cho ViSoBERT/PhoBERT, 200 mẫu × 1.000 mẫu nhiễu | ~20–35 phút mỗi mô hình (GPU) |
| SHAP Partition cho ViSoBERT/PhoBERT, 200 mẫu | ~20–40 phút mỗi mô hình (GPU) |
| Deletion test | ~5–10 phút |
| 1 lần giải thích trong app (LIME ~300 mẫu nhiễu, CPU) | ~15–45 giây (SVM: tức thì) |

**d. Suy luận (dự đoán)**

| Tác vụ | ML (SVM/LR) | Transformer multi-task | Sentence-pair |
|---|---|---|---|
| 1 bình luận, CPU | < 10 ms | ~50–150 ms | ~0,5–1 giây |
| 1.000 bình luận, CPU | ~1 giây | ~1–2 phút | ~10 phút |
| 1.000 bình luận, GPU | — | ~5–10 giây | ~1 phút |
| Phân tích 1 sản phẩm (~500 review) | crawl 1–2 phút + ~1 giây | crawl 1–2 phút + 30–60 giây | không nên dùng trong app |

→ App dùng **mô hình multi-task** (chạy được trên CPU, deploy miễn phí). Sentence-pair chỉ dùng để so sánh trong báo cáo.

**e. Tổng ngân sách GPU:** active learning ~1 giờ + huấn luyện ~12–17 giờ + learning curve ~1,5–2 giờ + giải thích mô hình ~1,5–2,5 giờ → **~15–22 giờ**, rải trong tuần 4–7. Kaggle cho ~30 giờ GPU/tuần nên đủ; nếu thiếu thì dùng `max_len=128` hoặc chỉ chạy sentence-pair cho ViSoBERT. Colab free hay ngắt phiên nên **lưu checkpoint lên Drive sau mỗi epoch**.

**Mẹo tiết kiệm thời gian:**
- Xem phân bố độ dài ở bước EDA; nếu đa số review < 100 từ thì dùng `max_len=128` (nhanh gấp khoảng 2 lần).
- Thử tham số với multi-task trước vì rẻ hơn; sentence-pair chỉ chạy với bộ tham số đã tốt.
- Chỉ chạy 3 seed cho cấu hình tốt nhất.
- LIME/SHAP chỉ chạy trên mẫu phân tầng (100–200 mẫu), không chạy trên toàn bộ tập test.

---

## 14. Rủi ro & hướng xử lý

| Rủi ro | Hướng xử lý |
|---|---|
| Gán nhãn ABSA lâu, dễ lệch giữa các thành viên | Pilot trước, guideline có nhiều ví dụ, pre-annotation từ khoá, đo Kappa từng khía cạnh |
| Gán nhãn đợt 2 bị trễ | Dừng sau vòng active learning 2 (~5.000 mẫu); dùng learning curve để chứng minh mức dữ liệu đó đã đủ hay chưa |
| Người gán "bấm theo" gợi ý của mô hình (thiên lệch) | Test chỉ dùng gợi ý từ khoá; 10% mẫu mỗi vòng gán lại không có gợi ý; theo dõi tỉ lệ giữ nguyên gợi ý (cao bất thường là dấu hiệu cần nhắc nhở) |
| Active learning làm train lệch phân bố | 10% mẫu chọn ngẫu nhiên mỗi vòng; val/test giữ phân bố tự nhiên; dùng class weights |
| Một số khía cạnh quá hiếm (ví dụ `PACKAGING` Tiêu cực) | Lấy mẫu có chủ đích bằng từ khoá; gộp khía cạnh; sentence-pair; class weights |
| Bình luận ngắn chỉ có `GENERAL` | Lọc ưu tiên bình luận ≥ 10 từ khi chọn mẫu gán nhãn |
| ~50% review không có nội dung, chỉ ~25% có ≥ 10 từ | Crawl 60.000–80.000 review thô; nếu vẫn thiếu thì thêm ngành hàng (laptop, thời trang nam, sách) hoặc hạ ngưỡng xuống ≥ 5 từ |
| Bị chặn khi crawl | Request tuần tự ~1 giây/lần, tự chờ và thử lại khi gặp 429/5xx, chạy tiếp được sau khi dừng; tăng `--delay` nếu vẫn bị chặn |
| Tiki đổi API (API không chính thức) | Tìm lại endpoint bằng F12 → Network (§3.2); crawl sớm (tuần 1) và lưu bản dữ liệu thô lên Drive |
| Thiếu GPU | Colab / Kaggle; fp16; giảm `max_len` |
| Teencode, sai chính tả | ViSoBERT xử lý trực tiếp; với ML/PhoBERT thì mở rộng từ điển chuẩn hoá |
| LIME/SHAP cho transformer chậm trong app | Giảm số mẫu nhiễu (~300), cache kết quả, hiển thị thanh tiến trình; mặc định giải thích bằng SVM (tức thì) |
| Giải thích LIME/SHAP không ổn định (chạy lại cho kết quả khác) | Cố định seed, tăng số mẫu nhiễu khi phân tích cho báo cáo, đối chiếu LIME với SHAP, chạy deletion test |
| PhoBERT cần VnCoreNLP (Java) khi triển khai app | Nếu ViSoBERT đạt kết quả tương đương hoặc tốt hơn → dùng ViSoBERT trong app |

---

## 15. Tài liệu tham khảo

- Pontiki, M. et al. (2016). *SemEval-2016 Task 5: Aspect Based Sentiment Analysis.*
- Sun, C., Huang, L., & Qiu, X. (2019). *Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence.* NAACL.
- Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP.
- Luc Phan, L. et al. (2021). *SA2SL: From Aspect-Based Sentiment Analysis to Social Listening System for Business Intelligence* (bộ dữ liệu **UIT-ViSFD**).
- Nguyen, H. T. M. et al. (2018). *VLSP Shared Task: Sentiment Analysis* (VLSP 2018 ABSA — nhà hàng, khách sạn).
- Nguyen, Q. N. et al. (2023). *ViSoBERT: A Pre-Trained Language Model for Vietnamese Social Media Text Processing.* EMNLP.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME). KDD.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Label Studio Documentation — https://labelstud.io/guide/
