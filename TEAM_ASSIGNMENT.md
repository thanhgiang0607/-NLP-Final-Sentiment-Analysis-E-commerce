# Phân công nhóm (4 thành viên)

> Đi kèm với [`PROJECT_PLAN.md`](PROJECT_PLAN.md). Số mục (§) trong file này trỏ tới các mục của kế hoạch.
> Đề tài: **trích xuất ý kiến** `(cụm khía cạnh, cụm ý kiến, cảm xúc)` từ bình luận Tiki.
> Số câu gán nhãn và giờ công là **ước tính**, có thể điều chỉnh sau buổi họp mỗi tuần.

## 1. Vai trò chính

Ai cũng **gán nhãn**. Ngoài ra mỗi người **phụ trách chính** một mảng (chịu trách nhiệm cuối cùng, nhưng có thể nhờ người khác hỗ trợ).

| Mã | Tên thành viên | Vai trò phụ trách chính | Code / file phụ trách |
|---|---|---|---|
| **A** | Giang | **Dữ liệu**: crawl, EDA, tiền xử lý lớp 1; sau đó **gom nhóm khía cạnh** và **dashboard** | `src/crawler/`, `src/preprocessing/cleaner.py`, `src/clustering/`, `notebooks/01–03, 08` |
| **B** | Phúc | **Gán nhãn & chất lượng**: Label Studio, guideline, đo độ đồng thuận, xuất bộ dữ liệu; sau đó **web app** | `labeling/label_config.xml`, `guideline.md`, `agreement.py`, `ls_to_dataset.py`, `app/` |
| **C** | Dũng | **Luật & ML truyền thống & active learning**, metrics đánh giá | `labeling/lexicon.json`, `preannotate.py`, `make_tasks.py`, `split_data.py`, `active_learning.py`, `src/models/rule_baseline.py`, `crf_tagger.py`, `ml_pairing.py`, `src/evaluation/`, `notebooks/04, 06` |
| **D** | Hằng | **Deep learning & giải thích mô hình**: BiLSTM-CRF, ViSoBERT, PhoBERT, giải thích | `src/preprocessing/tokenizer.py`, `src/models/bilstm_crf.py`, `transformer_tagger.py`, `transformer_pairing.py`, `predict.py`, `src/explain/`, `notebooks/05, 07` |

**Quy tắc chung:**
- Họp nhóm **2 lần/tuần** (15–30 phút): đầu tuần chia việc, cuối tuần báo cáo kết quả và vướng mắc.
- Code đẩy lên git theo **branch riêng → Pull Request → Giang duyệt → merge** vào `main`; cách đặt tên branch, viết commit, mở và duyệt PR xem [`CONTRIBUTING.md`](CONTRIBUTING.md). Dữ liệu thô (`data/raw/`) chia sẻ qua **Google Drive**, không đưa lên git.
- Khi gán chéo, **không ai gán lại câu do chính mình gán**. Vòng gán chéo: A → B → C → D → A (A gán lại một phần câu của B, B gán lại một phần câu của C...).
- Mỗi đợt gán nhãn đều theo **cùng một quy trình chuyển file** (C chia file → mọi người gán → B đo độ đồng thuận), xem **§11**.
- **Văn bản bình luận không được sửa sau khi đã gán nhãn** (nhãn dựa vào vị trí ký tự, §6 của kế hoạch).

---

## 2. Giai đoạn chuẩn bị (tuần 1–2)

### Tuần 1: Crawl dữ liệu & dựng công cụ

Bốn việc **chạy song song**, không ai phải chờ ai. Ai cần câu mẫu để thử thì lấy từ `tiki_reviews.csv` (đã có trên Drive).

| Người | Việc làm | Kết quả cuối tuần | Trạng thái |
|---|---|---|---|
| **A** Giang | Crawl review Tiki, xuất CSV, đưa lên Drive | `tiki_reviews.csv`, `tiki_products.jsonl` trên Drive | ✅ Xong (crawl 23/11, đã upload Drive) |
| **B** Phúc | Cài Label Studio, tạo project bằng `label_config.xml`; gán thử 5 câu; export JSON kiểm tra quan hệ; viết `guideline.md` v0 | Label Studio chạy ổn, file export có quan hệ, `guideline.md` v0 | ✅ Xong (PR #3) |
| **C** Dũng | Viết `lexicon.json` + `preannotate.py`; import thử 20 câu có gợi ý | 20 câu hiện sẵn cụm từ + mũi tên trong Label Studio | ✅ Xong (PR #1, #5: 4,2 cụm/câu, 10% câu không có gợi ý) |
| **D** Hằng | Môi trường GPU Kaggle/Colab; chạy thử ViSoBERT + PhoBERT; `teencode_dict.json` v0 | `notebooks/00_test_dl_env.ipynb` chạy thông, `teencode_dict.json` v0 | ✅ Xong (PR #4, #6) |

#### A – Giang: Crawl dữ liệu

**Đã làm** (23/11): crawl **24 ngành hàng** (không phải 6 như dự tính ban đầu), 60 sản phẩm/ngành, tối đa 300 review/sản phẩm. Chạy liên tục ~3,5 giờ, không lỗi.

| Chỉ số | Số liệu |
|---|---|
| Sản phẩm | 1.380 |
| Tổng review (đã bỏ trùng `review_id`) | 110.875 |
| Có nội dung | 51.569 |
| Dài 10–80 từ (đủ chuẩn để gán) | 25.605, trong đó **4.917 câu 1–3 sao** |
| Trùng nội dung (kiểu "Sản phẩm tốt") | ~9.500, bỏ ở `cleaner.py` tuần 2 |
| Rating (câu có nội dung) | 1★ 6,7% · 2★ 2,6% · 3★ 4,8% · 4★ 11,6% · 5★ 74,3% |

Ngành rất ít câu đủ chuẩn: Đồ ăn (55), Điện thoại (74), Giày dép nữ (229). Khi chọn câu gán nhãn không chia đều theo ngành mà chia theo tỉ lệ, mỗi ngành tối thiểu ~30–50 câu (ngành thiếu thì lấy hết).

**Còn lại:**
- [x] Upload `data/raw/tiki_reviews.csv` và `data/raw/tiki_products.jsonl` lên thư mục `raw/` trên Drive (§11), báo nhóm.

Lệnh đã dùng (macOS; Windows thay `python3` bằng `python`, bỏ `caffeinate -i`, thay `tee -a` bằng `Tee-Object -Append`):

```bash
caffeinate -i python3 -u src/crawler/tiki_crawler.py --categories all --max-products 60 --max-reviews 300 2>&1 | tee -a data/raw/crawl_log.txt
python3 src/crawler/tiki_crawler.py --export-csv
```

#### B – Phúc: Label Studio + guideline v0

- [x] **Cài và mở Label Studio**
  ```bash
  pip install label-studio
  label-studio start          # mở http://localhost:8080, tạo tài khoản
  ```
- [x] **Tạo file 5 câu thử** (ở thư mục gốc dự án, cần `tiki_reviews.csv` trong `data/raw/`):
  ```bash
  python3 -c "
  import pandas as pd
  df = pd.read_csv('data/raw/tiki_reviews.csv')
  df = df[df.content.fillna('').str.split().str.len().between(15, 60)]
  df.sample(5, random_state=1)[['review_id','content','rating','product_name','category']].to_json(
      'data/processed/ls_test_5.json', orient='records', force_ascii=False, indent=2)
  "
  ```
- [x] **Tạo project**: Create Project → *Labeling Setup* → *Code* → dán **toàn bộ** `labeling/label_config.xml` → *Data Import* chọn `ls_test_5.json` → Save.
  Không đổi các tên `span`, `skip`, `text` trong config (script đọc export dựa vào các tên này).
- [x] **Gán thử 5 câu**, cố tình có đủ các trường hợp: ít nhất 1 mũi tên quan hệ, 1 ý kiến **không nối** (khía cạnh ẩn), 1 câu tích **Spam**.
  Thao tác: phím `a`/`p`/`u`/`n` → bôi cụm từ → bấm cụm khía cạnh → *Create Relation* (Alt+R) → bấm cụm ý kiến → Submit.
- [x] **Export → JSON** (không chọn JSON-MIN). Mở file, kiểm tra:
  - có các mục `"type": "labels"` với `"from_name": "span"` (cụm từ),
  - có các mục `"type": "relation"` (mũi tên) → **đạt**.
  - Lưu file export vào Drive `exports/w1_test/` để Dũng và Hằng dùng thử.
- [x] **Viết `labeling/guideline.md` v0** dựa trên §2 của kế hoạch, gồm: định nghĩa khía cạnh / ý kiến / cảm xúc; ranh giới cụm từ (từ phủ định và từ mức độ nằm **trong** cụm ý kiến); bảng trường hợp đặc biệt (khía cạnh ẩn, 1 ý kiến nhiều khía cạnh, quan hệ xa, mỉa mai, câu hỏi, mô tả khách quan, spam); mỗi trường hợp 1–2 ví dụ lấy từ dữ liệu thật. Bản v0 sẽ được sửa sau pilot tuần 2.

**Xong khi:** mỗi người tự tạo được project trên máy mình bằng `label_config.xml` (§11), file export có `relation`, `guideline.md` v0 đã đẩy lên git.

#### C – Dũng: Lexicon + pre-annotate

- [x] **`labeling/lexicon.json`** gồm 6 nhóm, cộng `not_aspect` (cụm chứa từ khía cạnh nhưng không phải khía cạnh, `preannotate.py` phải bỏ qua):
  ```json
  {
    "aspect":   ["chất vải", "pin", "màn hình", "giao hàng", "shipper", "đóng gói", "giá", "shop", "..."],
    "not_aspect": ["mua hàng", "nhận hàng", "đơn hàng", "mở rộng", "..."],
    "op_pos":   ["đẹp", "tốt", "nhanh", "chắc chắn", "đáng tiền", "..."],
    "op_neg":   ["xấu", "mỏng", "chậm", "lỗi", "móp méo", "..."],
    "op_neu":   ["tạm", "tạm được", "bình thường", "..."],
    "negation": ["không", "ko", "chẳng", "chưa", "chả", "..."],
    "intensifier": {"before": ["rất", "khá", "hơi", "siêu", "..."], "after": ["quá", "lắm", "cực", "..."]}
  }
  ```
  Điểm xuất phát: danh sách từ trong `app/demo_backend.py` (đã có ~70 khía cạnh, ~150 từ ý kiến). Bổ sung từ khoá cho các ngành mới: sách (nội dung, bìa, giấy), đồ ăn (vị, hạn sử dụng), xe cộ, giày dép, túi ví. Viết chữ thường, có dấu.
- [x] **`labeling/preannotate.py`**: đọc CSV (`review_id`, `content`, ...) → tìm cụm từ theo lexicon (ưu tiên cụm dài nhất **trên tất cả các nhóm cùng lúc**, ví dụ `không hài lòng` thắng `không` + `hài lòng`, `chất lượng` thắng `chất`; bỏ qua cụm trong `not_aspect`; ghép từ phủ định / mức độ vào cụm ý kiến; phủ định thì đảo cảm xúc) → nối mỗi ý kiến với khía cạnh gần nhất trong cùng mệnh đề (tách mệnh đề theo `. ! ? ;` và "nhưng") → xuất JSON task Label Studio có `predictions`. Dùng `spans_to_result()` trong `labeling/ls_common.py` để tạo đúng định dạng (cụm từ + relation).
  ```bash
  python3 labeling/preannotate.py --input data/raw/tiki_reviews.csv --n 20 --out data/processed/preannotated_20.json
  ```
- [x] **Import thử** `preannotated_20.json` vào project tự tạo trên máy mình (Label Studio của Dũng, dán đúng `labeling/label_config.xml` như Phúc) → mở từng câu thấy cụm từ đã bôi màu và mũi tên sẵn → sửa thử 1 câu, Submit, export JSON, chạy `parse_result()` trong `ls_common.py` đọc lại được.

**Xong khi:** 20 câu hiện gợi ý đúng trong Label Studio; số liệu sơ bộ: trung bình mấy cụm/câu, bao nhiêu % câu không có gợi ý nào.

#### D – Hằng: Môi trường deep learning

- [x] **Tạo notebook** `notebooks/00_test_dl_env.ipynb` trên Kaggle (GPU T4/P100) hoặc Colab:
  ```python
  !pip -q install transformers torch
  from transformers import AutoTokenizer, AutoModelForTokenClassification
  for name in ["uitnlp/visobert", "vinai/phobert-base-v2"]:
      tok = AutoTokenizer.from_pretrained(name)
      model = AutoModelForTokenClassification.from_pretrained(name, num_labels=9)  # 9 nhãn BIO (§7.1)
      enc = tok("Áo đẹp nhưng chất vải thì mỏng quá", return_tensors="pt")
      print(name, tok.convert_ids_to_tokens(enc["input_ids"][0]))
      print(model(**enc).logits.shape)   # mong đợi: [1, số_token, 9]
  ```
- [x] **Kiểm tra và ghi lại** trong notebook:
  - Tokenizer của ViSoBERT nhận câu có dấu / teencode như thế nào (tách subword ra sao).
  - **PhoBERT cần câu đã tách từ** (`chất_vải`): thử thêm `pyvi` hoặc `underthesea` để tách từ trước khi đưa vào, ghi nhận khác biệt.
  - Kích thước đầu ra `[1, số_token, 9]` cho cả 2 mô hình, không lỗi thiếu thư viện / tràn bộ nhớ.
  - Thời gian chạy 1 batch 32 câu trên GPU (để ước tính thời gian train tuần 6).
- [x] **`src/preprocessing/teencode_dict.json` v0**: ~100–200 cặp hay gặp trong dữ liệu (`"ko": "không"`, `"sp": "sản phẩm"`, `"dc": "được"`, `"ship": "giao hàng"`...). Tìm bằng cách đếm các từ ngắn hay xuất hiện trong `tiki_reviews.csv`. Dùng **cho mô hình**, không sửa văn bản gốc đem đi gán nhãn (§6).

**Xong khi:** notebook chạy thông từ đầu đến cuối cho cả 2 mô hình, đẩy lên git; `teencode_dict.json` v0 đã đẩy lên git.

### Tuần 2: Làm sạch dữ liệu, gán thử 100 câu (pilot), chốt luật gán nhãn

Mỗi người **làm tiếp đúng mảng của mình ở tuần 1**:

| Người | Tuần 1 đã làm | Tuần 2 làm tiếp |
|---|---|---|
| **A** Giang | Crawl → `tiki_reviews.csv` | Làm sạch dữ liệu đó (`cleaner.py`) + EDA |
| **B** Phúc | Label Studio + `guideline.md` v0 (6 câu hỏi mở) | Tổ chức pilot, đo đồng thuận, chốt 6 câu hỏi → `guideline.md` v1 |
| **C** Dũng | `lexicon.json` + `preannotate.py` | Lexicon v2, chọn câu pilot / test / val (`split_data.py`), tạo file import |
| **D** Hằng | Notebook ViSoBERT / PhoBERT + `teencode_dict.json` v0 | `tokenizer.py` (cụm từ ↔ BIO ↔ sub-token) + `teencode_dict.json` v1 |
| **Cả nhóm** | | Gán **cùng 100 câu pilot** (~2,8 giờ/người) + họp thảo luận ~1,5 giờ |

**Thứ tự trong tuần** (việc của Giang phải xong trước để có văn bản cố định đem đi gán):

| Ngày | Giang | Phúc | Dũng | Hằng |
|---|---|---|---|---|
| **1–2** | `cleaner.py` → `clean.csv` lên Drive | Nhắc cả nhóm cài Label Studio, tạo project thử | Lexicon v2 | `tokenizer.py` phần cụm từ ↔ BIO |
| **3** | EDA | | `split_data.py` → file pilot lên Drive **tối ngày 3** | `tokenizer.py` phần sub-token |
| **4–5** | **Gán pilot** · EDA | **Gán pilot** | **Gán pilot** | **Gán pilot** · teencode v1 |
| **5 tối** | Export lên Drive | Export lên Drive | Export lên Drive | Export lên Drive |
| **6** | | Chạy `agreement.py`, gửi câu lệch · **chủ trì họp** | | Test `tokenizer.py` trên export pilot |
| **7** | PR EDA | PR `guideline.md` v1 | Cập nhật lexicon theo guideline v1 → file import tuần 3 lên Drive | PR `tokenizer.py` |

| Người | Kết quả cuối tuần | Trạng thái |
|---|---|---|
| **A** Giang | `cleaner.py`, `clean.csv` trên Drive, `02_eda.ipynb` | ⬜ Chưa làm |
| **B** Phúc | Kết quả đồng thuận pilot, `guideline.md` v1 | ⬜ Chưa làm |
| **C** Dũng | Lexicon v2, ID pilot / test / val đóng băng trên git, file import tuần 3 trên Drive | ⬜ Chưa làm |
| **D** Hằng | `tokenizer.py` có test, `teencode_dict.json` v1 | ⬜ Chưa làm |

#### A – Giang: Làm sạch dữ liệu (lớp 1) + EDA

- [ ] **`src/preprocessing/cleaner.py`** (§6 lớp 1). Văn bản ra ở bước này là văn bản **cố định**: đem đi gán nhãn rồi thì không sửa nữa.
  ```bash
  python3 src/preprocessing/cleaner.py --input data/raw/tiki_reviews.csv --out data/processed/clean.csv
  ```
  - Hàm `clean_text(text) -> str` dùng lại được (app của Phúc dùng để làm sạch câu người dùng nhập).
  - Bỏ HTML, URL; `\r\n` → `\n`; gộp khoảng trắng thừa.
  - Unicode **NFC**; quy **một kiểu bỏ dấu**: `hoà` → `hòa`, `khoá` → `khóa`, `khoẻ` → `khỏe`, `thuỷ` → `thủy` (giống hàm `canon()` trong `preannotate.py` và cách viết trong `lexicon.json`).
  - Ẩn thông tin cá nhân: số điện thoại, email → `<PHONE>`; tên người → `<NAME>` (đúng ký hiệu trong `guideline.md` §7). Tên khó bắt hết: dùng mẫu kiểu `mình là …`, `tên …`, chữ ký viết hoa cuối câu (vd *"NGỌC NAM ❤️"* ở review 20172079), rồi tự xem ngẫu nhiên 200 câu đầu ra.
  - Bỏ trùng nội dung (~9.500 câu kiểu "Sản phẩm tốt"), giữ 1 câu.
  - Đánh dấu spam quảng cáo thuần (vd *"AE qua nhóm mình săn sale…"*, 20223857). Câu vừa quảng cáo vừa có ý kiến thật thì **giữ lại** (đề xuất câu hỏi mở số 4).
  - Lọc 10–80 từ.
  - Cột đầu ra: `review_id, product_id, product_name, category, rating, content, n_words`. In thống kê: bao nhiêu câu bị bỏ ở mỗi bước.
- [ ] **Đưa `clean.csv` lên Drive `raw/`** và báo nhóm. **Hạn: hết ngày 2**, vì Dũng cần file này để chọn câu pilot.
- [ ] **`notebooks/02_eda.ipynb`** (§4, trên dữ liệu thô và dữ liệu sạch):
  - phân bố rating, độ dài, % review trống, theo ngành hàng, theo thời gian;
  - trùng lặp, spam, tần suất emoji;
  - đếm **từ ngắn không dấu hay gặp** → gửi Hằng làm teencode v1;
  - **top danh từ / tính từ** theo ngành (POS bằng `underthesea`), đánh dấu từ nào **chưa có trong `lexicon.json`** → gửi Dũng.
- [ ] **Gán pilot 100 câu** (ngày 4–5), export `w2_A_pilot.json` lên Drive `exports/w2/`.

**Xong khi:** `clean.csv` trên Drive trước ngày 3; `cleaner.py` + `02_eda.ipynb` đã merge; Dũng và Hằng đã nhận danh sách từ.

#### B – Phúc: Pilot + guideline v1

- [ ] **Ngày 1–2:** nhắc cả nhóm cài Label Studio và tạo project thử bằng `labeling/label_config.xml` (xem §11). Ai lỗi thì hỗ trợ trước ngày 4.
- [ ] **Ngày 4–5:** cả nhóm gán `w2_<người>_pilot.json`.
  - Pilot **không có gợi ý**, để đo đúng mức hiểu guideline của từng người.
  - Gán theo `guideline.md` v0. Câu nào phân vân thì ghi lại `review_id` + lý do.
  - **Tối ngày 5:** mọi người export **JSON**, đổi tên đúng quy ước, đưa lên `Drive/exports/w2/`.
- [ ] **Ngày 6:** đo độ đồng thuận:
  ```bash
  python3 labeling/agreement.py --exports labeling/exports/w2
  ```
  Gửi nhóm `w2_agreement.csv` (`triplet_f1`, `token_fleiss_kappa`) và `w2_disagreements.csv` **trước buổi họp**.
- [ ] **Chủ trì buổi họp ~1,5 giờ:**
  1. Đọc kết quả theo thứ tự `aspect_f1` → `opinion_f1` → `pair_f1` → `triplet_f1` để biết nhóm lệch ở bước nào.
  2. Chốt **6 câu hỏi mở** ở `guideline.md` §8.
  3. Xem ~15 câu lệch nhiều nhất và thống nhất đáp án.
- [ ] **`labeling/guideline.md` v1** (PR `docs/phuc-guideline-v1`):
  - đổi §8 từ "câu hỏi mở" thành **luật đã chốt**;
  - sửa §7 và đáp án §9 cho khớp (các dòng "chờ câu hỏi mở");
  - thêm ví dụ thật lấy từ pilot;
  - ghi số đồng thuận pilot.

**Xong khi:** có số `triplet_f1` của pilot; `guideline.md` v1 đã merge. Mục tiêu ≥ 0,7 là cho tập test; pilot thấp hơn cũng được, miễn là đã chốt luật cho các chỗ lệch.

#### C – Dũng: Lexicon v2 + chọn câu + file import

- [ ] **`labeling/lexicon.json` v2** (ngày 1–2):
  - `lâu` đứng riêng đang bị bôi là chê ("Cầm lâu", "Đeo lâu") → đổi thành cụm `giao lâu`, `hơi lâu`, `lâu quá`...
  - thêm `không có` vào `negation` ("không có mùi hôi" đang ra chê);
  - bổ sung từ trong danh sách top danh từ / tính từ của Giang;
  - mỗi lần sửa chạy lại `preannotate.py --n 200` và ghi số "% câu không có gợi ý".
- [ ] **`labeling/split_data.py`** (ngày 3, cần `clean.csv` của Giang):
  ```bash
  python3 labeling/split_data.py --input data/processed/clean.csv --pilot 100 --test 600 --val 400 --out data/splits
  ```
  - Ra `data/splits/pilot_ids.txt`, `test_ids.txt`, `val_ids.txt` + bảng thống kê theo ngành × rating.
  - **3 tập không trùng nhau**, cùng `--seed` thì ra giống hệt.
  - Phân tầng theo ngành hàng + rating (§5.2): chia theo tỉ lệ, mỗi ngành tối thiểu ~30–50 câu (ngành thiếu thì lấy hết); **lấy thêm câu 1–3 sao**, ghi rõ tỉ lệ.
  - Câu cùng 1 sản phẩm không nên dồn hết vào 1 tập.
  - **Commit 3 file ID lên git** để đóng băng: file nhỏ, không chứa nội dung review, `data/splits/` không bị `.gitignore` chặn.
- [ ] **File pilot** (tối ngày 3), không có gợi ý:
  ```bash
  # lấy 100 dòng theo pilot_ids.txt từ clean.csv → data/processed/pilot.csv, rồi:
  python3 labeling/make_tasks.py --input data/processed/pilot.csv --phase w2 --pilot
  ```
  Đưa 4 file `w2_*_pilot.json` lên `Drive/tasks/w2/`.
- [ ] **Thêm `--ids` cho `preannotate.py`** (bôi gợi ý cho đúng danh sách ID thay vì lấy ngẫu nhiên `--n`).
- [ ] **Ngày 7:** cập nhật lexicon theo luật mới của guideline v1, rồi tạo file import tuần 3:
  ```bash
  python3 labeling/preannotate.py --input data/processed/clean.csv --ids data/splits/test_ids.txt --out data/processed/test_preannotated.json
  python3 labeling/make_tasks.py --input data/processed/test_preannotated.json --phase w3 --quota A=120 B=180 C=150 D=150 --cross-n 45
  ```
  Đưa file và `w3_manifest.csv` lên `Drive/tasks/w3/` **trước ngày đầu tuần 3**.
- [ ] **Gán pilot 100 câu.**

**Xong khi:** 3 file ID đã trên git; file pilot lên Drive đúng hạn ngày 3; file tuần 3 lên Drive trước tuần 3.

#### D – Hằng: `tokenizer.py` + teencode v1

- [ ] **`src/preprocessing/tokenizer.py`** (§6 lớp 2):
  - **Tách từ:** dùng lại `words()` trong `labeling/ls_common.py` (cùng cách tách với `agreement.py`, để BIO 2 bên giống nhau).
  - **Cụm từ ↔ BIO:** `spans_to_bio(text, spans)` và `bio_to_spans(text, tags)`.
  - **ViSoBERT:** `align_visobert(text, spans, tok)` dùng `return_offsets_mapping=True`; nhãn gán cho sub-token đầu của mỗi từ, các sub-token còn lại và `<s>`, `</s>` = `-100`.
  - **PhoBERT:** `align_phobert(text, spans, tok)` tách từ bằng pyvi (theo notebook 00). Pyvi thêm khoảng trắng trước dấu câu và nối bằng `_`, nên phải **căn lại về vị trí ký tự gốc**.
  - **Chuẩn hoá từng từ** cho CRF / BiLSTM: `normalize_word(w, prev)`: chữ thường, teencode, `đẹppppp` → `đẹp`, emoji → `emo_pos` / `emo_neg`. **Không tra teencode khi từ đứng sau một số** (`5 kg`, `500 k`).
- [ ] **Test:** cụm từ → BIO → cụm từ phải ra **đúng như cũ**.
  - Ngày 3: thử trên `data/processed/preannotated_20.json` (gợi ý của Dũng), chưa cần chờ pilot.
  - Ngày 6: chạy lại trên 4 file export pilot.
  - In số câu khớp 100%, liệt kê câu lệch.
- [ ] **`src/preprocessing/teencode_dict.json` v1** (PR `data/hang-teencode-v1`), dùng danh sách từ ngắn của Giang hoặc ô phụ lục trong notebook 00:
  - thêm `vs` → với, `tui` → tôi, `rat` → rất...;
  - bỏ 12 khoá không xuất hiện lần nào (`k0`, `dzậy`, `jz`, `ghang`, `vchuyen`, `authentic`, `dthoai`, `cluong`, `oge`, `thôy`, `lmj`, `camon`);
  - cân nhắc bỏ `hông` ("bên hông").
- [ ] **Chạy lại notebook 00** trên Kaggle (*Run All*) để output khớp code đã sửa, rồi commit.
- [ ] **Gán pilot 100 câu.**

**Xong khi:** `tokenizer.py` đã merge, test khớp 100% trên export pilot; `teencode_dict.json` v1 đã merge.

---

## 3. Đợt 1: Người tự gán ~1.600 câu (tuần 3–4)

### Tuần 3: Gán tập test (600 câu)

Đây là "đề thi cuối kỳ", nên gán **cẩn thận nhất**. Chỉ có gợi ý từ điển, **không** dùng gợi ý từ mô hình.

| Người | Gán chính | Gán chéo (30%) | Tổng câu | Giờ công (~100 giây/câu) | Việc khác |
|---|---|---|---|---|---|
| **A** | 120 | 45 (câu của B) | 165 | ~4,6 giờ | Hoàn thiện EDA thô; hỗ trợ D kiểm tra `tokenizer.py` |
| **B** | 180 | 45 (câu của C) | 225 | ~6,3 giờ | Theo dõi tiến độ; trả lời thắc mắc về luật; viết `ls_to_dataset.py` (export → `aste_*.jsonl` + `bio_*.txt`) |
| **C** | 150 | 45 (câu của D) | 195 | ~5,4 giờ | Viết `rule_baseline.py` (§7.2) và `aste_metrics.py` (3 cấp đánh giá, §8.1) |
| **D** | 150 | 45 (câu của A) | 195 | ~5,4 giờ | Viết sẵn code `transformer_tagger.py`, chạy thử trên dữ liệu pilot |
| **Tổng** | **600** | **180** | **780** | | |

**Cuối tuần 3:** B đo độ đồng thuận cho tập test (`agreement.py`, §11) → cả nhóm thống nhất các câu bị lệch → người gán chính sửa lại trong Label Studio → **tập test đóng băng**, không ai sửa nữa.

### Tuần 4: Gán val + train (1.000 câu) và điểm kiểm tra

| Người | Gán chính | Gán chéo (15%) | Tổng câu | Giờ công (~95 giây/câu) | Việc khác |
|---|---|---|---|---|---|
| **A** | 280 | 38 | 318 | ~8,4 giờ | Thống kê tập test: số bộ ba theo cảm xúc, theo khoảng cách, tỉ lệ khía cạnh ẩn (chuẩn bị buổi kiểm tra) |
| **B** | 300 | 38 | 338 | ~8,9 giờ | Đo độ đồng thuận val/train |
| **C** | 140 | 38 | 178 | ~4,7 giờ | Train **CRF + SVM ban đầu**; viết `active_learning.py` (§5.4) |
| **D** | 280 | 38 | 318 | ~8,4 giờ | Train **ViSoBERT tagger ban đầu** (~10 phút GPU) |
| **Tổng** | **1.000** | **152** | **1.152** | | |

> C gán ít hơn vì phải hoàn thành `active_learning.py`. Không có script này thì đợt 2 không bắt đầu được.

### Buổi kiểm tra giữa 2 đợt (cuối tuần 4, cả nhóm ~1,5 giờ)

| Người trình bày | Nội dung | Quyết định cần đưa ra |
|---|---|---|
| **B** | Độ đồng thuận theo từng thành phần (khía cạnh, ý kiến, cặp, bộ ba); các lỗi hay gặp | Thành phần nào < 0,6 thì sửa guideline thế nào |
| **A** | Số bộ ba theo cảm xúc và khoảng cách trong tập test | Ý kiến Trung lập quá ít thì gộp hay giữ |
| **C** | Triplet F1 của luật và CRF + SVM trên val; chạy thử `active_learning.py` | Có bắt đầu đợt 2 được chưa |
| **D** | Kết quả ViSoBERT tagger trên val | Dùng CRF + SVM hay ViSoBERT để chọn câu và bôi nháp cho đợt 2 |
| **Cả nhóm** | | Chốt số câu đợt 2 (mặc định 1.400) |

---

## 4. Đợt 2: Máy chọn câu và bôi nháp, người kiểm tra (tuần 5)

2 vòng, mỗi vòng khoảng 2,5 ngày. Mỗi vòng ~700 câu do máy chọn, đã có cụm từ và quan hệ bôi sẵn. **Người kiểm tra và sửa tất cả các câu.**

### Một vòng diễn ra thế nào

```
Ngày 1 sáng:  C chạy active_learning.py → chọn ~700 câu → make_tasks.py → đưa lên Drive
Ngày 1–2,5:   A, B, D sửa nháp (bôi lại ranh giới, đổi cảm xúc, nối / bỏ quan hệ);
              C gán 70 câu KHÔNG có nháp để kiểm tra
Ngày 2,5 tối: B kiểm tra chất lượng → C gộp dữ liệu, train lại mô hình → sẵn sàng vòng sau
```

### Số câu mỗi người (tính cho cả 2 vòng)

| Người | Mỗi vòng | 2 vòng | Giờ công | Việc khác trong tuần 5 |
|---|---|---|---|---|
| **A** | 230 câu (sửa nháp, ~45 giây/câu) | 460 | ~5,8 giờ | EDA sau gán nhãn (§4) |
| **B** | 240 câu (sửa nháp) | 480 | ~6,0 giờ | **Kiểm tra chất lượng mỗi vòng**: tỉ lệ bộ ba gợi ý được giữ nguyên (cao bất thường = đang "bấm theo máy"), so với 70 câu kiểm tra của C |
| **C** | 70 câu **không có nháp** (~100 giây/câu) | 140 | ~3,9 giờ | **Chạy vòng active learning**; hoàn thiện CRF + LR / SVM / RF (§7.2); ghi Triplet F1 val sau mỗi vòng |
| **D** | 230 câu (sửa nháp) | 460 | ~5,8 giờ | Viết sẵn code `transformer_pairing.py`, `bilstm_crf.py`, `predict.py` cho tuần 6 |
| **Tổng** | | **~1.540** | | |

> **Nếu trễ:** dừng sau vòng 1 (~2.300 câu) và chuyển sang tuần 6 (§13 kế hoạch).

**Cuối tuần 5:** B chạy `ls_to_dataset.py` xuất **bộ dữ liệu cuối ~3.000 câu** vào `data/labeled/` và commit lên git.

---

## 5. Giai đoạn mô hình (tuần 6)

| Người | Việc làm | Kết quả |
|---|---|---|
| **A** | Hoàn thiện EDA sau gán nhãn; chia tập test theo mức "phi chuẩn" và theo khoảng cách khía cạnh – ý kiến (§7.3, §8.1); chạy thử gom nhóm khía cạnh trên kết quả CRF | `02_eda.ipynb` phần trích xuất, bản nháp phân cụm |
| **B** | Dựng **khung web app** (Streamlit): nhập 1 câu → gọi CRF + SVM → **bôi màu** cụm từ + bảng bộ ba; viết báo cáo chương Dữ liệu + Gán nhãn | App chạy được với CRF + SVM, bản nháp 2 chương |
| **C** | Tinh chỉnh luật, CRF (`c1`, `c2`), LR / SVM / RF nối quan hệ; **learning curve phần ML** (§8.3) | `04_ml_models.ipynb`, bảng kết quả ML |
| **D** | **BiLSTM-CRF**; **ViSoBERT & PhoBERT** tagger + pair; learning curve phần ViSoBERT | Checkpoint DL (lưu lên Drive), bảng so sánh ViSoBERT và PhoBERT |

> D tốn nhiều GPU nhất (~10–15 giờ). Nếu quota Kaggle của D không đủ, các thành viên khác cho dùng tài khoản Kaggle của mình để chạy thêm.

---

## 6. Đánh giá, giải thích & gom nhóm (tuần 7)

| Người | Việc làm | Kết quả |
|---|---|---|
| **A** | **Gom nhóm khía cạnh & đặt tên chủ đề** (§7.5) trên kho chưa gán nhãn; kiểm tra 300 cụm phổ biến. **Error analysis**: phân loại lỗi (lệch ranh giới, nối nhầm, phủ định, mỉa mai, khía cạnh ẩn) | `08_aspect_clustering.ipynb`, `aspect_topics.json`, bảng lỗi kèm ví dụ |
| **B** | Gắn ViSoBERT vào app; thêm nút **"Vì sao?"** (dùng code giải thích của D); phân tích CSV hàng loạt | App đủ 4 chức năng cơ bản (§9) |
| **C** | Đánh giá **3 cấp** cho tất cả mô hình trên tập test; **Triplet F1 theo khoảng cách**; vẽ learning curve hoàn chỉnh | `06_evaluation.ipynb`, bảng kết quả cuối |
| **D** | **Giải thích mô hình** (§8.2): trọng số CRF, hệ số + SHAP cho LR/SVM, LIME/SHAP cho ViSoBERT pair, deletion test; đóng gói hàm giải thích để B dùng trong app | `07_explainability.ipynb`, `src/explain/` |

---

## 7. App hoàn thiện & nâng cao (tuần 8)

| Người | Việc làm |
|---|---|
| **A** | Nâng cao #1: **dashboard sản phẩm**: dán link Tiki → crawl → bộ ba → gom theo chủ đề → "Top điều được khen / bị chê", so sánh 2 sản phẩm |
| **B** | Hoàn thiện giao diện app; deploy lên Streamlit Cloud hoặc Hugging Face Spaces |
| **C** | Nâng cao #2: so sánh với LLM trên tập test, hoặc #4: Label Studio ML Backend (chọn 1) |
| **D** | Nâng cao #3: mô hình end-to-end (GTS), hoặc #5: thêm CRF trên ViSoBERT / ensemble (chọn 1) |
| **Cả nhóm** | Cuối tuần, mỗi người dùng thử app và báo lỗi cho B |

> Nếu chậm tiến độ: **bỏ phần nâng cao** và dùng tuần 8 để hoàn thiện app cơ bản và bắt đầu viết báo cáo.

---

## 8. Báo cáo & bảo vệ (tuần 9)

| Người | Phần báo cáo | Khác |
|---|---|---|
| **A** | Thu thập dữ liệu (crawl & scrape), EDA, tiền xử lý, gom nhóm khía cạnh, error analysis | **Quay video demo** app |
| **B** | Gán nhãn (schema, guideline, độ đồng thuận, 2 đợt, active learning), ứng dụng | **Ghép báo cáo**, thống nhất định dạng |
| **C** | Baseline luật, mô hình ML, phương pháp đánh giá, kết quả, learning curve | Kiểm tra số liệu trong báo cáo khớp với notebook |
| **D** | Mô hình DL, so sánh ViSoBERT và PhoBERT, giải thích mô hình | **Làm slide** thuyết trình |
| **Cả nhóm** | Đọc chéo báo cáo; tập thuyết trình (mỗi người trình bày phần mình phụ trách) | |

---

## 9. Tổng hợp khối lượng gán nhãn

| Người | Pilot (T2) | Đợt 1 – test (T3) | Đợt 1 – val/train (T4) | Đợt 2 (T5) | **Tổng câu** | **Tổng giờ** |
|---|---|---|---|---|---|---|
| **A** | 100 | 165 | 318 | 460 | **1.043** | **~21,6 giờ** |
| **B** | 100 | 225 | 338 | 480 | **1.143** | **~24,0 giờ** |
| **C** | 100 | 195 | 178 | 140 | **613** | **~16,8 giờ** |
| **D** | 100 | 195 | 318 | 460 | **1.073** | **~22,4 giờ** |
| **Cả nhóm** | | | | | | **~85 giờ** (+ ~6 giờ họp pilot và kiểm tra) |

> C gán ít hơn vì gánh phần code luật, CRF, active learning trong tuần 2–5. D có khối lượng GPU lớn nhất ở tuần 6–7. Nếu nhóm muốn chia đều tuyệt đối thì đổi bớt câu của B sang C ở tuần 3.

## 10. Phụ thuộc giữa các thành viên (việc của ai cần xong trước)

| Việc | Ai làm | Cần có trước | Hạn |
|---|---|---|---|
| Dữ liệu thô trên Drive | A | — | Cuối tuần 1 |
| Label Studio (cụm từ + quan hệ) + gợi ý từ điển | B, C | Dữ liệu thô | Cuối tuần 1 |
| Văn bản sạch lớp 1 (cố định) | A | Dữ liệu thô | Giữa tuần 2 (trước pilot) |
| Tập test/val đóng băng + file import tuần 3 | C | Văn bản sạch, pilot xong | Cuối tuần 2 |
| `tokenizer.py` (căn chỉnh BIO) | D | Dữ liệu pilot | Cuối tuần 2 |
| `active_learning.py` + mô hình ban đầu | C, D | Đợt 1 xong | Cuối tuần 4 |
| Bộ dữ liệu cuối ~3.000 câu | B | Đợt 2 xong | Cuối tuần 5 |
| Checkpoint ViSoBERT tốt nhất | D | Bộ dữ liệu cuối | Cuối tuần 6 |
| Hàm giải thích cho app | D | Checkpoint | Giữa tuần 7 |
| `aspect_topics.json` (chủ đề khía cạnh) | A | Mô hình trích xuất | Cuối tuần 7 |

---

## 11. Quy trình chuyển file mỗi đợt gán nhãn

Mỗi người chạy **Label Studio trên máy mình**. File được chuyển qua **Google Drive**. Mọi project đều dùng **cùng file `labeling/label_config.xml`**.

### 11.1. Năm bước của một đợt

```
1. C   chạy make_tasks.py  → 8 file import + manifest → đưa lên Drive/tasks/<đợt>/
2. Mỗi người tải 2 file của mình → tạo 2 project trong Label Studio → import → gán nhãn
3. Mỗi người export JSON 2 project → đổi tên đúng quy ước → đưa lên Drive/exports/<đợt>/
4. B   tải cả thư mục exports/<đợt>/ → chạy agreement.py → bảng chính + độ đồng thuận + danh sách câu lệch
5. Cả nhóm họp chốt các câu lệch → người gán chính sửa trong Label Studio → export lại, ghi đè file trên Drive
   → B chạy lại agreement.py → bảng chính đã sửa
```

**Hai project của mỗi người:**

| Project | File import | Có gợi ý? | Dùng để |
|---|---|---|---|
| **Gán chính** | `<đợt>_<người>_chinh.json` | Có (từ điển ở đợt 1, mô hình ở đợt 2) | Dữ liệu cho mô hình |
| **Gán chéo** | `<đợt>_<người>_cheo.json` | **Không** (chỉ có chữ) | Đo độ đồng thuận |

> File gán chéo **không có gợi ý và không có nhãn của người kia**, để 2 lần gán thật sự độc lập.

### 11.2. Thư mục Drive & quy ước tên file

```
Drive/NLP-ABSA/
├── raw/                    tiki_reviews.csv (A), văn bản sạch lớp 1
├── tasks/                  file import do C tạo
│   ├── w2/                 w2_A_pilot.json ... w2_D_pilot.json
│   ├── w3/                 w3_A_chinh.json, w3_A_cheo.json, ..., w3_manifest.csv
│   ├── w4/
│   └── w5_vong1/  w5_vong2/
├── exports/                file export do từng người đưa lên (B thu gom)
│   ├── w2/  w3/  w4/  w5_vong1/  w5_vong2/
└── agreement/              kết quả B chạy (bảng chính, độ đồng thuận, câu lệch)
```

**Tên file:** `<đợt>_<người>_<chinh|cheo|pilot>.json`, ví dụ `w3_A_cheo.json`, `w5_vong2_D_chinh.json`.
- Label Studio export ra tên kiểu `project-3-at-2026-...json` → **phải đổi tên** trước khi đưa lên Drive, nếu không `agreement.py` sẽ bỏ qua file.
- Phải chọn kiểu export **JSON**. JSON-MIN làm mất quan hệ giữa các cụm từ; `agreement.py` sẽ báo lỗi nếu gặp file JSON-MIN.

Trên máy, đặt file vào `labeling/tasks/` và `labeling/exports/` (2 thư mục này **không đưa lên git**, Drive là bản gốc).

### 11.3. Lệnh cho C: chia file (`labeling/make_tasks.py`)

Đầu vào là CSV (cột `review_id`, `content`, có thể thêm `rating`, `category`) hoặc JSON task có gợi ý (đầu ra của `preannotate.py` / `active_learning.py`).

```bash
# Tuần 2 – pilot: cả 4 người cùng gán 100 câu
python labeling/make_tasks.py --input data/processed/pilot.json --phase w2 --pilot

# Tuần 3 – test: quota từng người, mỗi người gán chéo 45 câu theo vòng A→B→C→D→A
python labeling/make_tasks.py --input data/processed/test_preannotated.json --phase w3 \
    --quota A=120 B=180 C=150 D=150 --cross-n 45

# Tuần 4 – val + train
python labeling/make_tasks.py --input data/processed/w4_preannotated.json --phase w4 \
    --quota A=280 B=300 C=140 D=280 --cross-n 38

# Tuần 5 – mỗi vòng AL: A, B, D sửa nháp; C gán lại 70 câu KHÔNG có nháp để kiểm tra
python labeling/make_tasks.py --input data/processed/al_vong1.json --phase w5_vong1 \
    --quota A=230 B=240 D=230 --checker C --check-n 70
```

Cùng `--seed` (mặc định 42) thì chạy lại sẽ ra đúng cách chia cũ.

### 11.4. Lệnh cho B: đo độ đồng thuận (`labeling/agreement.py`)

```bash
python labeling/agreement.py --exports labeling/exports/w3
```

Script tự nhận ra đợt pilot (file `_pilot`) hay đợt thường (file `_chinh` + `_cheo`). Kết quả ghi vào `reports/agreement/<đợt>/`:

| File | Nội dung |
|---|---|
| `<đợt>_main.jsonl` | **Bảng chính**: mỗi dòng 1 câu, gồm các cụm từ (vị trí ký tự) và các bộ ba, `is_spam`, `labeled_by` |
| `<đợt>_agreement.csv` | Chỉ số đồng thuận cho cả nhóm (`ALL`) và từng cặp người (`D–A`, ...) |
| `<đợt>_disagreements.csv` | Mỗi dòng = 1 câu bị gán khác nhau, liệt kê **bộ ba chỉ người 1 có** và **bộ ba chỉ người 2 có** |

**Các cột đồng thuận** (bài trích xuất không dùng được Kappa kiểu "chọn ô", nên đo bằng F1 giữa 2 người):

| Cột | Ý nghĩa |
|---|---|
| `aspect_f1`, `opinion_f1` | 2 người có bôi **cùng cụm khía cạnh / cụm ý kiến** không (khớp đúng vị trí) |
| `pair_f1` | Có **nối giống nhau** không |
| `triplet_f1` | Khớp **cả bộ ba** (2 cụm từ + cảm xúc): **số chính để báo cáo**, mục tiêu ≥ 0,7 |
| `triplet_f1_partial` | Như trên nhưng cụm từ chỉ cần chồng lên nhau. Cao hơn `triplet_f1` nhiều → nhóm lệch chủ yếu ở **ranh giới** cụm từ |
| `polarity_agree_pct`, `polarity_kappa` | Trên các cụm ý kiến 2 người bôi trùng: % chọn cùng cảm xúc và Kappa. Dữ liệu lệch về Tích cực nên Kappa có thể thấp dù % khớp cao, **báo cáo cả hai** |
| `token_kappa` | Cohen's Kappa trên nhãn BIO của từng từ |
| `token_fleiss_kappa` | (chỉ pilot) Kappa theo từ cho cả 4 người cùng lúc |

Đọc các cột theo thứ tự `aspect_f1` → `opinion_f1` → `pair_f1` → `triplet_f1` để biết nhóm đang lệch ở bước nào: **tìm cụm từ**, **nối**, hay **cảm xúc**.

### 11.5. Hạn nộp file trong mỗi đợt

| Việc | Ai | Hạn |
|---|---|---|
| File import trên Drive | C | Sáng ngày đầu đợt |
| File export (JSON, đã đổi tên) trên Drive | Mọi người | Tối trước buổi họp cuối tuần 1 ngày |
| Độ đồng thuận + danh sách câu lệch | B | Trước buổi họp |
| Sửa câu lệch trong Label Studio, export lại | Người gán chính | 1 ngày sau buổi họp |
| Bảng chính cuối của đợt | B | Ngay sau khi nhận đủ file sửa |
