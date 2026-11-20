# Phân công nhóm (4 thành viên)

> Đi kèm với [`PROJECT_PLAN.md`](PROJECT_PLAN.md). Số mục (§) trong file này trỏ tới các mục của kế hoạch.
> Đề tài: **trích xuất ý kiến** `(cụm khía cạnh, cụm ý kiến, cảm xúc)` từ bình luận Tiki.
> Số câu gán nhãn và giờ công là **ước tính**, có thể điều chỉnh sau buổi họp mỗi tuần.

## 1. Vai trò chính

Ai cũng **gán nhãn**. Ngoài ra mỗi người **phụ trách chính** một mảng (chịu trách nhiệm cuối cùng, nhưng có thể nhờ người khác hỗ trợ).

| Mã | Tên thành viên | Vai trò phụ trách chính | Code / file phụ trách |
|---|---|---|---|
| **A** | *(điền tên)* | **Dữ liệu**: crawl, EDA, tiền xử lý lớp 1; sau đó **gom nhóm khía cạnh** và **dashboard** | `src/crawler/`, `src/preprocessing/cleaner.py`, `src/clustering/`, `notebooks/01–03, 08` |
| **B** | *(điền tên)* | **Gán nhãn & chất lượng**: Label Studio, guideline, đo độ đồng thuận, xuất bộ dữ liệu; sau đó **web app** | `labeling/label_config.xml`, `guideline.md`, `agreement.py`, `ls_to_dataset.py`, `app/` |
| **C** | *(điền tên)* | **Luật & ML truyền thống & active learning**, metrics đánh giá | `labeling/lexicon.json`, `preannotate.py`, `make_tasks.py`, `split_data.py`, `active_learning.py`, `src/models/rule_baseline.py`, `crf_tagger.py`, `ml_pairing.py`, `src/evaluation/`, `notebooks/04, 06` |
| **D** | *(điền tên)* | **Deep learning & giải thích mô hình**: BiLSTM-CRF, ViSoBERT, PhoBERT, giải thích | `src/preprocessing/tokenizer.py`, `src/models/bilstm_crf.py`, `transformer_tagger.py`, `transformer_pairing.py`, `predict.py`, `src/explain/`, `notebooks/05, 07` |

**Quy tắc chung:**
- Họp nhóm **2 lần/tuần** (15–30 phút): đầu tuần chia việc, cuối tuần báo cáo kết quả và vướng mắc.
- Code đẩy lên git theo nhánh riêng rồi gộp vào `main`. Dữ liệu thô (`data/raw/`) chia sẻ qua **Google Drive**, không đưa lên git.
- Khi gán chéo, **không ai gán lại câu do chính mình gán**. Vòng gán chéo: A → B → C → D → A (A gán lại một phần câu của B, B gán lại một phần câu của C...).
- Mỗi đợt gán nhãn đều theo **cùng một quy trình chuyển file** (C chia file → mọi người gán → B đo độ đồng thuận), xem **§11**.
- **Văn bản bình luận không được sửa sau khi đã gán nhãn** (nhãn dựa vào vị trí ký tự, §6 của kế hoạch).

---

## 2. Giai đoạn chuẩn bị (tuần 1–2)

### Tuần 1: Crawl dữ liệu & dựng công cụ

| Người | Việc làm | Kết quả cuối tuần |
|---|---|---|
| **A** | Chạy crawl thật (~60.000–80.000 review, chạy qua đêm), xuất CSV, đưa lên Drive cho cả nhóm | `tiki_reviews.csv` trên Drive |
| **B** | Cài Label Studio, tạo project với `label_config.xml` (§5.1: cụm từ + quan hệ); gán thử 5 câu, **export JSON kiểm tra quan hệ có được lưu**; viết bản nháp `guideline.md` | Label Studio chạy được, guideline v0 |
| **C** | Viết `lexicon.json` (cụm khía cạnh, từ ý kiến tích cực / tiêu cực / trung lập, từ phủ định, từ mức độ) và `preannotate.py` (bôi sẵn cụm từ + nối khía cạnh gần nhất) | Import thử 20 câu có gợi ý vào Label Studio thành công |
| **D** | Chuẩn bị môi trường Kaggle/Colab GPU; chạy thử ViSoBERT và PhoBERT **token classification** trên vài câu; bắt đầu từ điển teencode | Notebook chạy được 2 mô hình, `teencode_dict.json` v0 |

### Tuần 2: EDA, tiền xử lý lớp 1, gán thử 100 câu, chốt luật gán nhãn

| Người | Việc làm | Kết quả cuối tuần |
|---|---|---|
| **A** | EDA dữ liệu thô (§4): rating, độ dài, % review trống, theo ngành hàng; **top danh từ / tính từ** gửi C bổ sung từ điển. Viết `cleaner.py` (**tiền xử lý lớp 1**: làm sạch, Unicode, ẩn tên / số điện thoại, lọc spam, lọc 10–80 từ) | `02_eda.ipynb`, văn bản sạch **cố định** để gán nhãn |
| **B** | Tổ chức **pilot**: cả 4 người cùng gán **cùng 100 câu**; chạy `agreement.py`; chủ trì buổi thảo luận; cập nhật guideline (ranh giới cụm từ, khía cạnh ẩn, khi nào Trung lập) | `guideline.md` v1, schema nhãn đã chốt |
| **C** | Viết `split_data.py`: chọn và **đóng băng** 600 câu test + 400 câu val (phân tầng theo ngành hàng và rating, lấy thêm câu 1–3 sao); chạy `preannotate.py` cho tập test; chạy `make_tasks.py` cho tuần 3 | Danh sách ID test/val cố định, file import tuần 3 trên Drive |
| **D** | Viết `tokenizer.py`: tách từ giữ vị trí ký tự, đổi cụm từ ↔ nhãn BIO, căn chỉnh nhãn với token của ViSoBERT và PhoBERT; **test**: cụm từ → BIO → cụm từ phải ra đúng như cũ | Hàm căn chỉnh chạy đúng trên dữ liệu pilot |
| **Cả nhóm** | **Gán pilot 100 câu** (mỗi người ~2,8 giờ) + 1 buổi thảo luận ~1,5 giờ | |

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
