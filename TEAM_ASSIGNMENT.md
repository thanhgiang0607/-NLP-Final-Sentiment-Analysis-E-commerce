# Phân công nhóm (4 thành viên)

> Đi kèm với [`PROJECT_PLAN.md`](PROJECT_PLAN.md). Số mục (§) trong file này trỏ tới các mục của kế hoạch.
> Số câu gán nhãn và giờ công là **ước tính**, có thể điều chỉnh sau buổi họp mỗi tuần.

## 1. Vai trò chính

Ai cũng **gán nhãn**. Ngoài ra mỗi người **phụ trách chính** một mảng (chịu trách nhiệm cuối cùng, nhưng có thể nhờ người khác hỗ trợ).

| Mã | Tên thành viên | Vai trò phụ trách chính | Code / file phụ trách |
|---|---|---|---|
| **A** | *(điền tên)* | **Dữ liệu**: crawl, EDA, tiền xử lý | `src/crawler/`, `src/preprocessing/`, `notebooks/01–03` |
| **B** | *(điền tên)* | **Gán nhãn & chất lượng**: Label Studio, guideline, Kappa; sau đó **web app** | `labeling/label_config.xml`, `labeling/guideline.md`, `labeling/kappa.py`, `labeling/ls_to_table.py`, `app/` |
| **C** | *(điền tên)* | **ML truyền thống & active learning**, metrics đánh giá | `labeling/preannotate.py`, `make_tasks.py`, `split_data.py`, `active_learning.py`, `src/models/ml_absa.py`, `src/evaluation/`, `notebooks/04, 06` |
| **D** | *(điền tên)* | **Deep learning & giải thích mô hình**: ViSoBERT, PhoBERT, BiLSTM, LIME/SHAP | `src/models/transformer_absa.py`, `src/explain/`, `notebooks/05, 07` |

**Quy tắc chung:**
- Họp nhóm **2 lần/tuần** (15–30 phút): đầu tuần chia việc, cuối tuần báo cáo kết quả và vướng mắc.
- Code đẩy lên git theo nhánh riêng rồi gộp vào `main`. Dữ liệu thô (`data/raw/`) chia sẻ qua **Google Drive**, không đưa lên git.
- Khi gán chéo, **không ai gán lại câu do chính mình gán**. Vòng gán chéo: A → B → C → D → A (A gán lại một phần câu của B, B gán lại một phần câu của C...).
- Mỗi đợt gán nhãn đều theo **cùng một quy trình chuyển file** (C chia file → mọi người gán → B tính Kappa), xem **§11**.

---

## 2. Giai đoạn chuẩn bị (tuần 1–2)

### Tuần 1: Crawl dữ liệu & dựng công cụ

| Người | Việc làm | Kết quả cuối tuần |
|---|---|---|
| **A** | Chạy crawl thật (~60.000–80.000 review, chạy qua đêm), xuất CSV, đưa lên Drive cho cả nhóm | `tiki_reviews.csv` trên Drive |
| **B** | Cài Label Studio, tạo project với `label_config.xml` (§5.1); viết bản nháp `guideline.md` | Label Studio chạy được, guideline v0 |
| **C** | Viết `aspect_keywords.json` (từ khoá cho 7 khía cạnh, §2) và `preannotate.py` (tạo gợi ý từ khoá để import vào Label Studio) | Import thử 20 câu có gợi ý vào Label Studio thành công |
| **D** | Chuẩn bị môi trường Kaggle/Colab GPU; chạy thử ViSoBERT và PhoBERT trên vài câu; bắt đầu từ điển teencode | Notebook chạy được 2 mô hình, `teencode_dict.json` v0 |

### Tuần 2: EDA, gán thử 200 câu, chốt luật gán nhãn

| Người | Việc làm | Kết quả cuối tuần |
|---|---|---|
| **A** | EDA dữ liệu thô (§4): rating, độ dài, % review trống, theo ngành hàng, top từ, emoji | `02_eda.ipynb` + hình trong `reports/` |
| **B** | Tổ chức **pilot**: cả 4 người cùng gán **cùng 200 câu**; tính độ khớp; chủ trì buổi thảo luận; cập nhật guideline | `guideline.md` v1, schema khía cạnh đã chốt |
| **C** | Viết `split_data.py`: chọn và **đóng băng** 1.000 câu test + 800 câu val (phân tầng theo ngành hàng và rating, chỉ lấy câu ≥ 10 từ) | Danh sách ID test/val cố định |
| **D** | Viết `cleaner.py` (làm sạch, chuẩn hoá Unicode, teencode, emoji, lọc spam, ẩn tên người) và 2 phiên bản text (§6) | Pipeline tiền xử lý chạy được trên toàn bộ dữ liệu thô |
| **Cả nhóm** | **Gán pilot 200 câu** (mỗi người gán đủ 200 câu, ~2,5 giờ) + 1 buổi thảo luận ~1 giờ | |

---

## 3. Đợt 1: Người tự gán 3.000 câu (tuần 3–4)

### Tuần 3: Gán tập test (1.000 câu)

Đây là "đề thi cuối kỳ", nên gán **cẩn thận nhất**. Chỉ có gợi ý từ khoá, **không** dùng gợi ý từ mô hình.

| Người | Gán chính | Gán chéo (30%) | Tổng câu | Giờ công (~50 giây/câu) | Việc khác |
|---|---|---|---|---|---|
| **A** | 200 | 75 (câu của B) | 275 | ~3,8 giờ | Hoàn thiện EDA thô; hỗ trợ D ghép pipeline tiền xử lý |
| **B** | 300 | 75 (câu của C) | 375 | ~5,2 giờ | Theo dõi tiến độ gán; trả lời thắc mắc về luật; viết `ls_to_table.py` (xuất Label Studio → bảng) |
| **C** | 250 | 75 (câu của D) | 325 | ~4,5 giờ | Viết `absa_metrics.py` (3 cấp đánh giá, §8.1) |
| **D** | 250 | 75 (câu của A) | 325 | ~4,5 giờ | Chạy tiền xử lý trên tập test đã gán |
| **Tổng** | **1.000** | **300** | **1.300** | | |

**Cuối tuần 3:** B tính **Kappa** cho tập test (`kappa.py`, §11), cả nhóm thống nhất lại các câu bị gán khác nhau → **tập test đóng băng**, không ai sửa nữa.

### Tuần 4: Gán val + train (2.000 câu) và điểm kiểm tra

| Người | Gán chính | Gán chéo (15%) | Tổng câu | Giờ công (~45 giây/câu) | Việc khác |
|---|---|---|---|---|---|
| **A** | 550 | 75 | 625 | ~7,8 giờ | Đếm số câu của từng cặp (khía cạnh, cảm xúc) để chuẩn bị buổi kiểm tra |
| **B** | 600 | 75 | 675 | ~8,4 giờ | Tính Kappa từng khía cạnh |
| **C** | 300 | 75 | 375 | ~4,7 giờ | Train **SVM ban đầu**; viết `active_learning.py` (§5.4) |
| **D** | 550 | 75 | 625 | ~7,8 giờ | Train **ViSoBERT ban đầu** (~10 phút GPU) |
| **Tổng** | **2.000** | **300** | **2.300** | | |

> C gán ít hơn vì phải hoàn thành `active_learning.py`. Không có script này thì đợt 2 không bắt đầu được.

### Buổi kiểm tra giữa 2 đợt (cuối tuần 4, cả nhóm ~1,5 giờ)

| Người trình bày | Nội dung | Quyết định cần đưa ra |
|---|---|---|
| **B** | Kappa từng khía cạnh; các lỗi gán hay gặp | Khía cạnh nào Kappa < 0,6 thì sửa guideline thế nào |
| **A** | Số câu của từng cặp trong tập test | Cặp nào < 30 câu thì gộp lớp hay giữ nguyên |
| **C** | Điểm SVM ban đầu trên val; chạy thử `active_learning.py` | Có bắt đầu đợt 2 được chưa |
| **D** | Điểm ViSoBERT ban đầu trên val | Dùng SVM hay ViSoBERT để chọn câu và điền nháp cho đợt 2 |
| **Cả nhóm** | | Chốt số câu đợt 2 (mặc định 3.000) |

---

## 4. Đợt 2: Máy chọn câu và điền nháp, người kiểm tra (tuần 5)

3 vòng, mỗi vòng khoảng 2 ngày. Mỗi vòng ~1.000 câu do máy chọn, đã có nhãn nháp. **Người kiểm tra và sửa tất cả các câu.**

### Một vòng diễn ra thế nào

```
Ngày 1 sáng: C chạy active_learning.py → chọn ~1.000 câu → import vào Label Studio
Ngày 1–2:    A, B, D kiểm tra & sửa nhãn nháp; C gán 100 câu KHÔNG có nháp để kiểm tra
Ngày 2 tối:  B kiểm tra chất lượng → C gộp dữ liệu, train lại mô hình → sẵn sàng vòng sau
```

### Số câu mỗi người (tính cho cả 3 vòng)

| Người | Mỗi vòng | 3 vòng | Giờ công | Việc khác trong tuần 5 |
|---|---|---|---|---|
| **A** | 330 câu (sửa nháp, ~20 giây/câu) | 990 | ~5,5 giờ | EDA sau gán nhãn (§4); **keyword baseline** |
| **B** | 340 câu (sửa nháp) | 1.020 | ~5,7 giờ | **Kiểm tra chất lượng mỗi vòng**: tỉ lệ người gán giữ nguyên nháp (cao bất thường = đang "bấm theo máy"), so với 100 câu kiểm tra của C |
| **C** | 100 câu **không có nháp** (~45 giây/câu) | 300 | ~3,8 giờ | **Chạy vòng active learning**: chọn câu, import, gộp dữ liệu, train lại; ghi điểm val sau mỗi vòng |
| **D** | 330 câu (sửa nháp) | 990 | ~5,5 giờ | Viết sẵn code train ViSoBERT/PhoBERT multi-task và sentence-pair cho tuần 6 |
| **Tổng** | | **~3.300** | | |

> **Nếu trễ:** dừng sau vòng 2 (~5.000 câu) và chuyển sang tuần 6 (§13).

**Cuối tuần 5:** B xuất **bộ dữ liệu cuối ~6.000 câu** vào `data/labeled/` và commit lên git.

---

## 5. Giai đoạn mô hình (tuần 6)

| Người | Việc làm | Kết quả |
|---|---|---|
| **A** | Hoàn thiện EDA sau gán nhãn; tạo 2 phiên bản text (chuẩn hoá đầy đủ / giữ teencode); chia tập test theo mức "phi chuẩn" (§7.3) | `02_eda.ipynb` phần ABSA, dữ liệu đã xử lý |
| **B** | Dựng **khung web app** (Streamlit): nhập 1 câu → gọi SVM → hiện bảng 7 khía cạnh; viết báo cáo chương Dữ liệu + Gán nhãn | App chạy được với SVM, bản nháp 2 chương |
| **C** | NB / LR / SVM / RF, **Cách A và B** (§7.1–7.2), tinh chỉnh tham số; **learning curve phần ML** (§8.3) | `04_ml_models.ipynb`, bảng kết quả ML |
| **D** | **ViSoBERT & PhoBERT** (multi-task + sentence-pair), **BiLSTM**; learning curve phần ViSoBERT | Checkpoint DL (lưu lên Drive), bảng so sánh ViSoBERT và PhoBERT |

> D tốn nhiều GPU nhất (~15–20 giờ). Nếu quota Kaggle của D không đủ, các thành viên khác cho dùng tài khoản Kaggle của mình để chạy thêm.

---

## 6. Đánh giá & giải thích mô hình (tuần 7)

| Người | Việc làm | Kết quả |
|---|---|---|
| **A** | **Error analysis**: lấy các câu đoán sai, phân loại lỗi (mỉa mai, phủ định, nhầm `QUALITY`/`GENERAL`...) | Bảng lỗi kèm ví dụ |
| **B** | Gắn ViSoBERT vào app; thêm nút **"Vì sao?"** (dùng code giải thích của D); phân tích CSV hàng loạt | App đủ 4 chức năng cơ bản (§9) |
| **C** | Đánh giá **3 cấp** cho tất cả mô hình trên tập test; heatmap F1 theo khía cạnh; confusion matrix; vẽ learning curve hoàn chỉnh | `06_evaluation.ipynb`, bảng kết quả cuối |
| **D** | **LIME / SHAP** (§8.2): top từ ảnh hưởng, giải thích 100–200 câu, deletion test; đóng gói hàm giải thích để B dùng trong app | `07_explainability.ipynb`, `src/explain/` |

---

## 7. App hoàn thiện & nâng cao (tuần 8)

| Người | Việc làm |
|---|---|
| **A** | Nâng cao #1: **dashboard sản phẩm**: dán link Tiki → crawl → radar chart từng khía cạnh, so sánh 2 sản phẩm |
| **B** | Hoàn thiện giao diện app; deploy lên Streamlit Cloud hoặc Hugging Face Spaces |
| **C** | Nâng cao #5: so sánh với LLM trên tập test, hoặc #2: Label Studio ML Backend (chọn 1) |
| **D** | Nâng cao #4: ensemble ViSoBERT + PhoBERT hoặc focal loss (chọn 1) |
| **Cả nhóm** | Cuối tuần, mỗi người dùng thử app và báo lỗi cho B |

> Nếu chậm tiến độ: **bỏ phần nâng cao** và dùng tuần 8 để hoàn thiện app cơ bản và bắt đầu viết báo cáo.

---

## 8. Báo cáo & bảo vệ (tuần 9)

| Người | Phần báo cáo | Khác |
|---|---|---|
| **A** | Thu thập dữ liệu (crawl & scrape), EDA, tiền xử lý, error analysis | **Quay video demo** app |
| **B** | Gán nhãn (guideline, Kappa, 2 đợt, active learning), ứng dụng | **Ghép báo cáo**, thống nhất định dạng |
| **C** | Mô hình ML, phương pháp đánh giá, kết quả, learning curve | Kiểm tra số liệu trong báo cáo khớp với notebook |
| **D** | Mô hình DL, so sánh ViSoBERT và PhoBERT, giải thích mô hình | **Làm slide** thuyết trình |
| **Cả nhóm** | Đọc chéo báo cáo; tập thuyết trình (mỗi người trình bày phần mình phụ trách) | |

---

## 9. Tổng hợp khối lượng gán nhãn

| Người | Pilot (T2) | Đợt 1 – test (T3) | Đợt 1 – val/train (T4) | Đợt 2 (T5) | **Tổng câu** | **Tổng giờ** |
|---|---|---|---|---|---|---|
| **A** | 200 | 275 | 625 | 990 | **2.090** | **~19,6 giờ** |
| **B** | 200 | 375 | 675 | 1.020 | **2.270** | **~21,8 giờ** |
| **C** | 200 | 325 | 375 | 300 | **1.200** | **~15,5 giờ** |
| **D** | 200 | 325 | 625 | 990 | **2.140** | **~20,3 giờ** |
| **Cả nhóm** | | | | | | **~77 giờ** |

> C gán ít hơn vì gánh phần code của active learning và ML trong tuần 3–5. D có khối lượng GPU lớn nhất ở tuần 6–7. Nếu nhóm muốn chia đều tuyệt đối thì đổi bớt câu của B sang C ở tuần 3.

## 10. Phụ thuộc giữa các thành viên (việc của ai cần xong trước)

| Việc | Ai làm | Cần có trước | Hạn |
|---|---|---|---|
| Dữ liệu thô trên Drive | A | — | Cuối tuần 1 |
| Label Studio + gợi ý từ khoá | B, C | Dữ liệu thô | Cuối tuần 1 |
| Tập test/val đóng băng | C | Dữ liệu thô, pilot xong | Cuối tuần 2 |
| `active_learning.py` + mô hình ban đầu | C, D | Đợt 1 xong | Cuối tuần 4 |
| Bộ dữ liệu cuối ~6.000 câu | B | Đợt 2 xong | Cuối tuần 5 |
| Checkpoint ViSoBERT tốt nhất | D | Bộ dữ liệu cuối | Cuối tuần 6 |
| Hàm giải thích (LIME/SHAP) cho app | D | Checkpoint | Giữa tuần 7 |

---

## 11. Quy trình chuyển file mỗi đợt gán nhãn

Mỗi người chạy **Label Studio trên máy mình**. File được chuyển qua **Google Drive**. Mọi project đều dùng **cùng file `labeling/label_config.xml`**.

### 11.1. Năm bước của một đợt

```
1. C   chạy make_tasks.py  → 8 file import + manifest → đưa lên Drive/tasks/<đợt>/
2. Mỗi người tải 2 file của mình → tạo 2 project trong Label Studio → import → gán nhãn
3. Mỗi người export JSON 2 project → đổi tên đúng quy ước → đưa lên Drive/exports/<đợt>/
4. B   tải cả thư mục exports/<đợt>/ → chạy kappa.py → bảng chính + Kappa + danh sách câu lệch
5. Cả nhóm họp: điền cột `final` cho các câu lệch → B chạy lại kappa.py --resolved → bảng chính đã sửa
```

**Hai project của mỗi người:**

| Project | File import | Có gợi ý? | Dùng để |
|---|---|---|---|
| **Gán chính** | `<đợt>_<người>_chinh.json` | Có (từ khoá ở đợt 1, nhãn nháp của mô hình ở đợt 2) | Dữ liệu cho mô hình |
| **Gán chéo** | `<đợt>_<người>_cheo.json` | **Không** (chỉ có chữ) | Tính Kappa |

> File gán chéo **không có gợi ý và không có nhãn của người kia**, để 2 lần gán thật sự độc lập.

### 11.2. Thư mục Drive & quy ước tên file

```
Drive/NLP-ABSA/
├── raw/                    tiki_reviews.csv (A)
├── tasks/                  file import do C tạo
│   ├── w2/                 w2_A_pilot.json ... w2_D_pilot.json
│   ├── w3/                 w3_A_chinh.json, w3_A_cheo.json, ..., w3_manifest.csv
│   ├── w4/
│   └── w5_vong1/  w5_vong2/  w5_vong3/
├── exports/                file export do từng người đưa lên (B thu gom)
│   ├── w2/  w3/  w4/  w5_vong1/ ...
└── kappa/                  kết quả B chạy (bảng chính, Kappa, câu lệch)
```

**Tên file:** `<đợt>_<người>_<chinh|cheo|pilot>.json`, ví dụ `w3_A_cheo.json`, `w5_vong2_D_chinh.json`.
Label Studio export ra tên kiểu `project-3-at-2026-...json` → **phải đổi tên** trước khi đưa lên Drive, nếu không `kappa.py` sẽ bỏ qua file.

Trên máy, đặt file vào `labeling/tasks/` và `labeling/exports/` (2 thư mục này **không đưa lên git**, Drive là bản gốc).

### 11.3. Lệnh cho C: chia file (`labeling/make_tasks.py`)

Đầu vào là CSV (cột `review_id`, `content`, có thể thêm `rating`, `category` và 7 cột khía cạnh làm gợi ý) hoặc JSON task của `preannotate.py`.

```bash
# Tuần 2 – pilot: cả 4 người cùng gán 200 câu
python labeling/make_tasks.py --input data/processed/pilot.csv --phase w2 --pilot

# Tuần 3 – test: quota từng người, mỗi người gán chéo 75 câu theo vòng A→B→C→D→A
python labeling/make_tasks.py --input data/processed/test_preannotated.json --phase w3 \
    --quota A=200 B=300 C=250 D=250 --cross-n 75

# Tuần 4 – val + train
python labeling/make_tasks.py --input data/processed/w4_preannotated.json --phase w4 \
    --quota A=550 B=600 C=300 D=550 --cross-n 75

# Tuần 5 – mỗi vòng AL: A, B, D sửa nháp; C gán lại 100 câu KHÔNG có nháp để kiểm tra
python labeling/make_tasks.py --input data/processed/al_vong1.json --phase w5_vong1 \
    --quota A=330 B=340 D=330 --checker C --check-n 100
```

Cùng `--seed` (mặc định 42) thì chạy lại sẽ ra đúng cách chia cũ.

### 11.4. Lệnh cho B: tính Kappa (`labeling/kappa.py`)

```bash
python labeling/kappa.py --exports labeling/exports/w3
```

Script tự nhận ra đợt pilot (file `_pilot`) hay đợt thường (file `_chinh` + `_cheo`), và đọc được cả 2 kiểu export **JSON** và **JSON-MIN** của Label Studio. Kết quả ghi vào `reports/kappa/<đợt>/`:

| File | Nội dung |
|---|---|
| `<đợt>_main.csv` | **Bảng chính**: mỗi câu 1 dòng, 7 cột khía cạnh (0 = không đề cập, 1 = Tích cực, 2 = Trung lập, 3 = Tiêu cực), `is_spam`, `labeled_by` |
| `<đợt>_kappa.csv` | Kappa từng khía cạnh, cho cả nhóm (`ALL`) và từng cặp người (`D–A`, ...) |
| `<đợt>_disagreements.csv` | Mỗi dòng = 1 (câu, khía cạnh) bị gán khác nhau, có cột `final` để trống |

**Các cột Kappa:**

| Cột | Ý nghĩa |
|---|---|
| `agree_pct` | % câu 2 người chọn giống hệt nhau (dễ hiểu, đưa kèm Kappa trong báo cáo) |
| `kappa_4` | Kappa trên 4 nhãn (không đề cập / tích cực / trung lập / tiêu cực): **số chính để báo cáo** |
| `kappa_detect` | Kappa chỉ xét "có nhắc khía cạnh này không" |
| `kappa_sentiment` | Kappa cảm xúc, chỉ trên các câu **cả 2 người** cùng thấy khía cạnh |
| `fleiss_4`, `fleiss_detect` | (chỉ pilot) Kappa cho cả 4 người cùng lúc |

Nhìn `kappa_detect` và `kappa_sentiment` để biết nhóm đang lệch ở đâu: **thấy khía cạnh** hay **đánh giá cảm xúc**.

**Sau buổi họp:** mở `<đợt>_disagreements.csv`, điền cột `final` (`Tích cực` / `Trung lập` / `Tiêu cực` / `Không đề cập`), rồi chạy:

```bash
python labeling/kappa.py --exports labeling/exports/w3 --resolved reports/kappa/w3/w3_disagreements.csv
```

→ ra `<đợt>_main_resolved.csv`: bảng chính đã sửa theo kết luận của nhóm. Đây là file đưa vào `data/labeled/`.

### 11.5. Hạn nộp file trong mỗi đợt

| Việc | Ai | Hạn |
|---|---|---|
| File import trên Drive | C | Sáng ngày đầu đợt |
| File export (đã đổi tên) trên Drive | Mọi người | Tối trước buổi họp cuối tuần 1 ngày |
| Kappa + danh sách câu lệch | B | Trước buổi họp |
| Bảng chính đã sửa | B | Ngay sau buổi họp |
