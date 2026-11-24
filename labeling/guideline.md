# Guideline gán nhãn ASTE — bản nháp đầu tiên (rút ra từ 8 câu gán thử)

> Ví dụ đánh dấu **[thật]** lấy từ dữ liệu Tiki, **[tự đặt]** lấy từ kế hoạch.

## 1. Mục tiêu

Với mỗi bình luận, rút ra các **bộ ba** `(cụm khía cạnh, cụm ý kiến, cảm xúc)`.

Ví dụ [thật]: *"Hàng ổn đóng gói khá cẩn thận, nhưng dịch vụ ship bên tiki hơi chán"* cho 3 bộ ba:

| Khía cạnh | Ý kiến | Cảm xúc |
|---|---|---|
| Hàng | ổn | Tích cực |
| đóng gói | khá cẩn thận | Tích cực |
| dịch vụ ship | hơi chán | Tiêu cực |

## 2. Nguyên tắc chung

1. Đọc **cả câu** trước khi bôi.
2. Gán theo **ý thật của khách**, không theo số sao.
3. **Không sửa văn bản** trong Label Studio. Nhãn lưu theo vị trí ký tự.
4. Không bôi dấu câu (`. , ! ?`) vào đầu hoặc cuối cụm từ.
5. Mỗi bộ ba = 1 khía cạnh + 1 ý kiến + **1 mũi tên nối**. Quên kéo mũi tên thì ý kiến bị hiểu thành khía cạnh ẩn.

## 3. Thao tác trong Label Studio

| Việc | Cách làm |
|---|---|
| Bôi cụm khía cạnh | Phím `a`, bôi cụm từ |
| Bôi ý kiến tích cực / trung lập / tiêu cực | Phím `p` / `u` / `n`, bôi cụm từ |
| Nối | Chọn cụm khía cạnh → nút **Create Relation** (hoặc `Alt+R`) → bấm cụm ý kiến (mũi tên đi **từ khía cạnh sang ý kiến**) |
| Gửi câu | Nút **Submit** hoặc `Ctrl+Enter` |
| Câu không có ý kiến nào | Bấm **Submit** trống (không bấm Skip) |
| Câu spam | Tích ô **Spam / vô nghĩa (loại bỏ)**, Submit |
| Xuất dữ liệu | Export kiểu **JSON**. **Không dùng JSON-MIN** vì mất quan hệ |

## 4. Bốn loại nhãn

| Nhãn | Là gì | Ví dụ [thật] |
|---|---|---|
| `ASPECT` | Đối tượng được nhận xét | `thời gian giao hàng`, `đóng gói`, `Giao hàng`, `dịch vụ ship`, `giá cả` |
| `OP_POS` | Ý kiến tích cực | `khá cẩn thận`, `xài tốt`, `cứng cáp` |
| `OP_NEU` | Đánh giá mức trung bình ("tạm", "bình thường") | `tạm được` trong *"Hàng dùng tạm được"* (14195254), `bình thường` trong *"Sản phẩm bình thường"* (12862034) |
| `OP_NEG` | Ý kiến tiêu cực | `quá lâu`, `hơi chán`, `Dễ bị thấm nước` |

## 5. Quy tắc ranh giới cụm từ

### 5.1. Cụm khía cạnh
- Lấy **cụm từ ngắn nhất mà vẫn đủ nghĩa**, chỉ đúng thứ đang được đánh giá. Thường là cụm danh từ (`chất vải`, `giá cả`), nhưng cũng có thể là cụm động từ chỉ một hoạt động/dịch vụ (`đóng gói`, `Giao hàng`). [thật]
- **Bỏ** phần chỉ "của ai, ở đâu, cái nào": `bên tiki`, `của shop`, `này`.
  - Đúng: `dịch vụ ship`. Sai: `dịch vụ ship bên tiki`. [thật]
- Bôi **đối tượng được đánh giá**, không bôi chủ thể làm hành động.
  - *"Shop đóng gói cẩn thận"* → khía cạnh là `đóng gói`, không phải `Shop`. [thật]
- Danh từ ghép giữ nguyên: `chất vải`, `thời gian giao hàng`.

### 5.2. Cụm ý kiến
- Gồm **từ phủ định** và **từ chỉ mức độ** đi liền: `khá cẩn thận`, `hơi chán`, `quá lâu`, `không đảm bảo hàng mới`.
- Không gồm chủ ngữ hay trạng ngữ thời gian.

## 6. Quy tắc nối

1. **Chỉ nối khi khía cạnh và ý kiến cùng một câu**. Không nối xuyên qua dấu chấm.
   - *"Không có màng nilon cho hộp sản phẩm. Dễ bị thấm nước..."* → `Dễ bị thấm nước` không nối với `hộp sản phẩm` ở câu trước, để khía cạnh ẩn. Quy tắc tạm thời, đang bàn ở câu hỏi mở số 2. [thật]
2. **Một khía cạnh nhiều ý kiến**: nối nhiều mũi tên từ cùng khía cạnh.
   - *"Dt xài tốt, đẹp, cứng cáp"* → `Dt` nối với cả `xài tốt`, `đẹp`, `cứng cáp` (3 mũi tên). [thật]
3. **Nhiều khía cạnh một ý kiến**: nối mỗi khía cạnh với ý kiến đó.
   - *"Áo và quần đều đẹp"* → `Áo → đẹp`, `quần → đẹp`. [tự đặt]
4. **Khía cạnh ẩn**: ý kiến không nói rõ khen cái gì thì **bôi ý kiến, không nối**.
   - *"Dễ sử dụng nhưng thời gian giao hàng quá lâu"* → `Dễ sử dụng` là OP_POS không nối. [thật]

## 7. Các trường hợp đặc biệt

| Trường hợp | Cách gán | Ví dụ |
|---|---|---|
| **Câu hỏi** | Không bôi, Submit trống | *"cho mình hỏi... có phải bản 365 ko?"* [thật] |
| **Mô tả khách quan** | Không bôi | *"Áo màu xanh, size M"* [tự đặt] |
| **Mong muốn** | Không bôi | *"Mong shop giao nhanh hơn"* [tự đặt] |
| **Mỉa mai** | Gán theo ý thật (tiêu cực) | *"Giao nhanh thật, 2 tuần mới tới"* [tự đặt] |
| **Phủ định** | Từ phủ định nằm **trong** cụm ý kiến | `không đảm bảo hàng mới` [thật] |
| **Spam / quảng cáo** | Tích ô Spam, không bôi | *"Săn sale nó phải như này. AE qua nhóm mình săn sale mình trả công cao nhé"* (20223857) [thật]. Review vừa quảng cáo vừa có ý kiến thật: xem câu hỏi mở số 4 |
| **Tên người** | Đã được ẩn thành `<NAME>`, không bôi | `<NAME>` [thật] |

## 8. Câu hỏi còn mở (đem ra bàn ở buổi pilot tuần 2)

Sáu chỗ mà 8 câu gán thử chưa trả lời được. Cột cuối là đề xuất tạm, **chưa chốt**.

| # | Câu hỏi | Ví dụ [thật] | Đề xuất tạm |
|---|---|---|---|
| 1 | Có tính **động từ sử dụng** (`xài`, `dùng`, `thử dùng`) vào cụm ý kiến không? Hiện chưa nhất quán. | `xài tốt` có "xài", nhưng *"thử dùng thì ổn"* chỉ bôi `ổn`, bỏ "thử dùng" | Chưa chốt. Cần thống nhất một cách và áp dụng cho cả nhóm |
| 2 | "Không nối qua dấu chấm" có thể làm **sai nghĩa**. | 20068874: *"hộp sản phẩm. Dễ bị thấm nước"*, câu sau nói về cái hộp ở câu trước | Cân nhắc cho nối khi câu sau **không có chủ ngữ mới** |
| 3 | Câu **kể sự việc ngụ ý chê** (không dùng từ chê) có bôi không, và bôi dài tới đâu? | `đặt mua màu trắng giao màu đen`, `Không có màng nilon` | Có bôi (là OP_NEG). Bôi phần ngắn nhất vẫn đủ nghĩa chê, ví dụ `giao màu đen` |
| 4 | Review **vừa quảng cáo vừa chê thật** có tính là spam không? | 20172079: *"Bịp lắm"* (nói về giá) cùng câu *"AE săn sale qua mình nhé"* | Nếu có ý kiến thật về sản phẩm/giá thì **không loại**, chỉ bôi phần ý kiến thật |
| 5 | **Ranh giới cụm ý kiến dài**: lấy đủ hay lấy phần lõi? | *"đúng như thời gian để xuất"* hay `đúng`; `cũng tốt` có lấy "cũng" không | Chưa chốt. Hướng thử: lấy phần lõi + từ chỉ mức độ/phủ định đi liền |
| 6 | **Ý định / lời hứa** có bôi không? | *"Dt xài tốt, đẹp, cứng cáp, sẽ ủng hộ shop nữa"* | Không bôi `sẽ ủng hộ shop nữa` (là ý định, chưa phải đánh giá) |

Hai chỗ đã tạm chốt nên không còn là câu hỏi mở: `dịch vụ ship` (không cắt còn `ship`, giống `chất vải`) ở §5.1, và nối qua dấu phẩy được nếu vẫn cùng một câu.

## 9. Ví dụ đã gán (đáp án tham khảo, 8 câu thử)

| review_id | Câu | Bộ ba |
|---|---|---|
| 18418965 | Dễ sử dụng nhưng thời gian giao hàng quá lâu. Shop đóng gói cẩn thận | (∅, Dễ sử dụng, +) · (thời gian giao hàng, quá lâu, −) · (đóng gói, cẩn thận, +) |
| 16770197 | Hàng ổn đóng gói khá cẩn thận, nhưng dịch vụ ship bên tiki hơi chán | (Hàng, ổn, +) · (đóng gói, khá cẩn thận, +) · (dịch vụ ship, hơi chán, −) |
| 20068874 | Không có màng nilon cho hộp sản phẩm. Dễ bị thấm nước và không đảm bảo hàng mới. | (hộp sản phẩm, Không có màng nilon, −) · (∅, Dễ bị thấm nước, −) · (∅, không đảm bảo hàng mới, −). Chờ câu hỏi mở 2 và 3 |
| 20190530 | Giao hàng đúng như thời gian để xuất, hàng đóng gói cẩn thận, hàng đúng như mô tả, nhìn sản phẩm đúng cũng hãng nha, giá cả trả góp cũng tốt | (Giao hàng, đúng như thời gian để xuất, +) · (đóng gói, cẩn thận, +) · (hàng, đúng như mô tả, +) · (sản phẩm, đúng cũng hãng, +) · (giá cả trả góp, cũng tốt, +). Ranh giới ý kiến chờ câu hỏi mở 5 |
| 16874154 | Dt xài tốt, đẹp, cứng cáp, sẽ ủng hộ shop nữa | (Dt, xài tốt, +) · (Dt, đẹp, +) · (Dt, cứng cáp, +) |
| 19778668 | Hàng đẹp, thử dùng thì ổn, nhưng đặt mua màu trắng giao màu đen | (Hàng, đẹp, +) · (Hàng, ổn, +) · (∅, đặt mua màu trắng giao màu đen, −). Chờ câu hỏi mở 1 và 3 |
| 11302991 | cho mình hỏi, ... có phải bản 365 ko? | không có bộ ba |
| 20172079 | TỐP NHẬN HÀNG ĐẦU TIÊN... Bịp lắm nha... AE săn sale qua mình nhé | Tạm thời: spam. Nếu chốt đề xuất câu hỏi mở 4 thì đổi thành (∅, Bịp lắm, −) |

Ký hiệu: `+` Tích cực, `−` Tiêu cực, `∅` khía cạnh ẩn.

Câu gốc giữ nguyên chính tả ("để xuất", "cũng hãng"), vì Label Studio lưu nhãn theo vị trí ký tự. Sau buổi pilot, khi chốt các câu hỏi ở §8 thì sửa lại các dòng ghi "chờ câu hỏi mở" ở bảng này.
