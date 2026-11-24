# Guideline gán nhãn ASTE 

> Ví dụ đánh dấu **[thật]** lấy từ dữ liệu Tiki
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
| `ASPECT` | Đối tượng được nhận xét | `thời gian giao hàng`, `đóng gói`, `dịch vụ ship`, `giá cả` |
| `OP_POS` | Ý kiến tích cực | `khá cẩn thận`, `xài tốt`, `cứng cáp` |
| `OP_NEU` | Đánh giá mức trung bình ("tạm", "bình thường") | *(chưa có ví dụ thật, cần tìm ở pilot)* |
| `OP_NEG` | Ý kiến tiêu cực | `quá lâu`, `hơi chán`, `Dễ bị thấm nước` |

## 5. Quy tắc ranh giới cụm từ

### 5.1. Cụm khía cạnh
- Lấy **cụm danh từ ngắn nhất mà vẫn đủ nghĩa**.
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
   - *"Không có màng nilon cho hộp sản phẩm. Dễ bị thấm nước..."* → `Dễ bị thấm nước` không nối với `hộp sản phẩm` ở câu trước, để khía cạnh ẩn. [thật]
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
| **Spam / quảng cáo** | Tích ô Spam, không bôi | *"AE săn sale qua mình nhé"* [thật] |
| **Tên người** | Đã được ẩn thành `<NAME>`, không bôi | `<NAME>` [thật] |


## 8. Ví dụ đã gán (đáp án tham khảo, 8 câu thử)

| review_id | Câu | Bộ ba |
|---|---|---|
| 18418965 | Dễ sử dụng nhưng thời gian giao hàng quá lâu. Shop đóng gói cẩn thận | (∅, Dễ sử dụng, +) · (thời gian giao hàng, quá lâu, −) · (đóng gói, cẩn thận, +) |
| 16770197 | Hàng ổn đóng gói khá cẩn thận, nhưng dịch vụ ship bên tiki hơi chán | (Hàng, ổn, +) · (đóng gói, khá cẩn thận, +) · (dịch vụ ship, hơi chán, −) |
| 20068874 | Không có màng nilon cho hộp sản phẩm. Dễ bị thấm nước và không đảm bảo hàng mới. | (hộp sản phẩm, Không có màng nilon, −) · (∅, Dễ bị thấm nước, −) · (∅, không đảm bảo hàng mới, −) |
| 20190530 | Giao hàng đúng như thời gian để xuất, hàng đóng gói cẩn thận, ... giá cả trả góp cũng tốt | 5 bộ ba, đều tích cực |
| 16874154 | Dt xài tốt, đẹp, cứng cáp, sẽ ủng hộ shop nữa | (Dt, xài tốt, +) · (Dt, đẹp, +) · (Dt, cứng cáp, +) |
| 19778668 | Hàng đẹp, thử dùng thì ổn, nhưng đặt mua màu trắng giao màu đen | (Hàng, đẹp, +) · (Hàng, ổn, +) · (∅, đặt mua màu trắng giao màu đen, −) |
| 11302991 | cho mình hỏi, ... có phải bản 365 ko? | không có bộ ba |
| 20172079 | TỐP NHẬN HÀNG ĐẦU TIÊN... AE săn sale qua mình nhé | spam |

Ký hiệu: `+` Tích cực, `−` Tiêu cực, `∅` khía cạnh ẩn.
