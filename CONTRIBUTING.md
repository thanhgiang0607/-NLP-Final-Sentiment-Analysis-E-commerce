# Quy ước làm việc với Git (branch, commit, PR)

> Áp dụng cho cả 4 thành viên. Mục tiêu: `main` **luôn chạy được**, ai cũng biết ai đang sửa gì, không ai đè code của ai.

## 1. Nguyên tắc chung

1. **Không commit thẳng vào `main`.** Mọi thay đổi đi qua branch → Pull Request (PR) → **Giang duyệt** → merge.
2. **Mỗi branch làm 1 việc** (1 script, 1 notebook, 1 lần sửa lỗi). Việc lớn thì chia nhỏ, mỗi phần 1 PR.
3. **Branch sống ngắn**: tạo ra, làm, mở PR, merge trong **2–3 ngày**. Branch để lâu sẽ lệch xa `main` và dễ xung đột.
4. **Chỉ sửa file mình phụ trách** (bảng vai trò trong `TEAM_ASSIGNMENT.md` §1). Cần sửa file của người khác thì nhắn họ trước, hoặc mở PR nhỏ riêng và nhờ chính họ duyệt.
5. **Không đưa lên git:** dữ liệu (`data/raw/`, `data/processed/`), file task / export Label Studio (`labeling/tasks/`, `labeling/exports/`), checkpoint mô hình (`models/`, `*.pt`, `*.bin`...), file `.env` / mật khẩu / token. Những thứ này để trên **Google Drive** (`TEAM_ASSIGNMENT.md` §11). `.gitignore` đã chặn sẵn, **không dùng `git add -f`** để vượt qua.

## 2. Đặt tên branch

```
<loại>/<tên-người>-<mô-tả-ngắn>
```

- Viết **thường, không dấu**, nối bằng gạch ngang `-`.
- `<tên-người>`: `giang`, `phuc`, `dung`, `hang`.

| Loại | Dùng khi | Ví dụ |
|---|---|---|
| `feat` | Thêm script / chức năng mới | `feat/dung-preannotate`, `feat/phuc-ls-to-dataset` |
| `fix` | Sửa lỗi | `fix/giang-cleaner-unicode` |
| `exp` | Notebook thử nghiệm, train mô hình | `exp/hang-visobert-tagger`, `exp/dung-crf-baseline` |
| `data` | Thay đổi từ điển, cấu hình gán nhãn (không phải dữ liệu thô) | `data/dung-lexicon-v1`, `data/hang-teencode-v0` |
| `docs` | Kế hoạch, guideline, báo cáo, README | `docs/phuc-guideline-v1` |
| `refactor` | Sắp xếp lại code, không đổi kết quả | `refactor/phuc-app-layout` |

## 3. Viết commit message

```
<loại>(<phạm vi>): <mô tả ngắn bằng tiếng Việt>
```

- `<loại>`: giống bảng ở trên (`feat`, `fix`, `exp`, `data`, `docs`, `refactor`).
- `<phạm vi>`: thư mục / mảng bị sửa: `crawler`, `labeling`, `preprocessing`, `models`, `eval`, `explain`, `app`, `notebook`, `plan`.
- Mô tả **nói đã làm gì**, ≤ 72 ký tự. Cần giải thích thêm thì xuống dòng trống rồi viết tiếp.

```
feat(labeling): thêm preannotate.py bôi sẵn cụm từ theo lexicon
fix(preprocessing): giữ đúng vị trí ký tự khi chuẩn hoá Unicode
exp(models): train ViSoBERT tagger trên train 600 câu, F1 val 0.71
data(labeling): lexicon v1 bổ sung từ khoá ngành sách, đồ ăn
docs(guideline): v1 sau pilot, thêm luật cho câu mỉa mai
```

Commit **nhỏ và thường xuyên** (mỗi lần xong một bước chạy được), không gom cả tuần vào 1 commit.

## 4. Quy trình từ đầu đến cuối

```bash
# 1. Cập nhật main mới nhất
git switch main
git pull

# 2. Tạo branch mới từ main
git switch -c feat/dung-preannotate

# 3. Làm việc, commit dần
git status                      # xem đã sửa những gì
git add labeling/preannotate.py labeling/lexicon.json
git commit -m "feat(labeling): thêm preannotate.py bôi sẵn cụm từ theo lexicon"

# 4. Đẩy branch lên GitHub (lần đầu cần -u)
git push -u origin feat/dung-preannotate

# 5. Mở PR: vào trang repo trên GitHub → bấm "Compare & pull request"
#    (hoặc nếu có GitHub CLI: gh pr create --fill)
```

- **`git add` từng file cụ thể**, hạn chế `git add .` / `git add -A` để không lỡ đưa file rác lên. Luôn xem `git status` trước khi commit.
- Sau khi mở PR, **commit tiếp vào chính branch đó** rồi `git push` → PR tự cập nhật, không cần mở PR mới.

### Khi `main` đã có thay đổi mới (người khác vừa merge)

```bash
git switch main
git pull
git switch feat/dung-preannotate
git merge main                 # đưa thay đổi mới của main vào branch của mình
# nếu có xung đột: sửa file, rồi
git add <file đã sửa>
git commit
git push
```

Dùng `git merge main` (không dùng `rebase` + `push --force`) để tránh làm mất code khi chưa quen.

### Xử lý xung đột (conflict)

Git đánh dấu chỗ xung đột trong file như sau:

```
<<<<<<< HEAD
dòng của mình
=======
dòng mới trên main
>>>>>>> main
```

Giữ lại phần đúng (hoặc gộp cả hai), **xoá 3 dòng đánh dấu**, chạy lại code xem còn chạy không, rồi `git add` + `git commit`. Không chắc thì **hỏi người viết đoạn kia** trước khi chọn.

## 5. Pull Request

### Mở PR

- **Tiêu đề**: giống format commit, ví dụ `feat(labeling): thêm preannotate.py`.
- **Mô tả**: điền theo mẫu có sẵn (`.github/pull_request_template.md`, GitHub tự hiện khi mở PR):
  làm gì, cách kiểm tra, kết quả / ảnh chụp, liên quan tuần nào / mục § nào.
- **Reviewers**: chọn **Giang** (người duyệt duy nhất của nhóm, cũng là chủ repo).
- **Assignees**: chính mình.
- PR còn đang làm dở mà muốn Giang xem trước → chọn **"Create draft pull request"**; xong thì bấm **"Ready for review"**.
- Nhắn Giang trong nhóm chat kèm link PR để duyệt nhanh hơn.

**PR do chính Giang mở:** GitHub không cho tự duyệt PR của mình. Nhờ **1 bạn bất kỳ** xem qua và Approve (đặc biệt khi sửa kế hoạch, phân công, `.gitignore`, `requirements.txt`). Việc gấp hoặc chỉ sửa chữ thì Giang có thể merge luôn (quyền admin, xem §8).

### Duyệt PR (Giang)

Trong **24 giờ** sau khi được nhờ. Kiểm tra:

- [ ] Kéo branch về và **chạy thử** theo phần "Cách kiểm tra" trong PR (`git fetch` rồi `git switch <tên-branch>`).
- [ ] Không có dữ liệu, checkpoint, mật khẩu bị đưa lên (xem tab **Files changed**).
- [ ] Không sửa file của người khác mà không báo.
- [ ] Kết quả của người này khớp với thứ người khác cần (ví dụ: output của `cleaner.py` đúng định dạng `tokenizer.py` của Hằng đọc; export của Phúc đúng định dạng Dũng dùng để train).
- [ ] Đặt tên file / hàm / cột đúng với kế hoạch (ví dụ tên trường `span`, `skip`, `text` trong Label Studio).
- [ ] Notebook: chạy được từ đầu đến cuối, output không quá nặng (ảnh lớn / bảng hàng nghìn dòng thì xoá output).

Kết quả review:
- **Approve**: ổn, có thể merge.
- **Request changes**: ghi rõ cần sửa gì, tốt nhất kèm gợi ý. Người viết sửa, push, rồi bấm "Re-request review".
- **Comment**: góp ý nhỏ, không bắt buộc sửa.

### Merge

- Chỉ merge khi **Giang đã Approve** và không còn góp ý bắt buộc chưa xử lý.
- **Giang bấm merge** ngay sau khi Approve, chọn **"Squash and merge"** (gộp các commit thành 1, lịch sử `main` gọn), rồi bấm **"Delete branch"**.
- Sau khi merge, mọi người `git switch main && git pull` để cập nhật.

## 6. Notebook (`.ipynb`)

- **Mỗi người làm notebook của mình** (theo bảng vai trò). Notebook rất khó gộp khi 2 người cùng sửa.
- Muốn thử trên notebook của người khác → **tạo bản sao** (`05_deep_learning_dung_thu.ipynb`), không sửa bản gốc.
- Trước khi commit: *Restart & Run All* để chắc chạy được từ đầu; xoá output nặng.
- Đường dẫn dùng tương đối từ thư mục gốc dự án (`data/raw/tiki_reviews.csv`), không dùng đường dẫn riêng của máy mình (`C:\Users\...`, `/Users/...`). Notebook chạy trên Colab/Kaggle thì ghi rõ ô đầu tiên tải dữ liệu từ Drive thế nào.

## 7. Bảng lệnh nhanh

| Muốn | Lệnh |
|---|---|
| Xem đang ở branch nào, sửa gì | `git status` |
| Xem các branch | `git branch -a` |
| Chuyển sang branch của người khác để chạy thử | `git fetch` rồi `git switch <tên-branch>` |
| Xem thay đổi trước khi commit | `git diff` |
| Bỏ thay đổi chưa commit của 1 file | `git restore <file>` |
| Lỡ `git add` nhầm file | `git restore --staged <file>` |
| Sửa message commit vừa tạo (chưa push) | `git commit --amend -m "message mới"` |
| Xoá branch đã merge ở máy mình | `git branch -d <tên-branch>` |

**Đừng dùng** khi chưa hỏi cả nhóm: `git push --force`, `git reset --hard`, sửa lịch sử của `main`.

## 8. Khuyến nghị cài đặt trên GitHub (chủ repo làm 1 lần)

*Settings → Branches → Add branch protection rule* cho `main`:
- ✅ Require a pull request before merging, *Required approvals: 1*
- ⬜ **Không** tích *Do not allow bypassing the above settings*: để Giang (admin) vẫn merge được PR của chính mình khi gấp. Ba thành viên còn lại không phải admin nên vẫn bắt buộc phải có Giang duyệt.

*Settings → Collaborators*: thêm 3 bạn với quyền **Write** (không cấp Admin).

*Settings → General → Pull Requests*: chỉ bật **Allow squash merging**, bật **Automatically delete head branches**.
