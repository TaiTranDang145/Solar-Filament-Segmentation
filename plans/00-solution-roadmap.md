# Plan: Lộ trình giải Solar Filament Segmentation Challenge 2026

**Branch**: `plan/solar-filament-solution`
**Status**: Active

## Goal

Tạo một pipeline Kaggle tái lập được, tối ưu trực tiếp cho instance-level Panoptic Quality (PQ), xuất đúng CSV/RLE và đủ bằng chứng cho báo cáo cuối kỳ.

## Bằng chứng đầu vào đã kiểm tra

- 707 ảnh train, 180 ảnh test; toàn bộ là JPEG grayscale `2048 x 2048`, không có file hỏng và không trùng tên giữa train/test.
- COCO JSON có 1.154 image record, 8.199 filament instance và 707 tên ảnh vật lý duy nhất.
- 411 ảnh có 1 bộ annotation, 145 ảnh có 2 bộ, 151 ảnh có 3 bộ. Mọi split phải group theo `file_name`, không theo `image.id`.
- Số instance trên một annotation set: 1–26; diện tích median 1.228 px, p90 4.684 px, max 37.739 px. Filament rất nhỏ/mảnh so với toàn ảnh.
- Nhãn quan sát được: Left 2.535, Right 2.590, Unidentifiable 3.074, Ambiguous 0. Cuộc thi chỉ yêu cầu filament mask, nên baseline dùng một lớp `filament`.
- Train/test đều phủ 2011–2022 và sáu đài `Bh/Ch/Lh/Mh/Th/Uh`; phân bố tương đối gần nhau.
- Quan sát ảnh thật cho thấy limb, active region, vệt tối không phải filament, độ tương phản theo đài và filament dài/mảnh là các nguồn FP/FN chính.
- Submission là một hàng cho mỗi predicted instance, ID `<image_stem>_<unique_suffix>`, RLE COCO cho mask cố định `2048 x 2048`.

## Acceptance Criteria

- [ ] Một lệnh/notebook chạy từ dữ liệu thô đến `submission.csv` mà không cần chỉnh tay.
- [ ] Split local không có cùng `file_name` ở train và validation; thống kê fold được lưu lại.
- [ ] Local scorer khớp notebook self-evaluation của ban tổ chức trên fixture nhỏ.
- [ ] Mỗi thí nghiệm báo PQ, SQ/RQ hoặc thành phần tương đương, Dice, IoU, FP/FN, split/merge và thời gian inference.
- [ ] Có baseline end-to-end hợp lệ trước khi thử kiến trúc hoặc hậu xử lý phức tạp.
- [ ] Model cuối được chọn bằng out-of-fold evidence, không chỉ bằng một Kaggle public score.
- [ ] CSV cuối decode được, ID hợp lệ, không mask rỗng/trùng và phủ đúng 180 test image.
- [ ] Repository cuối có notebook toàn pipeline, `requirements.txt` ghim version và technical report 4 trang.

## Thứ tự thực hiện

1. [01-data-and-evaluation.md](01-data-and-evaluation.md): khóa data contract, split và scorer.
2. [02-baseline-and-experiments.md](02-baseline-and-experiments.md): dựng walking baseline, rồi chỉ cải thiện theo lỗi PQ đo được.
3. [03-inference-submission-and-report.md](03-inference-submission-and-report.md): khóa inference, RLE, submission và gói tái lập.

Mỗi plan là một cổng quyết định. Không bắt đầu sweep/model nâng cao trước khi cổng trước đạt acceptance criteria.

## Slice 1: Chấm được một dự đoán giả lập bằng đúng protocol local

**Value**: Người làm mô hình nhận được PQ và lỗi split/merge đáng tin cậy cho một ảnh thật.
**Path**: COCO JSON -> group ảnh/annotator -> mask instance -> scorer -> bảng metric và visualization.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `tdd`, `testing`; `refactoring` chỉ khi phát sinh trùng lặp; mutation testing `N/A` vì notebook/data pipeline, thay bằng fixture parity.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Các tiêu chí của [01-data-and-evaluation.md](01-data-and-evaluation.md) được xác nhận trước khi code.
**RED or preservation baseline**: Fixture cố tình chứa perfect match, FP, FN, một-to-many và many-to-one phải làm scorer hiện tại fail/chưa tồn tại.
**GREEN or preservation change**: Code tối thiểu parse dữ liệu, split theo file và cho metric kỳ vọng.
**REFACTOR**: Chỉ tách helper dùng chung cho train/scoring/submission sau khi có caller thứ hai.
**PRE-PR MUTATION or alternate evidence**: N/A; đối chiếu với notebook ban tổ chức và decode/encode round trip.
**PR-ready when**: Data audit, split manifest, fixture và scorer parity đều pass; người dùng duyệt commit.
**Slice complete when**: PR độc lập được merge.

## Slice 2: Có baseline end-to-end tạo instance masks hợp lệ

**Value**: Người dự thi có một mốc PQ tái lập và submission thử nghiệm hợp lệ.
**Path**: train image + annotation set -> train -> probability mask -> instance extraction -> local PQ -> CSV/RLE.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk sau Slice 1.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, thay bằng smoke train, fixed-seed validation và artifact checks.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Walking baseline trong [02-baseline-and-experiments.md](02-baseline-and-experiments.md) chạy hết trên một fold/subset và tạo CSV decode được.
**RED or preservation baseline**: Smoke test phải fail khi pipeline chưa tạo đủ mask/ID/RLE.
**GREEN or preservation change**: Một model semantic đơn giản cộng connected components; chưa thêm boundary head, ensemble hay TTA.
**REFACTOR**: N/A trừ khi notebook và training code thật sự dùng chung logic.
**PRE-PR MUTATION or alternate evidence**: N/A; dùng smoke training, metric report và submission validator.
**PR-ready when**: Baseline score, runtime, checkpoint và error gallery được lưu; người dùng duyệt commit.
**Slice complete when**: PR độc lập được merge.

## Slice 3: Nâng PQ bằng từng giả thuyết được ablation

**Value**: Model cuối cải thiện lỗi thật thay vì tích lũy thủ thuật không biết tác dụng.
**Path**: OOF predictions -> error buckets -> một thay đổi -> cùng fold/seed/budget -> quyết định keep/drop.
**Class**: Behavior change.
**Delivery**: Mỗi giả thuyết thắng là một PR độc lập; không stack mặc định.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, thay bằng controlled ablation và regression fixtures cho post-processing.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Chỉ giữ thay đổi tăng mean PQ hoặc cải thiện rõ một failure bucket mà không làm runtime vượt budget.
**RED or preservation baseline**: Báo cáo OOF chỉ ra failure bucket và metric mục tiêu trước khi thay code.
**GREEN or preservation change**: Áp dụng đúng một thay đổi nhỏ nhất giải quyết bucket đó.
**REFACTOR**: Xóa nhánh thí nghiệm thua; giữ config cuối ngắn và rõ.
**PRE-PR MUTATION or alternate evidence**: N/A; ablation cùng fold/seed và validator regression.
**PR-ready when**: Có bảng before/after, qualitative examples và quyết định keep/drop; người dùng duyệt commit.
**Slice complete when**: PR của thay đổi thắng được merge.

## Slice 4: Đóng gói model, submission và hồ sơ cuối

**Value**: Ban tổ chức có thể tái tạo prediction và đánh giá định lượng/định tính.
**Path**: frozen checkpoint/config -> test inference -> instance masks -> RLE CSV -> validation -> notebook/repo/report.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `testing`, `technical-writing`; mutation `N/A`, thay bằng clean-run reproduction và artifact validation.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Các tiêu chí trong [03-inference-submission-and-report.md](03-inference-submission-and-report.md) đều pass.
**RED or preservation baseline**: Clean Kaggle session chưa thể tái tạo chính xác checksum/artifact dự kiến.
**GREEN or preservation change**: Khóa version, path, seed và notebook tối thiểu để clean run thành công.
**REFACTOR**: Loại cell/debug artifact không cần cho reproduction.
**PRE-PR MUTATION or alternate evidence**: N/A; clean Kaggle execution, CSV decode audit và report checklist.
**PR-ready when**: Artifact cuối được tái tạo từ đầu, người dùng duyệt commit.
**Slice complete when**: PR được merge và submission/report được nộp.

## Pre-PR Quality Gate

1. Acceptance criteria của slice được xác nhận trước khi triển khai.
2. Test/smoke test và data/submission validators pass.
3. Lint/typecheck pass nếu repository đã cấu hình; không thêm framework chỉ để có gate.
4. Notebook chạy top-to-bottom trong môi trường sạch.
5. Artifact và metric report ghi seed, fold, config, checkpoint và runtime.

---
*Khi toàn bộ pipeline và hồ sơ đã nộp, xóa plan tạm này theo lifecycle của repository.*
