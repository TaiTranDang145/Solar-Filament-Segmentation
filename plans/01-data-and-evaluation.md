# Plan: Data Contract, Split và Local Evaluation

**Branch**: `feat/data-evaluation`
**Status**: Active

## Goal

Biến MAGFiLO thành tập train/validation không rò rỉ và một local scorer phản ánh đúng PQ cùng lỗi fragmentation/over-merging.

## Acceptance Criteria

- [ ] Audit xác nhận count, schema, mode, kích thước, file tồn tại, polygon hợp lệ và category mapping.
- [ ] Đơn vị split là ảnh vật lý `file_name`; mọi annotation set của cùng ảnh nằm chung fold.
- [ ] Manifest 5-fold cố định seed, cân bằng gần đúng theo year, observatory, số instance và tổng area.
- [ ] Train loader dùng một annotation set hợp lệ tại một thời điểm, không union mù các annotator thành instance trùng nhau.
- [ ] `official_pq` cộng TP/FP/FN trên từng annotator-image đúng notebook v6; `macro_pq` được giữ riêng làm diagnostic không dùng thay leaderboard metric.
- [ ] Scorer có perfect/FP/FN/fragment/merge fixtures và khớp notebook self-evaluation chính thức.
- [ ] Report lưu PQ, mean Dice/IoU, count TP/FP/FN, one-to-many, many-to-one và latency/image.
- [ ] Visualization hiển thị raw image, GT theo annotator, prediction, TP/FP/FN và các case split/merge.

## Data Contract cần khóa

- Input image: single-channel uint8, `2048 x 2048`; chuẩn hóa giữ một kênh trừ khi backbone pretrained bắt buộc ba kênh lặp.
- Sample identity: `file_name`/stem là ảnh vật lý; `image.id` là một annotation set có tiền tố annotator batch.
- Target: danh sách binary instance masks một lớp `filament`; `category_id` không phải mục tiêu cuộc thi.
- Polygon: một polygon/instance, có thể rất mảnh; rasterize tại full resolution trước rồi mới crop/resize để tránh mất cấu trúc.
- Empty target: dataset hiện không có, nhưng inference có thể không dự đoán instance; scorer/submission phải xử lý rõ.

## Slice 1: Audit dữ liệu tạo một manifest chuẩn duy nhất

**Value**: Trainer và evaluator dùng cùng identity/path/annotation mapping đã kiểm chứng.
**Path**: filesystem + COCO JSON -> validation -> `manifest` -> summary table/error report.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, dùng malformed in-memory fixtures.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Count phải bằng 707/180/1.154/8.199; thiếu file, sai size/mode, polygon lẻ tọa độ, tọa độ ngoài ảnh hoặc orphan ID phải fail loud.
**RED or preservation baseline**: Fixture schema sai và file thiếu chưa bị phát hiện.
**GREEN or preservation change**: Một audit command/notebook section bằng stdlib + thư viện ảnh đang dùng; không xây data framework riêng.
**REFACTOR**: Chỉ giữ một canonical manifest builder.
**PRE-PR MUTATION or alternate evidence**: N/A; malformed fixtures và audit toàn dataset.
**PR-ready when**: Audit report lưu được và không có lỗi chưa giải thích; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 2: Tạo split group-aware tái lập

**Value**: Local score không thấy cùng pixel image ở cả train và validation.
**Path**: manifest -> physical-image groups -> fold assignment -> leakage/balance report.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk sau Slice 1.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, dùng invariant checks.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Giao `file_name` giữa mọi cặp train/val bằng rỗng; tất cả annotator set đi cùng group; cùng seed sinh cùng fold; không fold nào lệch nghiêm trọng về đài/year/object-count.
**RED or preservation baseline**: Fixture có hai `image.id` chung `file_name` phải làm split naïve rò rỉ.
**GREEN or preservation change**: Grouped 5-fold tối thiểu; chỉ stratify thêm nếu balance report chứng minh cần.
**REFACTOR**: N/A.
**PRE-PR MUTATION or alternate evidence**: N/A; invariants và checksum split manifest.
**PR-ready when**: Leakage check pass và fold report được chấp nhận; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 3: Rasterize target và xử lý multi-annotator đúng nghĩa

**Value**: Model nhận target không chồng lặp giả và local evaluation phản ánh bất đồng annotation.
**Path**: physical image -> chọn annotation set -> polygon rasterization -> instance masks/semantic union -> training sample.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk sau Slice 2.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, dùng polygon/mask round-trip fixtures.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Train có thể random/cycle một annotation set mỗi epoch; validation giữ tất cả set riêng; rasterized area sai khác với COCO `area` chỉ trong tolerance đã định; overlap policy được ghi rõ.
**RED or preservation baseline**: Polygon vuông/filament mảnh fixture không cho đúng area/bbox sau rasterization.
**GREEN or preservation change**: Dùng COCO/Pillow rasterizer chuẩn; không tự viết polygon algorithm.
**REFACTOR**: Chỉ tách semantic-union helper nếu baseline cần.
**PRE-PR MUTATION or alternate evidence**: N/A; round-trip và area/bbox parity.
**PR-ready when**: Sample visualization và parity checks pass; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 4: Local scorer khớp protocol tổ chức

**Value**: Một thay đổi model/post-processing có thể được quyết định bằng metric đúng.
**Path**: predicted instances + GT instances -> overlap matching -> TP/FP/FN -> PQ/Dice/IoU/split/merge -> report.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk sau Slice 3.
**Required implementation skills**: `tdd`, `testing`; mutation testing chỉ cho logic scorer nếu rẻ, nếu không dùng exhaustive tiny masks.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Không tự đoán IoU threshold/matching edge cases; lấy đúng logic từ notebook ban tổ chức. Tiny fixtures có kết quả tính tay và parity test pass.
**RED or preservation baseline**: Perfect, all-FP, all-FN, split và merge fixtures fail trước khi scorer tồn tại.
**GREEN or preservation change**: Port scorer tối thiểu từ notebook v6; giữ official annotator-image aggregation và một macro diagnostic riêng.
**REFACTOR**: Tách visualization khỏi metric core để scorer thuần và deterministic.
**PRE-PR MUTATION or alternate evidence**: Mutation tập trung các dấu `>`, `>=`, hệ số 0.5 và mẫu số; hoặc exhaustive enumeration tiny masks.
**PR-ready when**: Official parity, fixtures và aggregation checks pass; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Pre-PR Quality Gate

1. Không leakage theo `file_name`.
2. Dataset audit và polygon parity pass.
3. Official scorer parity pass.
4. Một report trên validation subset được tạo cùng visualization.
5. Không dùng test set để chỉnh threshold hoặc chọn model.

---
*Đóng plan sau khi manifest, folds, loader và scorer đã trở thành contract ổn định của repository.*
