# Plan: Inference, Submission và Hồ sơ Tái lập

**Branch**: `feat/final-submission`
**Status**: Active

## Goal

Chạy inference ổn định trên 180 test images, tạo CSV/RLE hợp lệ và đóng gói notebook/repository/report để ban tổ chức tái lập toàn bộ pipeline.

## Acceptance Criteria

- [ ] Inference dùng duy nhất H-alpha test images như luật thi; không dùng ground-truth metadata bị cấm.
- [ ] Probability được dựng về đúng full resolution trước instance extraction và RLE encoding.
- [ ] Mỗi predicted instance là mask nhị phân không rỗng, thuộc đúng một image stem và có ID duy nhất.
- [ ] `segmentation_rle` là COCO compressed RLE counts không kèm quotes/size; decode lại đúng mask `2048 x 2048`.
- [ ] CSV có đúng hai cột `filament_id,segmentation_rle`, không NaN, không duplicate ID và không chứa train image ID.
- [ ] Log nêu số instance/image, area distribution, runtime, lỗi/ảnh và checksum checkpoint/config/CSV.
- [ ] Notebook chạy top-to-bottom trên Kaggle GPU từ dataset attach đến final CSV.
- [ ] Public repository có `requirements.txt` ghim version, hướng dẫn chạy ngắn, checkpoint acquisition và license/attribution cần thiết.
- [ ] Technical report 4 trang mô tả preprocessing, model, training, validation, ablation, morphology, runtime và limitations.

## Slice 1: Inference một ảnh tạo instances ổn định tại full resolution

**Value**: Một test image đi qua đúng production path và cho masks sẵn sàng encode.
**Path**: JPEG -> preprocessing/tiles -> model -> stitch/TTA nếu đã duyệt -> probability -> post-processing -> instance masks.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, dùng golden sample/invariants.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Deterministic với cùng checkpoint/config; mọi mask đúng shape, dtype, non-empty, non-overlap theo policy; peak VRAM/runtime được ghi.
**RED or preservation baseline**: Golden sample/invariants fail khi chưa có inference path.
**GREEN or preservation change**: Tái sử dụng preprocessing và post-processing đã thắng OOF; không tạo đường logic thứ hai trong notebook.
**REFACTOR**: Chỉ gom shared predict function nếu cả validation và test cùng gọi.
**PRE-PR MUTATION or alternate evidence**: N/A; golden sample checksum và invariant checks.
**PR-ready when**: Sample inference/report/visualization pass; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 2: Encode RLE và validate submission end-to-end

**Value**: Kaggle nhận file đúng schema và masks không đổi qua encode/decode.
**Path**: instance masks -> pycocotools encode -> UTF-8 counts -> IDs -> CSV -> decode audit.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk sau Slice 1.
**Required implementation skills**: `tdd`, `testing`; mutation phù hợp cho encoder/ID validation hoặc dùng exhaustive tiny masks.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Tiny masks round-trip exact; Fortran-order/COCO convention đúng; suffix chỉ cần unique; validator fail loud với empty RLE, duplicate ID, sai stem, quotes hoặc decoded size sai.
**RED or preservation baseline**: Tiny asymmetric mask bắt lỗi C-vs-Fortran order; duplicate/bad ID fixtures fail.
**GREEN or preservation change**: Gọi trực tiếp `pycocotools.mask.encode`; không tự viết compressed RLE.
**REFACTOR**: Một encoder và một validator dùng cho local/sample/final.
**PRE-PR MUTATION or alternate evidence**: Mutate order/shape/ID uniqueness hoặc exhaustive round-trip fixtures.
**PR-ready when**: Sample CSV và full decode audit pass; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 3: Chạy full test inference và khóa artifact

**Value**: Có final candidate CSV truy vết được về đúng code, config và checkpoint.
**Path**: frozen artifacts -> 180 test images -> rows -> validator -> checksums + run manifest.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk sau Slice 2.
**Required implementation skills**: `testing`; mutation `N/A`, dùng full artifact audit.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Mọi test stem được xử lý, kể cả 0 prediction; không silent skip/retry; count/area outliers được review bằng gallery; run hoàn thành trong budget; CSV checksum được lưu.
**RED or preservation baseline**: Full-run manifest chưa phủ 180 ảnh hoặc validator chưa pass.
**GREEN or preservation change**: Resume/checkpoint run tối thiểu nếu Kaggle time limit cần; không thêm distributed system.
**REFACTOR**: Xóa output thử nghiệm khỏi final package.
**PRE-PR MUTATION or alternate evidence**: N/A; 180-image completeness, decode audit và checksum.
**PR-ready when**: Candidate artifact được khóa và review; người dùng duyệt commit/submission.
**Slice complete when**: PR merge và CSV được submit.

## Slice 4: Reproduce notebook trong Kaggle session sạch

**Value**: Ban tổ chức chạy notebook không cần file hoặc thao tác ngoài hướng dẫn.
**Path**: clean Kaggle GPU + attached dataset -> install pinned packages -> train/load -> infer -> CSV.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `testing`, `technical-writing`; mutation `N/A`, dùng clean-run evidence.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Không absolute local path; seed/config hiển thị; package versions ghi lại; output paths rõ; notebook chạy tuần tự; cách lấy checkpoint hợp lệ và public.
**RED or preservation baseline**: Clean session fail vì path/dependency/state ẩn.
**GREEN or preservation change**: Sửa đúng dependency/path/state thiếu; không dựng packaging framework.
**REFACTOR**: Gộp/xóa cell debug và markdown thừa.
**PRE-PR MUTATION or alternate evidence**: N/A; clean Kaggle execution log và final validator.
**PR-ready when**: Clean run thành công và artifact tương đương được tạo; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 5: Hoàn thiện repository và technical report

**Value**: Bài dự thi đủ điều kiện qualitative review và final evaluation.
**Path**: experiment records + final pipeline -> README/notebook/requirements/report PDF -> public access check.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `technical-writing`, `testing`; mutation `N/A`, dùng document/repository checklist.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Report 4 trang dùng template yêu cầu; repo public không cần approval; `requirements.txt` có package/version; notebook thể hiện toàn pipeline; morphology examples gồm success và failure; limitations trung thực.
**RED or preservation baseline**: Checklist chỉ ra item thiếu hoặc link private.
**GREEN or preservation change**: Bổ sung đúng item thiếu, không thêm tài liệu trùng lặp.
**REFACTOR**: README là entry point duy nhất; report giữ nội dung đánh giá, không sao chép toàn README.
**PRE-PR MUTATION or alternate evidence**: N/A; fresh-reader reproduction checklist, link/access check và PDF visual check.
**PR-ready when**: Tất cả final-submission checklist pass; người dùng duyệt commit và nộp form.
**Slice complete when**: PR merge, form/report/repository links đã nộp.

## Final Submission Checklist

1. Freeze code, config, folds, checkpoint và package versions.
2. Chạy full inference một lần từ frozen state; lưu run log/checksums.
3. Decode audit 100% CSV rows và kiểm tra đủ 180 test stems trong run manifest.
4. Review gallery của ảnh có 0, rất ít và rất nhiều predicted instances.
5. Submit candidate lên Kaggle; ghi public/private score nhưng không thay pipeline chỉ theo public leaderboard.
6. Chạy notebook clean session, public-access check repository và visual-check PDF.
7. Nộp Google form trước deadline 15/11/2026.

## Pre-PR Quality Gate

1. Unit/fixture checks cho inference, ID và RLE pass.
2. Full submission validator pass, không warning chưa giải thích.
3. Kaggle clean run pass trong resource budget.
4. README, requirements, notebook và report nhất quán cùng frozen config.
5. Người dùng duyệt trước commit và trước submission.

---
*Đóng plan sau khi CSV, public repository và report đã được nộp thành công.*
