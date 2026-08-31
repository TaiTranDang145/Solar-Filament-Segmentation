# Plan: Baseline và Thí nghiệm Tối ưu PQ

**Branch**: `feat/model-experiments`
**Status**: Active

## Goal

Đạt một baseline instance segmentation đáng tin cậy rồi tăng PQ bằng các thí nghiệm một-biến, ưu tiên giữ filament mảnh liên tục và giảm split/merge.

## Acceptance Criteria

- [ ] Walking baseline train/infer trên một fold và tạo predicted instances tại full resolution.
- [ ] Baseline report có PQ, Dice/IoU, FP/FN, split/merge, runtime, peak memory và error gallery.
- [ ] Mỗi thí nghiệm có hypothesis, metric mục tiêu, fixed fold/seed/budget và quyết định keep/drop.
- [ ] Không dùng category Left/Right/Unidentifiable làm supervision trừ khi ablation chứng minh lợi ích cho filament masks.
- [ ] Không thêm ensemble/TTA/boundary head trước khi single-model baseline và post-processing được khóa.
- [ ] Model cuối được xác nhận ít nhất trên toàn bộ 5-fold OOF hoặc một protocol rút gọn đã ghi rõ rồi rerun full trước khi nộp.

## Kiến trúc thử theo thứ tự

1. **Walking baseline**: binary semantic segmentation, BCE + Dice, probability threshold, connected components, lọc area tối thiểu.
2. **Resolution/context**: so sánh whole-image resize với overlap tiles; stitch probability trước khi tách instance để tránh seam fragmentation.
3. **Backbone/loss**: chỉ thử một model pretrained mạnh hơn hoặc loss nhấn thin structures sau khi baseline lỗi FN mảnh rõ ràng.
4. **Instance separation**: boundary/distance target hoặc watershed chỉ khi many-to-one là failure bucket đáng kể.
5. **Continuity repair**: closing/bridge theo khoảng cách nhỏ chỉ khi one-to-many chiếm đáng kể; phải kiểm soát nguy cơ merge.
6. **Ensemble/TTA**: cuối cùng, chỉ giữ nếu gain PQ lớn hơn runtime/complexity cost.

## Slice 1: Walking baseline chứng minh toàn đường đi

**Value**: Người dự thi có score và submission-format prediction đầu tiên trong thời gian ngắn nhất.
**Path**: image + selected annotation set -> semantic target -> model -> probability -> threshold -> connected components -> PQ.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, dùng smoke train và deterministic post-processing fixtures.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Overfit được 2–4 ảnh nhỏ; one-fold run giảm loss; output mask đúng `2048 x 2048`; component IDs không overlap; report và gallery được lưu.
**RED or preservation baseline**: Tiny overfit và post-processing fixture fail trước implementation.
**GREEN or preservation change**: Dùng model/loss sẵn có trong PyTorch ecosystem; không viết architecture tùy biến.
**REFACTOR**: N/A cho đến khi pipeline chạy hết.
**PRE-PR MUTATION or alternate evidence**: N/A; tiny overfit, smoke fold và mask invariants.
**PR-ready when**: Baseline chạy top-to-bottom và config/checkpoint/report tái lập; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 2: Chọn resolution và sampling không làm mất filament mảnh

**Value**: Filament nhỏ/dài được nhìn đủ chi tiết mà vẫn giữ global context và vừa Kaggle GPU.
**Path**: full image -> resize hoặc overlapping crops -> model -> stitched probability -> instances -> bucketed PQ theo area/length.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk sau baseline.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, dùng stitch/seam fixtures và controlled ablation.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: So sánh ít nhất hai resolution/sampling policies cùng compute budget; báo PQ theo small/medium/large area và seam errors; chọn đúng một policy.
**RED or preservation baseline**: Synthetic filament cắt qua tile seam bị đứt hoặc double-count trước khi stitch đúng.
**GREEN or preservation change**: Overlap + weighted average probabilities; tách instance một lần sau stitch.
**REFACTOR**: Xóa policy thua khỏi đường chạy mặc định.
**PRE-PR MUTATION or alternate evidence**: N/A; seam fixtures và ablation.
**PR-ready when**: Policy thắng có gain/compute rõ ràng; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 3: Giảm FN/FP bằng đúng một thay đổi model hoặc loss

**Value**: Recall filament mảnh tăng mà không kéo theo nhiều dark-feature false positives.
**Path**: OOF error buckets -> một hypothesis -> retrain cùng protocol -> metric/error comparison.
**Class**: Behavior change.
**Delivery**: Mỗi hypothesis là một PR độc lập, chỉ PR thắng được giữ.
**Required implementation skills**: `tdd`, `testing`; mutation `N/A`, dùng fixed-seed ablation.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Chọn một trong backbone, loss hoặc augmentation mỗi run; report overall PQ và strata theo observatory/year/area; không kết luận từ train loss.
**RED or preservation baseline**: Failure gallery và bucket metric định nghĩa trước outcome cần cải thiện.
**GREEN or preservation change**: Thay đổi nhỏ nhất có thể; ưu tiên pretrained encoder và augmentations hình học/quang học thực tế.
**REFACTOR**: Drop code/config thua.
**PRE-PR MUTATION or alternate evidence**: N/A; controlled ablation trên cùng folds.
**PR-ready when**: Gain lặp lại trên nhiều fold hoặc bị drop có ghi lý do; người dùng duyệt commit.
**Slice complete when**: PR thắng được merge hoặc slice đóng với kết luận không cần thay đổi.

## Slice 4: Tối ưu instance extraction theo split/merge metrics

**Value**: Cùng probability map tạo ít fragmentation và over-merging hơn, tăng RQ/PQ.
**Path**: probability map -> threshold -> optional morphology/distance/watershed -> instances -> split/merge report.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `tdd`, `testing`; mutation testing phù hợp cho threshold/branch nhỏ hoặc dùng synthetic topology fixtures.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Tune threshold/min-area/connectivity trên OOF predictions, không retrain; synthetic close-filaments và broken-filament fixtures pass; keep chỉ khi PQ tăng ngoài một fold.
**RED or preservation baseline**: Một filament có gap nhỏ bị split và hai filament gần nhau bị merge trong fixture.
**GREEN or preservation change**: Connected components + tham số tối thiểu; thêm watershed/boundary cue chỉ nếu simple path không đạt.
**REFACTOR**: Một post-processing function canonical; xóa các nhánh thua.
**PRE-PR MUTATION or alternate evidence**: Mutate threshold, connectivity, min-area; hoặc exhaustive synthetic fixtures.
**PR-ready when**: Before/after PQ + split/merge + runtime được lưu; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Slice 5: Khóa model cuối bằng OOF và budget inference

**Value**: Chọn checkpoint/config có bằng chứng tổng quát và chạy được trong giới hạn Kaggle.
**Path**: fixed config -> 5 fold training/inference -> OOF aggregation -> final decision -> frozen config/checkpoints.
**Class**: Behavior change.
**Delivery**: Independent PR against trunk.
**Required implementation skills**: `testing`; mutation `N/A`, dùng full OOF and reproducibility checks.
**Reduction program**: N/A.
**Transition/terminal evidence**: N/A.
**Acceptance criteria**: Full OOF report; variance giữa folds; runtime/VRAM/disk; config immutable có checksum; ensemble chỉ nếu single model không đủ và gain rõ.
**RED or preservation baseline**: Candidate chưa có full-OOF evidence/checksum.
**GREEN or preservation change**: Train/evaluate config thắng, không thêm hypothesis mới trong freeze run.
**REFACTOR**: Xóa debug configs và artifact không dùng.
**PRE-PR MUTATION or alternate evidence**: N/A; full OOF, checksum và rerun sample inference.
**PR-ready when**: Model card ngắn và frozen artifacts đầy đủ; người dùng duyệt commit.
**Slice complete when**: PR được merge.

## Pre-PR Quality Gate

1. Một thay đổi mỗi ablation; cùng folds, seed và budget.
2. Report cả mean và per-fold; không cherry-pick ảnh đẹp.
3. Kiểm tra small-instance recall, split/merge và morphology gallery.
4. Mask invariants và inference timing pass.
5. Chỉ code/config thắng ở đường chạy mặc định.

---
*Đóng plan khi model/config/post-processing cuối đã freeze; kết quả bền vững chuyển vào README/report.*
