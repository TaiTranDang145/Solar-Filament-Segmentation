# Vòng 2: giảm positive weight

- Giả thuyết: giảm `positive_weight` từ `12` xuống `8` để giảm false positive (`1567`).
- Giữ nguyên fold, 20 epochs, kích thước ảnh và lưới threshold/min-area để chỉ đo một biến.
- Chạy riêng `lower-positive-weight-8`, tối đa một GPU run và một submission.
- Chấp nhận khi internal PQ và organizer self-evaluation khớp trong `1e-6`, đồng thời PQ đạt ít nhất `0.1754241482`.
- Nếu không đạt gate thì không submit; dùng metrics để chọn vòng kế tiếp.
