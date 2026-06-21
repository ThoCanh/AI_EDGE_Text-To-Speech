# Documentation Q&A — Bảo vệ thiết kế AI Edge S1
## Câu 1 — Backend: `whisper.cpp` hay `sherpa-onnx`?

**Chọn: `whisper.cpp`**

### So sánh nhanh

| Tiêu chí | whisper.cpp | sherpa-onnx |
|----------|-------------|-------------|
| Binary size | ~2 MB | ~25 MB (kèm ORT) |
| Dependencies | 0 (pure C++) | ONNX Runtime + protobuf |
| Quantization | GGML (Q4/Q5/Q8) native | INT8 cần calibration |
| Memory footprint | ~50 MB runtime | ~150 MB runtime |
| ARM NEON tuning | có sẵn, tác giả tự tune | qua ORT (generic) |
| Build trên Pi5 | `make -j4`, ~2 phút | cmake + ORT, ~15 phút |
| Streaming | Có (incremental) | Có (Zipformer) |
| Whisper support | Chính chủ | Wrap qua converter |

### Lý do chính

**1. whisper.cpp được viết riêng cho Whisper.**
ggerganov tune từng kernel GEMM cho ARM NEON. Trong khi sherpa-onnx phải đi qua ONNX Runtime — một runtime generic, support trăm model — nên không thể tối ưu sâu cho từng kiến trúc.

**2. Quantization GGML với INT8.**
GGML có sẵn Q4_0, Q5_0, Q8_0 — convert 1 lệnh, không cần calibration dataset. Trong khi INT8 quantization cho ONNX cần representative dataset tiếng Việt (mà Whisper-Tiny không có sẵn). Xem thêm [deep_dive_q5_quantization.md](deep_dive_q5_quantization.md).

**3. Footprint RAM nhỏ → quan trọng cho Pi5.**
Pi5 4GB phải share RAM cho OS, audio driver, TTS, dashboard. whisper.cpp ăn ~50MB. sherpa-onnx ăn ~150MB chỉ riêng runtime. Khác biệt 100MB là khác biệt giữa "chạy mượt" và "swap đĩa".

**4. Zero dependency = robot reproducible.**
Build whisper.cpp xong, copy 1 binary `main` + 1 file `.bin` model là chạy. sherpa-onnx phải kéo theo ONNX Runtime shared lib, libprotobuf, libabseil... Mỗi lần update OS có thể vỡ.

### Khi nào sherpa-onnx tốt hơn?

- Cần chạy nhiều model khác nhau (Whisper + SenseVoice + Zipformer) trong cùng app → ORT share runtime
- Cần Streaming Zipformer (whisper.cpp hỗ trợ kém streaming với chunk < 1s)
- Đã có pipeline ONNX cho TTS/VAD → đỡ thêm runtime mới

Trong scope S1 (1 model Whisper, offline batch), whisper.cpp thắng tuyệt đối.

---

## Câu 2 — Thread allocation: `num_threads = ?`

**Chọn: `num_threads = 2`** (không phải 4, dù Pi5 có 4 nhân).

### Lý do tóm tắt

| Vấn đề | 4 threads | 2 threads |
|--------|-----------|-----------|
| L3 cache 2MB chia | 512KB/thread → thrash | 1MB/thread → ổn |
| Memory bandwidth 34 GB/s | 4 thread tranh nhau, queue dài | mỗi thread ~17 GB/s |
| Fork-join overhead/inference | ~2.25 ms | ~1.05 ms |
| Cores còn lại cho OS/audio | 0 | 2 |

### Tại sao 4 thread chậm hơn 2?

AI inference trên ARM CPU là **memory-bound**, không phải compute-bound. 80% thời gian forward pass là fetch weights từ RAM, 20% là tính. Khi tăng thread:

- **Compute song song hơn** (lợi)
- **Memory request nhân lên 4x** → memory controller bottleneck (hại)
- **Cache contention** → cùng L3 2MB chia 4 phần → mỗi thread thiếu cache → fetch lại từ RAM (hại)
- **Sync barrier nhiều hơn** → mỗi GEMM op fork-join 4 thread tốn hơn 2 thread (hại)

Tổng kết: lợi nhỏ < hại lớn → chậm hơn.

### Rule of thumb

> Small model (< 100M params) trên ARM SBC: `num_threads = N_cores / 2`

Reserve nửa core cho:
- ALSA audio callback (real-time priority cao, không được delay)
- TTS subprocess
- Python GIL + GC
- Kernel + dashboard

Chi tiết breakdown từng yếu tố: [deep_dive_num_threads.md](deep_dive_num_threads.md).

---

## Câu 3 — Lưu PCM ở đâu để I/O tức thời, không hao mòn SD?

**Chọn: `/dev/shm/`**

### So sánh thư mục Linux

| Thư mục | Backend | R/W | Hao mòn SD | POSIX chuẩn? |
|---------|---------|-----|-----------|--------------|
| `/dev/shm/` | tmpfs (RAM) | ~4-8 GB/s | ❌ | ✅ (luôn tmpfs) |
| `/tmp/` | tùy distro | tùy | tùy | ❌ |
| `/run/` | tmpfs | ~4-8 GB/s | ❌ | ⚠️ (thường tmpfs nhưng quota nhỏ) |
| `/home/` | ext4 (SD) | ~50 MB/s | ✅ | ❌ |

### Tại sao không dùng `/tmp/`?

- Raspberry Pi OS: `/tmp/` là tmpfs ✅
- Ubuntu Server ARM64: `/tmp/` là ext4 trên SD ❌
- Debian minimal: `/tmp/` là ext4 ❌

Cùng 1 dòng code `open("/tmp/audio.raw", "wb")` chạy ở Pi OS thì OK, deploy sang Ubuntu là hao SD ngay. `/dev/shm/` theo chuẩn POSIX **luôn luôn** là tmpfs — không phụ thuộc distro.

### Tại sao quan trọng với robot?

MicroSD chịu được ~10K-100K write cycles/cell. Robot 24/7:
- 10 lần tương tác/phút × 60 × 24 = **14,400 writes/ngày**
- → SD card chết trong vài tháng nếu ghi vào ext4
- `/dev/shm/` = 0 writes xuống SD → robot chạy đến khi RAM hỏng (thường năm thứ 10+)

### Code

```python
TTS_OUT = "/dev/shm/voice_pipeline_tts.raw"  # Prefix tránh xung đột

with open(TTS_OUT, "wb") as f:
    f.write(pcm_bytes)  # Tốc độ RAM ~GB/s

# Process khác đọc
with open(TTS_OUT, "rb") as f:
    data = f.read()

os.remove(TTS_OUT)  # Trả RAM ngay
```

Chi tiết: [deep_dive_dev_shm.md](deep_dive_dev_shm.md).

---

## Câu 4 — Quantization Whisper-Tiny: FP16, INT8, Q8, Q5, Q4?

**Chọn: `Q5_0`** (~30MB).

### Bảng so sánh

| Format | Size | WER delta (vi) | Tốc độ | Verdict |
|--------|------|---------------|--------|---------|
| FP32 (gốc) | 75 MB | 0% | baseline | tham chiếu |
| FP16 | 75 MB | ~0% | **chậm hơn FP32** | ❌ A76 không có FP16 ALU |
| INT8 (ONNX) | 40 MB | 1.5-3% | trung bình | ❌ cần calibration tiếng Việt |
| Q8_0 | 42 MB | ~0.5% | nhanh | ⚠️ an toàn nhưng không tối ưu |
| **Q5_0** | **30 MB** | **~1.5%** | **nhanh nhất** | **✅ sweet spot** |
| Q4_0 | 25 MB | 2.0-2.5% | nhanh hơn 5% | ❌ vượt ngưỡng WER 2% |

### Tại sao Q5 nhanh hơn Q8 (dù tính phức tạp hơn)?

Trực giác sai: "ít bit = compute phức tạp hơn = chậm hơn".

Thực tế trên ARM CPU:

```
Breakdown 1 forward pass Whisper-Tiny:
                    Q8_0          Q5_0
Memory fetch:       60% time      55% time   ← bottleneck thật
Dequantize:         5%            9%         ← rất nhỏ
Compute (GEMM):     35%           36%
Total:              2.0ms         1.6ms      ← Q5 nhanh hơn 20%
```

**Memory fetch là bottleneck.** Model Q5 nhỏ hơn 28% (30MB vs 42MB) → ít bytes fetch → ít cache miss → nhanh hơn. Chi phí dequantize 5→32 bit chỉ ~1-5ns/element, nhỏ hơn 100ns latency RAM 20-100x.

### Tại sao loại từng option khác

**FP16** — Cortex-A76 không có FP16 ALU native. Pipeline phải `LOAD fp16 → CVT fp32 → COMPUTE → CVT fp16 → STORE`. Overhead convert ~2 cycles/op khiến FP16 **chậm hơn FP32**, trong khi RAM tiết kiệm bằng 0 (cùng 75MB). ARMv8.4-A+ mới có `FEAT_FP16` native — A76 là ARMv8.2.

**INT8** — Cần calibration dataset đại diện cho tiếng Việt. Whisper-Tiny training data tiếng Việt đã ít → calibration không chuẩn → WER có thể nhảy 1.5-3%, rủi ro vượt ngưỡng. Thêm dependency ONNX Runtime nặng.

**Q8_0** — An toàn (WER ~0.5%) nhưng 42MB > L3 cache 2MB nhiều lần → mỗi inference phải fetch toàn bộ 42MB. Q5 nhỏ hơn 28% → bandwidth tiết kiệm tương ứng.

**Q4_0** — Whisper-Tiny chỉ 39M params (so với 1.5B của Whisper-Large). Model càng nhỏ, mỗi parameter mang càng nhiều thông tin → nén 4-bit mất nhiều hơn. Cộng với tiếng Việt là low-resource (~0.1% training data Whisper) → decoder weights tiếng Việt vốn đã mỏng → Q4 đẩy WER delta lên 2.0-2.5%, vượt ngưỡng đề bài (2%).

Chi tiết phân tích kiến trúc bộ nhớ A76 + benchmark: [deep_dive_q5_quantization.md](deep_dive_q5_quantization.md).

---

## TL;DR

| Câu | Trả lời |
|-----|---------|
| 1. Backend | `whisper.cpp` — nhỏ, không dependency, GGML quantization tốt |
| 2. num_threads | `2` — match memory bandwidth, reserve cores cho OS/audio |
| 3. File temp | `/dev/shm/` — tmpfs RAM, POSIX chuẩn, không hao SD |
| 4. Quantization | `Q5_0` — sweet spot 30MB, WER < 2%, nhanh nhất do memory-bound |
