# Bảo vệ thiết kế AI Edge S2 — Q&A

## Câu 1 — Đọc trộn Anh-Việt với TTS dưới 100M params

Cách em chọn: **chuyển hết tiếng Anh thành phiên âm tiếng Việt ở tầng phần mềm**, trước khi đẩy vào TTS. Không động gì vào model.

thay sửa lexicon hoặc tokenizer của model, nhưng em nghĩ đây mới là cách hợp lý cho bài toán này. Để giải thích thì phải nhìn vào ràng buộc đề bài: TTS dưới 100M params, chạy trên Pi5, không có GPU để retrain. Nếu em đụng vào lexicon thì:

- Cần dataset song ngữ Anh-Việt aligned phoneme — ai chuẩn bị?
- Cần GPU để fine-tune — Pi5 chịu thua
- Mỗi lần thêm 1 thuật ngữ mới (ví dụ khách hàng yêu cầu thêm "BSD" hay "ESP32") = 1 release model mới

Trong khi đó, làm ở tầng text-norm thì thêm 1 dòng vào dictionary là xong. Live ngay không cần redeploy model. Đây là khác biệt rất lớn về khả năng vận hành thực tế.

Lúc đầu em lo cách này chậm vì regex trên string. Sau khi viết xong em test thử thì : 0.02 ms/call cho câu dài 50 từ. Pipeline TTS inference mất tầm 200ms thì text-norm chiếm 0.01% — bottleneck thật sự nằm ở TTS, không phải ở chỗ này.

Để regex nhanh được thế thì có vài thủ thuật. Cái quan trọng nhất là pre-compile regex 1 lần lúc startup, không recompile mỗi lần gọi:

```python
def __init__(self):
    # Compile 1 lần duy nhất
    self._tech_re = re.compile(r"\b(overcurrent|timeout|...)\b", re.IGNORECASE)
    self._unit_re = re.compile(r"(\d+\.?\d*)\s*(kWh|kHz|MHz|...)\b")
    self._acronym_lookup = {k.upper(): v for k, v in ACRONYM_MAP.items()}
```

Thứ hai là gộp 41 thuật ngữ kỹ thuật vào 1 regex duy nhất, thay vì 41 lần `re.sub` riêng. Python regex engine duyệt string 1 lần là xong. Thứ ba là dùng dict lookup O(1) trong callback, tránh loop O(n).

Về độ phủ: em build dictionary cho domain xe điện gồm 48 acronym (BMS, CAN, ECU...), 41 thuật ngữ (overcurrent, timeout, communication...), 14 đơn vị đo (V, A, °C, kWh...). Tổng cộng tầm 5KB Python code. Để so sánh, từ điển CMU tiếng Anh có 130k từ, nặng tầm 50MB. Nói cách khác em dùng cách **nhỏ hơn 10000 lần** mà vẫn đủ cho domain hẹp này.

Tất nhiên cách này có nhược điểm. Từ tiếng Anh nào không có trong dictionary thì sẽ rơi xuống fallback spell-out — đọc từng chữ cái kiểu "ti i ét ti" cho từ "test". Hơi máy móc nhưng vẫn hiểu được, và nếu thấy từ nào hay xuất hiện thì thêm vào dict là xong. Một vấn đề nữa là từ tiếng Việt không dấu (`loi`, `pin`, `xe`) sẽ bị tưởng nhầm là tiếng Anh và spell-out. em xử lý bằng `VIETNAMESE_PASSTHROUGH` frozenset 50 từ phổ biến — gặp các từ này thì giữ nguyên.

Còn về phát âm: phiên âm kiểu Việt hóa không hoàn hảo (ai mà nói chuẩn "ô-vơ-ca-rần" cho "overcurrent"). Nhưng mục tiêu là **người Việt nghe hiểu được trong xe đang chạy** — không phải native English pronunciation. Trong context xe điện, người dùng đã quen kiểu "BMS bi-em-ét" rồi, không ai chờ đợi giọng Mỹ chuẩn từ trợ lý ảo trên xe.

---

## Câu 2 — Cảnh báo khẩn cấp: nói nhanh + cao giọng, nhưng không tăng inference time?

Cái này VITS2 (kiến trúc của Valtec / VieNeu-TTS) cho em "freebie" rất đẹp. Trong forward pass, model expose 3 scalar parameters cho phép can thiệp ngữ điệu mà không thêm bất kỳ layer hay pass nào.

3 scalar đó là `length_scale`, `noise_scale`, và `noise_scale_w`:

- `length_scale` nhân vào duration prediction. Đặt 0.7 → đọc nhanh hơn 30%.
- `noise_scale` nhân vào std của latent z trong flow decoder. Thấp → pitch ổn định, ít "ngân nga". Nghe nghiêm túc hơn.
- `noise_scale_w` nhân vào std của stochastic duration predictor. Thấp → từng âm tiết đều nhịp.

Em build sẵn 3 profile cho 3 mức nghiêm trọng:

```python
PROSODY_PROFILES = {
    "normal":   {"length_scale": 1.0,  "noise_scale": 0.667, "noise_scale_w": 0.8},
    "warning":  {"length_scale": 0.85, "noise_scale": 0.5,   "noise_scale_w": 0.6},
    "critical": {"length_scale": 0.7,  "noise_scale": 0.3,   "noise_scale_w": 0.4},
}
```

Profile `critical` đọc nhanh 30%, pitch rất ổn định — nghe ra ngay là cảnh báo nghiêm túc, khác hẳn giọng mặc nhiên thư giãn của profile `normal`.

Lý do cách này không tăng inference time: trong forward pass VITS2, 3 params này chỉ là **3 phép nhân scalar**. Không thêm layer, không thêm pass, không thêm tensor op nào. Cost ở mức 3 cycles CPU, trong khi inference mất 200ms. 0% overhead đo được.

em có cân nhắc cách khác trước khi chốt:

- **Train multi-style** (1 model = 5 voice): model phình lên 3-5 lần, chậm hơn, RAM nhiều hơn. Pi5 không gánh nổi.
- **Post-process audio** (resample + pitch shift sau khi sinh waveform): thêm 30-50ms latency, mà output bị "robot hóa", nghe rất giả.
- **SSML tags**: cần TTS engine hỗ trợ SSML, mà Valtec/VieNeu không có sẵn. Tự implement parser cũng phải build pipeline mới.

Nên cuối cùng vẫn dùng VITS2 native — nó đã có sẵn cơ chế đó, sao không tận dụng.

Phần còn lại là tự động chọn profile. Em không bắt người dùng (hoặc upstream module) tag `[critical]` thủ công — detect tự động bằng keyword matching:

```python
class SeverityDetector:
    CRITICAL = frozenset(["khẩn cấp", "critical", "nguy hiểm",
                          "overheat", "shutdown", "emergency"])
    WARNING  = frozenset(["cảnh báo", "warning", "lỗi", "error",
                          "timeout", "fault"])

    @classmethod
    def detect(cls, text: str) -> str:
        t = text.lower()
        if any(kw in t for kw in cls.CRITICAL): return "critical"
        if any(kw in t for kw in cls.WARNING):  return "warning"
        return "normal"
```

Frozenset cho O(1) lookup. Đo thử thì dưới 0.1 ms/call. Thứ tự kiểm tra `critical` trước `warning` quan trọng — câu "lỗi nguy hiểm" có cả 2 keyword, em muốn nó được xếp `critical`.

Pipeline tổng thể chạy thế này:

```
"BMS overheat 85°C, shutdown khẩn cấp"
    ↓ detect severity                      → "critical"
    ↓ normalize text                       → "bi em ét ô-vơ-hít 85 độ xê, sát-đao khẩn cấp"
    ↓ get prosody profile                  → {0.7, 0.3, 0.4}
    ↓ TTS.synthesize(text, **prosody)
[audio: nhanh + nghiêm túc]
```

Cả chuỗi này xảy ra tuần tự trong **1 forward pass duy nhất** của TTS. Không có pass thứ hai.

---

## Câu 3 — Queue tràn vì user nói dài / phòng ồn ?

Câu trả lời là **cả 3 cơ chế** — VAD Timeout, Drop Frame, và Backpressure. Lúc đầu em cũng nghĩ chắc 1 trong 3 là đủ, nhưng đào sâu vào thì mỗi cái giải quyết 1 failure mode khác nhau.

Có 3 nguyên nhân khiến queue có thể tràn:

1. **User nói liên tục không nghỉ trên 10 giây.** Hoặc môi trường ồn (gió, động cơ) khiến VAD trigger liên tục, không bao giờ về trạng thái silence — speech_chunks tích lũy mãi trong producer, mà chưa kịp đẩy vào queue.
2. **Nhiều utterance ngắn dồn dập.** Ví dụ phòng họp đông người, 5-6 người cùng nói, mỗi câu 2-3 giây. Producer đẩy đều, Consumer xử lý không kịp, queue dày lên.
3. **CPU spike khi Consumer đang chạy ASR/TTS nặng.** Producer cũng cần CPU để VAD inference, nhưng không "nhường" thì cả 2 thread đều chậm, queue ùn ứ.

Với 3 nguyên nhân khác nhau thế thì phải có 3 cơ chế riêng. Đây là logic của 3 lớp.

**Lớp 1 — VAD Timeout 10 giây.** Khi 1 utterance kéo dài quá 10s mà chưa kết thúc, force-cut và đẩy phần đã có vào queue luôn. Reset state machine của VAD để bắt đầu utterance mới:

```python
elapsed = time.monotonic() - speech_start_time
if elapsed >= 10.0:
    self._enqueue_speech(speech_chunks)
    speech_chunks = []
    self._vad.reset()
    self._stats["vad_timeouts"] += 1
```

Tại sao 10 giây? Đây là con số em thử nghiệm — đủ dài cho hầu hết câu lệnh tự nhiên (xe điện, người dùng ít khi nói câu trên 10s liền), đủ ngắn để không tích quá nhiều RAM. Nếu user thực sự cần nói dài thì câu sẽ bị chia thành nhiều đoạn 10s — không lý tưởng nhưng còn hơn treo hệ thống.

**Lớp 2 — Bounded Queue + Drop Oldest.** Queue có maxsize=50. Khi đạt 80% (40 utterance đang chờ) → bỏ utterance cũ nhất, nhường chỗ cho mới. Nghe "tàn nhẫn" nhưng có lý do:

```python
threshold = int(50 * 0.8)
if self._audio_queue.qsize() >= threshold:
    dropped = self._audio_queue.get_nowait()
    del dropped
self._audio_queue.put(audio, timeout=0.1)
```

Lý do drop **OLDEST** chứ không phải **NEWEST**: trong context xe điện, người lái cần phản hồi câu vừa nói (ví dụ "tăng điều hòa") — không phải câu họ nói 30 giây trước. Dữ liệu mới luôn quan trọng hơn dữ liệu cũ. Nếu drop newest thì user sẽ gặp tình huống "ơ sao em vừa nói mà không phản hồi" — UX rất tệ.

**Lớp 3 — Adaptive Exponential Backoff.** Cái này là phần em tâm đắc nhất. Producer sleep tăng dần khi CPU vi phạm liên tục, nhưng reset về MIN ngay khi CPU ổn lại:

```
Background (chỉ VAD): CPU > 40% → sleep 5ms → 7.5ms → 11ms → ... max 50ms
Active (ASR + TTS):   CPU > 70% → sleep 2ms → 3ms → 4.5ms → ... max 20ms

CPU về ngưỡng → reset sleep về MIN ngay lập tức
```

Khác với sleep cố định: nếu sleep cố định 20ms thì khi CPU vừa hồi xong vẫn phải đợi 20ms — phí mất 20ms tiềm năng. Adaptive thì "kiềm chế" nhanh khi spike, nhưng "thả" cũng nhanh khi ổn.

Producer biết Consumer đang busy hay không qua `threading.Event`:

```python
self._cpu_gov.throttle_if_needed(
    is_active=self._consumer_busy.is_set(),
)
```

Khi Consumer đang ngủ (queue trống), Producer được dùng đến 40% CPU. Khi Consumer chạy ASR/TTS, Producer phải nhường, ngưỡng tổng nhảy lên 70%. Đề bài yêu cầu "Background ≤40%, Active ≤70%" — implementation này map 1:1 vào con số đó.

Một detail nhỏ nhưng quan trọng: shutdown sạch. `Queue.get(timeout=0.5)` thay vì blocking cứng → Consumer ngủ 0.5s rồi tỉnh, check `_running` flag. Khi cần stop, em `put(None)` vào queue làm sentinel — Consumer nhận None thì break loop:

```python
def stop(self):
    self._running.clear()
    self._audio_queue.put(None)
    self._consumer_thread.join(timeout=5.0)
```

Không có cái này thì khi muốn dừng pipeline, phải force-kill thread — mất sạch state, có thể leak resource (mic stream, file handle).

---

### Tác dụng phụ lên trải nghiệm người dùng

Đây là phần mà engineer hay quên: **mỗi lớp bảo vệ đều có giá phải trả về UX**. Nếu giấu chuyện đó thì người dùng cuối khó chịu mà không hiểu vì sao.

Khi user nói trên 10 giây liên tục → Lớp 1 cắt thành đoạn 10s. Phản hồi sẽ đến sớm hơn (không phải đợi user nói xong toàn bộ), nhưng câu dài bị chia → ASR có thể hiểu lệch context. Trong xe điện thì tình huống này hiếm — em chấp nhận trade-off.

Khi phòng quá ồn khiến VAD trigger liên tục → Lớp 1 timeout liên tiếp → 1-2 phản hồi đầu sẽ "lạc đề" (transcribe ra rác), sau đó VAD reset và tự ổn định. Ở đây có thể bổ sung lớp filter sau ASR — nếu confidence quá thấp thì bỏ qua.

Khi nhiều câu dồn dập (giả sử nhiều người trong xe cùng nói) → Lớp 2 drop câu cũ. User phải nói lại câu vừa bị drop. Tệ, nhưng còn hơn treo cả hệ.

Khi CPU spike → Lớp 3 sleep 5-50ms. Latency tăng nhẹ nhưng người dùng không cảm nhận được — phản xạ con người tầm 100ms, em thêm 50ms vào đó vẫn dưới ngưỡng cảm nhận.

Tóm lại triết lý là: **fail gracefully, never hang**. Worst case em mất 1-2 utterance, user nói lại. Còn không có lớp nào thì sau 1 ngày là OOM, xe phải reboot — không phải lựa chọn cho hệ thống chạy 24/7 trên xe điện.

---

### Tóm lại 3 câu

Câu 1: làm Regex/Rules ở tầng phần mềm. Model giữ nguyên 74.8M params, mỗi lần thêm thuật ngữ mới chỉ cần edit dict, ~0.02ms/call.

Câu 2: dùng 3 scalar params `length_scale`, `noise_scale`, `noise_scale_w` của VITS2. 3 phép nhân trong forward pass, 0% overhead. Map vào 3 profile normal/warning/critical, severity detect tự động bằng keyword matching.

Câu 3: defense-in-depth 3 lớp — VAD Timeout 10s + Drop Oldest @80% queue + Adaptive Backoff CPU. Mỗi lớp giải 1 failure mode, không lớp nào thừa. UX chấp nhận mất câu cũ và latency +5-50ms để đổi lại tính ổn định 24/7.
