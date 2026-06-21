# AI EDGE Text-To-Speech

Dự án này chứa các giải pháp pipeline giọng nói thông minh trên thiết bị biên (Edge Voice Pipeline), đặc biệt được tối ưu hóa cho phần cứng nhúng như Raspberry Pi 5.

Dự án bao gồm 2 phiên bản chính, được chia thành 2 thư mục:

## 1. [AI_EDGE_S1](./AI_EDGE_S1) - Push-to-Talk Pipeline
Hệ thống nhận dạng và tổng hợp giọng nói offline 100% cho robot hình người.
* **Mô hình hoạt động:** Bấm nút → nói → nhả nút → robot phản hồi bằng giọng nói.
* **Đặc điểm:** Chạy hoàn toàn trên ARM CPU, không cần GPU hay internet, tối ưu cực thấp độ trễ (RTF < 0.3) với `whisper.cpp` (ASR) và `Piper` (TTS).

👉 [Xem chi tiết AI_EDGE_S1](./AI_EDGE_S1/README.md)

## 2. [AI_EDGE_S2](./AI_EDGE_S2) - Always-On Assistant
Hệ thống trợ lý giọng nói luôn lắng nghe (always-on) cho bảng điều khiển xe điện thông minh (EV Dashboard).
* **Mô hình hoạt động:** Microphone luôn mở, tích hợp VAD để lọc tiếng ồn môi trường, tự động nhận diện và phản hồi lệnh thoại hỗn hợp Anh-Việt.
* **Đặc điểm:** Quản lý tài nguyên CPU cực kỳ chặt chẽ (background ≤ 40%, active ≤ 70%), sử dụng `Silero VAD`, `SenseVoiceSmall` (ASR) và `Valtec-TTS` (TTS), hỗ trợ Normalizer chuyển đổi thuật ngữ kỹ thuật tiếng Anh.

👉 [Xem chi tiết AI_EDGE_S2](./AI_EDGE_S2/README.md)
