# Deep Dive: /dev/shm/ — RAM-based Filesystem cho AI Edge

## Bài test hỏi:
> "Nếu bắt buộc phải lưu ra file, bạn lưu ở thư mục nào để tốc độ tức thời 
> và không hao mòn thẻ nhớ?"

## Kết luận: `/dev/shm/`

## So sánh thư mục

| Thư mục | Backend | Tốc độ R/W | Hao mòn SD | Luôn tmpfs? |
|---------|---------|-----------|-----------|------------|
| `/dev/shm/` | tmpfs (RAM) | ~4-8 GB/s | ❌ KHÔNG | ✅ LUÔN |
| `/tmp/` | tùy distro | tùy | tùy | ❌ KHÔNG chắc |
| `/run/` | tmpfs | ~4-8 GB/s | ❌ | ✅ | 
| `/home/pi/` | ext4 (SD) | ~50 MB/s | ✅ CÓ | ❌ |

## Tại sao KHÔNG dùng /tmp/?

- **Raspberry Pi OS**: `/tmp/` = tmpfs ✅
- **Ubuntu Server ARM64**: `/tmp/` = ext4 trên disk ❌  
- **Debian minimal**: `/tmp/` = ext4 ❌

`/dev/shm/` theo chuẩn POSIX luôn là tmpfs, bất kể distro.

## Đặc điểm /dev/shm/

```bash
# Kiểm tra mount type
mount | grep shm
# → tmpfs on /dev/shm type tmpfs (rw,nosuid,nodev)

# Dung lượng mặc định = 50% RAM
df -h /dev/shm
# Pi5 4GB → /dev/shm = 2GB
# Pi5 8GB → /dev/shm = 4GB

# Audio file ~1MB → dùng < 0.05% dung lượng
```

## Wear Leveling vấn đề

MicroSD card: ~10,000-100,000 write cycles per cell.
Robot hoạt động 24/7, mỗi interaction ghi 1 file audio:
- 10 interactions/phút × 60 × 24 = 14,400 writes/ngày
- MicroSD hết tuổi thọ trong vài tháng!

`/dev/shm/` = 0 writes xuống SD → robot chạy mãi mãi.

## Code sử dụng

```python
import os

# Ghi TTS output vào /dev/shm/
TTS_TEMP = "/dev/shm/voice_pipeline_tts.raw"

with open(TTS_TEMP, "wb") as f:
    f.write(pcm_bytes)  # Tốc độ RAM (~GB/s)

# Đọc lại (nếu process khác cần)
with open(TTS_TEMP, "rb") as f:
    data = f.read()     # Tốc độ RAM

# Cleanup
os.remove(TTS_TEMP)    # Giải phóng RAM
```

## Lưu ý

- Prefix file name (`voice_pipeline_`) để tránh xung đột
- File mất khi reboot → đúng ý đồ cho file tạm
- Nếu cần persist → ghi vào SD card (chấp nhận wear)
