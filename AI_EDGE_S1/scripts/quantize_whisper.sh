#!/usr/bin/env bash
# quantize_whisper.sh — Build whisper.cpp + quantize Whisper-Tiny sang Q5_0
# Target: Raspberry Pi 5 (ARM64, ARMv8.2-a + NEON + dotprod)
# Output: models/ggml-tiny-q5_0.bin (~30MB)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WHISPER_DIR="$PROJECT_ROOT/whisper.cpp"
MODEL_DIR="$PROJECT_ROOT/models"

echo "=== Step 1: Clone whisper.cpp ==="
if [ ! -d "$WHISPER_DIR" ]; then
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp "$WHISPER_DIR"
else
    echo "whisper.cpp already exists, pulling latest..."
    git -C "$WHISPER_DIR" pull --ff-only
fi

echo "=== Step 2: Build với ARM NEON + dotprod ==="
mkdir -p "$WHISPER_DIR/build"
cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod -O3 -ftree-vectorize"
cmake --build "$WHISPER_DIR/build" --target main quantize -j4

echo "=== Step 3: Download Whisper-Tiny gốc (FP32) ==="
mkdir -p "$MODEL_DIR"
bash "$WHISPER_DIR/models/download-ggml-model.sh" tiny "$MODEL_DIR"

FP32_MODEL="$MODEL_DIR/ggml-tiny.bin"
Q5_MODEL="$MODEL_DIR/ggml-tiny-q5_0.bin"

if [ ! -f "$FP32_MODEL" ]; then
    echo "ERROR: $FP32_MODEL not found after download"
    exit 1
fi

echo "=== Step 4: Quantize sang Q5_0 ==="
"$WHISPER_DIR/build/bin/quantize" "$FP32_MODEL" "$Q5_MODEL" q5_0

echo "=== Kết quả ==="
echo "FP32: $(du -sh "$FP32_MODEL" | cut -f1)"
echo "Q5_0: $(du -sh "$Q5_MODEL" | cut -f1)"
echo ""
echo "=== Step 5: Verify (optional) ==="
echo "Chạy lệnh sau để kiểm tra WER:"
echo "  $WHISPER_DIR/build/bin/main -m $Q5_MODEL -f <audio.wav> -l vi"
echo ""
echo "Q5_0 model sẵn sàng tại: $Q5_MODEL"
