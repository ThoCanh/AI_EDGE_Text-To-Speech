#!/bin/bash
# ══════════════════════════════════════════════════════
# setup_pi5.sh — Cài đặt môi trường trên Raspberry Pi 5
# ══════════════════════════════════════════════════════
set -e

echo "═══════════════════════════════════════════"
echo "  Voice Pipeline Setup — Raspberry Pi 5"
echo "═══════════════════════════════════════════"

# 1. System dependencies
echo "[1/5] Installing system dependencies..."
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  libportaudio2 libportaudio-dev \
  python3-pip python3-venv \
  alsa-utils

# 2. Python environment
echo "[2/5] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy sounddevice pywhispercpp

# 3. Build whisper.cpp (ARM NEON optimized)
echo "[3/5] Building whisper.cpp with ARM NEON..."
if [ ! -d "whisper.cpp" ]; then
  git clone https://github.com/ggerganov/whisper.cpp
fi
cd whisper.cpp
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_NO_ACCELERATE=ON \
  -DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod -O3 -ftree-vectorize"
make -j4
cd ../..

# 4. Download models
echo "[4/5] Downloading models..."
mkdir -p models
cd whisper.cpp
bash models/download-ggml-model.sh tiny
# Quantize to Q5_0
./build/bin/quantize \
  models/ggml-tiny.bin \
  ../models/ggml-tiny-q5_0.bin \
  q5_0
cd ..

# Download Piper TTS
echo "[4.5/5] Setting up Piper TTS..."
mkdir -p piper
cd piper
wget -q https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar xzf piper_arm64.tar.gz
rm piper_arm64.tar.gz
cd ..
# Download Vietnamese voice
wget -q -O models/vi_VN-vais1000-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN-vais1000-medium.onnx
wget -q -O models/vi_VN-vais1000-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN-vais1000-medium.onnx.json

# 5. Verify
echo "[5/5] Verifying setup..."
echo "Model files:"
ls -lh models/
echo ""
echo "whisper.cpp binary:"
./whisper.cpp/build/bin/main --help 2>&1 | head -5
echo ""
echo "Piper binary:"
./piper/piper --help 2>&1 | head -5

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo "  Run: python src/voice_pipeline.py"
echo "═══════════════════════════════════════════"
