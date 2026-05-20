# XEye

> XEye is a wearable, on-device AI assistant for the visually impaired, built on [Qualcomm Dragonwing™ QCS6490](https://www.qualcomm.com/internet-of-things/products/q6-series/qcs6490) Platform - [Thundercomm RUBIK Pi 3](https://rubikpi.ai/).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> XEye is currently among the **Top 28 shortlisted startups** of [**Qualcomm® Vietnam Innovation Challenge (QVIC) 2026**](https://www.qualcomm.com/company/locations/vietnam/vietnam-innovation-challenge#qvic-2026).  

## Pipeline

```
Mic → STT → Vietnamese question
                ↓
          VI→EN translation
                ↓
Image + EN question → VLM → EN answer
                ↓
          EN→VI translation
                ↓
          TTS → Speaker
```
## Hardware

| | |
|---|---|
| **Board** | [Thundercomm RUBIK Pi 3](https://rubikpi.ai/) |
| **SoC** | Qualcomm QCS6490 (4× Kryo A73 @ 2.7GHz + 4× Kryo A53 @ 1.9GHz) |
| **RAM** | 8GB LPDDR4x |
| **NPU** | Hexagon 780 HTP — 12 TOPS |
| **GPU** | Adreno 643 |
| **Storage** | 128GB UFS 3.1 |
| **OS** | Ubuntu 22.04 (LE) |

## Models

| Module | Model | Runtime | Quantization | Size | Speed |
|--------|-------|---------|--------------|------|-------|
| STT | [ZipFormer-30M RNNT](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) | ONNX Runtime | int8 | ~30MB | RTF 0.03x |
| VLM | [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | ONNX Runtime | FP32 | ~2.1GB | ~19 tok/s |
| NMT | [opus-mt-en-vi](https://huggingface.co/dekthedev/opus-mt-en-vi-ct2-int8) / [opus-mt-vi-en](https://huggingface.co/dekthedev/opus-mt-vi-en-ct2-int8) | CTranslate2 | int8 | ~145MB | — |
| TTS | [VieNeu-TTS-v2](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2) | llama-cpp | GGUF Q4-K-M | ~750MB | RTF ~2x |


> **Why CPU-only?**  
> The QCS6490's Hexagon NPU (12 TOPS) is designed for computer vision inference (object detection, classification) — not LLM/VLM workloads. It does not efficiently support attention mechanisms, dynamic KV cache, or large matrix multiplications required by language models. The Adreno GPU shares system RAM, making it unsuitable for models that require several GB of memory. After extensive testing across multiple approaches (llama.cpp Hexagon backend, ONNX Runtime QNN EP, Qualcomm AI Hub), CPU inference with highly quantized models was the only viable path on this hardware.

## Setup

```bash
# 1. Create environment and install dependencies
bash setup.sh

# 2. Activate environment
conda activate xeye

# 3. Download all models (~3.2GB total)
python download_models.py

# 4. Start server
python server.py
```

## Testing AI Services

Each script runs standalone — no server needed.

### STT

```bash
python scripts/stt_infer.py --file data/audio/audio.wav
```

### VLM

```bash
# Describe image (no question — uses default prompt)
python scripts/vlm_infer.py data/images/IMG_6817.jpg

# Ask a Vietnamese question
python scripts/vlm_infer.py data/images/IMG_6817.jpg --question "Đây là gì?"
```

### TTS

```bash
# Synthesize to file
python scripts/tts_infer.py "Xin chào" --out output.wav

# Synthesize with a different voice
python scripts/tts_infer.py "Xin chào" --out output.wav --voice Binh
```

## API Server

Runs at `http://0.0.0.0:8000`

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/health` | GET | — | `{"status": "ok"}` |
| `/stt` | POST | WAV file (multipart) | `{"text": "..."}` |
| `/vlm` | POST | `image` (file), `question` (Vietnamese text, optional) | `{"vi": "Vietnamese answer", "en": "English answer", "question_en": "translated question"}` |
| `/tts` | POST | `text` (Vietnamese), `voice` (name, optional) | WAV audio bytes |

### Example calls

```bash
# Health check
curl http://localhost:8000/health

# STT
curl -X POST http://localhost:8000/stt \
  -F "audio=@data/audio/test.wav"

# VLM
curl -X POST http://localhost:8000/vlm \
  -F "image=@data/images/IMG_6817.jpg" \
  -F "question=Đây là gì?"

# TTS
curl -X POST http://localhost:8000/tts \
  -F "text=Xin chào" \
  -F "voice=Ly" \
  --output response.wav
```

### TTS Voices

| Name | Description |
|------|-------------|
| `Ly` | Trúc Ly — nữ miền Bắc **(default)** |
| `Binh` | Thanh Bình — nam miền Bắc |
| `Tuyen` | Phạm Tuyên — nam miền Bắc |
| `Vinh` | Xuân Vĩnh — nam miền Nam |
| `Doan` | Thục Đoan — nữ miền Nam |
| `Sơn` | Thái Sơn — nam miền Nam |
| `Ngoc` | Bích Ngọc — nữ miền Bắc |

## Demo Pipeline (software-only)

No hardware required. Pass a WAV file (Vietnamese question) and an image file — get text output and a response audio file.

```bash
python pipeline.py data/audio/audio.wav --image data/images/IMG_6817.jpg
```

Optional flags:

```bash
python pipeline.py data/audio/audio.wav --image data/images/IMG_6817.jpg --output answer.wav --voice Binh
```

Output:
1. `[STT]` — transcribed Vietnamese question
2. `[VLM]` — translated question (EN) + answer (EN + VI)
3. Audio response saved to `output.wav` (default)

## Run as Service (PM2)

```bash
npm install -g pm2
pm2 start server.py --interpreter $(which python) --name xeye --cwd /home/ubuntu/xeye
pm2 save
pm2 startup
```

## Project Structure

```
xeye/
  server.py           # FastAPI server (STT / VLM / TTS endpoints)
  pipeline.py         # Demo pipeline (WAV + image → text + audio file)
  download_models.py  # Download all models from HuggingFace
  setup.sh            # Create conda env + install deps
  requirements.txt    # Python dependencies
  scripts/
    stt_infer.py      # STT: ZipFormer RNNT
    vlm_infer.py      # VLM: SmolVLM + NMT pipeline
    tts_infer.py      # TTS: VieNeu-TTS-v2
  models/
    vlm/              # SmolVLM-256M ONNX
    nmt/              # opus-mt-en-vi + opus-mt-vi-en
    stt/              # ZipFormer RNNT ONNX
    tts/              # VieNeu GGUF + voices
  data/
    images/           # Test images
    audio/            # Test audio
```
