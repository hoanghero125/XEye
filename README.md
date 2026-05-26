# XEye
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

> XEye is currently among the **Top 28 shortlisted startups** of [**Qualcomm® Vietnam Innovation Challenge (QVIC) 2026**](https://www.qualcomm.com/company/locations/vietnam/vietnam-innovation-challenge#qvic-2026).  

XEye is a wearable, on-device AI assistant for the visually impaired, built on [Qualcomm Dragonwing™ QCS6490](https://www.qualcomm.com/internet-of-things/products/q6-series/qcs6490) Platform - [Thundercomm RUBIK Pi 3](https://rubikpi.ai/).

**Technical Notes:** [English](docs/technical_report.md) · [Tiếng Việt](docs/bao_cao_ky_thuat.md) *(Last updated: 25/05/2026)*

## Pipeline

```
Mic ──→ STT ────→ VI question (text input)
                          │
     Camera ────→ Vision Language Model
                          │
                  VI answer (text output)
                          │
                         TTS ────→ VI answer (audio output) ────→ Speaker playback
```

## Hardware

| | |
|---|---|
| **Board** | [Thundercomm RUBIK Pi 3](https://rubikpi.ai/) |
| **SoC** | Qualcomm QCS6490 |
| **CPU** | 4× Cortex-A55 @1.96GHz + 3× Cortex-A78 @2.40GHz + 1× Cortex-X1 @2.71GHz |
| **RAM** | 8GB LPDDR4x |
| **NPU** | Hexagon 780 (V73) — 12 TOPS |
| **GPU** | Adreno 643 |
| **OS** | Ubuntu (Linux 6.8.0-1071-qcom) |

## Models

| Module | Model | Runtime | Quantization | Performance |
|--------|-------|---------|--------------|-------------|
| STT | [ZipFormer-30M RNNT](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) | sherpa-onnx | int8 | RTF <0.1x |
| VLM | [Vintern-1B-v3_5](https://huggingface.co/dekthedev/Vintern-1B-v3_5-GGUF) | llama-cpp-python | Q4_K_M | ~2.8-3.0 tok/s |
| TTS | [VieNeu-TTS-v2-Turbo](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF) | llama-cpp + VieNeu-Codec ONNX | Q4_K_M | ~4-5s latency |

> **Why CPU-only?**  
> The QCS6490's Hexagon NPU (12 TOPS) is designed for computer vision inference (object detection, classification) — not LLM/VLM workloads. It does not efficiently support attention mechanisms, dynamic KV cache, or large matrix multiplications required by language models. The Adreno GPU shares system RAM, making it unsuitable for models that require several GB of memory. After extensive testing across multiple approaches (llama.cpp Hexagon backend, ONNX Runtime QNN EP, Qualcomm AI Hub), CPU inference with highly quantized models was the only viable path on this hardware.

## Setup

```bash
# 1. Create environment and install dependencies
bash setup.sh

# 2. Activate environment
conda activate xeye

# 3. Download all models (~1.3GB total)
python download_models.py

# 4. Start server
python server.py
```

## Testing AI Services

Server must be running before using any script (`python server.py`).

### STT

```bash
python scripts/stt_infer.py --file data/audio/question.wav
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
python scripts/tts_infer.py "Xin chào" --out data/audio/output.wav

# Synthesize with a different voice
python scripts/tts_infer.py "Xin chào" --out data/audio/output.wav --voice "Phạm Tuyên (Nam - Miền Bắc)"
```

## API Server

Runs at `http://0.0.0.0:8000`

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/health` | GET | — | `{"status": "ok"}` |
| `/stt` | POST | WAV file (multipart) | `{"text": "..."}` |
| `/vlm` | POST | `image` (file), `question` (Vietnamese, optional) | `{"vi": "...", "tokens": N, "elapsed_s": N, "tok_s": N}` |
| `/tts` | POST | `text` (Vietnamese), `voice` (name, optional) | WAV audio bytes |

### Example calls

```bash
# Health check
curl http://localhost:8000/health

# STT
curl -X POST http://localhost:8000/stt \
  -F "audio=@data/audio/question.wav"

# VLM
curl -X POST http://localhost:8000/vlm \
  -F "image=@data/images/IMG_6817.jpg" \
  -F "question=Đây là gì?"

# TTS
curl -X POST http://localhost:8000/tts \
  -F "text=Xin chào" \
  -F "voice=Bích Ngọc (Nữ - Miền Bắc)" \
  --output response.wav
```

### TTS Voices

| Name | Gender | Dialect |
|------|--------|---------|
| `Bích Ngọc (Nữ - Miền Bắc)` | Female | Northern **(default)** |
| `Phạm Tuyên (Nam - Miền Bắc)` | Male | Northern |
| `Thục Đoan (Nữ - Miền Nam)` | Female | Southern |
| `Xuân Vĩnh (Nam - Miền Nam)` | Male | Southern |

## Demo

```bash
python pipeline.py demo/audio/question.wav --image demo/images/IMG_6817.jpg
```

**Input image:**

![Demo image](demo/images/IMG_6817.jpg)

**1. STT - Audio input:**

<video src="demo/video/question.mp4" controls width="480"></video>

```
[STT] Transcribing ...
[STT] 'mô tả khung cảnh trước mặt tôi'  (0.13s)
```

**2. VLM - image analysis:**

```
[VLM] Analyzing image ...
[VLM] Đây là một phòng họp với một người đàn ông đang ngồi trước màn hình máy tính. Trên bàn có một máy tính xách tay
[VLM] 29 tokens | 16.68s | 1.7 tok/s  (16.78s total)
```

**3. TTS — audio output:**

```
[TTS] Synthesizing ...
[TTS] Saved → data/audio/output.wav  (4.46s)
```

<video src="demo/video/output.mp4" controls width="480"></video>


```
[pipeline] Total: 21.38s
```




## Run as Service (PM2)

```bash
npm install -g pm2
pm2 start server.py --interpreter $(which python) --name xeye --cwd /home/ubuntu/xeye
pm2 save
pm2 startup
```

## Project Structure


```
xeye
├─ demo
│  ├─ audio/                    # Demo WAV files
│  ├─ images/                   # Demo images
│  └─ video/                    # Generated MP4s for README
├─ docs
│  ├─ bao_cao_ky_thuat.md       # Technical report (Vietnamese)
│  └─ technical_report.md       # Technical report (English)
├─ models
│  ├─ vintern/                  # Vintern-1B-v3_5 GGUF + mmproj
│  ├─ stt/                      # ZipFormer RNNT int8 ONNX
│  └─ tts/                      # VieNeu-TTS-v2-Turbo GGUF + voices
├─ scripts
│  ├─ stt_infer.py              # STT test script
│  ├─ vlm_infer.py              # VLM test script
│  └─ tts_infer.py              # TTS test script
├─ server.py                    # FastAPI server (STT / VLM / TTS endpoints)
├─ pipeline.py                  # Demo pipeline (WAV + image → text + audio)
├─ download_models.py           # Download all models from HuggingFace
├─ setup.sh                     # Create conda env + install deps
└─ requirements.txt
```