# Technical Report: XEye

**Last updated:** 25/05/2026 
**Author:** Do Pham Bao Hoang

---

## Table of Contents

- [1. System Information](#1-system-information)
- [2. Speech-to-Text (STT)](#2-speech-to-text-stt)
  - [2.1. Model Used](#21-model-used)
- [3. Text-to-Speech (TTS)](#3-text-to-speech-tts)
  - [3.1. Model Used](#31-model-used)
  - [3.2. Experiment History](#32-experiment-history)
    - [3.2.1. Experiment 1](#321-experiment-1)
    - [3.2.2. Experiment 2: Finding a Better Model](#322-experiment-2-finding-a-better-model)
    - [3.2.3. Experiment 3: Optimization](#323-experiment-3-optimization)
- [4. Vision-Language Model (VLM)](#4-vision-language-model-vlm)
  - [4.1. Model Used](#41-model-used)
  - [4.2. Experiment History](#42-experiment-history)
    - [4.2.1. Experiment 1](#421-experiment-1)
    - [4.2.2. Experiment 2: GPU Offload](#422-experiment-2-gpu-offload)
    - [4.2.3. Experiment 3: NPU Offload](#423-experiment-3-npu-offload)
    - [4.2.4. Experiment 4: ONNX + QNN Execution Provider](#424-experiment-4-onnx--qnn-execution-provider)
    - [4.2.5. Experiment 5: Smaller Model](#425-experiment-5-smaller-model)
    - [4.2.6. Experiment 6: Pipeline Optimization](#426-experiment-6-pipeline-optimization)
    - [4.2.7. Experiment 7: Better Model](#427-experiment-7-better-model)
    - [4.2.8. Experiment 8: Vintern-1B](#428-experiment-8-vintern-1b)
    - [4.2.9. Experiment 9: GPU Vulkan Offload](#429-experiment-9-gpu-vulkan-offload)
- [5. System Optimization](#5-system-optimization)
  - [5.1. Thread Affinity](#51-thread-affinity)
- [6. Summary](#6-summary)
  - [6.1. Selected Models](#61-selected-models)
  - [6.2. Current Limitations](#62-current-limitations)

---

## 1. System Information

| Component | Specification |
|-----------|---------------|
| Board | Thundercomm RUBIK Pi 3 |
| SoC | Qualcomm QCS6490 |
| RAM | 8GB LPDDR4x |
| CPU | 4× Cortex-A55 @1.96GHz + 3× Cortex-A78 @2.40GHz + 1× Cortex-X1 @2.71GHz |
| GPU | Adreno 643 |
| NPU | Hexagon 780 (V73), 12 TOPS |
| OS | Ubuntu (Linux 6.8.0-1071-qcom) |

---

## 2. Speech-to-Text (STT)

### 2.1. Model Used

#### 2.1.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | ZipFormer-30M RNNT |
| Params | ~30M |
| Quantization | int8 |
| Runtime | sherpa-onnx |
| Threads | 4 |
| Decoding Method | greedy search |
| Sample Rate | 16000 Hz |

#### 2.1.2. Performance

| Metric | Value |
|--------|-------|
| Latency (short utterance) | 0.07-0.17s |
| RTF | <0.1x |

---

## 3. Text-to-Speech (TTS)

### 3.1. Model Used

#### 3.1.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | VieNeu-TTS-v2-Turbo |
| Params | ~111M |
| Quantization | Q4_K_M |
| Format | GGUF |
| Codec | VieNeu-Codec (encoder + decoder ONNX) |
| Voice Format | 128-dim speaker embedding |
| Threads | 4 |
| Sample Rate | 24000 Hz |

#### 3.1.2. Available Voices

| Name | Gender | Dialect |
|------|--------|---------|
| Bích Ngọc | Female | Northern |
| Phạm Tuyên | Male | Northern |
| Thục Đoan | Female | Southern |
| Xuân Vĩnh | Male | Southern |

#### 3.1.3. Performance

| Metric | Value |
|--------|-------|
| Latency (short text) | ~4-5s |
| Speed vs. Standard | ~2x |

---

### 3.2. Experiment History

#### 3.2.1. Experiment 1

##### 3.2.1.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | OmniVoice Vietnamese |
| Quantization | float16 |
| Sample Rate | 24000 Hz |
| RTF (CUDA) | ~0.07 |

##### 3.2.1.2. Results

Model designed for CUDA — no effective CPU fallback.

→ CPU latency does not meet real-time requirements.

---

#### 3.2.2. Experiment 2: Finding a Better Model

##### 3.2.2.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | VieNeu-TTS-v2 Standard |
| Params | ~270M |
| Quantization | Q4_K_M |
| Format | GGUF |
| Codec | neuphonic neucodec ONNX int8 |
| Voice Format | codec token refs |

##### 3.2.2.2. Results

Acceptable latency, but speed can be further optimized.

---

#### 3.2.3. Experiment 3: Optimization

##### 3.2.3.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | VieNeu-TTS-v2-Turbo |
| Params | ~111M |
| Quantization | Q4_K_M |
| Format | GGUF |
| Codec | VieNeu-Codec (encoder + decoder ONNX) |
| Voice Format | 128-dim speaker embedding |

##### 3.2.3.2. Results

~2x faster.

---

## 4. Vision-Language Model (VLM)

### 4.1. Model Used

#### 4.1.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | Vintern-1B-v3_5 (InternVL2.5-1B fine-tuned Vietnamese) |
| Params | ~1B |
| Quantization | Q4_K_M |
| Format | GGUF |
| mmproj | ~620MB, F16 |
| Runtime | llama-cpp-python 0.3.16 |
| Threads | 4 |
| Context | 2048 |

#### 4.1.2. Performance

| Metric | Value |
|--------|-------|
| Throughput | ~2.8-3.0 tok/s |
| Language | Vietnamese |

---

### 4.2. Experiment History

#### 4.2.1. Experiment 1

##### 4.2.1.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | Qwen2.5-VL-3B-Instruct |
| Params | ~3B |
| Quantization | Q4_K_M |
| Format | GGUF |
| Runtime | llama.cpp |

##### 4.2.1.2. Results

3B model too large for on-device inference.

→ Token generation speed does not meet practical use threshold.

---

#### 4.2.2. Experiment 2: GPU Offload

##### 4.2.2.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | Qwen2.5-VL-3B-Instruct |
| Params | ~3B |
| Quantization | Q4_K_M |
| Format | GGUF |
| Runtime | llama.cpp Vulkan (GPU) |

##### 4.2.2.2. Results

Model + runtime overhead exceeds hardware limits.

→ OOM (Out of Memory) → Crash.

---

#### 4.2.3. Experiment 3: NPU Offload

##### 4.2.3.1. llama.cpp Hexagon Backend

Built `libggml-hexagon.so` + `libggml-htp-v73.so` from llama.cpp source, loaded into runtime.

During inference, backend loaded successfully but could not execute core LLM ops — attention, KV cache, and dynamic-shape matrix multiply were all rejected by Hexagon HTP. All computation still ran on CPU; no ops offloaded to NPU.

Hexagon 780 (V73) is designed for fixed-shape CV ops (conv, pool, activation) — not dynamic-shape transformer ops.

##### 4.2.3.2. Confirmation — Qualcomm AI Hub

Checked AI Hub: Qwen2-VL listed but no deployment available for target device. Qualcomm confirms Hexagon NPU is not designed for LLM inference — suitable for computer vision workloads only.

##### 4.2.3.3. Results

NPU not viable for LLM/VLM inference.

---

#### 4.2.4. Experiment 4: ONNX + QNN Execution Provider

> Qualcomm Neural Processing SDK (QNN SDK) is Qualcomm's official SDK for running inference on their accelerators — Hexagon NPU, Adreno GPU, CPU. ONNX Runtime integrates QNN SDK via the QNN Execution Provider (QNN EP), enabling offload of ONNX graph nodes to NPU.

→ Goal: bypass Hexagon NPU's lack of LLM op support via graph-level offload.

##### 4.2.4.1. Setup: ONNX + QNN EP

Export model to ONNX, run via ONNX Runtime with QNN EP to let it decide which ops to offload to NPU.

###### Issue: `backendValidateOpConfig` Bug (error 3110)

QNN SDK v2.43 bug: `backendValidateOpConfig` always returns error 3110, causing all ops to be rejected and fall back to CPU EP.

→ Fix: wrote custom shim `libQnnHtpShim.so` — overrides `backendValidateOpConfig` to always return `QNN_SUCCESS`, bypassing the bug.

##### 4.2.4.2. Qwen2.5-VL-3B ONNX (HuggingFace)

Tried the pre-built ONNX from HuggingFace first.

→ Only 36/1084 nodes run on QNN (trivial reshape/cast ops). Speed: 0.1 tok/s.

###### Issue: QNN EP cannot implement fused ORT extension ops (`MatMulNBits`, `GroupQueryAttention`).

##### 4.2.4.3. Custom Export Qwen2.5-VL-3B FP32 ONNX

Re-exported from PyTorch source with standard ONNX ops so QNN can accept them.

###### Issue: FP32 decoder ~17GB → cannot load on device.

##### 4.2.4.4. Quantize INT8 Qwen2.5-VL-3B

Quantized to reduce size to a loadable level. Decoder INT8 ~2.9GB.

###### Issue: ORT dequantizes to FP32 at load time → cannot load on device.

→ Fix: memory optimization (disabled oomd, overcommit, 4GB swap, 512MB zram) → still OOM.

→ 3B model not feasible.

##### 4.2.4.5. Qwen2-VL-2B Export + Deploy

Exported via same process. Decoder 1.5GB → loads successfully.

###### Issue: CPU EP still 0.1 tok/s. No NPU offload means no speed improvement.

##### 4.2.4.6. QNN EP for Qwen2-VL-2B

Applied QNN EP to 2B model.

###### Issue: OOM during Graph Optimization.

→ Fix: dropped `lm_head` to reduce memory → compiled successfully → inference still OOM.

Measured DDR bandwidth to verify NPU execution:
```
DDR write: 10KB  |  DDR read: 12KB
```
Near zero → QNN executed no ops.

→ Root cause: `DequantizeLinear` nodes cause CPU EP to pre-allocate FP32 buffers ~5GB before inference starts.

##### 4.2.4.7. Re-export with `do_constant_folding=True`

Tried folding `DequantizeLinear` nodes with constant inputs to reduce their count.

→ 2357 → 2352 nodes → negligible.

→ QNN still rejects ops.

##### 4.2.4.8. Results

0 ops offloaded to NPU after all attempts. Root cause: architectural mismatch — QNN EP designed for fixed-shape CV graphs, not dynamic-shape transformers with attention and KV cache.

---

#### 4.2.5. Experiment 5: Smaller Model

##### 4.2.5.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | SmolVLM-256M |
| Params | ~256M |
| Quantization | FP32 |
| Format | ONNX |
| Runtime | onnxruntime (CPU) |

##### 4.2.5.2. Results

~1.5 tok/s.

###### Issue: Model outputs English only and tends to hallucinate — generating overly long responses.

→ Fix: capped `max_tokens` + added instruction prompt requiring brief responses. Added MT model for text input/output.

---

#### 4.2.6. Experiment 6: Pipeline Optimization

##### 4.2.6.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | SmolVLM-256M + opus-mt-en-vi |
| Pipeline | Image → SmolVLM-256M (EN) → opus-mt-en-vi → Vietnamese |

##### 4.2.6.2. Results

~1.5 tok/s decode. Vietnamese text output achieved.

###### Issue: Output quality unnatural.

---

#### 4.2.7. Experiment 7: Better Model

##### 4.2.7.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | LFM2-VL-450M |
| Encoder | SigLIP2, 86M params |
| Decoder | LFM2, 350M params |
| Architecture | Hybrid (conv layers + full attention) |
| Quantization | int8 |
| Format | ONNX |

##### 4.2.7.2. Performance

| Metric | Value |
|--------|-------|
| Throughput | 1.2 tok/s |

###### Issue: Speed comparable to SmolVLM but output quality not significantly better.

---

#### 4.2.8. Experiment 8: Vintern-1B

##### 4.2.8.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | Vintern-1B-v3_5 (InternVL2.5-1B fine-tuned Vietnamese) |
| Backbone | ~462MB |
| Quantization | Q4_K_M |
| Format | GGUF |
| mmproj | ~620MB, F16 |
| Runtime | llama-cpp-python 0.3.16 |

##### 4.2.8.2. Issues & Solutions

`Llava16ChatHandler` uses `USER: ... ASSISTANT:` format → garbled output with Vintern.

Fix: subclassed `Llava15ChatHandler` — image injection occurs at C-level, overriding `CHAT_FORMAT` with Jinja2 ChatML:

```python
class InternVL2ChatHandler(Llava15ChatHandler):
    DEFAULT_SYSTEM_MESSAGE = None
    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "<|im_start|>user\n"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}{{ content.image_url.url }}\n{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}{{ content.text }}{% endif %}"
        "{% endfor %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
```

mmproj F16 ~620MB is the correct size — InternViT-300M × 2 bytes/param ≈ 600MB.

##### 4.2.8.3. Results

Vietnamese output quality significantly improved (without NMT) with better speed.

---

#### 4.2.9. Experiment 9: GPU Vulkan Offload

Offload VLM inference to GPU to reduce latency.

##### 4.2.9.1. Configuration

| Attribute | Value |
|-----------|-------|
| Driver | Turnip (Mesa open-source Adreno) |
| Vulkan | 1.4.318 |
| Build | llama-cpp `-DGGML_VULKAN=ON` + `glslc` (shaderc) |

##### 4.2.9.2. Results

```
MESA: error: Compute shader which has workgroup barrier cannot be used 
because it's impossible to have enough concurrent waves.
```

Turnip does not support compute shaders with workgroup barriers — required for clip/mmproj image encoder in llama.cpp. Qualcomm proprietary driver may support this but has no Linux release.

→ GPU Vulkan blocked by Turnip driver limitation. Reverted to CPU-only.

---

## 5. System Optimization

### 5.1. Thread Affinity

Mixing performance and efficiency cores in the same thread pool creates stall barriers, reducing throughput.

**Issue with process-level pinning:**

Pinning the entire process to performance cores restricts uvicorn/FastAPI threads too, competing with inference threads → total pipeline ≥22s (worse than unpinned <22s).

**Fix:** Pin affinity before model load so inference thread pools inherit the mask, then restore before uvicorn handles requests:

```python
os.sched_setaffinity(0, {4, 5, 6, 7})
models["stt"] = STTPipeline()   # inference threads inherit {4,5,6,7}
models["vlm"] = VLMPipeline()
models["tts"] = TTSPipeline()
os.sched_setaffinity(0, set(range(8)))  # restore for uvicorn
```

Verified: 9 inference threads → `[4,5,6,7]`, uvicorn threads → `[0-7]`.

---

## 6. Summary

### 6.1. Selected Models

| Component | Model | Params | Quantization | Runtime | Performance |
|-----------|-------|--------|--------------|---------|-------------|
| STT | ZipFormer-30M RNNT | ~30M | int8 | sherpa-onnx | ~0.07-0.17s latency, RTF <0.1x |
| TTS | VieNeu-TTS-v2-Turbo | ~111M | Q4_K_M | llama-cpp-python + VieNeu-Codec ONNX | ~4-5s latency |
| VLM | Vintern-1B-v3_5 | ~1B | Q4_K_M | llama-cpp-python 0.3.16 | ~2.8-3.0 tok/s, ~18-22s/image |

### 6.2. Current Limitations

~2.8-3.0 tok/s is the inference ceiling for a 1B VLM with current software configuration. Acceleration paths explored:

| Approach | Status | Reason |
|----------|--------|--------|
| GPU Vulkan (Turnip) | Blocked | Workgroup barrier shader not supported |
| NPU (QNN EP) | Ineffective | 0 ops offloaded to NPU |
| Increase thread count | Ineffective | Stall barriers when mixing core types |
| llama-cpp native recompile | Ineffective | LLAMAFILE=1 runtime dispatch already optimized |

No significant software leverage remaining at current configuration.
