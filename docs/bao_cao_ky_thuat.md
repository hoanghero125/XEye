# Báo Cáo Kỹ Thuật: XEye

**Cập nhật lần cuối:** 25/05/2026  
**Tác giả:** Đỗ Phạm Bảo Hoàng

---

## Mục Lục

- [1. Thông Tin Hệ Thống](#1-thông-tin-hệ-thống)
- [2. Speech-to-Text (STT)](#2-speech-to-text-stt)
  - [2.1. Mô Hình Sử Dụng](#21-mô-hình-sử-dụng)
- [3. Text-to-Speech (TTS)](#3-text-to-speech-tts)
  - [3.1. Mô Hình Sử Dụng](#31-mô-hình-sử-dụng)
  - [3.2. Lịch Sử Thử Nghiệm](#32-lịch-sử-thử-nghiệm)
    - [3.2.1. Thử Nghiệm 1](#321-thử-nghiệm-1)
    - [3.2.2. Thử Nghiệm 2: Tìm model phù hợp hơn](#322-thử-nghiệm-2-tìm-model-phù-hợp-hơn)
    - [3.2.3. Thử Nghiệm 3: Tối ưu](#323-thử-nghiệm-3-tối-ưu)
- [4. Vision-Language Model (VLM)](#4-vision-language-model-vlm)
  - [4.1. Mô Hình Sử Dụng](#41-mô-hình-sử-dụng)
  - [4.2. Lịch Sử Thử Nghiệm](#42-lịch-sử-thử-nghiệm)
    - [4.2.1. Thử Nghiệm 1](#421-thử-nghiệm-1)
    - [4.2.2. Thử Nghiệm 2: Kết hợp GPU](#422-thử-nghiệm-2-kết-hợp-gpu)
    - [4.2.3. Thử Nghiệm 3: Kết hợp NPU](#423-thử-nghiệm-3-kết-hợp-npu)
    - [4.2.4. Thử Nghiệm 4: ONNX + QNN Execution Provider](#424-thử-nghiệm-4-onnx--qnn-execution-provider)
    - [4.2.5. Thử Nghiệm 5: Mô hình nhỏ hơn](#425-thử-nghiệm-5-mô-hình-nhỏ-hơn)
    - [4.2.6. Thử Nghiệm 6: Tối ưu luồng](#426-thử-nghiệm-6-tối-ưu-luồng)
    - [4.2.7. Thử Nghiệm 7: Mô hình xịn hơn](#427-thử-nghiệm-7-mô-hình-xịn-hơn)
    - [4.2.8. Thử Nghiệm 8: Vintern-1B](#428-thử-nghiệm-8-vintern-1b)
    - [4.2.9. Thử Nghiệm 9: GPU Vulkan Offload](#429-thử-nghiệm-9-gpu-vulkan-offload)
- [5. Tối Ưu Hóa Hệ Thống](#5-tối-ưu-hóa-hệ-thống)
  - [5.1. Thread Affinity](#51-thread-affinity)
- [6. Tổng kết](#6-tổng-kết)
  - [6.1. Mô Hình Được Chọn](#61-mô-hình-được-chọn)
  - [6.2. Giới Hạn Hiện Tại](#62-giới-hạn-hiện-tại)

---

## 1. Thông Tin Hệ Thống

| Thành phần | Thông số |
|------------|----------|
| Board | Thundercomm RUBIK Pi 3 |
| SoC | Qualcomm QCS6490 |
| RAM | 8GB LPDDR4x |
| CPU | 4× Cortex-A55 @1.96GHz + 3× Cortex-A78 @2.40GHz + 1× Cortex-X1 @2.71GHz |
| GPU | Adreno 643 |
| NPU | Hexagon 780 (V73), 12 TOPS |
| OS | Ubuntu (Linux 6.8.0-1071-qcom) |

---

## 2. Speech-to-Text (STT)

### 2.1. Mô Hình Sử Dụng

#### 2.1.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | ZipFormer-30M RNNT |
| Params | ~30M |
| Quantization | int8 |
| Runtime | sherpa-onnx |
| Threads | 4 |
| Decoding method | greedy search |
| Sample rate | 16000 Hz |

#### 2.1.2. Hiệu Năng

| Metric | Giá trị |
|--------|---------|
| Latency (câu ngắn) | 0.07-0.17s |
| RTF | <0.1x |

---

## 3. Text-to-Speech (TTS)

### 3.1. Mô Hình Sử Dụng

#### 3.1.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | VieNeu-TTS-v2-Turbo |
| Params | ~111M |
| Quantization | Q4_K_M |
| Format | GGUF |
| Codec | VieNeu-Codec (encoder + decoder ONNX) |
| Voice format | 128-dim speaker embedding |
| Threads | 4 |
| Sample rate | 24000 Hz |

#### 3.1.2. Voices Khả Dụng

| Tên | Giới tính | Vùng |
|-----|-----------|------|
| Bích Ngọc | Nữ | Miền Bắc |
| Phạm Tuyên | Nam | Miền Bắc |
| Thục Đoan | Nữ | Miền Nam |
| Xuân Vĩnh | Nam | Miền Nam |

#### 3.1.3. Hiệu Năng

| Metric | Giá trị |
|--------|---------|
| Latency (đoạn ngắn) | ~4-5s |
| Tốc độ so với Standard | ~2x |

---

### 3.2. Lịch Sử Thử Nghiệm

#### 3.2.1. Thử Nghiệm 1

##### 3.2.1.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | OmniVoice Vietnamese |
| Quantization | float16 |
| Sample rate | 24000 Hz |
| RTF (CUDA) | ~0.07 |

##### 3.2.1.2. Kết Quả

Model được thiết kế cho CUDA - không có CPU fallback hiệu quả

→ Latency trên CPU không đạt yêu cầu real-time. 

---

#### 3.2.2. Thử Nghiệm 2: Tìm model phù hợp hơn

##### 3.2.2.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | VieNeu-TTS-v2 Standard |
| Params | ~270M |
| Quantization | Q4_K_M |
| Format | GGUF |
| Codec | neuphonic neucodec ONNX int8 |
| Voice format | codec token refs |

##### 3.2.2.2. Kết Quả

Latency chấp nhận được, vẫn có thể tối ưu hơn về tốc độ.

---

#### 3.2.3. Thử Nghiệm 3: Tối ưu

##### 3.2.3.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | VieNeu-TTS-v2-Turbo |
| Params | ~111M |
| Quantization | Q4_K_M |
| Format | GGUF |
| Codec | VieNeu-Codec (encoder + decoder ONNX) |
| Voice format | 128-dim speaker embedding |

##### 3.2.3.2. Kết Quả

Nhanh hơn ~2x. 

---

## 4. Vision-Language Model (VLM)

### 4.1. Mô Hình Sử Dụng

#### 4.1.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | Vintern-1B-v3_5 (InternVL2.5-1B fine-tuned VI) |
| Params | ~1B |
| Quantization | Q4_K_M |
| Format | GGUF |
| mmproj | ~620MB, F16 |
| Runtime | llama-cpp-python 0.3.16 |
| Threads | 4 |
| Context | 2048 |

#### 4.1.2. Hiệu Năng

| Metric | Giá trị |
|--------|---------|
| Throughput | ~2.8-3.0 tok/s |
| Ngôn ngữ | Tiếng Việt |

---

### 4.2. Lịch Sử Thử Nghiệm

#### 4.2.1. Thử Nghiệm 1

##### 4.2.1.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | Qwen2.5-VL-3B-Instruct |
| Params | ~3B |
| Quantization | Q4_K_M |
| Format | GGUF |
| Runtime | llama.cpp |

##### 4.2.1.2. Kết Quả

Model 3B quá lớn để inference trên thiết bị

→ Tốc độ sinh token không đạt ngưỡng sử dụng thực tế. 

---

#### 4.2.2. Thử Nghiệm 2: Kết hợp GPU

##### 4.2.2.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | Qwen2.5-VL-3B-Instruct |
| Params | ~3B |
| Quantization | Q4_K_M |
| Format | GGUF |
| Runtime | llama.cpp Vulkan (GPU) |

##### 4.2.2.2. Kết Quả

Model + runtime overhead vượt giới hạn phần cứng

→ OOM (Out of memory) → Crash

---

#### 4.2.3. Thử Nghiệm 3: Kết hợp NPU

##### 4.2.3.1. llama.cpp Hexagon Backend

Build `libggml-hexagon.so` + `libggml-htp-v73.so` từ source llama.cpp, load vào runtime.

Khi chạy inference, backend load thành công nhưng không thực thi được các op cốt lõi của LLM - attention, KV cache, matrix multiply với shape động đều không được Hexagon HTP chấp nhận. Toàn bộ computation vẫn chạy trên CPU, không có op nào offload lên NPU.

Hexagon 780 (V73) được thiết kế cho fixed-shape CV ops (conv, pool, activation) - không phải dynamic-shape transformer ops.

##### 4.2.3.2. Xác Nhận - Qualcomm AI Hub

Kiểm tra AI Hub: có Qwen2-VL nhưng không cung cấp bản deploy cho target device. Qualcomm xác nhận Hexagon NPU không được thiết kế cho LLM inference - chỉ phù hợp với computer vision workloads.

##### 4.2.3.3. Kết Quả

NPU không khả dụng cho LLM/VLM inference.

---

#### 4.2.4. Thử Nghiệm 4: ONNX + QNN Execution Provider

> Qualcomm Neural Processing SDK (QNN SDK) là bộ SDK chính thức của Qualcomm để chạy inference trên các accelerator của họ - Hexagon NPU, Adreno GPU, CPU. ONNX Runtime tích hợp QNN SDK qua QNN Execution Provider (QNN EP), cho phép offload ONNX graph nodes lên NPU.

→ Mục đích của thứ nghiệm là để bypass giới hạn không hỗ trợ LLM ops của Hexagon NPU

##### 4.2.4.1. Thiết Lập ONNX + QNN EP

Export model sang ONNX, chạy qua ONNX Runtime với QNN EP để tự quyết định op nào offload lên NPU.

###### Vấn đề: Bug `backendValidateOpConfig` (error 3110)

QNN SDK v2.43 có bug: hàm `backendValidateOpConfig` luôn trả về lỗi 3110, khiến toàn bộ ops bị từ chối và fallback về CPU EP. 

→ Giải pháp: tự viết shim `libQnnHtpShim.so` — override hàm `backendValidateOpConfig` để luôn trả về `QNN_SUCCESS` thay vì lỗi 3110, bypass bug.

##### 4.2.4.2. Qwen2.5-VL-3B ONNX (HuggingFace)

Thử bản ONNX có sẵn trên HuggingFace trước.

→ Chỉ 36/1084 nodes chạy trên QNN (reshape/cast ops trivial). Tốc độ: 0.1 tok/s.

###### Vấn đề: QNN EP không implement được các op fused ORT extensions (`MatMulNBits`, `GroupQueryAttention`)  

##### 4.2.4.3. Custom Export Qwen2.5-VL-3B FP32 ONNX

Export lại từ PyTorch source với standard ONNX ops để QNN có thể nhận.

###### Vấn đề: FP32 decoder ~17GB → không load được trên thiết bị.

##### 4.2.4.4. Quantize INT8 Qwen2.5-VL-3B

Quantize để giảm kích thước xuống mức có thể load. Decoder INT8 ~2.9GB.

###### Vấn đề: ORT dequantize sang FP32 lúc load → không load được trên thiết bị.

→ Giải pháp: Tối ưu lại bộ nhớ (tắt oomd, overcommit, swap 4GB, zram 512MB) → vẫn OOM.

→ Model 3B không khả thi.

##### 4.2.4.5. Qwen2-VL-2B Export + Deploy

Export theo cùng quy trình. Decoder 1.5GB → load thành công.

###### Vấn đề: CPU EP vẫn 0.1 tok/s. Không có NPU offload thì tốc độ không cải thiện.

##### 4.2.4.6. QNN EP cho Qwen2-VL-2B

Áp dụng QNN EP lên 2B. 

###### Vấn đề: OOM ngay trong Graph Optimization. 

→ Giải pháp: Loại bỏ `lm_head` để giảm memory → compile thành công → inference vẫn OOM.

Đo DDR bandwidth để kiểm tra NPU có thực sự chạy:
```
DDR write: 10KB  |  DDR read: 12KB
```
Gần bằng 0 → QNN không thực thi bất kỳ op nào.

→ Nguyên nhân: `DequantizeLinear` nodes khiến CPU EP pre-allocate FP32 buffers ~5GB trước khi inference bắt đầu.

##### 4.2.4.7. Re-export với `do_constant_folding=True`

Thử fold `DequantizeLinear` nodes có constant input để giảm số lượng.

→ 2357 → 2352 nodes → Không đáng kể. 

→ QNN vẫn không nhận ops. 

##### 4.2.4.8. Kết Quả

0 op offload lên NPU sau tất cả các lần thử. Vấn đề là mismatch kiến trúc: QNN EP thiết kế cho fixed-shape CV graph, không phải dynamic-shape transformer với attention và KV cache.

---

#### 4.2.5. Thử Nghiệm 5: Mô hình nhỏ hơn

##### 4.2.5.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | SmolVLM-256M |
| Params | ~256M |
| Quantization | FP32 |
| Format | ONNX |
| Runtime | onnxruntime (CPU) |

##### 4.2.5.2. Kết Quả

~1.5 tok/s. 

###### Vấn đề: Model sinh output là tiếng Anh và quá dài (có xu hướng hallucinate). 

→ Giải pháp: giới hạn `max_tokens` + thêm instruction prompt yêu cầu trả lời ngắn gọn. Bổ sung model dịch máy cho text input và output.

---

#### 4.2.6. Thử Nghiệm 6: Tối ưu luồng

##### 4.2.6.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | SmolVLM-256M + opus-mt-en-vi |
| Luồng | Ảnh → SmolVLM-256M (EN) → opus-mt-en-vi → Tiếng Việt |

##### 4.2.6.2. Kết Quả

~1.5 tok/s decode. Đã đảm bảo được text output là tiếng Việt

###### Vấn đề: Chất lượng output kém tự nhiên.

---

#### 4.2.7. Thử Nghiệm 7: Mô hình xịn hơn

##### 4.2.7.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | LFM2-VL-450M |
| Encoder | SigLIP2, 86M params |
| Decoder | LFM2, 350M params |
| Kiến trúc | Hybrid (conv layers + full attention) |
| Quantization | int8 |
| Format | ONNX |

##### 4.2.7.2. Hiệu Năng

| Metric | Giá trị |
|--------|---------|
| Throughput | 1.2 tok/s |

###### Vấn đề: Tốc độ ~SmolVLM nhưng chất lượng output không cải thiện đáng kể  

---

#### 4.2.8. Thử Nghiệm 8: Vintern-1B

##### 4.2.8.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | Vintern-1B-v3_5 (InternVL2.5-1B fine-tuned VI) |
| Backbone | ~462MB |
| Quantization | Q4_K_M |
| Format | GGUF |
| mmproj | ~620MB, F16 |
| Runtime | llama-cpp-python 0.3.16 |

##### 4.2.8.2. Vấn Đề & Giải Pháp

`Llava16ChatHandler` dùng format `USER: ... ASSISTANT:` → sinh output rác khi dùng với Vintern.

Giải pháp: subclass `Llava15ChatHandler` - image injection xảy ra ở C-level, override `CHAT_FORMAT` bằng Jinja2 ChatML:

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

mmproj F16 ~620MB là kích thước đúng - InternViT-300M × 2 bytes/param ≈ 600MB.

##### 4.2.8.3. Kết Quả

Chất lượng output tiếng Việt được cải thiện đáng kể (mà không cần dùng tới NMT) với tốc độ được cải thiện.

---

#### 4.2.9. Thử Nghiệm 9: GPU Vulkan Offload

Offload VLM inference sang GPU để giảm latency.

##### 4.2.9.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Driver | Turnip (Mesa open-source Adreno) |
| Vulkan | 1.4.318 |
| Build | llama-cpp `-DGGML_VULKAN=ON` + `glslc` (shaderc) |

##### 4.2.9.2. Kết Quả

```
MESA: error: Compute shader which has workgroup barrier cannot be used 
because it's impossible to have enough concurrent waves.
```

Turnip không hỗ trợ compute shader với workgroup barrier — loại shader bắt buộc cho clip/mmproj image encoder trong llama.cpp. Driver proprietary Qualcomm có thể hỗ trợ nhưng không có bản phát hành cho Linux.

→ GPU Vulkan blocked bởi giới hạn Turnip driver. Revert về CPU-only.

---

## 5. Tối Ưu Hóa Hệ Thống

### 5.1. Thread Affinity

Mixing performance và efficiency cores trong cùng thread pool tạo stall barriers, giảm throughput.

**Vấn đề phát sinh với process-level pinning:**

Pinning toàn bộ process vào performance cores khiến uvicorn/FastAPI threads cũng bị giới hạn, cạnh tranh với inference threads → pipeline tổng thể ≥22s (tệ hơn không pin <22s).

**Giải pháp:** Pin affinity trước khi load models để inference thread pools kế thừa mask, sau đó restore affinity trước khi uvicorn nhận request:

```python
os.sched_setaffinity(0, {4, 5, 6, 7})
models["stt"] = STTPipeline()   # inference threads kế thừa {4,5,6,7}
models["vlm"] = VLMPipeline()
models["tts"] = TTSPipeline()
os.sched_setaffinity(0, set(range(8)))  # restore cho uvicorn
```

Xác nhận: 9 inference threads → `[4,5,6,7]`, uvicorn threads → `[0-7]`.

---


## 6. Tổng kết

### 6.1. Mô Hình Được Chọn

| Thành phần | Model | Params | Quantization | Runtime | Hiệu Năng |
|------------|-------|--------|--------------|---------|-----------|
| STT | ZipFormer-30M RNNT | ~30M | int8 | sherpa-onnx | ~0.07-0.17s latency, RTF <0.1x |
| TTS | VieNeu-TTS-v2-Turbo | ~111M | Q4_K_M | llama-cpp + VieNeu-Codec ONNX | ~4-5s latency |
| VLM | Vintern-1B-v3_5 | ~1B | Q4_K_M | llama-cpp-python 0.3.16 | ~2.8-3.0 tok/s, ~18-22s/ảnh |

### 6.2. Giới Hạn Hiện Tại

~2.8-3.0 tok/s là ngưỡng trần inference cho VLM 1B với cấu hình phần mềm hiện tại. Các hướng tăng tốc đã khảo sát:

| Hướng | Trạng thái | Lý do |
|-------|-----------|-------|
| GPU Vulkan (Turnip) | Blocked | Workgroup barrier shader không được hỗ trợ |
| NPU (QNN EP) | Không hiệu quả | 0 ops được offload lên NPU |
| Tăng thread count | Không hiệu quả | Stall barriers khi mix core types |
| llama-cpp native recompile | Không hiệu quả | LLAMAFILE=1 runtime dispatch đã tối ưu |

Không còn đòn bẩy phần mềm đáng kể ở cấu hình hiện tại.
