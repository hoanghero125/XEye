# Báo Cáo Kỹ Thuật: XEye

**Cập nhật lần cuối:** 07/08/2026  
**Tác giả:** Đỗ Phạm Bảo Hoàng

Toàn bộ số liệu hiệu năng trong báo cáo này được đo trên server đang chạy ngày 23-24/07/2026 (RUBIK Pi 3, trạng thái warm — request đầu tiên sau khi khởi động server luôn chậm hơn).

---

## Mục Lục

- [1. Thông Tin Hệ Thống](#1-thông-tin-hệ-thống)
  - [1.1. Phần Cứng](#11-phần-cứng)
  - [1.2. Phần Mềm](#12-phần-mềm)
  - [1.3. Nguồn Điện](#13-nguồn-điện)
- [2. Speech-to-Text (STT)](#2-speech-to-text-stt)
  - [2.1. Mô Hình Sử Dụng](#21-mô-hình-sử-dụng)
- [3. Text-to-Speech (TTS)](#3-text-to-speech-tts)
  - [3.1. Mô Hình Sử Dụng](#31-mô-hình-sử-dụng)
  - [3.2. Lịch Sử Thử Nghiệm](#32-lịch-sử-thử-nghiệm)
    - [3.2.1. Thử Nghiệm 1](#321-thử-nghiệm-1)
    - [3.2.2. Thử Nghiệm 2: Tìm model phù hợp hơn](#322-thử-nghiệm-2-tìm-model-phù-hợp-hơn)
    - [3.2.3. Thử Nghiệm 3: Tối ưu](#323-thử-nghiệm-3-tối-ưu)
    - [3.2.4. Thử Nghiệm 4: Migrate sang v3 Turbo](#324-thử-nghiệm-4-migrate-sang-v3-turbo)
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
    - [4.2.10. Thử Nghiệm 10: Khảo Sát Ngưỡng Trần Tốc Độ](#4210-thử-nghiệm-10-khảo-sát-ngưỡng-trần-tốc-độ)
    - [4.2.11. Thử Nghiệm 11: Tinh Chỉnh Độ Dài Câu Trả Lời](#4211-thử-nghiệm-11-tinh-chỉnh-độ-dài-câu-trả-lời)
- [5. Tối Ưu Hóa Hệ Thống](#5-tối-ưu-hóa-hệ-thống)
  - [5.1. Thread Affinity](#51-thread-affinity)
  - [5.2. Chạy Song Song Trong Pipeline](#52-chạy-song-song-trong-pipeline)
  - [5.3. Độ Trễ End-to-End](#53-độ-trễ-end-to-end)
  - [5.4. Hành Vi Nhiệt](#54-hành-vi-nhiệt)
  - [5.5. Chụp Ảnh Từ Camera](#55-chụp-ảnh-từ-camera)
  - [5.6. Chế Độ Chạy](#56-chế-độ-chạy)
- [6. Tổng kết](#6-tổng-kết)
  - [6.1. Mô Hình Được Chọn](#61-mô-hình-được-chọn)
  - [6.2. Giới Hạn Hiện Tại](#62-giới-hạn-hiện-tại)

---

## 1. Thông Tin Hệ Thống

### 1.1. Phần Cứng

| Thành phần | Thông số |
|------------|----------|
| Board | Thundercomm RUBIK Pi 3 |
| SoC | Qualcomm QCS6490 |
| RAM | 8GB LPDDR4x |
| CPU | 4× Cortex-A55 @1.96GHz + 3× Cortex-A78 @2.40GHz + 1× Cortex-X1 @2.71GHz |
| GPU | Adreno 643 |
| NPU | Hexagon 780 (V73), 12 TOPS |
| Camera | Raspberry Pi Camera Module 2 (IMX219) trên CSI connector 1, chụp qua GStreamer `qtiqmmfsrc` ở 1280×720 NV12. Board yêu cầu FPC 22-pin 0.5mm; chỉ hỗ trợ bản standard (không hỗ trợ NoIR/wide-angle) |
| Audio | Seeed Studio ReSpeaker Lite (USB) — mic array vào, speaker ra |
| Nguồn | Pack Li-ion 3S2P, ~55.5 Wh, qua module DC-DC có ngõ ra USB-C PD — xem 1.3 |
| OS | Ubuntu (Linux 6.8.0-1071-qcom) |

### 1.2. Phần Mềm

Các phiên bản đã dùng khi đo số liệu trong báo cáo này. Ba runtime inference đã được pin trong `requirements.txt`; các dependency còn lại (FastAPI, uvicorn, numpy, Pillow) không pin nên có thể ra phiên bản mới hơn.

| Package | Phiên bản |
|---------|-----------|
| Python | 3.11.15 |
| llama-cpp-python | 0.3.16 (chỉ dùng cho VLM) |
| sherpa-onnx | 1.13.2 |
| vieneu | 3.2.3 (cài với `--no-deps`) |
| onnxruntime | 1.24.4 |
| sea-g2p / perth | 0.7.20 / 1.0.0 |
| FastAPI / uvicorn | 0.136.1 / 0.47.0 |

### 1.3. Nguồn Điện

XEye chạy bằng pin chứ không phải nguồn bàn — toàn bộ số liệu trong báo cáo này đều được đo ở
trạng thái đó.

| Thuộc tính | Giá trị |
|------------|---------|
| Pack | Li-ion 3S2P — 6× 18650, 3 nối tiếp × 2 song song |
| Điện áp danh định | 11.1V (3 × 3.7V) |
| Dải điện áp | 12.6V khi đầy → ~9.0V tại ngưỡng cắt của BMS |
| Dung lượng | 5Ah (2 × cell 2.5Ah mắc song song) |
| Năng lượng | ~55.5 Wh |
| Đường cấp nguồn | Pack → jack nguồn → module DC-DC có ngõ ra USB-C PD → board |

**Tầng chuyển đổi này là bắt buộc.** RUBIK Pi 3 nhận nguồn qua USB-C và yêu cầu thương lượng PD 3.0
ở mức 12V/3A; không có nó thì đèn báo nguồn không sáng và board không khởi động. Pack pin chỉ đưa ra
một đường điện thụ động, không có PD controller, nên không thể cấp nguồn trực tiếp cho board dù điện
áp có gần 12V đến đâu. Module ở giữa mới là thứ thực hiện thương lượng, và việc board khởi động được
từ pin chính là bằng chứng nó làm được điều đó.

Module đó phải boost chứ không chỉ ổn áp. Pack 3S chỉ ở trên 12V trong thời gian ngắn sau khi sạc
đầy, còn phần lớn đường xả nằm trong khoảng ~11.5V đến ~9V — thấp hơn mức điện áp phải cấp ra — nên
tổn hao chuyển đổi áp lên phần lớn năng lượng tích trữ chứ không phải một phần nhỏ. Dòng xả liên tục
cho phép của BMS cũng phải lớn hơn dòng *đầu vào* của module, vốn tăng dần khi pin cạn và tỉ số boost
lớn lên.

Ở mức ~55.5 Wh, pack nằm dưới ngưỡng 100 Wh mà các hãng hàng không áp dụng cho pin lithium dự phòng
trong hành lý xách tay.

Thời lượng pin đo được trên pack này:

| Điều kiện | Thời lượng |
|-----------|------------|
| Hỏi liên tục (full load) | 1-2h |
| Nhàn rỗi | 4-5h |

Khi hỏi liên tục, bốn core hiệu năng giữ ở điểm vận hành đã bị throttle mô tả trong 5.4, và đó là
khác biệt giữa hai con số.

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

Đo trên một câu nói tiếng Việt dài 1.8s.

| Metric | Giá trị |
|--------|---------|
| Latency decode (warm) | 0.05-0.11s |
| RTF (warm) | 0.03-0.07x |
| Latency decode (request đầu sau khi khởi động) | ~0.9s |

---

## 3. Text-to-Speech (TTS)

### 3.1. Mô Hình Sử Dụng

#### 3.1.1. Cấu Hình

| Thuộc tính | Giá trị |
|------------|---------|
| Model | VieNeu-TTS-v3-Turbo |
| Params | ~0.1B |
| Quantization | int8 |
| Format | ONNX (graph trong `onnx_int8/`) |
| Runtime | onnxruntime (engine CPU không cần torch) |
| Codec | MOSS-Audio-Tokenizer-Nano ONNX |
| Voice format | 192-dim speaker embedding + reference codes đã encode sẵn |
| Threads | 2 (intra-op) |
| Style | `tu_nhien`, cố định cho mọi voice |
| Watermark | tắt |
| Sample rate | 48000 Hz |

#### 3.1.2. Voices Khả Dụng

14 preset — 7 nữ, 7 nam, đủ cả ba vùng miền. Model hỗ trợ clone giọng từ đoạn mẫu 3-5 giây
nhưng XEye không expose tính năng này qua API.

| Tên | Giới tính | Vùng | Style gốc |
|-----|-----------|------|-----------|
| Mai Anh | Nữ | Miền Bắc | tin_tuc **(mặc định của XEye)** |
| Trúc Ly | Nữ | Miền Bắc | tu_nhien |
| Đoan Trang | Nữ | Miền Bắc | tu_nhien |
| Ngọc Linh | Nữ | Miền Bắc | doc_truyen |
| Phạm Tuyên | Nam | Miền Bắc | tu_nhien |
| Thanh Bình | Nam | Miền Bắc | doc_truyen |
| Minh Đức | Nam | Miền Bắc | tin_tuc |
| Thục Đoan | Nữ | Miền Nam | doc_truyen |
| Thùy Dung | Nữ | Miền Nam | tin_tuc |
| Xuân Vĩnh | Nam | Miền Nam | tu_nhien |
| Thái Sơn | Nam | Miền Nam | doc_truyen |
| Minh Triết | Nam | Miền Nam | tin_tuc |
| Ngọc Trân | Nữ | Miền Trung | tu_nhien |
| Quang Sơn | Nam | Miền Trung | tu_nhien |

XEye ghi đè style gốc của mọi voice thành `tu_nhien`. Style `tin_tuc` (đọc tin) và `doc_truyen`
(kể chuyện) chèn khoảng nghỉ giữa câu: với câu trả lời một câu, `Mai Anh` ở `tin_tuc` tạo ra
một khoảng ngắt 280ms giữa câu, trong khi ở `tu_nhien` là **không có khoảng nghỉ nào** với
cùng thời gian tổng hợp.

#### 3.1.3. Hiệu Năng

Đo qua server đang chạy. Thời gian tổng hợp tỉ lệ với độ dài text; RTF khá ổn định, chỉ kém đi
một chút với input dài.

| Input | Audio sinh ra | Thời gian tổng hợp | RTF |
|-------|---------------|--------------------|-----|
| 8 ký tự | 0.8s | 0.6-0.7s | 0.80-0.84x |
| 83 ký tự (độ dài câu trả lời VLM điển hình) | 3.9-4.6s | 3.3-4.0s | 0.84-0.87x |
| 203 ký tự | 9.8-10.6s | 9.1-10.2s | 0.93-0.97x |

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

Nhanh hơn ~2x. Được dùng cho tới khi migrate sang v3 bên dưới.

---

#### 3.2.4. Thử Nghiệm 4: Migrate sang v3 Turbo

v3 Turbo được phát hành như một model train from scratch (~10k giờ EN-VI), không phải fine-tune
từ v2. Model này không phát hành bản GGUF — trên CPU, package `vieneu` chạy engine ONNX Runtime
không cần torch, nên đường llama.cpp dùng cho v2 trở thành extra `legacy` của thư viện.

##### 3.2.4.1. So Sánh

Cả hai đều đo trên board này, cùng bộ text, trạng thái warm:

| | v2-Turbo | v3-Turbo |
|---|---|---|
| Runtime | llama-cpp-python + VieNeu-Codec | onnxruntime int8 |
| Sample rate | 24 kHz | 48 kHz |
| 8 ký tự | 0.7s | 0.6-0.7s |
| 83 ký tự | 3.2s | 3.3-4.0s |
| 203 ký tự | 8.5s | 9.1-10.2s |
| Peak RSS (chạy riêng) | 3293 MB | 1385 MB |
| Dung lượng đĩa | 655 MB | 286 MB |
| Voices | 4 | 14 + clone giọng |

Tốc độ xấp xỉ ngang nhau ở độ dài câu trả lời điển hình và chậm hơn ~10-20% với input dài, đổi
lại sample rate gấp đôi, peak memory giảm 58% và dung lượng đĩa giảm 56%.

##### 3.2.4.2. Vấn Đề: Oversubscription Thread

v3 tạo ~8 ONNX session. Giữ nguyên `threads=4` như v2 gây oversubscription trên 4 performance
core khi cả ba model cùng nằm trong một process:

| Intra-op threads | Trung vị (203 ký tự, trong server) |
|------------------|------------------------------------|
| 1 | 12.1-12.8s |
| **2** | **9.5s** |
| 3 | 10.5s |
| 4 | 10.9s |

→ Giải pháp: `threads=2`. Lưu ý giá trị tối ưu phụ thuộc ngữ cảnh — khi đo riêng lẻ, không có
model nào khác trong RAM, `threads=4` mới là nhanh nhất (~8.6s).

##### 3.2.4.3. Vấn Đề: Thread Gọi Làm Nghẽn Cả Phép Tính

ONNX Runtime dùng chính thread *gọi* làm một trong các intra-op worker. Request được phục vụ
bởi thread uvicorn, vốn đã được cố ý bỏ pin về `[0-7]` sau khi load model (5.1), nên thread gọi
có thể rơi vào một core A55 và làm nghẽn toàn bộ phép tính — 12.6-18.3s với text 203 ký tự,
dao động rất lớn.

→ Giải pháp: context manager `perf_cores()` trong `server.py` pin thread gọi vào `{4,5,6,7}`
trong suốt `infer()`, rồi khôi phục mask cũ sau đó.

##### 3.2.4.4. Kết Quả

Đã migrate. Output 48 kHz, 14 voices, peak memory chưa bằng một nửa v2.

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
| Max new tokens | 128 |
| Repeat penalty | 1.1 (override: `XEYE_VLM_REPEAT_PENALTY`) |

`llama-cpp-python` để mặc định `repeat_penalty` là 1.0 — tức tắt — trong khi mặc định của chính
llama.cpp là 1.1. Khi để 1.0, model thỉnh thoảng rơi vào vòng lặp lặp từ chạy đến hết token cap:
một trong mười lần chạy sinh ra `"... giá cả - Giá cả - Giá cả - Giá cả"`. Chi phí đo được của
penalty là **2 ms/token** (47 → 49 ms/token), tức ~0.1s cho câu trả lời 60 token.

#### 4.1.2. Hiệu Năng

Đo trên ảnh demo — 2568×1926, được server thu nhỏ còn 960×720 (mọi ảnh đầu vào đều được thu về vừa khung 1280×720).

| Giai đoạn | Thời gian |
|-----------|-----------|
| Encode ảnh (mmproj/clip) | 12.5-13.9s |
| Prefill ảnh (256 image tokens) | 3.8-3.9s |
| **Tổng mỗi ảnh** | **17-21s** |

Chi phí xử lý ảnh là cố định cho mỗi request và chiếm phần lớn thời gian: mất ~16-18s trước khi sinh ra token text đầu tiên. Giá trị `tok_s` mà server trả về là `completion_tokens ÷ tổng thời gian`, nên nó tăng theo độ dài câu trả lời chứ không phải tốc độ decode thuần:

| Độ dài câu trả lời | Throughput báo cáo |
|--------------------|--------------------|
| 58-75 tokens (mô tả đầy đủ, prompt mặc định) | 3.0-3.7 tok/s |
| 17-29 tokens (trả lời ngắn cho câu hỏi cụ thể) | 1.0-1.7 tok/s |

Ngôn ngữ output: Tiếng Việt.

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
| Backbone | ~491MB |
| Quantization | Q4_K_M |
| Format | GGUF |
| mmproj | ~620MB, F16 |
| Runtime | llama-cpp-python 0.3.16 |

##### 4.2.8.2. Vấn Đề & Giải Pháp

`Llava16ChatHandler` dùng format `USER: ... ASSISTANT:` → sinh output rác khi dùng với Vintern.

Giải pháp: subclass `Llava15ChatHandler` - image injection xảy ra ở C-level, override `CHAT_FORMAT` bằng Jinja2 ChatML:

```python
class InternVL2ChatHandler(Llava15ChatHandler):
    """Llava15ChatHandler image injection + InternVL2 ChatML prompt format."""
    DEFAULT_SYSTEM_MESSAGE = None
    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "<|im_start|>system\n{{ message.content }}<|im_end|>\n"
        "{% endif %}"
        "{% if message.role == 'user' %}"
        "<|im_start|>user\n"
        "{% if message.content is iterable and message.content is not string %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' %}"
        "{% if content.image_url is string %}{{ content.image_url }}\n{% endif %}"
        "{% if content.image_url is mapping %}{{ content.image_url.url }}\n{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}{{ content.text }}{% endif %}"
        "{% endfor %}"
        "{% else %}{{ message.content }}{% endif %}"
        "<|im_end|>\n"
        "{% endif %}"
        "{% if message.role == 'assistant' %}"
        "<|im_start|>assistant\n{{ message.content }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
```

Template xử lý luôn cả lượt system và assistant, content dạng chuỗi thuần, và cả hai dạng string lẫn mapping của `image_url`, vì llama-cpp-python có thể truyền vào bất kỳ dạng nào.

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

#### 4.2.10. Thử Nghiệm 10: Khảo Sát Ngưỡng Trần Tốc Độ

Chất lượng của Vintern-1B-v3_5 được đánh giá là đã đủ dùng, nên đợt khảo sát này chỉ tìm các
cách giảm độ trễ mà không đánh đổi chất lượng. Đường xử lý ảnh chiếm ~16s mỗi request
(12-13s encode + ~4s prefill) nên đó là mục tiêu.

##### 4.2.10.1. Encoder Có Bị Giới Hạn Bởi Compute Không?

| Cores | Threads | Trung vị (encode + prefill) |
|-------|---------|-----------------------------|
| `{4,5,6,7}` | 4 | **19.02s** |
| cả 8 | 8 | 19.20s |
| cả 8 | 6 | 19.29s |
| `{4,5,6,7}` | 3 | 21.84s |

Tăng gấp đôi số core không thay đổi gì; giảm xuống dưới 4 thì tệ đi. Độ phân giải đầu vào cũng
không ảnh hưởng — 960×720, 448×448 và 224×224 encode lần lượt hết 12.4s, 13.2s và 13.7s, vì
model luôn xử lý đúng một tile 448px cố định.

##### 4.2.10.2. mmproj F16 → Q8_0

Projector F16 nặng 592MB, là artifact lớn nhất trong đường xử lý ảnh. Việc requantize 146
tensor trọng số 2D sang Q8_0 (318MB) được kỳ vọng sẽ giảm lưu lượng bộ nhớ.

| mmproj | Dung lượng | Encode + prefill |
|--------|------------|------------------|
| **F16** | 592MB | **18.85s** |
| Q8_0 | 318MB | 20.84s |

→ **Loại bỏ: chậm hơn 10%.** Chất lượng output không đổi. CPU có `asimdhp` (FP16 native) nhưng
không có `i8mm` hay SVE, nên kernel F16 vốn đã hợp với phần cứng, trong khi Q8_0 thêm chi phí
dequantize mà không có đường matmul int8 nào để bù lại.

Lưu ý `llama-quantize` không làm được việc này — nó từ chối architecture `clip`. File được tạo
bằng cách requantize tensor trực tiếp qua `gguf-py`.

##### 4.2.10.3. llama.cpp Bản Mới

Build `llama-mtmd-cli` từ master và chạy đúng tấm ảnh đó: master áp dụng tiền xử lý dynamic
**4 tile** của InternVL, encode hết 52s + 13s so với 12-13s ở bản đang pin.
`--image-max-tokens` không ghi đè được hành vi này.

→ **Việc pin `llama-cpp-python==0.3.16` là thiết yếu.** Độ trễ hiện tại có được là nhờ
`Llava15ChatHandler` của bản này chỉ encode một tile duy nhất. Nâng version lên sẽ làm độ trễ
VLM tăng gấp 4 lần. Hệ quả kèm theo là XEye đang chạy model ở độ phân giải hiệu dụng thấp hơn
thiết kế gốc — một đánh đổi chất lượng lấy tốc độ có chủ đích.

##### 4.2.10.4. Build Lại Native

Wheel dựng sẵn của `llama-cpp-python` nhắm vào baseline aarch64 chung, nên build lại từ source với
`-march=native` trông như một khoản throughput miễn phí trên một CPU đã biết rõ đặc tính.

Thực tế thì không, vì các kernel chiếm phần lớn matmul không được chọn ở thời điểm compile.
llama.cpp build với `LLAMAFILE=1`, và đường sgemm của nó dispatch dựa trên đặc trưng CPU phát hiện
được lúc *chạy* — cùng một kernel sẽ thực thi bất kể build có được báo về target hay không. Cũng
không có đường nào rộng hơn đang chờ được mở: 4.2.10.2 đã xác định CPU này có `asimdhp` nhưng
không có `i8mm` lẫn SVE, nên kernel matmul rộng nhất khả dụng vốn đã là kernel đang được chọn.

→ **Không hiệu quả.** Build native chạy đúng những kernel mà wheel gốc đang chạy.

##### 4.2.10.5. Kết Quả

Không hướng nào cải thiện được cấu hình hiện tại. 17-21s mỗi ảnh là ngưỡng sàn cho model này
trên phần cứng này.

---

#### 4.2.11. Thử Nghiệm 11: Tinh Chỉnh Độ Dài Câu Trả Lời

Bản thân VLM luôn tốn ~20s cố định, nhưng độ dài câu trả lời tốn khoảng **0.25s tương tác mỗi
token** (thời gian tổng hợp TTS cộng thời gian người dùng nghe). Câu trả lời 62 token dành 32s
trong tổng số 52s chỉ để nói, nên prompt ngắn gọn hơn trông như một khoản latency miễn phí.

Năm biến thể prompt được chạy trên ba ảnh (một ảnh phòng họp nhiều chữ, hai khung hình camera
trực tiếp), mỗi biến thể hai lần, và được chấm theo nội dung thật của ảnh chứ không để model
tự chấm.

| Prompt | Độ bao phủ ở cảnh thực | Bịa đặt | Kiểu lỗi |
|--------|------------------------|---------|----------|
| **Hiện tại** `Mô tả những gì bạn thấy.` | chai nước, bàn phím hồng, tai nghe, cửa sổ, điện thoại | tiêu đề slide | một lần chạy tràn tới token cap |
| `Chỉ nêu vật thể chính…` | chỉ người + bàn | bịa "đang chơi game" | — |
| `Có gì trước mặt tôi?` | **bỏ sót hoàn toàn con người** (2/2) | bịa tiêu đề slide | — |
| `Nêu ngắn gọn những người và vật thể chính` | người + điện thoại | không | một lần thu về 3 token |
| `Mô tả … trong hai câu ngắn.` | người + điện thoại + tai nghe | gọi điện thoại là "máy tính" | — |

Các biến thể ngắn nhanh hơn 24s nhưng chỉ vì chúng bỏ qua đúng những vật thể làm thiết bị trở
nên hữu ích — chai nước trong tầm với, bàn phím, cửa sổ. Chúng cũng vẫn bịa ("đang chơi game"
trong khi người dùng đang nhìn điện thoại).

Thêm `Không đọc chữ.` vào prompt **không** ngăn được việc bịa tiêu đề slide — model 1B không
tuân thủ ổn định các chỉ dẫn phủ định. Việc bịa chữ là đặc tính của Vintern khi trong khung hình
có chữ, không phải lỗi của prompt.

→ **Không rút ngắn. Giữ nguyên prompt gốc.** Câu trả lời ngắn hơn ở đây không hiệu quả hơn,
chỉ kém hữu ích hơn.

Lưu ý: cả ba ảnh thử nghiệm đều là cảnh "người ngồi ở bàn". Ảnh thực tế khi đeo thiết bị — đi
lại, cửa ra vào, biển báo — có thể cho kết quả khác và cần thử lại khi có dữ liệu.

##### 4.2.11.1. Ép Output Tiếng Việt

Với câu hỏi rỗng hoặc vô nghĩa (ví dụ STT chỉ trả về `"rồi"`), model rơi sang **tiếng Anh** —
đo được 5/10 trên các input suy biến, gồm cả những câu từ chối như *"I'm unable to provide a
detailed description of the image."* Bốn cách khắc phục được thử trên đúng 10 input đó:

| Cách | Số lần trả lời tiếng Anh |
|------|--------------------------|
| Nguyên bản | 5/10 |
| System message (`Luôn trả lời bằng tiếng Việt`) | 3/10 |
| **Hậu tố prompt `" Trả lời bằng tiếng Việt."`** | **0/10** |
| System message + hậu tố | 0/10 |

Chỉ dùng system message là chưa đủ — model 1B không coi trọng vai trò system. Chỉ dẫn gắn trực
tiếp vào prompt người dùng mới có tác dụng. Hậu tố này giờ được thêm vào mọi prompt `/vlm`
(`VLM_LANG_SUFFIX`).

Chi phí: prefill và tốc độ decode không đổi (~8 token thêm nằm trong batch prefill 256 token của
ảnh). Độ chính xác không đổi — vẫn nhận đúng các vật thể, và việc bịa tiêu đề slide vẫn y nguyên.
Câu trả lời hơi dài hơn (~+20 token trung bình, dao động lớn). Nó không sửa độ chính xác, chỉ
sửa ngôn ngữ.

---

## 5. Tối Ưu Hóa Hệ Thống

### 5.1. Thread Affinity

Mixing performance và efficiency cores trong cùng thread pool tạo stall barriers, giảm throughput.

**Vấn đề phát sinh với process-level pinning:**

Pinning toàn bộ process vào performance cores khiến uvicorn/FastAPI threads cũng bị giới hạn, cạnh tranh với inference threads → pipeline tổng thể ≥22s, tệ hơn khi không pin (<22s) tại thời điểm đo. Tổng thời gian tuyệt đối thay đổi theo độ dài câu trả lời, nên đây chỉ là so sánh tương đối — xem 5.2 cho số liệu end-to-end hiện tại.

**Giải pháp:** Pin affinity trước khi load models để inference thread pools kế thừa mask, sau đó restore affinity trước khi uvicorn nhận request:

```python
os.sched_setaffinity(0, {4, 5, 6, 7})
models["stt"] = STTPipeline()   # inference threads kế thừa {4,5,6,7}
models["vlm"] = VLMPipeline()
models["tts"] = TTSPipeline()
os.sched_setaffinity(0, set(range(8)))  # restore cho uvicorn
```

Xác nhận trên server đang chạy (tổng cộng 41 threads):

| Số threads | Affinity | Nguồn |
|------------|----------|-------|
| 9 | `[4,5,6,7]` | Thread pool inference của llama.cpp — kế thừa mask đúng như thiết kế |
| 18 | `[0-7]` | uvicorn / FastAPI / asyncio — đã restore, đúng như thiết kế |
| 14 | mỗi thread một core, trải trên `[1-7]` | onnxruntime (STT sherpa-onnx + các session TTS) |

Nhóm cuối là ngoại lệ: onnxruntime tự pin intra-op threads của nó và bỏ qua mask kế thừa, nên một số rơi vào efficiency cores `[1,2,3]`. Chúng chỉ hoạt động trong giai đoạn STT và codec rất ngắn, không chạy trong lúc VLM decode, nên được giữ nguyên — việc pinning vẫn đạt mục tiêu cho giai đoạn chiếm phần lớn độ trễ.

Chỉ pin lúc load model là chưa đủ với onnxruntime, vì thread *gọi* cũng đóng vai trò một intra-op worker. Xem 3.2.4.3 cho phần pin theo từng request mà đường TTS cần đến.

---

### 5.2. Chạy Song Song Trong Pipeline

Ba giai đoạn trước đây chạy tuần tự một cách không cần thiết. Cả ba đã được cho chồng lấn:

**Chụp ảnh trong lúc ghi âm.** Việc chụp (khởi động gstreamer + ~2s để exposure ổn định) chạy
trước khi mở micro, dù hai việc này độc lập nhau. Khung hình giờ được chụp trong một worker
thread song song với lúc ghi câu hỏi, nên camera không tốn thêm thời gian với mọi cửa sổ ghi âm
dài hơn ~2.5s.

**Tổng hợp trước phát một câu.** Trước đây TTS phải tổng hợp toàn bộ câu trả lời rồi mới phát.
Câu trả lời giờ được tách theo câu, câu N+1 được tổng hợp trong khi câu N đang phát, và tất cả
các chunk được đưa vào một tiến trình `aplay` duy nhất đọc PCM thô từ stdin nên phát liền mạch.
Thời gian tới âm thanh đầu tiên giảm từ **4.05-4.39s** (tổng hợp cả câu trả lời) xuống
**1.1-2.7s** tùy độ dài câu đầu. Lợi ích tăng theo độ dài câu trả lời.

**Server không còn bị chặn.** Các endpoint vốn là `async def` nhưng chạy inference chặn, làm
đứng event loop của uvicorn suốt mỗi request — `/health` không thể trả lời khi VLM đang chạy.
Giờ chúng là `def` đồng bộ nên FastAPI đẩy sang threadpool, kèm `INFERENCE_LOCK` để tuần tự hóa
truy cập model (các model dùng chung 4 core; chạy song song chỉ gây tranh chấp). `/health` giờ
trả lời trong 5-9ms ngay khi VLM đang chạy.

Overhead còn lại ngoài model: ~10ms tiền xử lý ảnh, ~5ms HTTP.

### 5.3. Độ Trễ End-to-End

Pipeline đầy đủ, trạng thái warm, chạy `pipeline.py` với câu hỏi WAV và ảnh demo (thu nhỏ còn 960×720):

| Giai đoạn | Thời gian |
|-----------|-----------|
| STT | 0.15s |
| VLM | 17-21s |
| TTS chunk đầu | 1.1-2.7s |
| **Thời gian tới âm thanh đầu tiên** | **~20s** |
| **Tổng** | **~24-26s** |

Với phần cứng thật, camera được giấu trong cửa sổ ghi âm và độ dài câu hỏi do người dùng quyết
định, nên tổng thời gian trở thành `record_seconds + ~21s`. Câu trả lời dài hơn tốn thêm thời
gian ở cả VLM decode lẫn TTS. Lần chạy đầu sau khi khởi động server chậm hơn — STT và các session
codec cần warm up. Từ lúc khởi động server tới khi phục vụ được request đầu tiên là **12.9s**
với page cache đang nóng.

### 5.4. Hành Vi Nhiệt

Khi chạy inference liên tục, SoC ở mức **82-89°C** (các số đo 46-49°C lúc nhàn rỗi không phản
ánh đúng thực tế). Thời gian encode tăng dần từ lúc khởi động nguội rồi đi ngang:

| Số request | Thời gian encode |
|------------|------------------|
| 1 | 13.3s |
| 5 | 14.8s |
| 10 | 15.2s |
| 15-20 | **15.3s** (đi ngang, ±0.05s) |

Thiết bị không chậm đi liên tục — nó ổn định ở mức cao hơn khởi động nguội ~2s rồi giữ nguyên.
Tần số CPU xác nhận cơ chế: cpu7 giữ 2707MHz trong khoảng chục request đầu, sau đó tụt xuống
2208, 2515 và hai lần xuống **2035MHz** — giảm 25% xung nhịp khi throttling kích hoạt.

Số liệu này đo trên bàn thoáng. Khi đặt trong vỏ máy đeo sát người, throttling sẽ đến sớm hơn và
sâu hơn; không nên giả định các con số ở đây vẫn đúng.

---

### 5.5. Chụp Ảnh Từ Camera

Khung hình được chụp bằng GStreamer `qtiqmmfsrc` ở 1280×720 NV12 (`capture_frame` trong
`pipeline.py`), encode JPEG rồi ghi qua `multifilesink`. Hai đặc tính của cảm biến quyết định cách
làm này.

**Auto-exposure cần thời gian để ổn định.** Những khung hình đầu của mọi luồng đều tối — cần khoảng
5 frame thì AE mới hội tụ. Vì vậy quá trình chụp chạy `gst-launch-1.0` khoảng 2s warmup, ghi ra các
frame được đánh số, rồi giữ lại frame *mới nhất* và bỏ phần còn lại. Lấy frame đầu tiên đồng nghĩa
với việc lấy mẫu giữa lúc cảm biến còn đang hội tụ. Chính khoảng warmup này được 5.2 giấu vào trong
cửa sổ ghi âm.

**Exposure mặc định quá tối khi ở trong nhà.** `exposure-compensation` nhận giá trị −12..12; đo trên
board này với một cảnh trong nhà thiếu sáng:

| exposure-compensation | Kết quả |
|-----------------------|---------|
| 0 | độ sáng trung bình 122 |
| **+2** | **độ sáng trung bình 138 — không cháy sáng** |
| +4 | 22% pixel bị cháy sáng |
| +6 | 29% pixel bị cháy sáng |

→ **`EXPOSURE = 2`.** Mức này lấy lại được chi tiết vùng tối mà không làm cháy vùng sáng; từ +4 trở
lên chỉ là đổi lỗi này lấy lỗi kia.

**Chỉ một consumer.** Camera chỉ cho phép một tiến trình đọc, nên việc chụp ảnh và
`scripts/camera_preview.py` không thể chạy đồng thời. Server preview theo dõi tiến trình
`gst-launch-1.0` đang sống và kill nó khi có viewer mới kết nối, để một luồng cũ không khóa mất
camera.

### 5.6. Chế Độ Chạy

`pipeline.py` chạy ở một trong hai chế độ, chọn bằng `--mode`, biến môi trường `XEYE_MODE`, hoặc
hằng `MODE`:

| Chế độ | Hành vi |
|--------|---------|
| `dev` *(mặc định)* | Ghi câu trả lời đã tổng hợp ra `data/audio/output.wav` |
| `prod` | Không ghi gì xuống đĩa — âm thanh chỉ đi ra loa |

Ở `prod`, các chunk PCM được đẩy thẳng vào `aplay` và không bao giờ được gom lại, nên câu trả lời
chỉ tồn tại trong bộ nhớ. Điều này có ý nghĩa ở hai mặt. Âm thanh đầu ra 48kHz 16-bit mono tốn
96 KB/s, và 3.1.3 đo được một câu trả lời điển hình dài 3.9-4.6s, nên `dev` ghi khoảng 0.4MB mỗi
lượt hỏi và tới ~1MB với câu dài — hao mòn flash liên tục trên một thiết bị được kỳ vọng trả lời
suốt cả ngày. Đây đồng thời là một tính chất về quyền riêng tư: một thiết bị đeo ghi lại người dùng
đã hỏi gì và trước mặt họ có gì sẽ để lại toàn bộ lịch sử đó trên đĩa, còn `prod` không để lại gì.

`--output PATH` ghi đè cả hai và luôn lưu, dùng cho việc debug từng lần.

---


## 6. Tổng kết

### 6.1. Mô Hình Được Chọn

| Thành phần | Model | Params | Quantization | Runtime | Hiệu Năng |
|------------|-------|--------|--------------|---------|-----------|
| STT | ZipFormer-30M RNNT | ~30M | int8 | sherpa-onnx | 0.05-0.11s latency, RTF 0.03-0.07x |
| TTS | VieNeu-TTS-v3-Turbo | ~0.1B | int8 | onnxruntime 1.24.4 | RTF 0.80-0.97x (~3.5s mỗi câu trả lời), 48 kHz |
| VLM | Vintern-1B-v3_5 | ~1B | Q4_K_M | llama-cpp-python 0.3.16 | 17-21s/ảnh (3.0-3.7 tok/s báo cáo) |

End-to-end: ~24-26s cho mỗi câu hỏi, âm thanh đầu tiên phát ở ~20s (5.3). Khi chạy liên tục,
thời gian encode ảnh ổn định ở mức cao hơn ~2s do SoC bị throttle (5.4).

### 6.2. Giới Hạn Hiện Tại

17-21s mỗi ảnh là ngưỡng trần cho VLM 1B với cấu hình phần mềm hiện tại, và ~16-18s trong số đó là chi phí cố định để encode + prefill ảnh chứ không phải sinh token. Các hướng tăng tốc đã khảo sát:

| Hướng | Trạng thái | Lý do |
|-------|-----------|-------|
| GPU Vulkan (Turnip) | Blocked | Workgroup barrier shader không được hỗ trợ |
| NPU (QNN EP) | Không hiệu quả | 0 ops được offload lên NPU |
| Tăng thread count | Không hiệu quả | Stall barriers khi mix core types |
| llama-cpp native recompile | Không hiệu quả | LLAMAFILE=1 runtime dispatch đã tối ưu |

Không còn đòn bẩy phần mềm đáng kể ở cấu hình hiện tại.
