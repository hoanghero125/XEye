# Technical Report: XEye

**Last updated:** 17/08/2026 
**Author:** Do Pham Bao Hoang

All performance figures in this report were measured on the running server on 23-24/07/2026 (RUBIK Pi 3, warm — the first request after startup is slower).

---

## Table of Contents

- [1. System Information](#1-system-information)
  - [1.1. Hardware](#11-hardware)
  - [1.2. Software](#12-software)
  - [1.3. Power](#13-power)
  - [1.4. Button](#14-button)
- [2. Speech-to-Text (STT)](#2-speech-to-text-stt)
  - [2.1. Model Used](#21-model-used)
  - [2.2. Voice Activity Detection (VAD)](#22-voice-activity-detection-vad)
  - [2.3. Noisy Environments](#23-noisy-environments)
- [3. Text-to-Speech (TTS)](#3-text-to-speech-tts)
  - [3.1. Model Used](#31-model-used)
  - [3.2. Experiment History](#32-experiment-history)
    - [3.2.1. Experiment 1](#321-experiment-1)
    - [3.2.2. Experiment 2: Finding a Better Model](#322-experiment-2-finding-a-better-model)
    - [3.2.3. Experiment 3: Optimization](#323-experiment-3-optimization)
    - [3.2.4. Experiment 4: Migration to v3 Turbo](#324-experiment-4-migration-to-v3-turbo)
  - [3.3. Playback Level](#33-playback-level)
  - [3.4. Interaction Cues](#34-interaction-cues)
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
    - [4.2.10. Experiment 10: Speed Ceiling Investigation](#4210-experiment-10-speed-ceiling-investigation)
    - [4.2.11. Experiment 11: Prompt Length Tuning](#4211-experiment-11-prompt-length-tuning)
- [5. System Optimization](#5-system-optimization)
  - [5.1. Thread Affinity](#51-thread-affinity)
  - [5.2. Pipeline Concurrency](#52-pipeline-concurrency)
  - [5.3. End-to-End Latency](#53-end-to-end-latency)
  - [5.4. Thermal Behaviour](#54-thermal-behaviour)
  - [5.5. Camera Capture](#55-camera-capture)
    - [5.5.1. Module Substitution: IMX219 → IMX708](#551-module-substitution-imx219--imx708)
  - [5.6. Runtime Modes](#56-runtime-modes)
- [6. Summary](#6-summary)
  - [6.1. Selected Models](#61-selected-models)
  - [6.2. Current Limitations](#62-current-limitations)

---

## 1. System Information

### 1.1. Hardware

| Component | Specification |
|-----------|---------------|
| Board | Thundercomm RUBIK Pi 3 |
| SoC | Qualcomm QCS6490 |
| RAM | 8GB LPDDR4x |
| CPU | 4× Cortex-A55 @1.96GHz + 3× Cortex-A78 @2.40GHz + 1× Cortex-X1 @2.71GHz |
| GPU | Adreno 643 |
| NPU | Hexagon 780 (V73), 12 TOPS |
| Camera | Raspberry Pi Camera Module 3 (IMX708) on CSI connector 2, captured via GStreamer `qtiqmmfsrc` at 1280×720 NV12, rotated 180° on capture. Substituted after the Module 2 failed to probe (5.5.1); autofocus is unsupported on this board. Requires a 22-pin 0.5mm FPC; standard variant only (no NoIR/wide-angle) |
| Audio | Seeed Studio ReSpeaker Lite (USB) — mic array in, speaker out |
| Button | 3-pin momentary module with onboard pull-up — signal on pin 16 (`GPIO_26`), active-low, see 1.4 |
| Power | 3S2P Li-ion pack, ~55.5 Wh, through a DC-DC module with USB-C PD output — see 1.3 |
| OS | Ubuntu (Linux 6.8.0-1071-qcom) |

### 1.2. Software

Versions the measurements in this report were taken with. The three inference runtimes are pinned in `requirements.txt`; the remaining dependencies (FastAPI, uvicorn, numpy, Pillow) are unpinned and may resolve to newer versions.

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| llama-cpp-python | 0.3.16 (VLM only) |
| sherpa-onnx | 1.13.2 |
| vieneu | 3.2.3 (installed `--no-deps`) |
| onnxruntime | 1.24.4 |
| sea-g2p / perth | 0.7.20 / 1.0.0 |
| FastAPI / uvicorn | 0.136.1 / 0.47.0 |

### 1.3. Power

XEye runs on battery, not a bench supply — every figure in this report was measured that way.

| Attribute | Value |
|-----------|-------|
| Pack | 3S2P Li-ion — 6× 18650, 3 in series × 2 in parallel |
| Nominal voltage | 11.1V (3 × 3.7V) |
| Voltage range | 12.6V charged → ~9.0V at BMS cutoff |
| Capacity | 5Ah (2 × 2.5Ah cells in parallel) |
| Energy | ~55.5 Wh |
| Delivery | Pack → power connector → DC-DC module with USB-C PD output → board |

**The conversion stage is not optional.** The RUBIK Pi 3 takes power over USB-C and requires a
PD 3.0 negotiation at 12V/3A; without one the power LED stays off and the board does not boot. A
battery pack presents a passive rail with no PD controller, so it cannot drive the board directly
no matter how close its voltage sits to 12V. The module in between is what negotiates, and the
board booting from the pack is the evidence that it does.

That module has to boost, not merely regulate. A 3S pack sits above 12V only briefly after a full
charge and spends most of its discharge curve between ~11.5V and ~9V — below the voltage it has to
supply — so conversion loss applies to the majority of the stored energy rather than a corner of
it. The pack's BMS continuous-current rating must also clear the module's *input* draw, which
rises as the pack drains and the boost ratio grows.

At ~55.5 Wh the pack is under the 100 Wh threshold airlines apply to spare lithium batteries in
carry-on baggage.

Measured runtime on this pack:

| Condition | Runtime |
|-----------|---------|
| Continuous querying (full load) | 1-2h |
| Idle | 4-5h |

Under continuous querying the four performance cores hold the throttled operating point described
in 5.4, which is what separates the two figures.

### 1.4. Button

A press starts a query. Until this, the only trigger was running `pipeline.py` from a shell —
which the device's own user cannot do.

| Attribute | Value |
|-----------|-------|
| Switch | 3-pin momentary button module, with an onboard pull-up resistor |
| Connection | Signal to physical pin 16 (`GPIO_26`), plus the module's own VCC and GND |
| Line address | `/dev/gpiochip4` (`f100000.pinctrl`) offset 26 |
| Logic | Active-low — the line rests at 3.3V and the press pulls it to 0V |
| Edge | Falling, `gpiod.line.Edge.FALLING` with `Bias.PULL_UP` |
| Debounce | 50ms, in the kernel via libgpiod |

**Momentary, not latching, because VAD already owns the other end.** The press means only
"start listening"; 2.2 decides when the question finished. Nothing is ever held down, which
also suits a user who cannot see how long they are meant to hold it.

**A bare 2-pin switch does not work on this board, and the reason is worth recording.** With
only a switch between the pin and ground, the line is driven only while the contacts are
closed; when they open it is left floating, and a CMOS input holds its last charge. The pin
therefore latched at whatever the switch last connected it to and never returned — a single
press registered, and every press after it was invisible because the line was already at the
pressed level. Reversing the wiring to 3.3V produced the identical fault mirrored: the line
latched high instead of low. The module's onboard resistor is what actively restores the
released level, and so what makes the release edge exist at all.

The internal pull-up cannot substitute for it. `libgpiod`'s bias request is silently ignored by
this pinctrl driver — verified across five separate lines, none of which responded to
`PULL_UP` or `PULL_DOWN`. Writing the TLMM pin-config register directly does work, but it is
lost on reboot and is no substitute for a resistor in the circuit.

**Pin choice.** The 40-pin header is Raspberry Pi-compatible in layout, and pin 16 is a plain
GPIO with no default function. Pin 14 (GND) sits adjacent to it in the same row.

**Header labels are not TLMM pin numbers, and this is the trap that cost the most time.** The
header's `GPIO_n` names are the board's own signal names; the SoC's pinctrl separately names its
pins `GPIO_0`–`GPIO_175`. The two schemes share a format and do not interchange — the header's
pin 3 is labelled "GPIO_2 (I2C1_SDA)" while TLMM pin 2 is owned by `1c08000.pcie`. A summarised
pinout table claiming pin 16 was `GPIO_23` sent every measurement to an unrelated line.

The authority is the pinout diagram in the vendor's 40-pin LS connector page, which gives
**pin 16 = GPIO_26**, and hence offset 26 on `/dev/gpiochip4`. It is emphatically not
`gpiochip0`, which is a PMIC (`c440000.spmi:pmic@8`) with 12 lines and no relationship to the
header.

**The vendor's sysfs numbers are misleading on this kernel.** They assume a TLMM base of 535
where this kernel uses 547, so arithmetic on them yields a wrong — and typically also free —
line that fails silently rather than loudly. Chip plus offset is stable across kernels where the
global sysfs number is not.

**`gpiofind` is not usable here.** The gpiod CLI tools are not installed, and no chip exposes
line names in any case — the device tree sets no `gpio-line-names`, so all six chips report zero
named lines. The pinctrl table is a separate mapping that `gpiofind` does not consult.

**GPIO access requires configuration.** `/dev/gpiochip*` is `crw------- root root` and the board
has no `gpio` group, so button mode fails with `Permission denied` on an unprepared board. A
udev rule plus group membership grants it without running the pipeline as root; `pipeline.py`
checks for access at startup and prints that fix rather than failing on the first press.

**A held button is reported, not hung on.** A falling edge cannot arrive on a line already low,
so `wait_for_button()` reads the level before arming: if the button is held or stuck closed it
says so and waits for release, rather than blocking silently forever.

**It also turns `pipeline.py` into a loop.** `--button` waits for a press, answers, and returns
to waiting, so the device needs no terminal after startup. A failed query is reported and the
loop continues — on a wearable, a camera hiccup or a server restart must not end the session.

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

Measured on a 1.8s Vietnamese utterance.

| Metric | Value |
|--------|-------|
| Decode latency (warm) | 0.05-0.11s |
| RTF (warm) | 0.03-0.07x |
| Decode latency (first request after startup) | ~0.9s |

---

### 2.2. Voice Activity Detection (VAD)

Recording used to run for a fixed window — `arecord -d 5` — which always waited the full five
seconds and cut off anyone still speaking at the end of it. A user who cannot see the device has
no way to know when that window opened or closed. Silero VAD replaces it: the recording now ends
when the speaker does, and only the trimmed speech segment reaches the STT model.

#### 2.2.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | Silero VAD |
| Size | 629KB, fp32 ONNX |
| Runtime | sherpa-onnx — already the STT runtime, so no new dependency |
| Threads | 1 |
| Cores | `{0,1,2,3}` — A55 efficiency cores |
| Window size | 512 samples (32ms at 16kHz) |
| Threshold | 0.6 (Silero default is 0.5 — see 2.3) |
| Min speech duration | 0.25s |
| Min silence duration | 0.8s |
| Max speech duration | 8s |
| Sample rate | 16000 Hz |

VAD runs in `pipeline.py`, not in the server. The latency it saves comes from ending the recording
early, and recording happens client-side; server-side VAD could trim a completed WAV but could not
give back wall-clock already spent waiting.

It is also the one component pinned to the *efficiency* cores, inverting 5.1. At 629KB an A55 keeps
up with realtime comfortably, which leaves cpu4-7 to the camera during the question — and if
listening ever becomes continuous, the efficiency cores are where it belongs.

#### 2.2.2. Endpoint Threshold

`min_silence_duration` is the parameter that matters. It is added to every query, because the
recording always waits it out after speech stops. Silero defaults to 0.5s, tuned for conversational
agents where turn latency is the product. XEye is not that system.

The costs are asymmetric. Being too generous costs exactly the excess. Being too aggressive
truncates the question — STT sees half of it, the VLM confidently answers the wrong thing, and the
user waits out the full ~21s pipeline before discovering they have to ask again, losing ~25s.
Against a cost of `d + p(truncation) × 25s`:

| min_silence_duration | Truncation rate | Expected cost |
|----------------------|-----------------|---------------|
| 0.4s | 5% | 1.65s |
| 0.7s | 1% | 0.95s |
| **0.8s** | **~0.7%** | **0.98s** |
| 1.0s | 0.3% | 1.08s |

The curve is steep on the short side and nearly flat on the long side, so the setting errs long.
0.8s also clears the range of normal between-clause pauses (300-600ms) while staying below a
turn-final pause (700ms+) — which matters for Vietnamese question phrasing, where pausing mid-question
is ordinary.

→ **0.8s**, exposed as `--silence`.

The truncation rates above are a cost model, not measurements. They establish the shape of the
curve, not the exact optimum. A sweep against real recordings on the board is still to be run, and
it is the measurement that should replace this table.

#### 2.2.3. Guards

The fixed window was the only thing guaranteeing that recording ever ended. Without it, a VAD held
in speech by sustained noise would listen indefinitely, so these bounds are load-bearing rather
than defensive:

| Guard | Value | Behaviour |
|-------|-------|-----------|
| Hard cap | 12s | Flush the VAD, keep whatever speech it holds, stop |
| No-speech timeout | 6s | Abort with an error rather than listen forever |

---

### 2.3. Noisy Environments

**The audio input path has not been measured in noise.** Every figure in 2.1 and 2.2 comes from
clean recordings — a 1.8s utterance in a quiet room. A device worn outdoors will not see that.
This section records what the path relies on and what is expected to fail, so the gap is
explicit rather than implied.

#### 2.3.1. What the Hardware Provides

The ReSpeaker Lite's XMOS XU316 runs acoustic echo cancellation, noise suppression, automatic
gain control, interference cancellation and voice-to-noise ratio estimation, and is rated for
far-field capture to 3m. XEye has never configured any of it — the algorithms run at whatever
the shipped firmware defaults are.

**It does not beamform.** Beamforming and dereverberation belong to the ReSpeaker Mic Array v2.0,
not the Lite. This is the material limitation for crowded settings: without a steerable beam
there is no rejection by *direction*, so a competing talker is attenuated only by NS and IC. Both
target stationary or point-like noise, and babble — many overlapping voices — is neither. Babble
occupies the same spectrum and the same modulation rates as the target speech, which makes it the
hardest case for any single-channel suppressor.

A 3m pickup radius is also as much liability as feature here. In a crowd it guarantees the array
hears the crowd, and the microphones sit on the board rather than near the mouth, so the signal-to
-noise ratio at the capsule is set by where the device is worn more than by anything downstream.

#### 2.3.2. Threshold

`silero_vad.threshold` is set to **0.6**, above Silero's 0.5 default. Silero's own guidance is to
raise the threshold in noisy conditions to suppress false positives, and this device is not a
desk accessory. The negative threshold derives as `threshold - 0.15`, so it rises to 0.45 with
it — speech has to be more confident to start a segment and to sustain one.

This is a reasoned starting point, not a measured optimum. It trades some sensitivity to quiet
speech for resistance to being held open by background noise, and the correct value can only come
from recordings made where the device is actually used.

#### 2.3.3. Expected Failure Modes

| Failure | Mechanism | Current mitigation |
|---------|-----------|--------------------|
| Every query runs to the 12s cap | Sustained noise holds the VAD in speech, so it never endpoints | The hard cap in 2.2.3 keeps the device alive, but the interaction is slow |
| Question truncated early | Noise dips below the negative threshold during a pause | Raise `--silence` |
| Nothing captured | Speech never clears 0.6 over the noise floor | 6s no-speech timeout aborts |
| Confident wrong answer | STT returns garbage, the VLM answers a question that was never asked | None — see below |

The last is the one that matters. 4.2.11 records that degenerate STT output made the VLM reply in
English, which is why `VLM_LANG_SUFFIX` exists; that was measured on quiet-room inputs where such
output was an edge case. In a crowd it becomes the normal case, and nothing in the pipeline
currently distinguishes a transcript that is wrong from one that is right. ZipFormer-30M is also a
small model, and small ASR degrades faster under noise than large.

#### 2.3.4. What to Measure

Recordings from the environments the device is meant for — a street, a café, a market — then:

- VAD endpoint rate versus noise floor, sweeping `threshold` across 0.5-0.7
- Proportion of queries reaching the 12s cap
- STT word error rate against a clean-room baseline
- Whether the XU316's VNR output is usable as a confidence gate, so a hopeless transcript can be
  rejected before ~21s of VLM time is spent answering it

---

## 3. Text-to-Speech (TTS)

### 3.1. Model Used

#### 3.1.1. Configuration

| Attribute | Value |
|-----------|-------|
| Model | VieNeu-TTS-v3-Turbo |
| Params | ~0.1B |
| Quantization | int8 |
| Format | ONNX (`onnx_int8/` graphs) |
| Runtime | onnxruntime (torch-free CPU engine) |
| Codec | MOSS-Audio-Tokenizer-Nano ONNX |
| Voice Format | 192-dim speaker embedding + pre-encoded reference codes |
| Threads | 2 (intra-op) |
| Style | `tu_nhien`, pinned for every voice |
| Watermark | disabled |
| Sample Rate | 48000 Hz |

#### 3.1.2. Available Voices

14 presets — 7 female, 7 male, across all three dialects. Voice cloning from a 3-5s reference
clip is supported by the model but not exposed through the XEye API.

| Name | Gender | Dialect | Preset style |
|------|--------|---------|--------------|
| Mai Anh | Female | Northern | tin_tuc **(XEye default)** |
| Trúc Ly | Female | Northern | tu_nhien |
| Đoan Trang | Female | Northern | tu_nhien |
| Ngọc Linh | Female | Northern | doc_truyen |
| Phạm Tuyên | Male | Northern | tu_nhien |
| Thanh Bình | Male | Northern | doc_truyen |
| Minh Đức | Male | Northern | tin_tuc |
| Thục Đoan | Female | Southern | doc_truyen |
| Thùy Dung | Female | Southern | tin_tuc |
| Xuân Vĩnh | Male | Southern | tu_nhien |
| Thái Sơn | Male | Southern | doc_truyen |
| Minh Triết | Male | Southern | tin_tuc |
| Ngọc Trân | Female | Central | tu_nhien |
| Quang Sơn | Male | Central | tu_nhien |

XEye overrides every voice's preset style with `tu_nhien`. The `tin_tuc` (news-reader) and
`doc_truyen` (storytelling) styles insert mid-sentence pauses: on a one-sentence answer,
`Mai Anh` at `tin_tuc` produces a 280ms mid-sentence break, versus **zero pauses** at
`tu_nhien` for the same synthesis time.

#### 3.1.3. Performance

Measured through the running server. Synthesis time scales with text length; RTF is roughly
stable, degrading slightly on long input.

| Input | Audio produced | Synthesis time | RTF |
|-------|----------------|----------------|-----|
| 8 chars | 0.8s | 0.6-0.7s | 0.80-0.84x |
| 83 chars (typical VLM answer) | 3.9-4.6s | 3.3-4.0s | 0.84-0.87x |
| 203 chars | 9.8-10.6s | 9.1-10.2s | 0.93-0.97x |

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

~2x faster. Shipped until the v3 migration below.

---

#### 3.2.4. Experiment 4: Migration to v3 Turbo

v3 Turbo was released as a from-scratch model (~10k hours EN-VI), not a fine-tune of v2. It
publishes no GGUF — on CPU the `vieneu` package runs a torch-free ONNX Runtime engine, so the
llama.cpp path used for v2 became the library's `legacy` extra.

##### 3.2.4.1. Comparison

Both measured on this board, same texts, warm:

| | v2-Turbo | v3-Turbo |
|---|---|---|
| Runtime | llama-cpp-python + VieNeu-Codec | onnxruntime int8 |
| Sample rate | 24 kHz | 48 kHz |
| 8 chars | 0.7s | 0.6-0.7s |
| 83 chars | 3.2s | 3.3-4.0s |
| 203 chars | 8.5s | 9.1-10.2s |
| Peak RSS (standalone) | 3293 MB | 1385 MB |
| Disk footprint | 655 MB | 286 MB |
| Voices | 4 | 14 + cloning |

Speed is roughly at parity for typical answer lengths and ~10-20% slower on long input, in
exchange for double the sample rate, 58% less peak memory and 56% less disk.

##### 3.2.4.2. Issue: Thread Oversubscription

v3 builds ~8 ONNX sessions. Carrying over v2's `threads=4` oversubscribed the 4 performance
cores once all three models shared the process:

| Intra-op threads | Median (203 chars, in-server) |
|------------------|-------------------------------|
| 1 | 12.1-12.8s |
| **2** | **9.5s** |
| 3 | 10.5s |
| 4 | 10.9s |

→ Fix: `threads=2`. Note the optimum is context-dependent — measured standalone, with nothing
else resident, `threads=4` is instead the fastest (~8.6s).

##### 3.2.4.3. Issue: Calling Thread Stalls the Op

ONNX Runtime uses the *calling* thread as one of its intra-op workers. Requests are served on
uvicorn threads, which are deliberately unpinned to `[0-7]` after model load (5.1), so the
calling thread could land on an A55 efficiency core and stall the whole operation — 12.6-18.3s
on a 203-char text, with high variance.

→ Fix: a `perf_cores()` context manager in `server.py` pins the calling thread to `{4,5,6,7}`
for the duration of `infer()`, restoring the previous mask afterwards.

##### 3.2.4.4. Results

Migrated. 48 kHz output, 14 voices, less than half the peak memory of v2.

---

### 3.3. Playback Level

Nothing in XEye sets the output volume. `/tts` scales the model's float output linearly to int16
(`audio * 32767`) with no normalisation and no limiter, so the digital level is whatever the model
produced for that utterance, and `aplay` writes the PCM to the device without touching gain.
Loudness is therefore whatever the ReSpeaker's ALSA mixer happens to be set to.

It is set by hand with `alsamixer -c <ReSpeaker card>`, and the mixer is the correct layer for it:
`pipeline.py` addresses the card as `plughw:`, which bypasses PulseAudio entirely, so a mixer
change applies directly and nothing at the desktop level can reroute or rescale it.

**`alsamixer` does not persist the change when it is made.** `alsa-restore.service` restores state
at boot but writes it out only from its `ExecStop`, on a clean shutdown. A device running from a
battery pack is hard-powered-off routinely — the pack runs down, or power is simply pulled — and
that write never happens, so the level silently reverts to whatever was last stored.
`alsactl store` writes `/var/lib/alsa/asound.state` immediately, and is what makes a setting
survive.

Two gaps remain open. The level does not adapt to the environment, so a setting that carries
indoors can be lost under traffic — and unlike the capture path there is no automatic gain to fall
back on, since the XU316's AGC operates on the microphones only. The user also cannot change the
volume without a shell, which is the same class of problem the button solved for triggering.

### 3.4. Interaction Cues

The device has no screen, so a user who cannot see it has no way to tell whether it is
listening, working, or broken. Four generated cues carry that state.

| Cue | Fires when | Sound |
|-----|------------|-------|
| Ready | The button loop starts, after models have loaded | 600 → 900 → 1200Hz, 390ms |
| Listening | The button is pressed, before the microphone opens | 800 → 1200Hz, 130ms |
| Captured | VAD has ended the question (2.2) | 1200 → 800Hz, 130ms |
| Error | A query failed — no speech, camera, or server | 300Hz twice, 290ms |

**Rising means open, falling means closed.** Listening and captured are deliberate mirror images
so the pair is learned as a single gesture bracketing the question rather than as two unrelated
sounds. Error is lower and doubled rather than a chirp at all: a failure should not sound like a
variant of success. The 800-1200Hz band is where hearing is most sensitive and sits above
low-frequency traffic noise, which matters given 2.3.

Ready exists because the 12.9s of model loading is otherwise silent, and a wearable that has not
finished booting is indistinguishable from one that is broken.

**Generated, not sampled.** The tones are synthesised with numpy and written to `aplay` through
the same raw-PCM path as the answer at the same 48kHz, so there are no audio assets to ship and
no resampling. Each segment carries a 5ms attack and release; a bare sine burst starts and ends
on a discontinuity and clicks audibly.

**The listening cue blocks.** It has to be out of the speaker before `arecord` opens the
microphone, or the VAD scores the tone as speech and ends the question before the user has
spoken. The XU316's echo canceller might suppress it, but a pure tone is an awkward case for a
canceller tuned on speech, and ~130ms of waiting is cheaper than depending on it. Captured is
played after the recorder has closed, for the same reason.

Cues follow `--no-play`, so a run asked to stay silent stays silent, and a cue that fails to play
is never fatal — a missing sound card degrades the interaction rather than ending the query.

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
| Max new tokens | 128 |
| Repeat penalty | 1.1 (override: `XEYE_VLM_REPEAT_PENALTY`) |

`llama-cpp-python` defaults `repeat_penalty` to 1.0 — disabled — where llama.cpp's own default
is 1.1. Left at 1.0, the model occasionally falls into a repetition loop that runs to the token
cap: one run in ten produced `"... giá cả - Giá cả - Giá cả - Giá cả"`. Measured cost of the
penalty is **2 ms/token** (47 → 49 ms/token), or ~0.1s on a 60-token answer.

#### 4.1.2. Performance

Measured on the demo photo — 2568×1926, downscaled by the server to 960×720 (every input is fitted into a 1280×720 box).

| Stage | Time |
|-------|------|
| Image encode (mmproj/clip) | 12.5-13.9s |
| Image prefill (256 image tokens) | 3.8-3.9s |
| **Total per image** | **17-21s** |

The image cost is fixed per request and dominates: ~16-18s elapses before the first text token. The `tok_s` value the server reports is `completion_tokens ÷ total elapsed`, so it rises with answer length rather than describing a decode rate:

| Answer length | Reported throughput |
|---------------|---------------------|
| 58-75 tokens (full description, default prompt) | 3.0-3.7 tok/s |
| 17-29 tokens (short answer to a specific question) | 1.0-1.7 tok/s |

Output language: Vietnamese.

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
| Backbone | ~491MB |
| Quantization | Q4_K_M |
| Format | GGUF |
| mmproj | ~620MB, F16 |
| Runtime | llama-cpp-python 0.3.16 |

##### 4.2.8.2. Issues & Solutions

`Llava16ChatHandler` uses `USER: ... ASSISTANT:` format → garbled output with Vintern.

Fix: subclassed `Llava15ChatHandler` — image injection occurs at C-level, overriding `CHAT_FORMAT` with Jinja2 ChatML:

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

The template also covers system and assistant turns, plain-string content, and both string and mapping forms of `image_url`, since llama-cpp-python can hand over any of them.

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

#### 4.2.10. Experiment 10: Speed Ceiling Investigation

Quality at Vintern-1B-v3_5 was judged sufficient, so this pass looked only for latency wins
that preserve it. The image path costs ~16s of every request (12-13s encode + ~4s prefill),
so that was the target.

##### 4.2.10.1. Is the Encoder Compute-Bound?

| Cores | Threads | Median (encode + prefill) |
|-------|---------|---------------------------|
| `{4,5,6,7}` | 4 | **19.02s** |
| all 8 | 8 | 19.20s |
| all 8 | 6 | 19.29s |
| `{4,5,6,7}` | 3 | 21.84s |

Doubling the core count changes nothing; dropping below 4 hurts. Input resolution is equally
inert — 960×720, 448×448 and 224×224 encode in 12.4s, 13.2s and 13.7s respectively, because
the model always processes one fixed 448px tile.

##### 4.2.10.2. mmproj F16 → Q8_0

The F16 projector is 592MB, the largest single artifact in the image path. Requantizing its
146 2D weight tensors to Q8_0 (318MB) was expected to cut memory traffic.

| mmproj | Size | Encode + prefill |
|--------|------|------------------|
| **F16** | 592MB | **18.85s** |
| Q8_0 | 318MB | 20.84s |

→ **Rejected: 10% slower.** Output quality was unchanged. The CPU exposes `asimdhp` (native
FP16) but not `i8mm` or SVE, so the F16 kernels are already well matched to the hardware while
Q8_0 adds dequantization overhead with no int8 matmul path to recover it.

Note `llama-quantize` cannot do this — it rejects architecture `clip`. The file was produced
by requantizing tensors directly through `gguf-py`.

##### 4.2.10.3. Upstream llama.cpp

Built `llama-mtmd-cli` from master and ran the same photo: master applies InternVL's dynamic
**4-tile** preprocessing, encoding in 52s + 13s versus 12-13s on the pinned version.
`--image-max-tokens` does not override it.

→ **The `llama-cpp-python==0.3.16` pin is load-bearing.** Current latency exists because that
version's `Llava15ChatHandler` encodes a single tile. Bumping it quadruples VLM latency.
The corollary is that XEye runs the model at lower effective resolution than upstream intends —
a deliberate quality-for-speed trade.

##### 4.2.10.4. Native Recompilation

The prebuilt `llama-cpp-python` wheel targets a generic aarch64 baseline, so rebuilding from
source with `-march=native` looked like free throughput on a CPU whose features are known.

It is not, because the kernels that dominate matmul are not selected at compile time. llama.cpp
builds with `LLAMAFILE=1`, whose sgemm path dispatches on CPU features detected at *runtime* —
the same kernel executes whether or not the build was told about the target. Nor is there a wider
path waiting to be unlocked: 4.2.10.2 established that this CPU exposes `asimdhp` but neither
`i8mm` nor SVE, so the widest matmul kernel available is already the one being chosen.

→ **Ineffective.** A native rebuild runs the same kernels as the stock wheel.

##### 4.2.10.5. Results

No lever improved on the current configuration. 17-21s per image is the floor for this model
on this hardware.

---

#### 4.2.11. Experiment 11: Prompt Length Tuning

The VLM itself is a fixed ~20s, but answer length costs roughly **0.25s of interaction per
token** (TTS synthesis plus the speech the user listens through). A 62-token answer spends
32s of a 52s interaction on speaking alone, so terser prompts looked like free latency.

Five prompt variants were run against three images (one text-heavy meeting room, two live
camera frames), two runs each, and graded against the actual image content rather than by
the model itself.

| Prompt | Coverage on live scenes | Fabrication | Failure mode |
|--------|-------------------------|-------------|--------------|
| **Current** `Mô tả những gì bạn thấy.` | bottle, pink keyboard, headphones, window, phone | slide titles | one runaway to token cap |
| `Chỉ nêu vật thể chính…` | person + desk only | invented "playing a game" | — |
| `Có gì trước mặt tôi?` | **missed the person entirely** (2/2) | invented a slide title | — |
| `Nêu ngắn gọn những người và vật thể chính` | person + phone | none | collapsed to 3 tokens once |
| `Mô tả … trong hai câu ngắn.` | person + phone + headphones | said "computer" for a phone | — |

The concise variants were 24s faster but only because they omitted the objects that make the
device useful — a water bottle within reach, the keyboard, the window. They also still
fabricated ("playing a game" when the subject was looking at a phone).

Appending `Không đọc chữ.` ("do not read text") did **not** suppress the invented slide
titles — a 1B model does not reliably follow negative instructions. The fabrication is a
property of Vintern when text is in frame, not a prompt defect.

→ **No change to length. The original prompt is retained.** Shorter answers are not more
efficient here, they are less useful.

Caveat: all three test images are "person at a desk" scenes. Real wearable imagery — walking,
doorways, signage — may score differently and should be re-tested when available.

##### 4.2.11.1. Forcing Vietnamese Output

On empty or nonsense questions (e.g. STT returning just `"rồi"`), the model fell back to
**English** — measured 5/10 across degenerate inputs, including outright refusals like
*"I'm unable to provide a detailed description of the image."* Four fixes were tested against
those 10 inputs:

| Strategy | English replies |
|----------|-----------------|
| Baseline | 5/10 |
| System message (`Luôn trả lời bằng tiếng Việt`) | 3/10 |
| **Prompt suffix `" Trả lời bằng tiếng Việt."`** | **0/10** |
| System message + suffix | 0/10 |

A system message alone was insufficient — a 1B model does not weight the system role strongly.
The instruction appended directly to the user prompt is what holds. The suffix is now added to
every `/vlm` prompt (`VLM_LANG_SUFFIX`).

Cost: prefill and decode rate are unchanged (~8 extra tokens ride in the 256-token image
prefill batch). Answer correctness is unchanged — the same objects are identified, and the
slide-title hallucination persists identically. Answers trend slightly longer (~+20 tokens on
average, high variance). It does not fix accuracy, only language.

---

## 5. System Optimization

### 5.1. Thread Affinity

Mixing performance and efficiency cores in the same thread pool creates stall barriers, reducing throughput.

**Issue with process-level pinning:**

Pinning the entire process to performance cores restricts uvicorn/FastAPI threads too, competing with inference threads → total pipeline ≥22s, worse than leaving it unpinned (<22s) as measured at the time. Absolute totals vary with answer length, so treat this as a relative comparison — see 5.2 for the current end-to-end figure.

**Fix:** Pin affinity before model load so inference thread pools inherit the mask, then restore before uvicorn handles requests:

```python
os.sched_setaffinity(0, {4, 5, 6, 7})
models["stt"] = STTPipeline()   # inference threads inherit {4,5,6,7}
models["vlm"] = VLMPipeline()
models["tts"] = TTSPipeline()
os.sched_setaffinity(0, set(range(8)))  # restore for uvicorn
```

Verified on the running server (41 threads total):

| Threads | Affinity | Origin |
|---------|----------|--------|
| 9 | `[4,5,6,7]` | llama.cpp inference pools — inherited the mask as intended |
| 18 | `[0-7]` | uvicorn / FastAPI / asyncio — restored, as intended |
| 14 | one core each, spread over `[1-7]` | onnxruntime (sherpa-onnx STT + TTS sessions) |

The last group is the exception: onnxruntime pins its own intra-op threads and ignores the inherited mask, so some land on efficiency cores `[1,2,3]`. They are only active during the short STT and codec stages, not during VLM decode, so this was left alone — the pinning does what it was intended to do for the stage that dominates latency.

Load-time pinning is not sufficient on its own for onnxruntime, because the *calling* thread also acts as an intra-op worker. See 3.2.4.3 for the per-request pinning that the TTS path needs.

---

### 5.2. Pipeline Concurrency

Three stages were serialized for no reason. All three were overlapped:

**Camera capture during recording.** Capture (gstreamer startup + ~2s exposure settle) ran
before the microphone opened, though the two are independent. The camera now streams for the
duration of the question and the frame is taken at the end of it, so it costs nothing unless
the question ends before exposure has settled — see 5.5.

**Synthesis one sentence ahead of playback.** TTS previously rendered the whole answer before
any sound played. The answer is now split into sentences, sentence N+1 renders while sentence N
plays, and all chunks feed one `aplay` process reading raw PCM from stdin so playback is
gapless. Time to first sound dropped from **4.05-4.39s** (whole-answer synthesis) to
**1.1-2.7s** depending on the first sentence's length. The gain grows with answer length.

**Non-blocking server.** The endpoints were `async def` performing blocking inference, which
stalled the uvicorn event loop for the duration of every request — `/health` could not answer
while the VLM was running. They are now sync `def`, so FastAPI dispatches them to its
threadpool, with an `INFERENCE_LOCK` serializing model access (the models share 4 cores;
parallel requests would only thrash). `/health` now responds in 5-9ms during a VLM request.

Remaining overhead outside the models: ~10ms image preprocessing, ~5ms HTTP.

### 5.3. End-to-End Latency

Full pipeline, warm, `pipeline.py` with a WAV question and the demo photo (downscaled to 960×720):

| Stage | Time |
|-------|------|
| STT | 0.15s |
| VLM | 17-21s |
| TTS first chunk | 1.1-2.7s |
| **Time to first sound** | **~20s** |
| **Total** | **~24-26s** |

With live hardware the camera is hidden inside the question and the recording ends when the
speaker does, so the total becomes `spoken_question + 0.8s + ~21s`. Longer answers add time at both
VLM decode and TTS. The first run after server startup is slower — STT and the codec sessions
warm up on first use. Server startup to first servable request is **12.9s** with warm page cache.

### 5.4. Thermal Behaviour

Under sustained inference the SoC runs at **82-89°C** (idle readings of 46-49°C are not
representative). Encode time rises from a cold start and then plateaus:

| Requests | Encode time |
|----------|-------------|
| 1 | 13.3s |
| 5 | 14.8s |
| 10 | 15.2s |
| 15-20 | **15.3s** (flat, ±0.05s) |

The device does not degrade continuously — it settles ~2s above cold start and holds. CPU
frequency confirms the mechanism: cpu7 holds 2707MHz for the first dozen requests, then drops
to 2208, 2515 and twice to **2035MHz** — a 25% clock reduction as throttling engages.

This was measured on an open desk. Inside an enclosure worn against the body, throttling will
arrive sooner and cut deeper; the numbers here should not be assumed to transfer.

---

### 5.5. Camera Capture

The frame is captured with GStreamer `qtiqmmfsrc` at 1280×720 NV12 (`RollingCamera` in
`pipeline.py`), JPEG-encoded and written through `multifilesink`. Two properties of the sensor
shape how this is done.

**Auto-exposure needs time to settle.** The first frames of any stream are dark — roughly 5 frames
pass before AE converges. The stream therefore stays open for the whole question, writing into a
10-frame ring buffer, and a frame is taken once the speaker stops. Taking the first frame
instead would sample the sensor mid-convergence.

This replaced a one-shot ~2s warmup burst, which was safe only while recording used a fixed 5s
window. With VAD ending the recording as soon as the speaker stops, a short question would have
finished before the burst did and put the camera back on the critical path. Streaming for the
duration removes the coupling entirely: the sensor is converged however long the question runs,
and the frame is contemporaneous with the question rather than with the start of listening. The
only remaining wait is when a question ends sooner than AE converges, which is floored at ~2s.

The ring buffer is written to tmpfs where available. At 30fps a long question is several MB of
JPEG, which does not belong on the board's flash — see 5.6 for why that matters.

**The sharpest buffered frame is chosen, not the newest.** Motion blur scales with the camera's
instantaneous angular velocity, which rises and falls through a walking gait, so a window spanning
part of a step usually holds a stiller moment than its final frame. Each frame is scored by
variance of the Laplacian on a ¼-scale draft decode — blur is a low-frequency property that
survives downscaling, so scoring at 320×180 is equally valid and roughly ten times faster.

| Measurement | Result |
|-------------|--------|
| Discrimination (sharp vs Gaussian-blurred copy) | 1263 vs 160 — **7.9×** |
| Cost to score 10 frames | **43ms** (against a ~18s VLM call) |
| Static camera (nothing to gain) | picks a frame 1.00× the newest — a clean no-op |
| Synthetic burst, sharp frame buried at position 7 | selected it, **17.69×** sharper than newest |

Buffer depth is 10 frames, ~333ms at 30fps: wide enough to span part of a gait cycle while the
frame stays contemporaneous with the question. Unreadable frames score 0 and can never win, and if
every frame fails to decode the newest is used — a torn write must not be able to fail a query.

**Exposure.** `exposure-compensation` accepts −12..12. The IMX708 tolerates it far better than the
IMX219 did; measured on the same indoor scene:

| exposure-compensation | Brightness | Contrast | Blown out |
|-----------------------|-----------|----------|-----------|
| +2 | 126.9 | 44.1 | 4.2% |
| **+4** | **133.7** | **44.7** | 5.8% |
| +6 | — | 31 | — |

→ **`EXPOSURE = 2`.** The figures for the IMX219 were much harsher (+4 blew out 22% of pixels), so
this setting is sensor-specific and had to be re-measured when the module changed. On the IMX708
there is headroom above 2: contrast holds to +4 and only flattens between +4 and +6.

**Exposure stays on auto — the shutter is not pinned.** Shortening it would cut motion blur:
10ms with ISO 1600 measured identical brightness (132.2 vs 129.9) and identical noise (0.76 vs
0.77) at a 3.3× shorter shutter, and the noise trade is unusually favourable here because
downscaling to 448×448 averages noise away but cannot undo blur. It was rejected anyway. Pinning
shutter and gain costs the ~7.6 stops of adaptation between a 400-lux room and 80,000-lux
sunlight; outdoors the frame would be pure white with no mechanism to recover. A blurred frame
still describes, a blown one does not — and for a mobility aid, outdoors is not an edge case.

**Resolution and aspect ratio are not levers.** The model sees one fixed 448×448 tile regardless
of input (4.2.10), encode time is flat against input size, and server-side preprocessing is 10ms.
1280×720 sits above 448 in both axes so nothing is upscaled, and downscaling from it supplies free
antialiasing. Requesting 4:3 (1280×960) does not add view — it **crops ~33% of the horizontal
field**, measured by comparing a landmark's pixel width across both modes. A wearable should not
trade peripheral awareness for a marginally smaller aspect squeeze.

> **Mode support is narrow and fails hard.** 960×720 and 1920×1080 both **crash `cam-server`**
> outright rather than erroring; systemd restarts it in ~2s, but the capture is lost and Ubuntu
> raises a crash report. Stay on 1280×720.

**One consumer only.** The camera admits a single reader, so a capture and
`scripts/camera_preview.py` cannot run at the same time. The preview server tracks the live
`gst-launch-1.0` process and terminates it when a new viewer connects, so a stale stream cannot
lock the camera out.

#### 5.5.1. Module Substitution: IMX219 → IMX708

The Camera Module 2 stopped being detected — every I2C probe returned `read id: 0x0` with a NACK,
on both CSI connectors and every sensor address, while both CSI PHYs and both sensor components
bound normally and no package, kernel or device-tree change had occurred since the last working
boot. A NACK proves the controller clocked out an address correctly and nothing acknowledged, so
the fault is electrical, not software. Substituting a Module 3 on the same connector and software
probed immediately (`sensor_id:0x708`), isolating the fault to the **IMX219 module or its ribbon
cable**. Both remain untested individually.

The Module 3 is a working substitute but is **not** the better sensor for this application:

| | Module 2 (IMX219) | Module 3 (IMX708) |
|---|---|---|
| Horizontal FOV | 62.2° | 66° |
| Aperture / focal length | f/2.0, 3.04mm | f/1.8, 4.74mm |
| Hyperfocal | ~1.5m → sharp from **~0.8m** to ∞ | ~2.6m → sharp from **~1.3m** to ∞ |
| Focus | fixed by design | VCM, **no actuator driver bound** |

**The Module 3's autofocus does not work and cannot be made to.** Thundercomm's documentation
states plainly that *"the current software version does not support the autofocus (AF) function of
the Module 3 camera"*, and `dmesg` confirms it at the hardware level — no `CAM-ACTUATOR` is ever
probed, so the voice-coil motor has no driver. `qtiqmmfsrc` accepts `focus-mode` without error and
does nothing with it, which is the dangerous part: it fails silently.

The lens therefore rests wherever its spring parks it unpowered, a position that is neither
selectable nor guaranteed between units. On the unit tested it lands usefully far — a face at
~0.5–0.8m scored 179 for sharpness against 180–311 for a wall 2–3m behind, i.e. near and far
comparably sharp with no visible focal plane. That is the deep-depth-of-field rendering this
application wants, but it is luck rather than configuration, and the Module 2 delivers it by
design with a nearer limit. Restoring the Module 2 remains preferable once a replacement ribbon
cable identifies whether the module or the cable failed.

**Orientation.** The Module 3 as mounted delivers frames rotated 180°, so the capture path
rotates the grabbed frame back before it is sent (`ROTATE = 180` in `pipeline.py`, overridable
with `XEYE_CAMERA_ROTATE`). Orientation measurably affects description quality — asked the same
question about the same frame, the VLM answered *"một chiếc máy móc"* (a machine) inverted
against *"một chiếc laptop"* (a laptop) upright. The rotation is applied to the single chosen
frame rather than the video stream, so it costs ~10ms per query instead of running on every
streamed frame.

**`CAMERA` is a camera index, not a connector number.** `qtiqmmfsrc` enumerates *detected*
cameras, so with one module attached the index is 0 whichever CSI connector it is plugged into —
the module currently sits on connector 2 and is still `camera=0`. Requesting an index with no
camera behind it does not raise an error; the pipeline simply fails to preroll. The kernel probe
line reports the real slot:

```
$ dmesg | grep "Probe success"
Probe success,slot:1,slave_addr:0x34,sensor_id:0x708
```

### 5.6. Runtime Modes

`pipeline.py` runs in one of two modes, selected by `--mode`, the `XEYE_MODE` environment variable,
or the `MODE` constant:

| Mode | Behaviour |
|------|-----------|
| `dev` *(default)* | Writes the synthesized answer to `data/audio/output.wav` |
| `prod` | Writes nothing to disk — audio goes only to the speaker |

In `prod` the PCM chunks are streamed straight into `aplay` and never accumulated, so the answer
exists only in memory. This matters twice over. Answer audio at 48kHz 16-bit mono costs 96 KB/s,
and 3.1.3 measures a typical answer at 3.9-4.6s of audio, so `dev` writes roughly 0.4MB per query
and up to ~1MB on a long one — continuous flash wear on a device expected to answer questions all
day. It is also a privacy property: a wearable that records what its user asked and what was in
front of them leaves that history on disk, and `prod` leaves none.

`--output PATH` overrides both and always saves, for one-off debugging.

---

## 6. Summary

### 6.1. Selected Models

| Component | Model | Params | Quantization | Runtime | Performance |
|-----------|-------|--------|--------------|---------|-------------|
| STT | ZipFormer-30M RNNT | ~30M | int8 | sherpa-onnx | 0.05-0.11s latency, RTF 0.03-0.07x |
| TTS | VieNeu-TTS-v3-Turbo | ~0.1B | int8 | onnxruntime 1.24.4 | RTF 0.80-0.97x (~3.5s per answer), 48 kHz |
| VLM | Vintern-1B-v3_5 | ~1B | Q4_K_M | llama-cpp-python 0.3.16 | 17-21s/image (3.0-3.7 tok/s reported) |

End-to-end: ~24-26s per question, with first sound at ~20s (5.3). Under sustained use the
image encode settles ~2s higher as the SoC throttles (5.4).

### 6.2. Current Limitations

17-21s per image is the ceiling for a 1B VLM with the current software configuration, and ~16-18s of that is the fixed image encode + prefill cost rather than token generation. Acceleration paths explored:

| Approach | Status | Reason |
|----------|--------|--------|
| GPU Vulkan (Turnip) | Blocked | Workgroup barrier shader not supported |
| NPU (QNN EP) | Ineffective | 0 ops offloaded to NPU |
| Increase thread count | Ineffective | Stall barriers when mixing core types |
| llama-cpp native recompile | Ineffective | LLAMAFILE=1 runtime dispatch already optimized |

No significant software leverage remaining at current configuration.

**Camera.** The Module 2 no longer probes and a Module 3 is substituted (5.5.1). Two consequences
carry into the current build:

| Limitation | Effect |
|------------|--------|
| No autofocus driver on the IMX708 | Focus is fixed at the lens's unpowered rest position — usable on the unit tested, but neither selectable nor guaranteed across units |
| Near limit ~1.3m vs the Module 2's ~0.8m | Objects within arm's reach are less well resolved than the Module 2 would render them |
| Frames arrive rotated 180° | Corrected in the capture path (`ROTATE = 180`), so the VLM receives an upright scene |

The first two resolve by restoring a Module 2 once a replacement ribbon cable establishes whether
the module or the cable failed.
