import base64
import io
import os
import threading
import time
import wave

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import sherpa_onnx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from PIL import Image


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

ROOT            = Path(__file__).resolve().parent
STT_DIR         = ROOT / "models" / "stt"
VINTERN_DIR     = ROOT / "models" / "vintern"

VINTERN_MODEL  = VINTERN_DIR / "vintern-1b-v3_5-q4_k_m.gguf"
VINTERN_MMPROJ = VINTERN_DIR / "mmproj-vintern-1b-v3_5-f16.gguf"

STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 48000
MAX_NEW_TOKENS  = 128

VLM_DEFAULT_PROMPT = "Mô tả những gì bạn thấy."
# Vintern falls back to English on empty/nonsense questions (measured 5/10 of the time).
# Appending this to every prompt forces Vietnamese — a system message alone was not enough
# for a 1B model; the instruction beside the question is what holds (measured 10/10 VI).
VLM_LANG_SUFFIX = " Trả lời bằng tiếng Việt."
# llama-cpp-python defaults this to 1.0 (disabled), which lets the model fall into
# repetition loops that run to the token cap. llama.cpp's own default is 1.1.
VLM_REPEAT_PENALTY = float(os.getenv("XEYE_VLM_REPEAT_PENALTY", "1.1"))

TTS_DEFAULT_VOICE = "Mai Anh"
TTS_STYLE         = "tu_nhien"  # pinned: preset styles (tin_tuc/doc_truyen) pause mid-sentence

PERF_CORES = {4, 5, 6, 7}  # A78 ×3 + X1, cpu4-7 on QCS6490


@contextmanager
def perf_cores():
    """Pin the calling thread to the performance cores.

    ONNX Runtime uses the calling thread as one of its intra-op workers, so a request
    served on an unpinned uvicorn thread can land on an A55 and stall the whole op.
    """
    prev = os.sched_getaffinity(0)
    os.sched_setaffinity(0, PERF_CORES)
    try:
        yield
    finally:
        os.sched_setaffinity(0, prev)


# ── STT ──────────────────────────────────────────────────────────────────────

class STTPipeline:
    def __init__(self):
        print("[STT] Loading ZipFormer RNNT int8 ...")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(STT_DIR / "encoder-epoch-20-avg-10.int8.onnx"),
            decoder=str(STT_DIR / "decoder-epoch-20-avg-10.int8.onnx"),
            joiner=str(STT_DIR  / "joiner-epoch-20-avg-10.int8.onnx"),
            tokens=str(STT_DIR / "config.json"),
            num_threads=4,
            sample_rate=STT_SAMPLE_RATE,
            decoding_method="greedy_search",
        )
        print("[STT] Ready.\n")

    def transcribe(self, audio: np.ndarray, sample_rate: int = STT_SAMPLE_RATE) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        t0 = time.time()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=audio)
        self.recognizer.decode_stream(stream)
        elapsed = time.time() - t0

        text = stream.result.text.strip().lower()
        audio_sec = len(audio) / sample_rate
        print(f"[STT] {audio_sec:.1f}s audio → {elapsed:.2f}s decode (RTF {elapsed/audio_sec:.2f}x)")
        return text


# ── VLM ──────────────────────────────────────────────────────────────────────

class VLMPipeline:
    def __init__(self):
        print("[VLM] Loading Vintern-1B Q4_K_M + mmproj F16 ...")
        handler = InternVL2ChatHandler(clip_model_path=str(VINTERN_MMPROJ), verbose=False)
        self.llm = Llama(
            model_path=str(VINTERN_MODEL),
            chat_handler=handler,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        print("[VLM] Ready.\n")

    def describe(self, image: Image.Image, prompt: str) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"

        t0 = time.time()
        resp = self.llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=MAX_NEW_TOKENS,
            repeat_penalty=VLM_REPEAT_PENALTY,
        )
        text    = resp["choices"][0]["message"]["content"].strip()
        elapsed = time.time() - t0
        n_tok   = resp["usage"]["completion_tokens"]
        tok_s   = n_tok / elapsed
        print(f"[VLM] {n_tok} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)")
        return text, n_tok, round(elapsed, 2), round(tok_s, 1)


# ── TTS ──────────────────────────────────────────────────────────────────────

class TTSPipeline:
    def __init__(self):
        print("[TTS] Loading VieNeu-TTS-v3-Turbo (ONNX int8) ...")
        from vieneu import Vieneu
        # threads=2 measured fastest in-server: v3 builds ~8 ONNX sessions and 4 intra-op
        # threads each oversubscribes the 4 performance cores (9.5s vs 10.9s on a 203-char text).
        self.tts = Vieneu(mode="v3turbo", device="cpu", threads=2)
        self.voices = [vid for _, vid in self.tts.list_preset_voices()]
        print(f"[TTS] Voices: {self.voices}")
        print("[TTS] Ready.\n")

    def synthesize(self, text: str, voice: Optional[str] = TTS_DEFAULT_VOICE) -> np.ndarray:
        name = voice or TTS_DEFAULT_VOICE
        if name not in self.voices:
            raise ValueError(f"Unknown voice '{name}'. Available: {self.voices}")

        t0 = time.time()
        with perf_cores():
            audio = self.tts.infer(text, voice=name, style=TTS_STYLE, apply_watermark=False)
        elapsed = time.time() - t0
        audio_sec = len(audio) / TTS_SAMPLE_RATE
        print(f"[TTS] {len(text)} chars → {audio_sec:.1f}s audio in {elapsed:.1f}s (RTF {elapsed/audio_sec:.2f}x)")
        return audio


# ── Server ───────────────────────────────────────────────────────────────────

models: dict = {}

# Endpoints are sync `def`, so FastAPI runs them in its threadpool and the event loop stays
# free (/health answers during inference). The lock keeps the models single-consumer — they
# share the same 4 performance cores, so parallel requests would only thrash.
INFERENCE_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] Loading models ...")
    os.sched_setaffinity(0, PERF_CORES)  # inference threads inherit perf cores
    models["stt"] = STTPipeline()
    models["vlm"] = VLMPipeline()
    models["tts"] = TTSPipeline()
    os.sched_setaffinity(0, set(range(8)))  # restore for uvicorn
    print("[server] All models ready.")
    yield
    models.clear()


app = FastAPI(title="XEye API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "models": list(models.keys())}


@app.post("/stt")
def stt(audio: UploadFile = File(...)):
    """WAV audio → Vietnamese text."""
    data = audio.file.read()
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    with INFERENCE_LOCK:
        text = models["stt"].transcribe(pcm, sample_rate=sr)
    return {"text": text}


@app.post("/vlm")
def vlm(
    image:    UploadFile = File(...),
    question: str        = Form(default=""),
):
    """Image + Vietnamese question → Vietnamese answer."""
    data = image.file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1280, 720))

    prompt = (question.strip() or VLM_DEFAULT_PROMPT) + VLM_LANG_SUFFIX
    with INFERENCE_LOCK:
        vi_text, n_tok, elapsed, tok_s = models["vlm"].describe(img, prompt)
    return {"vi": vi_text, "tokens": n_tok, "elapsed_s": elapsed, "tok_s": tok_s}


@app.post("/tts")
def tts(
    text:  str = Form(...),
    voice: str = Form(default=TTS_DEFAULT_VOICE),
):
    """Vietnamese text → WAV audio bytes."""
    with INFERENCE_LOCK:
        audio = models["tts"].synthesize(text, voice=voice)
    pcm   = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
