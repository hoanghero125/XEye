import base64
import io
import os
import time
import wave

# os.sched_setaffinity(0, {4, 5, 6, 7})  # pin to perf cores (A78+X1, cpu4-7 on QCS6490)
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

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
TTS_SAMPLE_RATE = 24000
MAX_NEW_TOKENS  = 128

VLM_DEFAULT_PROMPT = "Mô tả những gì bạn thấy."


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
        print("[TTS] Loading VieNeu-TTS-v2-Turbo GGUF + VieNeu-Codec ONNX ...")
        from vieneu import Vieneu
        self.tts = Vieneu(mode="turbo", device="cpu", n_threads=4)
        print(f"[TTS] Voices: {list(self.tts._preset_voices.keys())}")
        print("[TTS] Ready.\n")

    def _get_voice(self, name: Optional[str]) -> Optional[Any]:
        if name is None:
            return None
        v = self.tts._preset_voices.get(name)
        if v is None:
            raise ValueError(f"Unknown voice '{name}'. Available: {list(self.tts._preset_voices.keys())}")
        return v

    def synthesize(self, text: str, voice: Optional[str] = "Bích Ngọc (Nữ - Miền Bắc)") -> np.ndarray:
        t0 = time.time()
        audio = self.tts.infer(text, voice=self._get_voice(voice))
        elapsed = time.time() - t0
        audio_sec = len(audio) / TTS_SAMPLE_RATE
        print(f"[TTS] {len(text)} chars → {audio_sec:.1f}s audio in {elapsed:.1f}s (RTF {elapsed/audio_sec:.2f}x)")
        return audio


# ── Server ───────────────────────────────────────────────────────────────────

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] Loading models ...")
    os.sched_setaffinity(0, {4, 5, 6, 7})  # inference threads inherit perf cores
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
async def stt(audio: UploadFile = File(...)):
    """WAV audio → Vietnamese text."""
    data = await audio.read()
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr  = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    text = models["stt"].transcribe(pcm, sample_rate=sr)
    return {"text": text}


@app.post("/vlm")
async def vlm(
    image:    UploadFile = File(...),
    question: str        = Form(default=""),
):
    """Image + Vietnamese question → Vietnamese answer."""
    data = await image.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1280, 720))

    prompt                    = question.strip() if question.strip() else VLM_DEFAULT_PROMPT
    vi_text, n_tok, elapsed, tok_s = models["vlm"].describe(img, prompt)
    return {"vi": vi_text, "tokens": n_tok, "elapsed_s": elapsed, "tok_s": tok_s}


@app.post("/tts")
async def tts(
    text:  str = Form(...),
    voice: str = Form(default="Bích Ngọc (Nữ - Miền Bắc)"),
):
    """Vietnamese text → WAV audio bytes."""
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
