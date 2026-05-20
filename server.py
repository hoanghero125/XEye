import io
import wave
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response

from scripts.stt_infer import STTPipeline
from scripts.vlm_infer import VLMPipeline
from scripts.tts_infer import TTSPipeline

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[server] Loading models ...")
    models["stt"] = STTPipeline()
    models["vlm"] = VLMPipeline()
    models["tts"] = TTSPipeline()
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
    from PIL import Image
    data = await image.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1280, 720))

    pipeline = models["vlm"]
    if question.strip():
        en_prompt = pipeline.vi_to_en(question)
    else:
        en_prompt = "Describe the scene in front of me."

    en_text = pipeline.describe(img, en_prompt)
    vi_text = pipeline.en_to_vi(en_text)
    return {"question_en": en_prompt, "en": en_text, "vi": vi_text}


@app.post("/tts")
async def tts(
    text:  str = Form(...),
    voice: str = Form(default="Ly"),
):
    """Vietnamese text → WAV audio bytes."""
    audio = models["tts"].synthesize(text, voice=voice)
    pcm   = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm.tobytes())

    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
