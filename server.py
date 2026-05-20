import io
import json
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import ctranslate2
import numpy as np
import onnxruntime as ort
import sherpa_onnx
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from PIL import Image
from transformers import AutoConfig, AutoProcessor, MarianTokenizer

ROOT          = Path(__file__).resolve().parent
STT_DIR       = ROOT / "models" / "stt"
VLM_DIR       = ROOT / "models" / "vlm"
NMT_EN_VI_DIR = ROOT / "models" / "nmt" / "opus-mt-en-vi-int8"
NMT_VI_EN_DIR = ROOT / "models" / "nmt" / "opus-mt-vi-en-int8"
TTS_DIR       = ROOT / "models" / "tts"
ONNX_DIR      = VLM_DIR / "onnx"

STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
MAX_NEW_TOKENS  = 128

VLM_PROMPT_PREFIX = (
    "Answer based only on what is visible in the image. "
    "Be accurate and descriptive — include relevant details about the subject, surroundings, and context. "
    "Do not fabricate anything not present. "
)



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
        print("[VLM] Loading config and processor ...")
        self.config    = AutoConfig.from_pretrained(str(VLM_DIR))
        self.processor = AutoProcessor.from_pretrained(str(VLM_DIR))

        text_cfg             = self.config.text_config
        self.eos_token_id    = self.processor.tokenizer.eos_token_id
        self.image_token_idx = self.config.image_token_id
        self._num_kv_heads   = text_cfg.num_key_value_heads
        self._head_dim       = text_cfg.head_dim
        self._num_layers     = text_cfg.num_hidden_layers

        providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.inter_op_num_threads = 4
        so.intra_op_num_threads = 4

        print("[VLM] Loading ONNX sessions (FP32) ...")
        self.vision_sess  = ort.InferenceSession(str(ONNX_DIR / "vision_encoder.onnx"),      sess_options=so, providers=providers)
        self.embed_sess   = ort.InferenceSession(str(ONNX_DIR / "embed_tokens.onnx"),         sess_options=so, providers=providers)
        self.decoder_sess = ort.InferenceSession(str(ONNX_DIR / "decoder_model_merged.onnx"), sess_options=so, providers=providers)

        self._kv_keys = sorted(
            [inp.name for inp in self.decoder_sess.get_inputs() if inp.name.startswith("past_key_values.")],
            key=lambda s: (int(s.split(".")[1]), s.split(".")[2]),
        )

        print("[NMT] Loading EN→VI translator ...")
        self.en_vi     = ctranslate2.Translator(str(NMT_EN_VI_DIR), device="cpu", inter_threads=2)
        self.en_vi_tok = MarianTokenizer.from_pretrained(str(NMT_EN_VI_DIR))

        print("[NMT] Loading VI→EN translator ...")
        self.vi_en     = ctranslate2.Translator(str(NMT_VI_EN_DIR), device="cpu", inter_threads=2)
        self.vi_en_tok = MarianTokenizer.from_pretrained(str(NMT_VI_EN_DIR))

        print("[VLM] Ready.\n")

    def _init_cache(self, batch_size: int) -> dict:
        return {
            f"past_key_values.{layer}.{kv}": np.zeros(
                [batch_size, self._num_kv_heads, 0, self._head_dim], dtype=np.float32
            )
            for layer in range(self._num_layers)
            for kv in ("key", "value")
        }

    def describe(self, image: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": VLM_PROMPT_PREFIX + prompt}]}]
        text   = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="np", do_image_splitting=False)

        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids   = np.cumsum(attention_mask, axis=-1)
        pixel_values   = inputs["pixel_values"]
        pix_attn_mask  = inputs["pixel_attention_mask"].astype(np.bool_)

        batch_size  = input_ids.shape[0]
        past_cache  = self._init_cache(batch_size)
        image_features = None
        generated   = np.array([[]], dtype=np.int64)

        t0 = time.time()
        for _ in range(MAX_NEW_TOKENS):
            embeds = self.embed_sess.run(None, {"input_ids": input_ids})[0]

            if image_features is None:
                print("[VLM] Running vision encoder ...", flush=True)
                image_features = self.vision_sess.run(
                    ["image_features"],
                    {"pixel_values": pixel_values, "pixel_attention_mask": pix_attn_mask},
                )[0]
                img_mask = inputs["input_ids"] == self.image_token_idx
                embeds[img_mask] = image_features.reshape(-1, image_features.shape[-1])
                print("[VLM] Prefill ...", flush=True)

            logits, *present_cache = self.decoder_sess.run(None, dict(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_cache,
            ))

            input_ids      = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)], axis=-1)
            position_ids   = position_ids[:, -1:] + 1
            for j, key in enumerate(past_cache):
                past_cache[key] = present_cache[j]

            generated = np.concatenate([generated, input_ids], axis=-1)
            if (input_ids == self.eos_token_id).all():
                break

        elapsed = time.time() - t0
        n_tok   = generated.shape[-1]
        print(f"[VLM] {n_tok} tokens in {elapsed:.1f}s ({n_tok/elapsed:.1f} tok/s)")
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def _translate(self, text: str, translator, tokenizer) -> str:
        encoded    = tokenizer([text], return_tensors=None, padding=False)
        src_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in encoded["input_ids"]]
        results    = translator.translate_batch(src_tokens)
        tgt_tokens = results[0].hypotheses[0]
        tgt_ids    = tokenizer.convert_tokens_to_ids(tgt_tokens)
        return tokenizer.decode(tgt_ids, skip_special_tokens=True)

    def vi_to_en(self, text: str) -> str:
        return self._translate(text, self.vi_en, self.vi_en_tok)

    def en_to_vi(self, text: str) -> str:
        return self._translate(text, self.en_vi, self.en_vi_tok)


# ── TTS ──────────────────────────────────────────────────────────────────────

def _load_voices() -> dict:
    p = TTS_DIR / "voices.json"
    return json.loads(p.read_text()).get("presets", {}) if p.exists() else {}


class TTSPipeline:
    def __init__(self):
        print("[TTS] Loading VieNeu-TTS-v2 (GGUF Q4-K-M) + neucodec ONNX int8 ...")
        from vieneu import Vieneu
        self.tts = Vieneu(
            mode="standard",
            backbone_repo="pnnbao-ump/VieNeu-TTS-v2",
            gguf_filename="VieNeu-TTS-v2-Q4-K-M.gguf",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec-onnx-decoder-int8",
            codec_device="cpu",
        )
        self._voices = _load_voices()
        print(f"[TTS] Voices: {list(self._voices.keys())}")
        print("[TTS] Ready.\n")

    def _get_voice(self, name: Optional[str]) -> Optional[dict]:
        if name is None:
            return None
        v = self._voices.get(name)
        if v is None:
            raise ValueError(f"Unknown voice '{name}'. Available: {list(self._voices.keys())}")
        return v

    def synthesize(self, text: str, voice: Optional[str] = "Ly") -> np.ndarray:
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
    data = await image.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((1280, 720))

    pipeline  = models["vlm"]
    en_prompt = pipeline.vi_to_en(question) if question.strip() else "Describe the scene in front of me in 2-3 concise sentences."

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
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
