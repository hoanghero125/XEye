"""TTS inference pipeline: VieNeu-TTS-v2 GGUF Q4-K-M + neucodec ONNX int8 (text → Vietnamese speech)."""
import json
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
TTS_DIR = ROOT / "models" / "tts"

SAMPLE_RATE = 24000

# Presets: Binh, Tuyen, Vinh, Doan, Ly (default), Sơn, Ngoc
def _load_voices() -> dict:
    p = TTS_DIR / "voices.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


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
        self._voices = _load_voices().get("presets", {})
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
        audio_sec = len(audio) / SAMPLE_RATE
        print(f"[TTS] {len(text)} chars → {audio_sec:.1f}s audio in {elapsed:.1f}s (RTF {elapsed/audio_sec:.2f}x)")
        return audio

    def speak(self, text: str, voice: Optional[str] = "Ly") -> np.ndarray:
        import sounddevice as sd
        audio = self.synthesize(text, voice=voice)
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()
        return audio

    def save(self, text: str, output_path: str, voice: Optional[str] = "Ly") -> np.ndarray:
        audio = self.synthesize(text, voice=voice)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        print(f"[TTS] Saved → {output_path}")
        return audio


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Vietnamese text to synthesize")
    parser.add_argument("--out", default="output.wav", help="Output WAV path (default: output.wav)")
    parser.add_argument("--play", action="store_true", help="Play audio after synthesis")
    parser.add_argument("--voice", default="Ly",
                        help="Voice preset: Binh, Tuyen, Vinh, Doan, Ly, Sơn, Ngoc (default: Ly)")
    args = parser.parse_args()

    pipeline = TTSPipeline()
    if args.play:
        pipeline.speak(args.text, voice=args.voice)
    else:
        pipeline.save(args.text, args.out, voice=args.voice)
