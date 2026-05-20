"""STT inference pipeline: ZipFormer-30M RNNT int8 (Vietnamese speech → text)."""
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

ROOT    = Path(__file__).resolve().parent.parent
STT_DIR = ROOT / "models" / "stt"

SAMPLE_RATE = 16000


class STTPipeline:
    def __init__(self):
        print("[STT] Loading ZipFormer RNNT int8 ...")
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(STT_DIR / "encoder-epoch-20-avg-10.int8.onnx"),
            decoder=str(STT_DIR / "decoder-epoch-20-avg-10.int8.onnx"),
            joiner=str(STT_DIR  / "joiner-epoch-20-avg-10.int8.onnx"),
            tokens=str(STT_DIR / "config.json"),
            num_threads=4,
            sample_rate=SAMPLE_RATE,
            decoding_method="greedy_search",
        )
        print("[STT] Ready.\n")

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
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

    def transcribe_file(self, path: str) -> str:
        import wave, array as arr
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return self.transcribe(pcm, sample_rate=sr)

    def record_and_transcribe(self, duration: float = 5.0) -> str:
        print(f"[STT] Recording {duration}s ... (speak now)", flush=True)
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("[STT] Recording done.")
        return self.transcribe(audio.flatten())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to WAV file")
    group.add_argument("--mic", type=float, metavar="SECONDS", help="Record N seconds from mic")
    args = parser.parse_args()

    pipeline = STTPipeline()
    if args.file:
        text = pipeline.transcribe_file(args.file)
    else:
        text = pipeline.record_and_transcribe(args.mic)
    print(f"\n{text}")
