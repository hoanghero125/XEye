"""TTS CLI — calls /tts endpoint on the running XEye server."""
import argparse
from pathlib import Path

import requests

SERVER = "http://localhost:8000"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Vietnamese text to synthesize")
    parser.add_argument("--out", default="data/audio/output.wav", help="Output WAV path (default: data/audio/output.wav)")
    parser.add_argument("--voice", default="Ly",
                        help="Voice preset: Binh, Tuyen, Vinh, Doan, Ly, Sơn, Ngoc (default: Ly)")
    args = parser.parse_args()

    r = requests.post(f"{SERVER}/tts", data={"text": args.text, "voice": args.voice})
    r.raise_for_status()
    Path(args.out).write_bytes(r.content)
    print(f"Saved → {args.out}")
