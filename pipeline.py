"""Demo pipeline — WAV + image (or live camera capture) → text + audio file."""
import subprocess
import tempfile
import time
from pathlib import Path

import requests

SERVER = "http://localhost:8000"

GST_CAPTURE_CMD = [
    "gst-launch-1.0", "-e", "qtiqmmfsrc", "!",
    "video/x-raw,width=1280,height=720,format=NV12", "!",
    "videoconvert", "!", "jpegenc", "!", "filesink", "location={path}",
]


def capture_frame(path: str, warmup: float = 2.0):
    """Capture one frame from Raspberry Pi Camera Module V3 via qtiqmmfsrc."""
    print("[camera] Capturing frame ...", flush=True)
    cmd = [c.format(path=path) if "{path}" in c else c for c in GST_CAPTURE_CMD]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(warmup)
    proc.terminate()
    proc.wait()
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        raise RuntimeError("Camera capture failed — no frame written.")
    print(f"[camera] Frame saved → {path}")


def run(audio_path: str, image_path: str | None, output_path: str = "output.wav", voice: str = "Ly"):
    # 1. Camera capture if no image provided
    _tmp = None
    if image_path is None:
        _tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        _tmp.close()
        capture_frame(_tmp.name)
        image_path = _tmp.name

    # 2. STT
    print("[pipeline] Transcribing ...")
    wav_bytes = Path(audio_path).read_bytes()
    r = requests.post(f"{SERVER}/stt", files={"audio": (Path(audio_path).name, wav_bytes, "audio/wav")})
    r.raise_for_status()
    question_vi = r.json()["text"]
    print(f"[STT] {question_vi!r}")

    # 3. VLM
    print("[pipeline] Analyzing image ...")
    img_bytes = Path(image_path).read_bytes()
    suffix    = Path(image_path).suffix.lstrip(".")
    r = requests.post(
        f"{SERVER}/vlm",
        files={"image": (Path(image_path).name, img_bytes, f"image/{suffix}")},
        data={"question": question_vi},
    )
    r.raise_for_status()
    result = r.json()
    print(f"[VLM] Question (EN): {result['question_en']}")
    print(f"[VLM] Answer (EN):   {result['en']}")
    print(f"[VLM] Answer (VI):   {result['vi']}")

    # 4. TTS
    print("[pipeline] Synthesizing speech ...")
    r = requests.post(f"{SERVER}/tts", data={"text": result["vi"], "voice": voice})
    r.raise_for_status()

    Path(output_path).write_bytes(r.content)
    print(f"[pipeline] Audio saved → {output_path}")

    if _tmp:
        Path(_tmp.name).unlink(missing_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XEye demo pipeline")
    parser.add_argument("audio", help="Input WAV file (Vietnamese question)")
    parser.add_argument("--image", default=None, help="Input image file (default: capture from camera)")
    parser.add_argument("--output", default="output.wav", help="Output WAV file (default: output.wav)")
    parser.add_argument("--voice", default="Ly", help="TTS voice (default: Ly)")
    args = parser.parse_args()

    print(f"[pipeline] Audio:  {args.audio}")
    print(f"[pipeline] Image:  {args.image or 'camera (qtiqmmfsrc)'}")
    print(f"[pipeline] Server: {SERVER}\n")

    run(args.audio, args.image, args.output, args.voice)


if __name__ == "__main__":
    main()
