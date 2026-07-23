"""Demo pipeline — WAV + image (or live camera capture) → text + audio file."""
import io
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SERVER = "http://localhost:8000"

# dev  — save the answer WAV so you can listen back and verify.
# prod — write nothing to disk; the answer only goes to the speaker.
# Change the default here, export XEYE_MODE=prod, or pass --mode prod for one run.
MODE = os.getenv("XEYE_MODE", "dev")
DEV_OUTPUT = "data/audio/output.wav"

CAMERA = 0    # CSI connector: 0 = Camera connector 1, 1 = Camera connector 2
EXPOSURE = 2  # exposure-compensation (-12..12); +2 lifts indoor scenes without blowing highlights

SOUND_CARD = "ReSpeaker"  # matched against /proc/asound/cards
STT_RATE   = 16000        # what /stt expects
TTS_RATE   = 48000        # what /tts returns


def alsa_device() -> str:
    """ALSA device for the ReSpeaker, looked up by name so card order can change."""
    try:
        for line in Path("/proc/asound/cards").read_text().splitlines():
            if SOUND_CARD.lower() in line.lower():
                return f"plughw:{line.split('[')[0].strip().split()[0]},0"
    except OSError:
        pass
    return "default"


def record_question(path: str, seconds: float):
    """Record a Vietnamese question from the ReSpeaker mic array."""
    dev = alsa_device()
    print(f"[mic] Recording {seconds:.0f}s from {dev} ... speak now", flush=True)
    subprocess.run(
        ["arecord", "-D", dev, "-f", "S16_LE", "-r", str(STT_RATE), "-c", "1",
         "-d", str(max(1, round(seconds))), path],  # arecord -d takes whole seconds
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[mic] Done.")


def play(path: str):
    """Play a WAV file through the ReSpeaker's speaker output."""
    dev = alsa_device()
    print(f"[speaker] Playing → {dev}", flush=True)
    subprocess.run(["aplay", "-D", dev, path],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def split_sentences(text: str, min_chars: int = 40) -> list[str]:
    """Split into speakable chunks, merging short fragments so playback isn't choppy."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]
    chunks: list[str] = []
    for p in parts:
        if chunks and len(chunks[-1]) < min_chars:
            chunks[-1] += " " + p
        else:
            chunks.append(p)
    return chunks or [text.strip()]


def synthesize(text: str, voice: str) -> bytes:
    """One /tts call → raw PCM (no WAV header)."""
    r = requests.post(f"{SERVER}/tts", data={"text": text, "voice": voice})
    r.raise_for_status()
    with wave.open(io.BytesIO(r.content), "rb") as wf:
        return wf.readframes(wf.getnframes())


def speak(text: str, voice: str, output_path: str | None, playback: bool = True) -> float:
    """Synthesize sentence-by-sentence, playing each while the next is still rendering.

    Cuts time-to-first-sound to one sentence instead of the whole answer, and hides the
    remaining synthesis behind playback.
    """
    chunks = split_sentences(text)
    dev = alsa_device()
    player = None
    if playback:
        print(f"[speaker] Streaming → {dev}", flush=True)
        player = subprocess.Popen(
            ["aplay", "-q", "-D", dev, "-f", "S16_LE", "-r", str(TTS_RATE), "-c", "1", "-"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    keep = output_path is not None          # prod streams straight to the speaker
    pcm_all, t0, first_audio = [], time.time(), None
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(synthesize, chunks[0], voice)
        for i, chunk in enumerate(chunks):
            pcm = pending.result()
            if i + 1 < len(chunks):                       # start the next one before playing
                pending = pool.submit(synthesize, chunks[i + 1], voice)
            if first_audio is None:
                first_audio = time.time() - t0
                print(f"[TTS] First audio after {first_audio:.2f}s ({len(chunks)} chunk(s))")
            if keep:
                pcm_all.append(pcm)
            if player:
                try:
                    player.stdin.write(pcm)
                except BrokenPipeError:
                    break

    if player:
        player.stdin.close()
        player.wait()

    if keep:
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TTS_RATE)
            wf.writeframes(b"".join(pcm_all))
    return first_audio or 0.0


def capture_frame(path: str, warmup: float = 2.0):
    """Capture one frame from the CSI camera (IMX219 / Camera Module 2) via qtiqmmfsrc.

    Writes numbered frames to a temp dir and keeps the newest one: auto-exposure needs a
    moment to settle, so the last frame of the warmup is usable where the first is not.
    """
    print("[camera] Capturing frame ...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "gst-launch-1.0", "-e", "qtiqmmfsrc", f"camera={CAMERA}",
            f"exposure-compensation={EXPOSURE}", "!",
            "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1", "!",
            "queue", "!", "jpegenc", "!", "queue", "!",
            "multifilesink", f"location={tmpdir}/f_%04d.jpg", "max-files=5",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(warmup)
        proc.terminate()
        _, err = proc.communicate(timeout=10)

        frames = sorted(Path(tmpdir).glob("f_*.jpg"))
        frames = [f for f in frames if f.stat().st_size > 0]
        if not frames:
            msg = err.decode(errors="replace").strip().splitlines()[-3:] if err else []
            raise RuntimeError(
                "Camera capture failed — no frame written. Check that the CSI camera is "
                "detected (`dmesg | grep 'Probe success'`).\n" + "\n".join(msg))
        shutil.copy(frames[-1], path)  # newest frame: exposure has settled
    print(f"[camera] Frame saved → {path}")


def run(audio_path: str | None, image_path: str | None, output_path: str | None = None,
        voice: str = "Mai Anh", record_seconds: float = 5.0, playback: bool = True):
    t_total = time.time()

    # 1+2. Capture the frame while the question is being recorded — the camera needs a
    # couple of seconds to settle exposure, and that fits entirely inside the mic window.
    _tmp = _tmp_wav = None
    cam_future = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        if image_path is None:
            _tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            _tmp.close()
            cam_future = pool.submit(capture_frame, _tmp.name)
            image_path = _tmp.name

        if audio_path is None:
            _tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _tmp_wav.close()
            record_question(_tmp_wav.name, record_seconds)
            audio_path = _tmp_wav.name

        if cam_future is not None:
            cam_future.result()   # surface camera errors before we go on

    # 3. STT
    print("[STT] Transcribing ...", flush=True)
    t0 = time.time()
    wav_bytes = Path(audio_path).read_bytes()
    r = requests.post(f"{SERVER}/stt", files={"audio": (Path(audio_path).name, wav_bytes, "audio/wav")})
    r.raise_for_status()
    question_vi = r.json()["text"]
    print(f"[STT] {question_vi!r}  ({time.time()-t0:.2f}s)")

    # 4. VLM
    print("[VLM] Analyzing image ...", flush=True)
    t0 = time.time()
    img_bytes = Path(image_path).read_bytes()
    suffix    = Path(image_path).suffix.lstrip(".")
    r = requests.post(
        f"{SERVER}/vlm",
        files={"image": (Path(image_path).name, img_bytes, f"image/{suffix}")},
        data={"question": question_vi},
    )
    r.raise_for_status()
    result = r.json()
    print(f"[VLM] {result['vi']}")
    print(f"[VLM] {result['tokens']} tokens | {result['elapsed_s']}s | {result['tok_s']} tok/s  ({time.time()-t0:.2f}s total)")

    # 5+6. Synthesize and speak, one sentence ahead of playback
    print("[TTS] Synthesizing ...", flush=True)
    t0 = time.time()
    first_audio = speak(result["vi"], voice, output_path, playback)
    where = f"Saved → {output_path}" if output_path else "Not saved (prod mode)"
    print(f"[TTS] {where}  ({time.time()-t0:.2f}s total)")

    if playback:
        print(f"\n[pipeline] Time to first sound: {(t0 - t_total) + first_audio:.2f}s")
    print(f"[pipeline] Total: {time.time()-t_total:.2f}s")

    if _tmp:
        Path(_tmp.name).unlink(missing_ok=True)
    if _tmp_wav:
        Path(_tmp_wav.name).unlink(missing_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="XEye demo pipeline")
    parser.add_argument("audio", nargs="?", default=None,
                        help="Input WAV file (default: record from the ReSpeaker mic)")
    parser.add_argument("--image", default=None, help="Input image file (default: capture from camera)")
    parser.add_argument("--mode", choices=["dev", "prod"], default=MODE,
                        help=f"dev saves the answer WAV, prod writes nothing to disk "
                             f"(default: {MODE}; set XEYE_MODE to change)")
    parser.add_argument("--output", default=None,
                        help=f"Where to save the answer WAV (dev default: {DEV_OUTPUT}). "
                             f"Passing it saves even in prod mode.")
    parser.add_argument("--voice", default="Mai Anh", help="TTS voice (default: Mai Anh)")
    parser.add_argument("--record", type=float, default=5.0, metavar="SECONDS",
                        help="Seconds to record when no WAV is given (default: 5)")
    parser.add_argument("--no-play", action="store_true", help="Do not play the answer aloud")
    args = parser.parse_args()

    # Explicit --output always wins; otherwise only dev mode writes anything.
    output = args.output or (DEV_OUTPUT if args.mode == "dev" else None)

    print(f"[pipeline] Mode:   {args.mode} ({'saving ' + output if output else 'no disk writes'})")
    print(f"[pipeline] Audio:  {args.audio or f'microphone ({alsa_device()}, {args.record:.0f}s)'}")
    print(f"[pipeline] Image:  {args.image or 'camera (qtiqmmfsrc)'}")
    print(f"[pipeline] Server: {SERVER}\n")

    run(args.audio, args.image, output, args.voice, args.record, not args.no_play)


if __name__ == "__main__":
    main()
