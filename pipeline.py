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
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import requests

ROOT   = Path(__file__).resolve().parent
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

# ── VAD ──────────────────────────────────────────────────────────────────────
# Silero VAD ends the recording when the speaker stops, replacing the old fixed window.
VAD_MODEL      = ROOT / "models" / "vad" / "silero_vad.onnx"
VAD_SILENCE    = 0.8    # trailing silence that ends the question
VAD_THRESHOLD  = 0.5    # Silero default; raise toward 0.6 if ambient noise holds it in speech
VAD_MIN_SPEECH = 0.25   # rejects coughs, door slams, mic knocks
VAD_MAX_SPEECH = 8.0    # past this Silero raises its own threshold to 0.9 to force a split
VAD_WINDOW     = 512    # 32ms at 16kHz — must be a trained size (512/1024/1536)
VAD_CORES      = {0, 1, 2, 3}  # A55 efficiency cores

# The fixed window used to be the only guarantee that recording ever ended. Without it a VAD
# held in speech by sustained noise would listen forever, so both guards are load-bearing.
MAX_RECORD     = 12.0   # hard cap on one question
SPEECH_TIMEOUT = 6.0    # give up if nobody speaks at all


def alsa_device() -> str:
    """ALSA device for the ReSpeaker, looked up by name so card order can change."""
    try:
        for line in Path("/proc/asound/cards").read_text().splitlines():
            if SOUND_CARD.lower() in line.lower():
                return f"plughw:{line.split('[')[0].strip().split()[0]},0"
    except OSError:
        pass
    return "default"


@contextmanager
def efficiency_cores():
    """Run the VAD loop on the A55s.

    Every other model in XEye is pinned to the performance cores; VAD is the exception. At 629KB
    it is small enough that an A55 keeps up with realtime comfortably, and staying off cpu4-7
    leaves them to the camera now and to continuous listening later.
    """
    try:
        prev = os.sched_getaffinity(0)
        os.sched_setaffinity(0, VAD_CORES)
    except (AttributeError, OSError):
        yield          # affinity unavailable — run unpinned
        return
    try:
        yield
    finally:
        os.sched_setaffinity(0, prev)


def record_question(path: str, silence: float = VAD_SILENCE, max_seconds: float = MAX_RECORD):
    """Record until the speaker stops, using Silero VAD to find the end of the question.

    The old fixed window (`arecord -d N`) always waited the full N seconds and cut off anyone
    still talking at the end of it. Recording now stops `silence` seconds after speech does,
    and what reaches /stt is the trimmed speech segment with no leading or trailing room tone.
    """
    import sherpa_onnx   # lazy: replaying from a WAV should not need the VAD stack

    if not VAD_MODEL.is_file():
        raise RuntimeError(f"VAD model missing: {VAD_MODEL}\nRun: python download_models.py")

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model                = str(VAD_MODEL)
    config.silero_vad.threshold            = VAD_THRESHOLD
    config.silero_vad.min_silence_duration = silence
    config.silero_vad.min_speech_duration  = VAD_MIN_SPEECH
    config.silero_vad.max_speech_duration  = VAD_MAX_SPEECH
    config.silero_vad.window_size          = VAD_WINDOW
    config.sample_rate = STT_RATE
    config.num_threads = 1
    config.provider    = "cpu"
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=20)

    dev = alsa_device()
    print(f"[mic] Listening on {dev} ... speak now", flush=True)
    proc = subprocess.Popen(
        ["arecord", "-D", dev, "-f", "S16_LE", "-r", str(STT_RATE), "-c", "1", "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    segment, spoke, t0 = None, False, time.time()
    try:
        with efficiency_cores():
            while True:
                data = proc.stdout.read(VAD_WINDOW * 2)     # int16 → 2 bytes per sample
                if len(data) < VAD_WINDOW * 2:
                    break                                   # arecord stopped
                vad.accept_waveform(
                    np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
                spoke = spoke or vad.is_speech_detected()

                if not vad.empty():                         # speech, then `silence` of quiet
                    segment = vad.front.samples
                    break

                elapsed = time.time() - t0
                if not spoke and elapsed > SPEECH_TIMEOUT:
                    raise RuntimeError(
                        f"No speech in {SPEECH_TIMEOUT:.0f}s — check the mic ({dev}).")
                if elapsed > max_seconds:                    # still talking at the cap
                    vad.flush()
                    if not vad.empty():
                        segment = vad.front.samples
                    print(f"[mic] Hit the {max_seconds:.0f}s cap.")
                    break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if segment is None or len(segment) == 0:
        raise RuntimeError("No speech captured.")

    pcm = (np.asarray(segment, dtype=np.float32) * 32768).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(STT_RATE)
        wf.writeframes(pcm.tobytes())
    print(f"[mic] {len(pcm)/STT_RATE:.1f}s of speech  ({time.time()-t0:.1f}s listening)")


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


class RollingCamera:
    """Stream the CSI camera (IMX219 / Camera Module 2) and hand out the newest frame.

    The old one-shot capture ran a ~2s warmup burst hidden inside the fixed 5s mic window.
    With VAD a short question can end before that, which would put the camera back on the
    critical path — so the stream runs for the whole question and the frame is taken at the
    end. It is also the more correct frame: contemporaneous with the question rather than
    with the start of listening.

    `multifilesink max-files=5` is a ring buffer, and it lands in tmpfs where available so a
    long question does not push megabytes of JPEG through the board's flash.
    """
    WARMUP = 2.0   # auto-exposure needs ~5 frames to converge

    def __init__(self):
        shm = Path("/dev/shm")
        self._dir = tempfile.TemporaryDirectory(dir=str(shm) if shm.is_dir() else None)
        # stderr to a file, not a pipe: nothing drains a pipe during a long stream and gst
        # would eventually block on a full buffer.
        self._err = open(Path(self._dir.name) / "gst.err", "w+b")
        self._t0  = time.time()
        print("[camera] Streaming ...", flush=True)
        self._proc = subprocess.Popen(
            ["gst-launch-1.0", "-e", "qtiqmmfsrc", f"camera={CAMERA}",
             f"exposure-compensation={EXPOSURE}", "!",
             "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1", "!",
             "queue", "!", "jpegenc", "!", "queue", "!",
             "multifilesink", f"location={self._dir.name}/f_%04d.jpg", "max-files=5"],
            stdout=subprocess.DEVNULL, stderr=self._err)

    def grab(self, path: str):
        """Copy the newest settled frame to `path`."""
        wait = self.WARMUP - (time.time() - self._t0)
        if wait > 0:
            time.sleep(wait)   # question ended before auto-exposure had converged
        frames = [f for f in sorted(Path(self._dir.name).glob("f_*.jpg")) if f.stat().st_size > 0]
        if not frames:
            raise RuntimeError(
                "Camera capture failed — no frame written. Check that the CSI camera is "
                "detected (`dmesg | grep 'Probe success'`).\n" + self._stderr_tail())
        shutil.copy(frames[-1], path)
        print(f"[camera] Frame saved → {path}")

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._err.close()
        self._dir.cleanup()

    def _stderr_tail(self) -> str:
        self._err.flush()
        self._err.seek(0)
        return "\n".join(self._err.read().decode(errors="replace").strip().splitlines()[-3:])


def run(audio_path: str | None, image_path: str | None, output_path: str | None = None,
        voice: str = "Mai Anh", silence: float = VAD_SILENCE,
        max_record: float = MAX_RECORD, playback: bool = True):
    t_total = time.time()

    # 1+2. The camera streams for as long as the question takes. Exposure settles during the
    # question rather than before it, and the frame taken is the one from the moment the user
    # stopped speaking — so the camera never lands on the critical path, however short the
    # question turns out to be.
    _tmp = _tmp_wav = None
    cam = None
    if image_path is None:
        _tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        _tmp.close()
        image_path = _tmp.name
        cam = RollingCamera()

    try:
        if audio_path is None:
            _tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            _tmp_wav.close()
            record_question(_tmp_wav.name, silence, max_record)
            audio_path = _tmp_wav.name

        if cam is not None:
            cam.grab(image_path)
    finally:
        if cam is not None:
            cam.close()   # never leave the camera streaming, even if recording failed

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
    parser.add_argument("--silence", type=float, default=VAD_SILENCE, metavar="SECONDS",
                        help=f"Trailing silence that ends the question (default: {VAD_SILENCE})")
    parser.add_argument("--max-record", type=float, default=MAX_RECORD, metavar="SECONDS",
                        help=f"Hard cap on a single question (default: {MAX_RECORD:.0f})")
    parser.add_argument("--no-play", action="store_true", help="Do not play the answer aloud")
    args = parser.parse_args()

    # Explicit --output always wins; otherwise only dev mode writes anything.
    output = args.output or (DEV_OUTPUT if args.mode == "dev" else None)

    print(f"[pipeline] Mode:   {args.mode} ({'saving ' + output if output else 'no disk writes'})")
    mic = f"microphone ({alsa_device()}, VAD: {args.silence}s silence, {args.max_record:.0f}s cap)"
    print(f"[pipeline] Audio:  {args.audio or mic}")
    print(f"[pipeline] Image:  {args.image or 'camera (qtiqmmfsrc)'}")
    print(f"[pipeline] Server: {SERVER}\n")

    run(args.audio, args.image, output, args.voice, args.silence, args.max_record,
        not args.no_play)


if __name__ == "__main__":
    main()
