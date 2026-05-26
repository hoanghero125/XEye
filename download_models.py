"""Download all models. Run once — after this, the server works fully offline."""
import os

from huggingface_hub import snapshot_download, hf_hub_download

ROOT     = os.path.dirname(os.path.abspath(__file__))
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def hf_cached(repo_id: str, filename: str) -> bool:
    snapshots = os.path.join(HF_CACHE, "models--" + repo_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(snapshots):
        return False
    return any(
        os.path.isfile(os.path.join(snapshots, snap, filename))
        for snap in os.listdir(snapshots)
    )


# VLM — Vintern-1B Q4_K_M + mmproj F16
vintern_dir = os.path.join(ROOT, "models", "vintern")
vintern_files = [
    "vintern-1b-v3_5-q4_k_m.gguf",
    "mmproj-vintern-1b-v3_5-f16.gguf",
]
if all(os.path.isfile(os.path.join(vintern_dir, f)) for f in vintern_files):
    print("\n[skip] VLM — already downloaded.")
else:
    print("\n==> Downloading VLM (Vintern-1B Q4_K_M + mmproj F16) ...")
    os.makedirs(vintern_dir, exist_ok=True)
    for filename in vintern_files:
        hf_hub_download(
            repo_id="dekthedev/Vintern-1B-v3_5-GGUF",
            filename=filename,
            local_dir=vintern_dir,
        )

# STT — ZipFormer-30M RNNT int8
stt_dir   = os.path.join(ROOT, "models", "stt")
stt_files = [
    "encoder-epoch-20-avg-10.int8.onnx",
    "decoder-epoch-20-avg-10.int8.onnx",
    "joiner-epoch-20-avg-10.int8.onnx",
    "config.json",
]
if all(os.path.isfile(os.path.join(stt_dir, f)) for f in stt_files):
    print("\n[skip] STT — already downloaded.")
else:
    print("\n==> Downloading STT (ZipFormer-30M RNNT int8) ...")
    snapshot_download(repo_id="hynt/Zipformer-30M-RNNT-6000h", local_dir=stt_dir)

# TTS — VieNeu-TTS-v2-Turbo + VieNeu-Codec
tts_downloads = [
    ("pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF", "vieneu-tts-v2-turbo.gguf"),
    ("pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF", "voices.json"),
    ("pnnbao-ump/VieNeu-Codec",              "vieneu_decoder.onnx"),
    ("pnnbao-ump/VieNeu-Codec",              "vieneu_encoder.onnx"),
]
missing = [(r, f) for r, f in tts_downloads if not hf_cached(r, f)]
if not missing:
    print("\n[skip] TTS — already cached.")
else:
    print("\n==> Downloading TTS (VieNeu-TTS-v2-Turbo + VieNeu-Codec) ...")
    for repo_id, filename in missing:
        print(f"    {repo_id}/{filename}")
        hf_hub_download(repo_id=repo_id, filename=filename)

print("\nDone. Server runs fully offline from here.")
