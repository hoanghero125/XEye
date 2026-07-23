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

# TTS — VieNeu-TTS-v3-Turbo (ONNX int8) + MOSS audio tokenizer
# The vieneu package loads these from the HF cache, not from models/.
V3_REPO   = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"
tts_downloads = [
    (V3_REPO,   "config.json"),
    (V3_REPO,   "denoiser.onnx"),
    (V3_REPO,   "onnx_int8/config.json"),
    (V3_REPO,   "onnx_int8/tokenizer.json"),
    (V3_REPO,   "onnx_int8/vieneu_prefill.onnx"),
    (V3_REPO,   "onnx_int8/vieneu_decode_step.onnx"),
    (V3_REPO,   "onnx_int8/vieneu_acoustic_cached.onnx"),
    (V3_REPO,   "onnx_int8/vieneu_backbone_shared.data"),
    (V3_REPO,   "onnx_int8/vieneu_v3_heads.npz"),
    (MOSS_REPO, "codec_browser_onnx_meta.json"),
    (MOSS_REPO, "moss_audio_tokenizer_encode.onnx"),
    (MOSS_REPO, "moss_audio_tokenizer_encode.data"),
    (MOSS_REPO, "moss_audio_tokenizer_decode_full.onnx"),
    (MOSS_REPO, "moss_audio_tokenizer_decode_step.onnx"),
    (MOSS_REPO, "moss_audio_tokenizer_decode_shared.data"),
]
missing = [(r, f) for r, f in tts_downloads if not hf_cached(r, f)]
if not missing:
    print("\n[skip] TTS — already cached.")
else:
    print("\n==> Downloading TTS (VieNeu-TTS-v3-Turbo ONNX int8 + MOSS tokenizer) ...")
    for repo_id, filename in missing:
        print(f"    {repo_id}/{filename}")
        hf_hub_download(repo_id=repo_id, filename=filename)

print("\nDone. Server runs fully offline from here.")
