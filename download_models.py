"""Download and convert all models. Run once — after this, the server works fully offline."""
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


# VLM — SmolVLM-256M-Instruct FP32 ONNX (main pipeline)
vlm_dir      = os.path.join(ROOT, "models", "vlm")
vlm_sentinel = os.path.join(vlm_dir, "onnx", "decoder_model_merged.onnx")
if os.path.isfile(vlm_sentinel):
    print("\n[skip] VLM — already downloaded.")
else:
    print("\n==> Downloading VLM (SmolVLM-256M-Instruct FP32 ONNX) ...")
    snapshot_download(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        allow_patterns=[
            "onnx/decoder_model_merged.onnx",
            "onnx/embed_tokens.onnx",
            "onnx/vision_encoder.onnx",
            "*.json",
            "*.jinja",
            "tokenizer*",
            "preprocessor*",
        ],
        local_dir=vlm_dir,
    )

# NMT EN→VI — pre-converted CTranslate2 int8
nmt_en_vi_dir      = os.path.join(ROOT, "models", "nmt", "opus-mt-en-vi-int8")
nmt_en_vi_sentinel = os.path.join(nmt_en_vi_dir, "model.bin")
if os.path.isfile(nmt_en_vi_sentinel):
    print("\n[skip] NMT EN→VI — already downloaded.")
else:
    print("\n==> Downloading NMT EN→VI (dekthedev/opus-mt-en-vi-ct2-int8) ...")
    snapshot_download(
        repo_id="dekthedev/opus-mt-en-vi-ct2-int8",
        local_dir=nmt_en_vi_dir,
    )

# NMT VI→EN — pre-converted CTranslate2 int8
nmt_vi_en_dir      = os.path.join(ROOT, "models", "nmt", "opus-mt-vi-en-int8")
nmt_vi_en_sentinel = os.path.join(nmt_vi_en_dir, "model.bin")
if os.path.isfile(nmt_vi_en_sentinel):
    print("\n[skip] NMT VI→EN — already downloaded.")
else:
    print("\n==> Downloading NMT VI→EN (dekthedev/opus-mt-vi-en-ct2-int8) ...")
    snapshot_download(
        repo_id="dekthedev/opus-mt-vi-en-ct2-int8",
        local_dir=nmt_vi_en_dir,
    )

# STT — local dir (sherpa-onnx loads by path)
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
    print("\n==> Downloading STT (Zipformer-30M RNNT int8) ...")
    snapshot_download(repo_id="hynt/Zipformer-30M-RNNT-6000h", local_dir=stt_dir)

# TTS — HF cache (vieneu + codec load via huggingface_hub)
tts_downloads = [
    ("pnnbao-ump/VieNeu-TTS-v2",             "VieNeu-TTS-v2-Q4-K-M.gguf"),
    ("pnnbao-ump/VieNeu-TTS-v2",             "voices.json"),
    ("neuphonic/neucodec-onnx-decoder-int8", "model.onnx"),
]
missing = [(r, f) for r, f in tts_downloads if not hf_cached(r, f)]
if not missing:
    print("\n[skip] TTS — already cached.")
else:
    print("\n==> Downloading TTS (VieNeu-TTS-v2 backbone + codec) ...")
    for repo_id, filename in missing:
        print(f"    {repo_id}/{filename}")
        hf_hub_download(repo_id=repo_id, filename=filename)

print("\nDone. Server runs fully offline from here.")
