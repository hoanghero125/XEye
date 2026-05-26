#!/bin/bash
set -e

CONDA="/home/ubuntu/miniconda3/bin/conda"
PIP="/home/ubuntu/miniconda3/envs/xeye/bin/pip"
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> [1/3] Creating conda environment: xeye (Python 3.11)"
if $CONDA env list | grep -q "^xeye "; then
    echo "    Already exists, skipping."
else
    $CONDA create -n xeye python=3.11 -y
fi

echo "==> [2/3] Installing Python dependencies"
$PIP install --upgrade pip wheel setuptools
$PIP install -r "$ROOT/requirements.txt"

echo "==> [3/3] Creating directories"
mkdir -p "$ROOT/models/vintern"
mkdir -p "$ROOT/models/stt"
mkdir -p "$ROOT/models/tts"
mkdir -p "$ROOT/data/audio"
mkdir -p "$ROOT/data/images"

echo ""
echo "Setup complete."
echo "Next: conda activate xeye && python download_models.py"
