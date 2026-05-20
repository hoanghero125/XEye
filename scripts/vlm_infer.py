"""VLM CLI — calls /vlm endpoint on the running XEye server."""
import argparse
from pathlib import Path

import requests

SERVER = "http://localhost:8000"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--question", default="", help="Question in Vietnamese (optional)")
    args = parser.parse_args()

    suffix    = Path(args.image).suffix.lstrip(".")
    img_bytes = Path(args.image).read_bytes()
    r = requests.post(
        f"{SERVER}/vlm",
        files={"image": (Path(args.image).name, img_bytes, f"image/{suffix}")},
        data={"question": args.question},
    )
    r.raise_for_status()
    result = r.json()
    print(f"Question (EN): {result['question_en']}")
    print(f"Answer   (EN): {result['en']}")
    print(f"Answer   (VI): {result['vi']}")
