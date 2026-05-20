"""STT CLI — calls /stt endpoint on the running XEye server."""
import argparse
import requests

SERVER = "http://localhost:8000"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to WAV file")
    args = parser.parse_args()

    wav_bytes = open(args.file, "rb").read()
    r = requests.post(f"{SERVER}/stt", files={"audio": (args.file, wav_bytes, "audio/wav")})
    r.raise_for_status()
    print(r.json()["text"])
