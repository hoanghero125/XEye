"""Live camera preview — serves the CSI camera as MJPEG over HTTP.

    python scripts/camera_preview.py            # then open http://<board-ip>:8080/

Useful for aiming the camera and checking focus before mounting it on the wearable.
Only one viewer at a time: the camera allows a single consumer.
"""
import argparse
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAMERA = 0
WIDTH, HEIGHT, FPS = 1280, 720, 15
EXPOSURE = 2  # exposure-compensation, -12..12
FLIP = {"90": "clockwise", "180": "rotate-180", "270": "counterclockwise"}

# The camera allows one consumer. Track the live gstreamer process so a new viewer can
# take over from a stale one instead of finding the camera locked.
_stream_lock = threading.Lock()
_current = None


def _claim(proc):
    global _current
    with _stream_lock:
        if _current and _current.poll() is None:
            _current.terminate()
            try:
                _current.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _current.kill()
        _current = proc

PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>XEye camera</title>
<style>body{margin:0;background:#111;color:#bbb;font:14px system-ui;text-align:center}
img{max-width:100%;height:auto}p{padding:8px;margin:0}</style>
<p>XEye &mdash; live camera (CSI __CAM__, __W__x__H__)</p><img src="/stream">
"""


def gst_cmd(rotate):
    pipe = [
        "gst-launch-1.0", "-q", "-e", "qtiqmmfsrc", f"camera={CAMERA}",
        f"exposure-compensation={EXPOSURE}", "!",
        f"video/x-raw,format=NV12,width={WIDTH},height={HEIGHT},framerate={FPS}/1", "!", "queue",
    ]
    if rotate != "0":
        pipe += ["!", "videoconvert", "!", "videoflip", f"method={FLIP[rotate]}"]
    pipe += ["!", "jpegenc", "!", "multipartmux", "boundary=frame", "!", "fdsink", "fd=1"]
    return pipe


class Handler(BaseHTTPRequestHandler):
    rotate = "0"

    def do_GET(self):
        if self.path == "/":
            body = (PAGE.replace("__CAM__", str(CAMERA))
                        .replace("__W__", str(WIDTH))
                        .replace("__H__", str(HEIGHT))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/stream":
            self.send_error(404)
            return

        proc = subprocess.Popen(gst_cmd(self.rotate), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        _claim(proc)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()   # surface the disconnect promptly, else gst leaks
        except (BrokenPipeError, ConnectionResetError):
            pass  # viewer closed the tab
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def log_message(self, fmt, *args):
        pass  # keep the console quiet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--rotate", default="0", choices=["0", "90", "180", "270"],
                    help="rotate the preview if the module is mounted sideways")
    ap.add_argument("--ev", type=int, default=EXPOSURE, choices=range(-12, 13), metavar="-12..12",
                    help=f"exposure compensation (default {EXPOSURE}; higher = brighter)")
    args = ap.parse_args()
    Handler.rotate = args.rotate
    EXPOSURE = args.ev

    ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    host = ips[0] if ips else "localhost"
    print(f"Camera preview:  http://{host}:{args.port}/   (Ctrl-C to stop)")
    try:
        ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
