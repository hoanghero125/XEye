"""Live button state monitor. Ctrl-C to stop.

    python scripts/button_monitor.py

Prints the line level continuously and logs every change with a timestamp, so a press,
a release, or a line stuck at one level is visible as it happens.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import BUTTON_CHIP, BUTTON_LINE  # noqa: E402

import gpiod                                    # noqa: E402
from gpiod.line import Bias, Direction, Value   # noqa: E402


def main():
    offset = int(BUTTON_LINE)
    cfg = {offset: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP)}
    print(f"{BUTTON_CHIP} line {offset} (pin 16 / GPIO_26) — Ctrl-C to stop")
    print("HIGH = released    LOW = pressed\n")

    with gpiod.request_lines(BUTTON_CHIP, consumer="xeye-monitor", config=cfg) as req:
        prev, changes, t0 = None, 0, time.time()
        while True:
            level = "LOW " if req.get_value(offset) == Value.INACTIVE else "HIGH"
            if level != prev:
                if prev is not None:
                    changes += 1
                    print(f"\r  {time.strftime('%H:%M:%S')}  {prev} -> {level}"
                          f"   (change #{changes})        ")
                prev = level
            state = "PRESSED " if level == "LOW " else "released"
            print(f"\r  {level}  {state}   changes: {changes}   "
                  f"{time.time()-t0:5.0f}s ", end="", flush=True)
            time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
