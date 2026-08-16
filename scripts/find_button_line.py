"""Find which GPIO line the button is wired to. Ctrl-C to stop.

    sudo python scripts/find_button_line.py

Watches the level of every GPIO on every chip — including lines claimed by a driver, which
libgpiod will not let us request — by polling the kernel's own debugfs table. Press the button
a few times; any line that changes is reported immediately.

Needs root: /sys/kernel/debug is root-only.
"""
import re
import sys
import time
from collections import Counter

GPIO_DEBUG = "/sys/kernel/debug/gpio"
CHIP = re.compile(r"^(gpiochip\d+):")
LINE = re.compile(r"^\s*(gpio\d+)\s*:\s*(?:in|out)\s+(hi|lo)\w*")


def snapshot() -> dict:
    """Level of every line, keyed by (chip, line).

    Each chip block numbers its lines from zero, so `gpio0` exists on all six chips. Keying by
    name alone silently collapses them onto each other and loses most of the board.
    """
    levels, chip = {}, "?"
    with open(GPIO_DEBUG) as fh:
        for line in fh:
            c = CHIP.match(line)
            if c:
                chip = c.group(1)
                continue
            m = LINE.match(line)
            if m:
                levels[(chip, m.group(1))] = m.group(2)
    return levels


def main():
    try:
        base = snapshot()
    except PermissionError:
        sys.exit("needs root — run:  sudo python scripts/find_button_line.py")

    print(f"watching {len(base)} GPIO lines across all chips")
    print("press the button several times — each press and release is one change")
    print("Ctrl-C to stop\n")

    hits: Counter = Counter()
    t0 = time.time()
    while True:
        for name, level in snapshot().items():
            if name in base and level != base[name]:
                hits[name] += 1
                print(f"\r  {time.strftime('%H:%M:%S')}  {name[0]} {name[1]}: {base[name]} -> {level}"
                      f"   (total {hits[name]})           ")
                base[name] = level
        top = (f"  best guess: {hits.most_common(1)[0][0][0]} {hits.most_common(1)[0][0][1]}"
               if hits else "  nothing yet")
        print(f"\r{top}   elapsed {time.time()-t0:4.0f}s ", end="", flush=True)
        time.sleep(0.03)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
