# Board Checklist

What to run on the RUBIK Pi 3 after pulling changes. Steps 1–4 are setup and only need repeating
when something they depend on changes; 5 onward is the smoke test.

For the reasoning behind any of these settings, see the [technical report](technical_report.md).

---

## Quick reference

For a board that has been through this before:

```bash
conda activate xeye
pip install -r requirements.txt
python download_models.py
python server.py                  # wait for "All models ready"
# second shell:
python pipeline.py --button
```

GPIO access must already be granted (step 3) and the volume already stored. If either is fresh,
work through the full list below.

---

## 1. Update dependencies

```bash
conda activate xeye
pip install -r requirements.txt
```

`gpiod` was added for the button. `vieneu` stays installed `--no-deps` — do not let anything
reinstall it with dependencies, it pulls in gradio.

## 2. Fetch models

```bash
python download_models.py
```

Idempotent; each section skips if the files are already present. Silero VAD (629KB) is the
recent addition and lands in `models/vad/`.

## 3. Grant GPIO access

Once per board. The line address itself no longer needs looking up — it was resolved on this
hardware and is now the default: **`/dev/gpiochip4` offset 26** (`f100000.pinctrl`, the SoC
TLMM), which is the signal on **header pin 16**. `XEYE_BUTTON_CHIP` / `XEYE_BUTTON_LINE` still
override it if a board differs.

What *does* need doing is permissions. `/dev/gpiochip*` is `crw------- root root` and the board
has no `gpio` group, so `--button` fails with `Permission denied` until:

```bash
sudo groupadd -f gpio && sudo usermod -aG gpio $USER
echo 'SUBSYSTEM=="gpio", KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"' \
  | sudo tee /etc/udev/rules.d/60-gpio.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Log out and back in for the group to apply. `pipeline.py` checks this at startup and prints the
same commands rather than failing on the first press.

> **Where the offset comes from.** The header's `GPIO_n` labels are board signal names, not
> TLMM pin numbers, and the two only coincide by luck — read the mapping off Thundercomm's
> official 40-pin diagram, which gives pin 16 = `GPIO_26`. On the SoC side that signal is TLMM
> pin 26, confirmed with
> `sudo cat /sys/kernel/debug/pinctrl/f100000.pinctrl/pinmux-pins | grep 'pin 26'`.
>
> Do not use `gpiofind` — the gpiod CLI tools are not installed, and no chip exposes line names
> anyway (the device tree sets no `gpio-line-names`). Do not derive the offset from the vendor's
> sysfs number either: those numbers assume a TLMM base of 535 where this kernel uses 547, and
> the arithmetic lands on a different, also-free line that fails silently.

## 4. Set and persist the volume

```bash
alsamixer -c <ReSpeaker card>       # F6 to pick the card — the board's ES8316 is separate
sudo alsactl store
```

**`alsactl store` is not optional on this device.** `alsamixer` does not save the change when you
make it; `alsa-restore.service` writes state out only from its `ExecStop`, on a *clean* shutdown.
A battery-powered device gets hard-powered-off routinely, so without an explicit store the level
silently reverts.

## 5. Start the server

```bash
python server.py
```

Wait for `All models ready` — about 13s with warm page cache.

## 6. Smoke-test each stage

From a second shell, using the tracked demo files:

```bash
curl http://localhost:8000/health
python scripts/stt_infer.py --file demo/audio/question.wav
python scripts/vlm_infer.py demo/images/IMG_6817.jpg
python scripts/tts_infer.py "Xin chào" --out data/audio/output.wav
```

## 7. Test VAD

Speak, then stop. Recording should end on its own about 0.8s after you finish.

```bash
python pipeline.py --image demo/images/IMG_6817.jpg   # VAD only, camera bypassed
python pipeline.py                                    # full live path
```

Look for `[mic] N.Ns of speech`. That number should be close to how long you actually spoke — if
it is much longer, the VAD is not endpointing where you think it is.

## 8. Test the button and cues

```bash
python pipeline.py --button
```

Expect the ready triad at startup, a rising chirp when you press, and a falling chirp when you
stop speaking.

The button is a **3-pin module with an onboard pull-up**, signal to pin 16 (`GPIO_26`), read
active-low (`Edge.FALLING`, 50ms debounce). A bare 2-pin switch does *not* work on this board:
with nothing holding the line it floats on release, the CMOS input keeps its last charge, and the
pin latches pressed after one press. The internal pull-up cannot substitute — `libgpiod`'s bias
request is silently ignored by this pinctrl driver. If `--button` reports `Device or resource
busy`, another process still holds line 26; check for a leftover `button_monitor.py`.

`python scripts/button_monitor.py` prints the level continuously and logs every transition, which
is the fastest way to tell a wiring fault from a code one.

Cues are synthesised at 0.25 of full scale, so they are **quieter than the spoken answer**. If you
set the volume in step 4 against speech and the chirps then turn out too faint, adjust and re-run
`sudo alsactl store` — otherwise the correction is lost at the next power cut. If the *gap* between
cues and speech is wrong rather than the overall level, move `CUE_AMPLITUDE` at the top of
`pipeline.py` instead.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `--button` exits with "No access to /dev/gpiochip4" | udev rule / gpio group not set up | Step 3 |
| Button does nothing, no error | Wrong line offset, or the signal wire is not on pin 16 | `python scripts/button_monitor.py`; if the level never moves, sweep the free TLMM lines while pressing (see below) |
| Button registers once, then never again | Bare 2-pin switch — line floating on release | Use the 3-pin module with its onboard pull-up; step 8 |
| Image described upside down | Module mounted inverted | `ROTATE` in `pipeline.py` (default 180) / `XEYE_CAMERA_ROTATE` |
| Every question hits `Hit the 12s cap` | Background noise holds the VAD open | Technical report 2.3; raise `VAD_THRESHOLD` |
| `No speech in 6s` but the mic works | ReSpeaker not enumerating — `alsa_device()` falls back to `default` and listens to the wrong card | `cat /proc/asound/cards`, confirm ReSpeaker is present |
| Question cut off mid-sentence | Endpoint too aggressive | `--silence 1.0` |
| Volume reset after a reboot | `alsactl store` never ran | Step 4 |
| Speech audible, cues not | Cue amplitude relative to speech | Step 8 |
| No audio at all | Wrong card, or muted in the mixer | `alsamixer -c <ReSpeaker>`, unmute with `m` |
| `Camera capture failed — no frame written` | CSI camera not detected | `dmesg \| grep 'Probe success'`; never hot-plug the camera |

### Finding the button line empirically

If the button is wired but the configured offset is wrong, this sweeps every free TLMM line
while you press and reports which one moves. Each press and release is one transition, so press
several times — a line reporting a single transition is floating, not the switch.

```bash
sudo python - <<'EOF'
import gpiod, time
from gpiod.line import Direction, Bias
CHIP = '/dev/gpiochip4'
c = gpiod.Chip(CHIP); free = [o for o in range(c.get_info().num_lines)
                              if not c.get_line_info(o).used]; c.close()
reqs, base, hits = [], {}, {}
for i in range(0, len(free), 60):          # a request is capped at 64 lines
    batch = free[i:i+60]
    r = gpiod.request_lines(CHIP, consumer='xeye-scan', config={
        o: gpiod.LineSettings(direction=Direction.INPUT, bias=Bias.PULL_UP) for o in batch})
    reqs.append((r, batch))
    base.update({o: r.get_value(o) for o in batch})
print('press the button repeatedly for 20s'); t0 = time.time()
while time.time() - t0 < 20:
    for r, batch in reqs:
        for o in batch:
            v = r.get_value(o)
            if v != base[o]: hits[o] = hits.get(o, 0) + 1; base[o] = v
    time.sleep(0.005)
for r, _ in reqs: r.release()
print(sorted(hits.items(), key=lambda kv: -kv[1]) or 'no line moved')
EOF
```

---

## Hardware verification status

Checked on the board 07-17/08/2026. All three original suspicions are resolved:

| Item | Status |
|------|--------|
| `sherpa_onnx` VAD API vs pinned `1.13.2` | ✅ works — endpointed real speech at `1.1s of speech (4.3s listening)` |
| libgpiod major version | ✅ v2 (`gpiod 2.5.0`), so the `request_lines` path is the live one |
| `RollingCamera` ring buffer on `/dev/shm` | ✅ works — streams, buffers and grabs; sharpest-frame selection scores 10 frames in 43ms |
| Cues (`listening`, `captured`, `ready`, `error`) | ✅ audible on the ReSpeaker |
| Button | ✅ works — 3-pin module on pin 16 (`GPIO_26`), full press→answer round trip in 26.4s |

**The camera module was replaced.** The Module 2 stopped answering I2C entirely (`read id: 0x0`,
zero `Probe success`, NACK on both connectors) with no software change — a hardware fault in the
module or its ribbon. A Module 3 (IMX708) on the same connector probed immediately. Consequences
to know when testing:

- **Autofocus does not work** on the Module 3 and cannot be enabled — no actuator driver is bound.
  Focus sits at the lens's unpowered rest position, which happens to be usefully far on this unit.
- **Frames arrive rotated 180°** with the current mounting, and the capture path now corrects it
  (`ROTATE = 180`, override with `XEYE_CAMERA_ROTATE`). It is not cosmetic: the VLM answered
  *"một chiếc máy móc"* on an inverted frame against *"một chiếc laptop"* on the same frame upright.
- The module now sits on **CSI connector 2**. `CAMERA` in `pipeline.py` stays `0` — it is an index
  into detected cameras, not a connector number. `dmesg | grep 'Probe success'` reports the real slot.
- The near limit is ~1.3m against the Module 2's ~0.8m.

**The cues fail quietly by design** — `beep()` swallows errors so a missing sound card can never
end a query. So silence from the cues means checking `aplay` and the mixer, not that the tone
generation was wrong.
