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

`XEYE_BUTTON_LINE` must already be exported, and the volume already stored. If either is fresh,
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

## 3. Find the button's GPIO line

Once per board. libgpiod addresses a line as chip + offset, which is **neither** the physical pin
number (13) **nor** the sysfs number the vendor docs use (559). `--button` refuses to start until
this is set, rather than waiting silently on a guessed line.

```bash
gpiofind GPIO_24                    # → "<chip> <offset>"; fall back to `gpioinfo`
export XEYE_BUTTON_LINE=<offset>
export XEYE_BUTTON_CHIP=/dev/gpiochipN   # only if not gpiochip0
```

Put both in `~/.bashrc` or the PM2 environment so they survive a reboot.

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

Cues are synthesised at 0.25 of full scale, so they are **quieter than the spoken answer**. If you
set the volume in step 4 against speech and the chirps then turn out too faint, adjust and re-run
`sudo alsactl store` — otherwise the correction is lost at the next power cut. If the *gap* between
cues and speech is wrong rather than the overall level, move `CUE_AMPLITUDE` at the top of
`pipeline.py` instead.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `--button` exits with "Button line not configured" | `XEYE_BUTTON_LINE` unset | Step 3 |
| Button does nothing, no error | Wrong line offset, or the switch is not across pins 13/14 | Re-check with `gpioinfo`; confirm continuity across the switch |
| Every question hits `Hit the 12s cap` | Background noise holds the VAD open | Technical report 2.3; raise `VAD_THRESHOLD` |
| `No speech in 6s` but the mic works | ReSpeaker not enumerating — `alsa_device()` falls back to `default` and listens to the wrong card | `cat /proc/asound/cards`, confirm ReSpeaker is present |
| Question cut off mid-sentence | Endpoint too aggressive | `--silence 1.0` |
| Volume reset after a reboot | `alsactl store` never ran | Step 4 |
| Speech audible, cues not | Cue amplitude relative to speech | Step 8 |
| No audio at all | Wrong card, or muted in the mixer | `alsamixer -c <ReSpeaker>`, unmute with `m` |
| `Camera capture failed — no frame written` | CSI camera not detected | `dmesg \| grep 'Probe success'`; never hot-plug the camera |

---

## Untested on hardware

The VAD, rolling camera, button and cue code has been written and verified as far as a
non-Linux machine allows — it compiles, and the cue waveforms were checked for clipping and
click-free edges — but **none of it has run on the board**.

The likeliest failure points:

- the `sherpa_onnx` VAD API against the pinned `1.13.2`
- `RollingCamera` writing its ring buffer to `/dev/shm`
- which libgpiod major version is installed (both v1 and v2 are handled, but only one is real here)

Those three fail loudly. **The cues fail quietly by design** — `beep()` swallows errors so a
missing sound card can never end a query. So silence from the cues means checking `aplay` and the
mixer, not that the tone generation was wrong.
