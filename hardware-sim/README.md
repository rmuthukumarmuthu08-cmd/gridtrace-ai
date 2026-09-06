# Stage 1 — Wokwi Hardware / Telemetry Simulation

**Project:** AI-Powered Real-Time Energy Loss Investigation & Localization System
**Author:** Nirmal (Muthukumar R) — Team Squad Mavericks
**Scope of this stage:** ESP32 simulation + clean NDJSON telemetry **only**. No ML, no n8n, no backend, no dashboard.

---

## 1. What's in the box

| File | Purpose |
|---|---|
| `diagram.json` | Wokwi wiring (ESP32 + OLED + 2 buttons + slider + 2 LEDs) |
| `sketch.ino` | Firmware — use this on **wokwi.com** |
| `src/main.cpp` | The *same* firmware — use this with **VS Code + PlatformIO** |
| `libraries.txt` | Library list for wokwi.com |
| `platformio.ini` | Build config for the VS Code route |
| `wokwi.toml` | Points the Wokwi VS Code extension at the built firmware |
| `test/` | Host-side test harness + `validate.py` (see §7) |

`sketch.ino` and `src/main.cpp` are byte-identical. Keep only the one your workflow needs.

---

## 2. Hardware in the simulation

| Part | Pin | Role |
|---|---|---|
| SSD1306 OLED 128×64 (`wokwi-ssd1306`) | `DATA`→`GPIO21`, `CLK`→`GPIO22`, `VIN`→3V3, `GND`→GND | Live meter readout |

> Pin-name gotcha: Wokwi's `wokwi-ssd1306` names its I²C pins **`DATA`/`CLK`**, not `SDA`/`SCL`, and its supply **`VIN`**, not `VCC`. Wire it with the wrong names and Wokwi silently drops the connection — the display just stays black. The firmware also scans both `0x3C` and `0x3D` because Adafruit breakouts ship on the latter.
| Pushbutton "MODE" | `GPIO25` → GND, `INPUT_PULLUP` | Cycle the 4 operating modes |
| Pushbutton "VIEW" | `GPIO26` → GND, `INPUT_PULLUP` | Cycle OLED page (C001…C005, feeder) |
| Slide potentiometer | `GPIO34` (ADC1, input-only) | Grid demand scaler ×0.60 … ×1.40 |
| Green LED "NORMAL" | `GPIO18` + 220 Ω | Solid = NORMAL mode |
| Red LED "ANOMALY" | `GPIO19` + 220 Ω | Blinks = any anomaly mode active |

There is no current transformer or voltage divider part in Wokwi that models a real mains meter, so **voltage, current, power factor and energy are synthesised in firmware** from a physically consistent model (§4). Nothing here touches mains and nothing here should ever be wired to mains.

---

## 3. Network hierarchy

```
Feeder F01
 ├── Zone Z01
 │    ├── C001  residential_low     ~0.45 kW base
 │    ├── C002  residential_medium  ~1.10 kW base
 │    └── C003  residential_high    ~2.20 kW base
 └── Zone Z02
      ├── C004  commercial          ~3.60 kW base
      └── C005  variable_load       ~1.60 kW base
```

---

## 4. How the numbers are generated (not random)

Each 1-second tick advances a **virtual clock by 1 simulated minute**, so a full 24-hour load curve plays out in ~24 real minutes.

1. **Load shape** — each consumer has a time-of-day profile: residential double peak (morning + evening), commercial 09:00–18:30 plateau, variable = oscillating. Multiplied by a slow random walk (±15 %, AR(1) smoothed) and by the slider's grid-demand scaler.
2. **Bus voltage** — `V = 233.0 − 0.42 × ΣkW + noise`, clamped 200–250 V, then a per-zone drop (Z01 ≈ −1.5 V, Z02 ≈ −3.3 V). Voltage therefore **sags when the feeder is loaded**, which is what a real feeder does.
3. **Power factor** — per-consumer nominal (0.89 commercial … 0.97 low residential) with ±0.015 jitter.
4. **Current** — derived, not invented: `I = P / (V × PF)`. This makes `P = V × I × PF` hold exactly, by construction.
5. **Energy** — `E += P × Δt`, Δt = 1 simulated minute. Two accumulators per consumer: *true delivered* and *meter registered*.
6. **Feeder balance** — `input = ΣtrueKW × (1 + 3.2 % technical loss) + injected unaccounted block`.
   `unaccounted = feeder_input_energy − Σ meter_energy`.

The RNG is a fixed-seed LCG, so **the same run produces the same dataset every time** — useful when you start training and debugging models.

---

## 5. The four operating modes

| # | Mode | What it injects | Ground-truth `label` |
|---|---|---|---|
| 1 | `NORMAL` | Nothing. Baseline unaccounted ≈ 3.1 % (technical loss only) | `normal` |
| 2 | `ABNORMAL_CONSUMPTION` | C003 sudden ×2.4 current surges; C005 unexpected drops to 25 % of registered load | `abnormal_consumption` |
| 3 | `PERSISTENT_ANOMALY` | C002 meter persistently under-registers at 45 %; C004 draws load at implausible hours (00:00–05:00) | `persistent_deviation` |
| 4 | `NETWORK_IMBALANCE` | Consumer meters all read normal; a 9–17 % block appears only at the feeder head | `network_imbalance` |

Every record also carries `anomaly_injected: true/false` per consumer. Keep `label` / `anomaly_injected` for supervised training and **drop them at inference** — they're ground truth, not features.

> **Framing that matters for your judges:** these are *measurement* anomalies, not theft mechanisms. An energy gap at the feeder head has at least four competing explanations — technical loss, a faulty/drifting meter, a metering or data gap, and unmetered consumption. Your Stage-2 classifier is what distinguishes them; the simulator deliberately does not pre-label a gap as theft.

---

## 6. Running it

### Route A — wokwi.com (fastest)

1. Go to `https://wokwi.com/projects/new/esp32`.
2. Open the **`diagram.json`** tab → select all → paste in this repo's `diagram.json`.
3. Open the **`sketch.ino`** tab → select all → paste in this repo's `sketch.ino`.
4. Open **Library Manager** → **+** → search `Adafruit SSD1306` → add it. (`Adafruit GFX Library` is pulled in automatically.)
5. Press the green **▶ Play** button. First build takes ~20–40 s in the free build queue.
6. The **Serial Monitor** panel is at the bottom of the simulation pane.

> **Known Wokwi UI bug:** on some Chrome profiles wokwi.com throws a React hydration error (`Minified React error #418`) at page load and the Serial Monitor panel never mounts — the board runs fine, you just can't see its output. It is caused by an extension mutating the DOM before React hydrates. If your Serial Monitor is missing, open Wokwi in an **Incognito window** (extensions off) or a clean Chrome profile. Check with F12 → Console for error #418.

> **Keep the Wokwi tab in the foreground.** Chrome throttles background tabs and the ESP32 clock will crawl.

### Route C — no Wokwi at all (works today, on your laptop)

The firmware's simulation core is ported 1:1 to `simulate.py`, so you can generate
labelled telemetry with nothing but Python:

```bat
run.bat                                     :: generate + validate in one go
python simulate.py --ticks 900 --out data.ndjson
python simulate.py --mode NETWORK_IMBALANCE --ticks 600
python simulate.py --seed 12345             :: independent validation set
python validate.py data.ndjson
```

`telemetry_sample.ndjson` in this folder is already generated: 480 minutes in each
of the four modes, 15,368 records, ready to load into pandas.

The port is verified against the firmware, not assumed equivalent: running both on
the same seed and comparing mean power per consumer per mode and unaccounted-energy
percentage gives a **worst-case relative difference of 0.011 %** (32-bit float on the
ESP32 vs 64-bit on the host). Both pass the same 12-check suite with identical
verdicts. Use Python for datasets, the firmware for the hardware demo.

### Route B — VS Code + PlatformIO + Wokwi extension

```bash
pio run                 # builds .pio/build/esp32dev/firmware.bin + .elf
# then: F1 → "Wokwi: Start Simulator"
```

`wokwi.toml` already points at the right artefacts. If you use this route you can delete `sketch.ino` and `libraries.txt`.

---

## 7. Switching modes

Three ways, all equivalent:

- **MODE button** on the breadboard — cycles NORMAL → ABNORMAL → PERSISTENT → IMBALANCE → NORMAL.
- **Serial Monitor** — type `1`, `2`, `3` or `4` and press Enter. (`m` = next mode.)
- Other serial keys: `n` = next OLED page, `p` = pause/resume the stream, `r` = reset energy accumulators.

Every change emits an audit record:

```json
{"record_type":"event","schema":"els.v1","uptime_ms":41700,"event":"MODE_CHANGE","mode":"NETWORK_IMBALANCE","mode_index":3}
```

---

## 8. What you should see in the Serial Monitor

A **meta** record and an **OLED_INIT** record at boot, then **8 lines per second**: 5 consumer + 2 zone + 1 feeder.

```json
{"record_type":"event","schema":"els.v1","event":"OLED_INIT","present":true,"ok":true}
```

`OLED_INIT` tells you whether the I2C display actually answered. `present:false` means the OLED isn't wired or isn't responding — the telemetry stream keeps running regardless, by design: serial output is deliberately printed *before* the display is touched, and the I2C probe has a 50 ms timeout, so a dead display can never silence your data.

```json
{"record_type":"consumer","schema":"els.v1","timestamp":"2026-09-05T06:00:00Z","uptime_ms":1700,"tick":0,"consumer_id":"C001","zone_id":"Z01","feeder_id":"F01","load_type":"residential_low","voltage":229.24,"current":1.106,"power":244.1,"power_factor":0.962,"energy":0.0041,"status":"OK","meter_status":"ONLINE","mode":"NORMAL","label":"normal","anomaly_injected":false}
{"record_type":"zone","schema":"els.v1","timestamp":"2026-09-05T06:00:00Z","uptime_ms":1700,"tick":0,"zone_id":"Z01","feeder_id":"F01","metered_power":3041.5,"metered_energy":0.0507,"mode":"NORMAL","label":"normal"}
{"record_type":"feeder","schema":"els.v1","timestamp":"2026-09-05T06:00:00Z","uptime_ms":1700,"tick":0,"feeder_id":"F01","consumers":5,"input_power":4726.9,"input_energy":0.0788,"sum_meter_energy":0.0763,"unaccounted_energy":0.0024,"unaccounted_pct":3.10,"technical_loss_pct_expected":3.10,"grid_load_scale":1.00,"mode":"NORMAL","label":"normal","note":"SIMULATED NETWORK IMBALANCE - unaccounted energy is not proof of theft"}
```

Sanity checks you can eyeball:

- `229.24 × 1.106 × 0.962 = 244.0 W` ✓ matches `power`
- Voltage sits ~225–231 V and dips when the slider raises demand
- `unaccounted_pct` ≈ 3.1 % in NORMAL, climbs to ~7–9 % in NETWORK_IMBALANCE

Capture to a file for Stage 2 — it's already NDJSON, so:

```python
import json
rows = [json.loads(l) for l in open("telemetry.ndjson") if l.strip()]
consumers = [r for r in rows if r["record_type"] == "consumer"]
```

---

## 9. Verification actually performed

`src/main.cpp` is compiled **natively** against thin Arduino/Adafruit stubs (`test/stub/`),
driven through all four modes for 1920 telemetry frames (15,366 records), and checked by
`validate.py`:

```
PASS  1.  all 15366 lines are valid JSON (bad lines: 0)
PASS  2.  every consumer record has all required fields
PASS  3.  P = V*I*PF holds on all 9600 records (worst rel. error 0.14 %)
PASS  4.  per-consumer energy is monotonically non-decreasing
PASS  5.  feeder balance closes: in - metered - unaccounted = 0 (residual 1e-04 kWh)
PASS  6a. every feeder record carries the instantaneous `unaccounted_power_pct`
PASS  6b. NORMAL instantaneous gap ~ technical loss (3.10 % of feeder input)
PASS  7a. NETWORK_IMBALANCE lifts the gap to 13.79 % (+10.7 pts)
PASS  7b. PERSISTENT_ANOMALY under-registration widens it to 8.25 % (+5.1 pts)
PASS  7c. ABNORMAL_CONSUMPTION also shows a raised gap (4.40 %)
PASS  7d. gap metric is history-free: NORMAL drifts only 0.000 pts first-quarter vs last
PASS  8.  ABNORMAL mode raises SURGE/DIP on ['C003', 'C004', 'C005']
PASS  9.  no NaN/inf/negatives; voltage range 224.2..230.5 V
PASS  10. frame integrity: 1920 frames x (5 consumer + 2 zone + 1 feeder)
```

Reproduce it:

```bash
g++ -std=c++17 -O1 -I test/stub src/main.cpp test/host_harness.cpp -o sim
./sim 480 > telemetry.ndjson
python3 validate.py telemetry.ndjson
```

The 0.14 % worst-case error on `P = V·I·PF` is serial-print rounding (`%.3f` on current),
not a modelling error.

### The cumulative-gap trap (worth knowing for Stage 2)

The first version of this simulator reported the energy gap **only** cumulatively:

```
unaccounted_pct = (feeder_input_energy − Σ meter_energy) / feeder_input_energy
```

Both terms accumulate from boot, so the metric gets steadily **less sensitive the longer
the meter runs** — a real gap opening on day 30 barely moves a number averaged over 30
days of clean history. It was caught because the validation thresholds passed at 300
frames per mode and failed at 480: a metric whose value depends on how long you ran is
not a metric.

Every feeder record now carries **both**:

| Field | Meaning | Use it for |
|---|---|---|
| `unaccounted_energy` / `unaccounted_pct` | cumulative since boot | billing-period reconciliation, long-horizon reporting |
| `unaccounted_power` / `unaccounted_power_pct` | **instantaneous**, this frame only | anomaly detection, localisation, model features |

Mode separation on the instantaneous metric is clean and run-length independent:

| Mode | Instantaneous gap |
|---|---|
| `NORMAL` | 3.10 % (technical loss floor) |
| `ABNORMAL_CONSUMPTION` | 4.40 % |
| `PERSISTENT_ANOMALY` | 8.25 % |
| `NETWORK_IMBALANCE` | 13.79 % |

**Feed your model `unaccounted_power_pct`, not `unaccounted_pct`.**

### Firmware vs Python port

Both implementations are run on the same seed and compared across 24 series (mean power
per consumer per mode, plus gap % per mode). Worst relative difference: **0.0107 %**,
which is 32-bit float on the ESP32 against 64-bit on the host. Both pass the same suite
with identical verdicts.

## 10. "Simulation Ready" checklist

Tick these before you move to the ML stage:

- [ ] Wokwi compiles with no errors and the ▶ timer advances (tab in foreground)
- [ ] Serial Monitor shows 8 JSON lines per second, no truncated lines
- [ ] OLED cycles pages every 3 s and shows V / I / P / E / PF / status
- [ ] Green LED solid in NORMAL; red LED blinks in the other three modes
- [ ] MODE button cycles all four modes and each emits a `MODE_CHANGE` event
- [ ] Moving the slider visibly changes current on the OLED **and** drops voltage
- [ ] `unaccounted_power_pct` ≈ 3 % in NORMAL, ~14 % in NETWORK_IMBALANCE
- [ ] Copy 5–10 minutes of Serial output to `telemetry.ndjson` and it parses in Python with zero errors
- [ ] Run left going 10+ minutes with no reset, no watchdog reboot, no `Guru Meditation`
- [ ] You can state out loud why an imbalance ≠ theft (technical loss / faulty meter / data gap / unmetered load)

---

## 11. Knobs worth tuning before Stage 2

All at the top of `main.cpp`:

- `SIM_SECONDS_PER_TICK` (60) — raise to 300 for a 24 h curve in ~5 min, lower for finer resolution
- `TECH_LOSS_FRAC` (0.032) — your baseline "normal" loss; the AI's decision threshold sits above it
- `DEV_THRESHOLD` (0.35) — the naive EWMA rule the firmware uses for `status`; your model should beat it
- `CFG[]` — add consumers, change base loads, add more zones
- `g_rng` seed (`0xC0FFEE`) — change it to generate an independent dataset for validation
