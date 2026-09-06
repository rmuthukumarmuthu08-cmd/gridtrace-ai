# GridTrace AI

**Finding energy loss on a distribution feeder — and knowing when not to blame anyone.**

Hybrid Isolation Forest + autoencoder anomaly detection with hierarchical localization
and explainable investigation scoring. Runs end to end on one laptop. No cloud.

Built by **Nirmal (Muthukumar R)** · Team Squad Mavericks · Amrita Vishwa Vidyapeetham,
Nagercoil Campus — for a 36-hour hackathon under the *Sustainable Energy & Resource
Innovation* theme.

---

## The problem

**16.16%** of the electricity entering India's distribution network is never accounted
for — Aggregate Technical & Commercial (AT&C) loss, FY25, Ministry of Power. That number
has barely moved since 2023, when it hit 15.5%.

The difficulty is not measuring the gap. It is explaining it. Four very different things
produce an identical signature in billing data:

| Cause | What it actually is |
|---|---|
| Technical loss | Resistive heating in lines and transformers. Physics. Always present, roughly 3%. |
| Faulty metering | A meter that drifts or under-registers looks exactly like loss. |
| Metering / data gaps | A meter offline for a week produces a hole that looks exactly like theft. |
| Unmetered consumption | Genuine unbilled draw — the case a utility actually wants to find. |

A system that reports "loss detected" has told the utility nothing it did not already
know. What is missing is a defensible answer to *"so where do I send the inspector, and
why?"*

---

## What this does

1. **Detects** anomalous meter readings with two independently trained models.
2. **Localizes** the finding up a hierarchy — Network → Feeder → Zone → Consumer.
3. **Explains** every score in terms of the values that produced it.
4. **Refuses to accuse** when the evidence does not support it.

That last point is the design centre of the project. When a feeder shows a large energy
gap but no individual meter looks unusual, the system reports **unattributed loss at
feeder level** rather than pinning it on the highest-scoring consumer. It knows when the
honest answer is *nobody*.

The system never outputs the word "theft". It outputs **high-risk anomaly, investigation
required** — because theft, a failed meter and an unrecorded connection are not
distinguishable from telemetry alone, and claiming otherwise is how people get falsely
accused.

---

## Repository layout

```
gridtrace-ai/
├── hardware-sim/        Stage 1 — ESP32 / Wokwi field-device simulation
│   ├── sketch.ino           firmware (identical to src/main.cpp)
│   ├── diagram.json         Wokwi wiring: ESP32 + SSD1306 + 2 buttons + pot + 2 LEDs
│   ├── simulate.py          Python port of the firmware, verified to 0.011%
│   └── validate.py          16 physical-consistency checks on emitted telemetry
│
├── gridtrace/           Stage 2 — the ML pipeline
│   ├── gridtrace/           broker · publisher · features · models · localization · dashboard
│   ├── tests/               10 end-to-end checks
│   └── run_demo.py          one-command demo launcher
│
├── docs/
│   └── GridTrace_AI_Pitch.pptx
└── DEMO_RUNBOOK.md      minute-by-minute demo script + judge Q&A
```

Each stage has its own README with the full detail. Start with
[`gridtrace/README.md`](gridtrace/README.md) if you want the ML; start with
[`hardware-sim/README.md`](hardware-sim/README.md) if you want the device side.

---

## Quick start

```bash
cd gridtrace
pip install -r requirements.txt
python -m gridtrace.train        # ~10 s — REQUIRED before anything else
python tests/test_pipeline.py    # expect: ALL CHECKS PASSED
python run_demo.py               # broker + publisher + inference + dashboard
```

Then open <http://127.0.0.1:5000>.

> **Train first.** The model bundle is deliberately not committed — it is a
> version-sensitive scikit-learn pickle, and loading one written by a different
> version can give silently wrong results. `train.py` regenerates it in about ten
> seconds and warns you loudly if a stale bundle is ever loaded.

---

## How it works

### Two models, because they fail differently

**Isolation Forest** is strong on readings that sit in a sparse region of feature space.
The **autoencoder** (9–6–3–6–9, NumPy, no framework dependency) is strong on readings
whose *internal relationships* are wrong — power that does not reconcile with
voltage × current × power factor, for instance.

Their raw scores are not comparable, so each is calibrated against the distribution of
known-normal behaviour into a percentile, then passed through a severity transform:

```
severity(p) = clip( -log10(1 - p) / 3, 0, 1 )
```

and combined:

```
hybrid = 0.5 · mean(s_if, s_ae) + 0.5 · max(s_if, s_ae)
```

The mean term means agreement between the two raises confidence. The max term means
either one alone can still raise a flag. A test asserts the hybrid always lies between
the two component scores, so neither model can dominate by scale.

### Trained on normal data only

Both models see **only NORMAL-mode telemetry** during training, across three random
seeds and three demand levels. Nothing at training time has ever seen an anomaly, so
neither model can have memorised the specific ones the simulator injects — they can only
report *"this does not look like what I learned."* The test suite verifies attribution
against `anomaly_injected`, a ground-truth field the models are never given.

### Features are consumer-relative

A large commercial load is compared to **its own history**, not to a residential
neighbour: z-score against the consumer's own baseline, rolling-mean ratio, rolling
coefficient of variation, hourly-profile ratio. This is what lets one trained model
generalise from 5 meters to 50,000 without a per-meter threshold.

### The feeder gap is evidence, not a feature

An early version put feeder imbalance into the model's feature vector. It destroyed
localization: every consumer on a feeder shares the same gap value, so when the gap rose,
all five scores rose together and the system could no longer say *which* meter to look
at — the one thing it exists to do.

So the gap sits outside the model as a capped evidence term (12 of 100 points). It can
escalate a finding. It can never create one on its own. This was a deliberate departure
from the original design, and the reasoning is documented in `hybrid.py`.

### Risk score

```
risk = 100 · hybrid  +  12 · network_evidence  +  12 · persistence
```

| Band | Score |
|---|---|
| NORMAL | 0–30 |
| SUSPICIOUS | 31–60 |
| HIGH RISK | 61–80 |
| CRITICAL | 81–100 |

---

## Measured results

Every figure below is produced by `tests/test_pipeline.py` on the working prototype.
None of it is projected.

| | |
|---|---|
| End-to-end checks passing | **10 / 10** |
| Normal readings left in the NORMAL band | **92.2%** |
| Normal readings reaching HIGH RISK | **0.1%** |
| Network-only events localized to the feeder, no consumer accused | **93%** of frames |
| MQTT messages through the full path | **1,120 received → 680 scored**, zero loss |
| Malformed / NaN telemetry | rejected without interrupting the stream |
| Full retrain | **~10 s** on a laptop |

---

## Honest limits

- Baselines need roughly two weeks of clean per-consumer history to be meaningful.
- Real meters are noisier than the simulator; expect the false-positive rate to rise on
  field data before tuning.
- The technical-loss baseline (3.1% here) is a per-feeder constant a utility would
  calibrate from its own records, not take from this repository.
- The system produces a ranked investigation queue with reasons attached. **A human
  decides.** It does not and cannot prove theft.

---

## Stage 1 — hardware simulation

The `hardware-sim/` firmware models one feeder (F01), two zones (Z01/Z02) and five
consumers (C001–C005), and emits NDJSON telemetry (`els.v1`) over serial. The electrical
model is physically consistent by construction — current is derived as `I = P/(V·PF)`, so
`P = V·I·PF` holds exactly — with voltage droop `V = 233 − 0.42·ΣkW`.

Four operating modes: `NORMAL`, `ABNORMAL_CONSUMPTION`, `PERSISTENT_ANOMALY`,
`NETWORK_IMBALANCE`.

**These simulate measurement-level anomalies only.** Nothing in this repository
describes, enables, or instructs the physical theft of electricity. The anomaly modes
perturb what a meter *reports*; they are a test fixture for a detector.

`simulate.py` is a Python port of the same model, verified against the firmware to a
0.011% worst-case relative difference, so the ML stage can be trained and demonstrated
without an ESP32 attached.

---

## License

MIT — see [LICENSE](LICENSE).
