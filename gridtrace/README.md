# GridTrace AI

**AI-powered real-time energy loss detection and localization — local prototype.**
Nirmal (Muthukumar R) · Team Squad Mavericks

Hybrid Isolation Forest + autoencoder anomaly detection over simulated ESP32 smart-meter
telemetry, with hierarchical localization and explainable risk scoring. Runs entirely on
one laptop: no cloud, no Docker, no Mosquitto install, no TensorFlow.

---

## Quick start

```bash
pip install -r requirements.txt

python -m gridtrace.train        # ALWAYS run this first  (~10 s)
python run_demo.py               # broker + inference + dashboard + telemetry
```

> **Train before you run anything else.** `models_store/gridtrace_bundle.joblib` ships as a
> convenience, but scikit-learn estimators are pickles: loading one saved by a *different*
> scikit-learn version raises `InconsistentVersionWarning` and can give wrong results
> silently. `train.py` records the version it trained with and warns you on load if they
> differ. Retraining takes ten seconds and makes the warning — and the risk — go away.

The dashboard opens at **http://127.0.0.1:8000**. `run_demo.py` plays a scripted tour —
NORMAL → ABNORMAL → NORMAL → PERSISTENT → NORMAL → IMBALANCE → NORMAL — so all four
scenarios appear without you touching anything.

To prove it works:

```bash
python tests/test_pipeline.py    # 10 end-to-end checks (~2 min)
```

---

## Pipeline

```
ESP32 / Wokwi simulator          gridtrace/simulator_core.py   (same NDJSON the firmware emits)
        │
        ▼  MQTT  gridtrace/telemetry/{consumer,zone,feeder}
   local broker                  gridtrace/broker.py           (pure-Python, ~180 lines)
        │
        ▼
   collector + validation        gridtrace/infer.py            (malformed messages dropped, never fatal)
        │
        ▼
   feature processing            gridtrace/features.py         (consumer-relative, streaming)
        │
        ├──► Isolation Forest    scikit-learn
        └──► Autoencoder         gridtrace/autoencoder.py      (NumPy, 9-6-3-6-9)
                 │
                 ▼
        Hybrid anomaly score     gridtrace/hybrid.py           (calibrated, then combined)
                 ▼
        Risk score 0–100 + band
                 ▼
        Localization + reasons   gridtrace/localize.py         (Network→Feeder→Zone→Consumer)
                 ▼
        SQLite  +  MQTT alert    gridtrace/storage.py          data/gridtrace.db
                 ▼
        Dashboard                gridtrace/dashboard.py        http://127.0.0.1:8000
```

The simulator's record format is **unchanged** — the same `record_type` / `consumer_id` /
`voltage` / `current` / `power` / `energy` / `power_factor` / `timestamp` schema the Wokwi
firmware prints on serial. Swapping in the real ESP32 means replacing the loop in
`publisher.py` with a `pyserial` reader; nothing downstream changes.

---

## How the two models are trained

Both are trained **on normal data only** (`mode 0`), across 3 RNG seeds × 3 demand levels
so the models see genuine variety rather than one narrow run. Nothing at training time has
ever seen an anomaly, so the models cannot have memorised the specific ones the simulator
injects — they can only report *"this doesn't look like what I learned."*

| | Isolation Forest | Autoencoder |
|---|---|---|
| Library | scikit-learn | NumPy (hand-written) |
| Shape | 200 trees, `max_samples=256` | 9 → 6 → **3** → 6 → 9, tanh, Adam |
| Learns | where normal points sit, by how few splits isolate them | how to reconstruct normal behaviour through a 3-unit bottleneck |
| Anomaly signal | short isolation path | high reconstruction MSE |
| Good at | coordinate-wise outliers (one value far out) | broken *relationships* between features |

They fail differently, which is the entire reason for running both.

**Preprocessing.** `StandardScaler` fitted on normal data only. The autoencoder holds out
15% for validation and early-stops on it (patience 30) — training MSE is not used to decide
when to stop, so it can't overfit its way to a low score.

### Consumer-specific baselines

C004 (commercial, ~2.9 kW mean) and C001 (low residential, ~0.23 kW mean) aren't comparable
in raw watts. A model trained on raw power would just learn *"C004 is big"* and flag C001's
evening peak as an outlier. So the features are **consumer-relative**:

| Feature | What it means |
|---|---|
| `voltage`, `current`, `power_kw`, `power_factor` | raw electrical quantities |
| `energy_delta_kwh` | energy registered this interval (catches meters that under-register) |
| `dev_from_own_baseline` | z-score against **this consumer's** training mean/σ |
| `roll_mean_ratio` | 15-frame rolling mean ÷ this consumer's baseline |
| `roll_cv` | rolling coefficient of variation — volatility |
| `hourly_profile_ratio` | power ÷ this consumer's own mean **for this hour of day** |

One global model then generalises across all five meters — which is also how you'd scale
this to 50,000 of them, instead of training 50,000 models.

> **Design note — why feeder imbalance is *not* in this vector.** The spec suggested
> including it as a model feature. It's shared by all five consumers, so during a purely
> network-level event *every* meter looks anomalous and localization collapses — the system
> would blame whichever consumer happened to score highest. It enters as separate
> **supporting evidence** in the risk score instead (§ Risk score). This is what makes the
> NETWORK_IMBALANCE scenario localize correctly to the feeder.

---

## How the two outputs are combined

**The problem.** Isolation Forest returns a path-length score around [-0.5, +0.5]. The
autoencoder returns an MSE in [0, ∞) whose scale depends on how training converged.
Averaging those directly is meaningless — whichever has the larger numeric range dominates,
and the "hybrid" is one model wearing two names.

**Step 1 — calibrate both to the same unit.** At training time each model scores the normal
data; those distributions are stored. At inference a raw score becomes its **empirical
percentile within normal**: *"what fraction of known-normal behaviour is less anomalous than
this?"* Both models now emit a number in [0,1] with identical meaning.

**Step 2 — percentile → severity.** A percentile alone is still the wrong scale for risk: on
normal data percentiles are uniform, so half of perfectly healthy readings sit above 0.5 and
would score 50/100. What matters is how far into the *tail* a reading is:

```
severity = clip( −log₁₀(1 − percentile) / 3 , 0, 1 )
```

Each extra "nine" of rarity adds a fixed amount of risk: median normal → 0.10, 90th → 0.33,
99th → 0.67, 99.9th → 1.00.

**Step 3 — combine.**

```
hybrid = 0.5 · mean(s_IF, s_AE)  +  0.5 · max(s_IF, s_AE)
```

The **mean** term rewards *agreement* — two independent models finding the same point unusual
is much stronger evidence, and averaging suppresses single-model false positives. The **max**
term preserves *sensitivity* — a real anomaly only one model can see must still survive. Pure
mean is too conservative, pure max too jumpy; the even split is the useful middle. The hybrid
always lies between the two inputs, which test #8 asserts.

## How the risk score is calculated

```
risk = 100 · hybrid                      ← the body of the score, from the models
     + 12 · network_evidence             ← feeder gap above the 3.1% technical-loss floor
     + 12 · persistence                  ← fraction of last 12 intervals flagged
     → clipped to 0–100
```

| Band | Score | Meaning |
|---|---|---|
| NORMAL | 0–30 | nothing to do |
| SUSPICIOUS | 31–60 | watch |
| HIGH RISK | 61–80 | investigation required |
| CRITICAL | 81–100 | investigate first |

The two evidence terms are deliberately capped at 12 points each, so **no amount of feeder
imbalance alone can push a well-behaved consumer into HIGH RISK.** Network imbalance is a hint
about *where to look*, never a verdict about *who did it*.

## How localization works

`localize.py` rolls consumer scores up **Network → Feeder → Zone → Consumer** and picks the
level to investigate. The judgement that matters:

```
if feeder_excess ≥ 3 points  AND  no consumer scores ≥ 61:
        → FEEDER level, flagged UNATTRIBUTED
```

A large gap that no meter explains is a *feeder* finding — inspect the segment, check for
unmetered connections or a metering fault. Blaming the highest-scoring consumer there is
exactly the false accusation this system exists to avoid. In testing this fires on **93%** of
NETWORK_IMBALANCE frames.

Otherwise the highest-risk consumer is named, with its zone and feeder:

```
Consumer: C003   Zone: Z01   Feeder: F01
Hybrid Risk Score: 100/100   Status: CRITICAL
```

## Explainability

Reasons are derived from the values that actually produced the score — never hard-coded per
mode. Sources: calibrated model percentiles, the autoencoder's **per-feature** reconstruction
error (which input it failed to reproduce), the consumer's own deviation, the hour-of-day
ratio, persistence, and the feeder gap. Every high-risk result carries ≥ 2 factors (test #7).

```
HIGH-RISK ENERGY ANOMALY — investigation required
Consumer: C003   Zone: Z01   Feeder: F01   Risk: 100/100  CRITICAL
  - Very high autoencoder reconstruction error - the pattern does not match any
    normal behaviour the model learned
  - Isolation Forest isolates this reading from the normal population
  - Both models agree independently, which raises confidence
  - Consumption is 4.2 standard deviations from this consumer's own baseline
  - Draw is 2.4x this consumer's typical level for this hour
  - Pattern is persistent - flagged in 92% of the last 12 intervals
```

The system says **"high-risk anomaly / investigation required."** It never claims theft is
proven — an energy gap is equally consistent with technical loss, a drifting meter, a
metering/data gap, or unmetered load.

---

## Validation results

`python tests/test_pipeline.py` — all 10 checks pass:

```
PASS  1. trained model bundle loads
PASS  2. feature contract matches (9 features, 17,980 normal samples)
PASS  3. malformed / missing / NaN telemetry rejected without crashing
PASS  4. NORMAL data stays normal: 92.2% NORMAL band, 0.1% HIGH RISK+ (mean risk 14.9)
PASS  5a. ABNORMAL_CONSUMPTION: mean 37.4 > 14.9; top ['C003','C005'] = injected ['C003','C005']
PASS  5b. PERSISTENT_ANOMALY:   mean 30.5 > 14.9; top ['C002','C004'] includes injected ['C002']
PASS  6. NETWORK_IMBALANCE localizes to FEEDER as unattributed loss in 93% of frames
PASS  7. every high-risk result carries >=2 contributing factors (369 examined)
PASS  8. both scores calibrated to [0,1]; hybrid always lies between them
PASS  9. full MQTT path: 1120 received -> 680 scored -> 5 consumers in SQLite,
         93 alerts, 1 malformed rejected
```

Checks 4 and 5 are the ones that matter: normal stays normal, and the flagged consumers are
the ones the simulator actually perturbed — known from `anomaly_injected`, a ground-truth
field **the models never see**.

| Mode | Mean risk | % NORMAL band | Localizes to |
|---|---|---|---|
| NORMAL | 14.9 | 92.2% | nothing |
| ABNORMAL_CONSUMPTION | 37.4 | 57.2% | C003 (158×), C005 (83×) — both injected |
| PERSISTENT_ANOMALY | 30.5 | 65.1% | C002 (injected) + feeder/unattributed |
| NETWORK_IMBALANCE | 25.5 | 74.5% | **FEEDER, unattributed** — no consumer blamed |

---

## Test procedure — exact commands

```bash
# 0. install
pip install -r requirements.txt

# 1. train on normal data only (~10 s)
python -m gridtrace.train
#    → data/train_normal.ndjson, models_store/gridtrace_bundle.joblib

# 2. full validation suite (~2 min)
python tests/test_pipeline.py

# 3. NORMAL data -> predominantly normal results
python run_demo.py --mode 0 --fresh
#    dashboard stays green; risk mostly < 30

# 4. ABNORMAL data -> elevated risk, correct consumer named
python run_demo.py --mode 1 --fresh      # C003 surges, C005 under-registers
python run_demo.py --mode 2 --fresh      # C002 under-registers persistently
python run_demo.py --mode 3 --fresh      # feeder-level gap, no consumer at fault

# 5. scripted tour of all four (this is the demo)
python run_demo.py --fresh
```

Run the parts separately in 4 terminals if you prefer:

```bash
python -m gridtrace.broker        # 1. MQTT broker on 127.0.0.1:1883
python -m gridtrace.infer         # 2. inference service
python -m gridtrace.dashboard     # 3. http://127.0.0.1:8000
python -m gridtrace.publisher --script demo   # 4. telemetry
```

Already running Mosquitto? Skip step 1 — everything detects port 1883 is busy and uses yours.

### Retraining on more normal data

```bash
python -m gridtrace.train --append --ticks 600   # add data, retrain on all of it
python -m gridtrace.train --no-collect           # retrain on the existing file
```

---

## Demonstrating it at the hackathon

A five-minute run that lands every point:

1. **Start clean.** `python run_demo.py --fresh`. Dashboard opens; all five meters green,
   risk under 30. *"Both models were trained only on normal data — this is what agreement
   looks like."*
2. **Wait for ABNORMAL (~40 s in).** C003 goes CRITICAL. Point at the row: Isolation Forest
   and autoencoder scores rise *together*, and localization names C003 / Z01 / F01 with
   reasons. *"Nothing here is a threshold I picked — both models were trained without ever
   seeing this."*
3. **PERSISTENT.** C002 is under-registering at 45%. It's subtle per-reading, so it shows up
   as SUSPICIOUS plus a *feeder* finding — this is the case where per-consumer and
   network-level evidence combine.
4. **NETWORK_IMBALANCE — the money moment.** Every consumer meter reads perfectly normal, but
   the feeder is losing 14%. The system localizes to the **feeder** and says *unattributed
   loss*. *"A naive system would accuse whoever scored highest. Ours says the loss exists but
   no consumer explains it — inspect the segment first."*
5. **Close on framing.** Read a reason list aloud, then: *"Every output says 'investigation
   required', never 'theft proven'. Four things produce this signature — technical loss, a
   faulty meter, a data gap, unmetered load — and separating them is the point."*

Have `tests/test_pipeline.py` output on a second screen. Judges asking "is the ML real or
hard-coded?" get check 5 as the answer: the models flag the consumers the simulator perturbed,
and the ground-truth field is never fed to them.

---

## Design decisions worth defending

| Decision | Why |
|---|---|
| **No n8n** | Every step is an in-process function call on a message arriving every 0.7 s. An external workflow engine adds a service, a queue hop and JSON round-trips without simplifying anything. n8n earns its place when steps span systems — subscribe it to `gridtrace/alert` and it fits without touching a line. |
| **NumPy autoencoder, not TensorFlow** | ~150 parameters. Pulling a 500 MB runtime onto a demo laptop for that is the wrong trade. Still a real network: dense layers, Adam, mini-batches, early stopping. |
| **Built-in MQTT broker** | Real MQTT on the wire with ordinary paho clients, zero install. If you run Mosquitto it steps aside. |
| **SQLite in WAL mode** | Dashboard reads never block the inference writer. One file, no server. |
| **Instantaneous, not cumulative, feeder gap** | The cumulative figure is diluted by every hour of history — a meter running a week barely reacts to a new gap. |

## Project layout

```
gridtrace/
  config.py           all tunables in one place
  simulator_core.py   ESP32 telemetry model (same schema as the firmware)
  broker.py           minimal local MQTT 3.1.1 broker
  publisher.py        simulator/file → MQTT
  mqtt_util.py        paho 1.x/2.x shim
  features.py         consumer-relative streaming features + baselines
  autoencoder.py      NumPy autoencoder
  hybrid.py           calibration, severity, hybrid score, risk score
  localize.py         hierarchy roll-up + explanations
  storage.py          SQLite
  infer.py            real-time service (validate → score → localize → store → alert)
  infer_state.py      shared localization state for the dashboard
  dashboard.py        Flask + single page
  train.py            collect → preprocess → train both → calibrate → save
run_demo.py           one-command launcher
tests/test_pipeline.py  10 end-to-end checks
requirements.txt
```

---

*This is a measurement-analysis prototype on synthetic data. It detects patterns that warrant
investigation; it does not and cannot prove wrongdoing by anyone.*
