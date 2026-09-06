# GridTrace AI — Demo Runbook

Team Squad Mavericks · Nirmal (Muthukumar R) · Sustainable Energy & Resource Innovation

Everything below runs on one laptop, offline. Nothing here is scripted output — every
number the judges see is produced live by the trained models on the data the simulator
generates in front of them.

---

## 0 · Before you walk up (5 minutes, do this once)

```bat
cd D:\WATTWATCH\gridtrace
python -m pip install -r requirements.txt
python -m gridtrace.train
python tests\test_pipeline.py
```

You want to see `ALL CHECKS PASSED` and 10 PASS lines. If check 1 warns about a
scikit-learn version mismatch, run `python -m gridtrace.train` once more — that retrains
the bundle against the sklearn you actually have installed and the warning disappears.

Then leave **three terminals open** in `D:\WATTWATCH\gridtrace`, and a browser tab on
`http://127.0.0.1:5000`. Do not start anything yet.

---

## 1 · The 6-minute demo

### Minute 0–1 — Frame the problem before showing anything

> "16.16% of the electricity entering India's distribution network never gets accounted
> for. Part of that is physics. Part of it is broken meters. Part of it is genuinely
> unbilled consumption. In the billing data all three look identical — and that is the
> actual problem. Not detecting the gap. Explaining it."

Slide 2 is on screen. Don't touch the laptop yet.

### Minute 1–2 — Start the system, live

Terminal 1:

```bat
python -m gridtrace.broker
```

Terminal 2:

```bat
python -m gridtrace.publisher
```

Terminal 3:

```bat
python -m gridtrace.infer
```

Then open the dashboard:

```bat
python -m gridtrace.dashboard
```

> "Broker, simulated field meters, inference service, dashboard. MQTT between them —
> the same protocol real AMI head-ends speak. No cloud, no internet."

### Minute 2–3 — NORMAL: prove it stays quiet

Leave it in NORMAL mode for ~45 seconds. Point at the risk column.

> "Five meters, all in the NORMAL band. This matters more than the alerts. A theft
> detector that fires on normal data is worse than useless — the utility learns to
> ignore it. Ours leaves 92.2% of normal readings completely alone, and that number is
> measured, not claimed."

Point at the feeder gap reading, hovering around 3%.

> "That 3% is technical loss. Resistive heating. It is *supposed* to be there. The
> system knows that and does not raise anything."

### Minute 3–4 — ABNORMAL_CONSUMPTION: one meter, named and explained

Press **2** in the publisher terminal (or the mode button).

Wait ~30 seconds. One consumer climbs into SUSPICIOUS then HIGH RISK.

> "C003's risk score is climbing. And here is the part that matters —"

Click that consumer. Read its reasons out loud, exactly as they appear:

> "Elevated autoencoder reconstruction error, rarer than 99.7% of normal readings.
> Isolation Forest isolates this reading from the normal population. Both models agree
> independently. Consumption is 4.2 sigma from *this consumer's own* baseline.
>
> An inspector can check every one of those before knocking on a door. That is the
> difference between an alarm and evidence."

### Minute 4–5 — NETWORK_IMBALANCE: the slide that wins it

Press **4**.

The feeder gap jumps to ~13%. Consumer scores stay low. The banner turns to a
FEEDER-level finding.

> "Big loss on the feeder. Every meter reads normal. Almost every system built for this
> would now blame the highest-scoring consumer — and that consumer is innocent.
>
> Ours says: unattributed loss at feeder F01, inspect the segment, check for unmetered
> connections or a metering gap, before suspecting any consumer. It knows when the
> honest answer is *nobody*."

Let that sit for a beat. This is the strongest moment in the demo.

### Minute 5–6 — Close

Press **1** to return to NORMAL and watch the scores decay back down.

> "Detection, localization, and an explanation an inspector can act on — running end to
> end on a laptop, on a five-meter feeder, with a model that generalises to fifty
> thousand because every feature is relative to each consumer's own history rather than
> to an absolute threshold.
>
> And it never says the word 'theft'. It says high-risk anomaly, investigation required.
> That is deliberate."

---

## 2 · Questions the judges will ask

**"How do you know it isn't just memorising the anomalies you inject?"**
Both models are trained on NORMAL mode data only, across three random seeds and three
demand levels. Nothing at training time has ever seen an anomaly. They cannot recognise
an attack pattern — they can only report that a reading does not resemble what they
learned. The test file checks this against `anomaly_injected`, a ground-truth field the
models are never given.

**"Why two models instead of one good one?"**
They fail in different directions. Isolation Forest is strong on readings that sit in a
sparse region of feature space; the autoencoder is strong on readings whose *internal
relationships* are wrong — power that doesn't match voltage × current × power factor,
for instance. Their raw scores aren't comparable, so both are calibrated to percentiles
against the normal distribution first, then combined as `0.5 × mean + 0.5 × max`. The
mean term means the two agreeing raises confidence; the max term means either one alone
can still raise a flag. Test 8 proves the hybrid always lands between the two, so neither
model can dominate by scale.

**"Isn't the feeder imbalance the obvious feature? Why isn't it in your model?"**
We tried it. It destroys localization. Every consumer on a feeder shares the same gap
value, so when the gap rises, all five scores rise together and the system can no longer
tell you *which* meter to look at — the thing it exists to do. So the gap sits outside
the model as a separate evidence term, capped at 12 points of the 100. It can escalate a
finding. It can never create one on its own. That was a deliberate departure from the
original design and it is documented in `hybrid.py`.

**"What stops it from falsely accusing someone?"**
Three things, structurally. The evidence terms are capped, so network imbalance alone
cannot push a consumer into HIGH RISK. The unattributed rule fires when the feeder gap
exceeds technical loss but no consumer scores above 61, and reports at feeder level
instead. And features are consumer-relative — a large commercial load is compared to its
own history, not to a residential neighbour.

**"Will this work on a real utility feeder?"**
The transport is already what real AMI uses. Swap `publisher.py` for a serial reader or
a head-end feed and nothing downstream changes. The honest limits: it needs roughly two
weeks of clean per-consumer history to learn baselines, real meters are noisier than the
simulator, and the technical-loss baseline is a per-feeder constant that a utility would
calibrate from its own records rather than take from us.

**"What does the system do about theft?"**
Nothing. It produces a ranked investigation queue with reasons attached. A human decides.
The word "theft" does not appear in any output, because we cannot distinguish theft from
a failed meter from an unrecorded connection — and neither can any of the deployed
systems that claim to.

---

## 3 · If something breaks on stage

| Symptom | Fix |
|---|---|
| `Address already in use` on the broker | A previous run is still up. Close that terminal, or change `MQTT_PORT` in `config.py`. |
| Dashboard empty | Inference service isn't running, or was started before the broker. Start broker → publisher → infer, in that order. |
| sklearn version warning | `python -m gridtrace.train` — 10 seconds, retrains against your installed version. |
| Scores all sitting at 0 | The feature engine needs ~20 frames of history per consumer before it scores. Wait 30 seconds. |
| Everything is on fire | `python tests\test_pipeline.py` runs the whole pipeline in one process with no networking and prints the same evidence. It is your fallback demo. |

---

## 4 · One-line summary, if you only get one sentence

> GridTrace AI finds energy loss on a distribution feeder, tells you whether it belongs
> to a consumer or to the network itself, and gives the inspector a reason they can check
> before they knock on anybody's door.
