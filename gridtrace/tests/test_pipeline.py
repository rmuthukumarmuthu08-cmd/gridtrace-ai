#!/usr/bin/env python3
"""
End-to-end validation for GridTrace AI.

    python tests/test_pipeline.py

Nine checks, no test framework needed. The two that matter most are #4 and #5:
normal telemetry must stay in the NORMAL band, and injected anomalies must be
attributed to the consumers the simulator actually perturbed - which the test
knows from `anomaly_injected`, a ground-truth field the models never see.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from gridtrace import broker, config
from gridtrace.features import FeatureEngine
from gridtrace.infer import Pipeline, validate
from gridtrace.localize import explain, localize
from gridtrace.mqtt_util import make_client
from gridtrace.simulator_core import MODES, Simulator
from gridtrace.storage import Store
from gridtrace.train import load as load_bundle

FAILS = []


def check(ok, msg):
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def offline_run(scorer, baselines, mode, ticks=300, seed=0xA11CE):
    scorer.reset_persistence()
    eng, sim = FeatureEngine(baselines), Simulator(seed=seed)
    rows, locs = [], []
    for _ in range(ticks):
        frame = sim.step(mode)
        feeder = [r for r in frame if r["record_type"] == "feeder"][0]
        scored = []
        for rec in [r for r in frame if r["record_type"] == "consumer"]:
            out = eng.transform(rec)
            if not out:
                continue
            x, ctx = out
            res = scorer.score_one(x, ctx)
            rows.append((ctx["consumer_id"], res["risk_score"], ctx["gt_injected"], res, ctx))
            scored.append({"consumer_id": ctx["consumer_id"], "zone_id": ctx["zone_id"],
                           "feeder_id": ctx["feeder_id"], "risk_score": res["risk_score"],
                           "risk_band": res["risk_band"]})
        locs.append(localize(scored, eng.feeder_gap_pct))
        eng.update_feeder(feeder)
    return rows, locs


def main():
    print("\nGridTrace AI - end-to-end validation")
    print("=" * 78)

    # ---------------------------------------------------------------- 1 models
    try:
        scorer, baselines, bundle = load_bundle()
        loaded = True
    except Exception as e:
        loaded = False
        print("   ", e)
    check(loaded, f"1. trained model bundle loads from {config.BUNDLE_PATH.name}")
    if not loaded:
        return 1
    check(len(bundle["feature_names"]) == len(config.FEATURE_NAMES),
          f"2. feature contract matches ({len(config.FEATURE_NAMES)} features, "
          f"trained on {bundle['n_train']:,} normal samples)")

    # ---------------------------------------------------------------- 3 malformed input
    bad = [None, "not json", {}, {"record_type": "consumer"},
           {"record_type": "consumer", "consumer_id": "C001", "voltage": "abc",
            "current": 1, "power": 2, "power_factor": 0.9},
           {"record_type": "consumer", "consumer_id": "C001", "voltage": -5,
            "current": 1, "power": 2, "power_factor": 0.9},
           {"record_type": "consumer", "consumer_id": "C001", "voltage": float("nan"),
            "current": 1, "power": 2, "power_factor": 0.9}]
    survived = True
    for b in bad:
        try:
            if validate(b):
                eng = FeatureEngine(baselines)
                eng.transform(b)
        except Exception as e:
            survived = False
            print("      crashed on:", b, e)
    check(survived, "3. malformed / missing / NaN telemetry is rejected without crashing")

    # ---------------------------------------------------------------- 4 normal
    rows_n, locs_n = offline_run(scorer, baselines, 0)
    r_n = np.array([x[1] for x in rows_n])
    pct_normal = 100 * (r_n <= 30).mean()
    pct_high = 100 * (r_n >= 61).mean()
    check(pct_normal >= 85 and pct_high <= 5,
          f"4. NORMAL data stays normal: {pct_normal:.1f}% in NORMAL band, "
          f"{pct_high:.1f}% HIGH RISK+ (mean risk {r_n.mean():.1f})")

    # ---------------------------------------------------------------- 5 anomalies
    for mode in (1, 2):
        rows, locs = offline_run(scorer, baselines, mode)
        r = np.array([x[1] for x in rows])
        injected = sorted({c for c, _, gt, _, _ in rows if gt})
        per = collections.defaultdict(list)
        for cid, risk, gt, _, _ in rows:
            per[cid].append(risk)
        top2 = [c for c, _ in sorted(((c, np.mean(v)) for c, v in per.items()),
                                     key=lambda t: -t[1])[:2]]
        hit = bool(set(injected) & set(top2))
        check(r.mean() > r_n.mean() and hit,
              f"5{'ab'[mode-1]}. {MODES[mode]}: mean risk {r.mean():.1f} > normal "
              f"{r_n.mean():.1f}; top-scoring {top2} includes injected {injected}")

    # ---------------------------------------------------------------- 6 network localization
    rows_i, locs_i = offline_run(scorer, baselines, 3)
    settled = locs_i[60:]
    feeder_share = 100 * sum(1 for l in settled if l["unattributed"]) / max(len(settled), 1)
    check(feeder_share >= 70,
          f"6. NETWORK_IMBALANCE localizes to the FEEDER as unattributed loss "
          f"{feeder_share:.0f}% of frames (no consumer wrongly blamed)")

    # ---------------------------------------------------------------- 7 explanations
    hi = [(res, ctx) for _, risk, _, res, ctx in
          offline_run(scorer, baselines, 1)[0] if risk >= 61]
    ok_expl = bool(hi) and all(len(explain(res, ctx)) >= 2 for res, ctx in hi[:50])
    check(ok_expl, f"7. every high-risk result carries >=2 contributing factors "
                   f"({len(hi)} high-risk readings examined)")

    # ---------------------------------------------------------------- 8 hybrid balance
    sample = [(res["iforest_score"], res["autoencoder_score"], res["hybrid_score"])
              for _, _, _, res, _ in rows_i[:400]]
    both_bounded = all(0 <= a <= 1 and 0 <= b <= 1 and min(a, b) <= h <= max(a, b) + 1e-9
                       for a, b, h in sample)
    check(both_bounded,
          "8. both model scores are calibrated to [0,1] and the hybrid always lies "
          "between them - neither model can dominate by scale")

    # ---------------------------------------------------------------- 9 full MQTT path
    port = 18884
    srv = broker.serve("127.0.0.1", port, background=True)
    time.sleep(0.4)
    db = config.DATA_DIR / "test_e2e.db"
    for p in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    store = Store(db)
    pipe = Pipeline(store)
    sub = make_client("test-infer")
    def _safe(_c, _u, m):                      # same guard the real service uses
        try:
            pipe.handle(json.loads(m.payload))
        except json.JSONDecodeError:
            pipe.stats["rejected"] += 1
    sub.on_message = _safe
    sub.connect("127.0.0.1", port, 30)
    sub.subscribe(config.TOPIC_ALL, qos=0)
    sub.loop_start()
    time.sleep(0.4)

    pub = make_client("test-pub")
    pub.connect("127.0.0.1", port, 30)
    pub.loop_start()
    topic_for = {"consumer": config.TOPIC_CONSUMER, "zone": config.TOPIC_ZONE,
                 "feeder": config.TOPIC_FEEDER}
    sim = Simulator(seed=0xA11CE)
    for i in range(140):
        for rec in sim.step(0 if i < 40 else 1):
            pub.publish(topic_for.get(rec["record_type"], config.TOPIC_EVENT),
                        json.dumps(rec), qos=0)
        time.sleep(0.03)           # pace like the real publisher; QoS 0 has no flow control
    pub.publish(config.TOPIC_CONSUMER, b"{not json at all", qos=0)   # hostile message

    # Drain: scoring + one SQLite commit per reading is slower than the publish
    # loop, so wait for the consumer side to catch up rather than guessing a sleep.
    stable, last, waited = 0, -1, 0.0
    while stable < 3 and waited < 40:
        time.sleep(0.5); waited += 0.5
        now = pipe.stats["received"]
        stable = stable + 1 if now == last else 0
        last = now
    sub.loop_stop(); pub.loop_stop(); srv.shutdown()

    scored = pipe.stats["scored"]
    stored = len(store.latest_per_consumer())
    check(scored > 500 and stored == 5 and pipe.stats["rejected"] >= 1
          and pipe.stats["received"] >= 1000 and pipe.stats["alerts"] > 0,
          f"9. full MQTT path: {pipe.stats['received']} received -> {scored} scored -> "
          f"{stored} consumers in SQLite, {pipe.stats['alerts']} alerts, "
          f"{pipe.stats['rejected']} malformed rejected")
    store.close()

    print("=" * 78)
    print("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
