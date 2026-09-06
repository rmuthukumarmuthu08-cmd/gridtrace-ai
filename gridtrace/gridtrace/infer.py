"""
Real-time inference service: MQTT -> validate -> features -> hybrid AI -> risk ->
localize -> SQLite -> alert.

    python -m gridtrace.infer

This is the orchestration layer. n8n was considered and deliberately not used:
every step here is an in-process Python function call on a message that arrives
every ~0.7 s, so routing each one through an external workflow engine would add a
service to install, a queue hop, and JSON round-trips without simplifying
anything. n8n earns its place when steps span systems (email, ticketing, Slack) -
if you later want an inspection ticket raised, subscribe n8n to `gridtrace/alert`
and it fits without touching this file.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading

from gridtrace import config
from gridtrace import infer_state
from gridtrace.features import FeatureEngine
from gridtrace.localize import explain, localize
from gridtrace.mqtt_util import make_client
from gridtrace.storage import Store
from gridtrace.train import load as load_bundle

log = logging.getLogger("gridtrace.infer")

REQUIRED = ("consumer_id", "voltage", "current", "power", "power_factor")


def validate(rec) -> bool:
    """Reject anything that would poison the pipeline. Never raises."""
    if not isinstance(rec, dict):
        return False
    rt = rec.get("record_type")
    if rt not in ("consumer", "zone", "feeder", "event", "meta"):
        return False
    if rt == "consumer":
        if any(rec.get(k) is None for k in REQUIRED):
            return False
        try:
            v, c, p = float(rec["voltage"]), float(rec["current"]), float(rec["power"])
        except (TypeError, ValueError):
            return False
        # physically implausible readings are data-quality problems, not anomalies
        if not (50.0 <= v <= 500.0) or c < 0 or p < 0:
            return False
    return True


class Pipeline:
    """Holds model + feature state and processes one frame at a time."""

    def __init__(self, store: Store, client=None):
        self.scorer, self.baselines, self.bundle = load_bundle()
        self.engine = FeatureEngine(self.baselines)
        self.store = store
        self.client = client
        self.frame: list[dict] = []
        self.stats = {"received": 0, "rejected": 0, "scored": 0, "alerts": 0}
        self.last_localization: dict | None = None
        self.current_mode = "?"
        self._lock = threading.Lock()

    def handle(self, rec: dict) -> None:
        with self._lock:
            self.stats["received"] += 1
            if not validate(rec):
                self.stats["rejected"] += 1
                log.debug("rejected malformed record")
                return

            rt = rec["record_type"]
            if rt == "event":
                self.current_mode = rec.get("mode", self.current_mode)
                return
            if rt == "zone" or rt == "meta":
                return
            if rt == "consumer":
                self.frame.append(rec)
                return

            # feeder record closes the frame
            self._close_frame(rec)

    def _close_frame(self, feeder_rec: dict) -> None:
        scored, per_consumer = [], {}
        for rec in self.frame:
            out = self.engine.transform(rec)
            if not out:
                continue
            x, ctx = out
            try:
                res = self.scorer.score_one(x, ctx)
            except Exception:                       # a scoring failure must not kill the stream
                log.exception("scoring failed for %s", ctx.get("consumer_id"))
                continue
            reasons = explain(res, ctx)
            self.store.save_reading(ctx, res, reasons)
            self.stats["scored"] += 1
            scored.append({
                "consumer_id": ctx["consumer_id"], "zone_id": ctx["zone_id"],
                "feeder_id": ctx["feeder_id"], "risk_score": res["risk_score"],
                "risk_band": res["risk_band"],
            })
            per_consumer[ctx["consumer_id"]] = (res, ctx, reasons)

        loc = localize(scored, self.engine.feeder_gap_pct)
        tick = int(feeder_rec.get("tick", 0))
        ts = feeder_rec.get("timestamp", "")

        reasons: list[str] = []
        if loc["consumer_id"] and loc["consumer_id"] in per_consumer:
            reasons = per_consumer[loc["consumer_id"]][2]
        elif loc.get("unattributed"):
            reasons = [
                f"Feeder gap {loc['feeder_gap_pct']:.1f}% vs "
                f"{config.TECHNICAL_LOSS_PCT:.1f}% technical-loss baseline",
                "No individual consumer meter explains the difference",
                "Consistent with unmetered load, a metering gap, or distribution loss",
            ]

        if loc["risk_score"] >= 61:
            self.store.save_alert(loc, tick, ts, reasons)
            self.stats["alerts"] += 1
            if self.client:
                self.client.publish(config.TOPIC_ALERT, json.dumps(
                    {**loc, "tick": tick, "timestamp": ts, "reasons": reasons,
                     "note": "high-risk anomaly - investigation required; not proof of theft"}
                ), qos=0)
            log.warning("ALERT %-9s %-5s risk %.0f  %s",
                        loc["risk_band"], loc.get("consumer_id") or loc["level"],
                        loc["risk_score"], loc["summary"][:90])

        self.last_localization = {**loc, "tick": tick, "timestamp": ts, "reasons": reasons}
        infer_state.write_localization(self.last_localization)

        # feeder gap for the NEXT frame's features (causal: never uses its own future)
        self.engine.update_feeder(feeder_rec)
        self.frame = []


def main(argv=None):
    ap = argparse.ArgumentParser(description="GridTrace AI real-time inference service.")
    ap.add_argument("--host", default=config.MQTT_HOST)
    ap.add_argument("--port", type=int, default=config.MQTT_PORT)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s  %(levelname)-7s %(message)s")

    store = Store()
    client = make_client("gridtrace-infer")
    pipe = Pipeline(store, client)
    print(f"loaded models trained {pipe.bundle['created']} on {pipe.bundle['n_train']:,} samples")

    def on_message(_c, _u, msg):
        try:
            pipe.handle(json.loads(msg.payload.decode("utf-8", errors="replace")))
        except json.JSONDecodeError:
            pipe.stats["rejected"] += 1                 # not JSON at all - drop it, keep going
        except Exception:
            log.exception("handler error")

    client.on_message = on_message
    client.connect(a.host, a.port, keepalive=30)
    client.subscribe(config.TOPIC_ALL, qos=0)
    print(f"subscribed to {config.TOPIC_ALL} on mqtt://{a.host}:{a.port}")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    client.loop_start()
    try:
        while not stop.is_set():
            stop.wait(5)
            s = pipe.stats
            print(f"  received {s['received']:>6}  scored {s['scored']:>6}  "
                  f"alerts {s['alerts']:>4}  rejected {s['rejected']:>4}")
    finally:
        client.loop_stop()
        client.disconnect()
        store.close()
        print("inference stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
