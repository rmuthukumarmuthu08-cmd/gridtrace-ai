"""
Telemetry publisher: simulator -> MQTT.
=======================================

Stands in for the ESP32. It emits exactly the records the Wokwi firmware emits,
unchanged, so swapping in the real board later means pointing a serial-to-MQTT
bridge at the same topics - no schema migration.

    python -m gridtrace.publisher                      # NORMAL, forever
    python -m gridtrace.publisher --mode 3 --frames 200
    python -m gridtrace.publisher --script demo        # scripted mode tour for the demo
    python -m gridtrace.publisher --from-file data.ndjson

If you have a real ESP32 on a serial port, replace this file's source loop with a
`pyserial` reader - everything downstream is unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from gridtrace import config
from gridtrace.mqtt_util import make_client
from gridtrace.simulator_core import MODES, Simulator

TOPIC_FOR = {
    "consumer": config.TOPIC_CONSUMER,
    "zone": config.TOPIC_ZONE,
    "feeder": config.TOPIC_FEEDER,
    "event": config.TOPIC_EVENT,
    "meta": config.TOPIC_EVENT,
}

# A scripted tour so the demo is repeatable: 60 normal frames, then each anomaly
# mode for 90, then back to normal. Roughly 6 minutes at 0.7 s/frame.
DEMO_SCRIPT = [(0, 60), (1, 90), (0, 30), (2, 90), (0, 30), (3, 90), (0, 40)]


def publish_records(client, records) -> int:
    n = 0
    for rec in records:
        topic = TOPIC_FOR.get(rec.get("record_type"), config.TOPIC_EVENT)
        client.publish(topic, json.dumps(rec), qos=0)
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="Publish simulated meter telemetry over MQTT.")
    ap.add_argument("--mode", type=int, default=0, choices=[0, 1, 2, 3],
                    help="0 NORMAL, 1 ABNORMAL, 2 PERSISTENT, 3 IMBALANCE")
    ap.add_argument("--frames", type=int, default=0, help="0 = run until interrupted")
    ap.add_argument("--interval", type=float, default=0.7, help="seconds between frames")
    ap.add_argument("--load", type=float, default=1.0, help="grid demand scaler 0.6-1.4")
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=0xA11CE)
    ap.add_argument("--script", choices=["demo"], help="run the scripted mode tour")
    ap.add_argument("--from-file", help="replay an existing NDJSON file instead of simulating")
    ap.add_argument("--host", default=config.MQTT_HOST)
    ap.add_argument("--port", type=int, default=config.MQTT_PORT)
    a = ap.parse_args(argv)

    client = make_client("gridtrace-publisher")
    client.connect(a.host, a.port, keepalive=30)
    client.loop_start()
    print(f"publisher -> mqtt://{a.host}:{a.port}")

    sent = 0
    try:
        if a.from_file:
            with open(a.from_file, "r", encoding="utf-8") as fh:
                batch = []
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    batch.append(rec)
                    if rec.get("record_type") == "feeder":     # end of a frame
                        sent += publish_records(client, batch)
                        batch = []
                        time.sleep(a.interval)
                if batch:
                    sent += publish_records(client, batch)
            print(f"replayed {sent} records from {a.from_file}")
            return 0

        sim = Simulator(seed=a.seed, load=a.load)
        plan = DEMO_SCRIPT if a.script == "demo" else [(a.mode, a.frames or 10 ** 9)]

        for mode, count in plan:
            print(f"  mode -> {MODES[mode]}  ({count} frames)")
            client.publish(config.TOPIC_EVENT, json.dumps({
                "record_type": "event", "schema": "els.v1", "event": "MODE_CHANGE",
                "mode": MODES[mode], "mode_index": mode}), qos=0)
            for _ in range(count):
                sent += publish_records(client, sim.step(mode))
                time.sleep(a.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        time.sleep(0.3)
        client.loop_stop()
        client.disconnect()
        print(f"published {sent} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
