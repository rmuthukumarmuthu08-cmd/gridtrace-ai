#!/usr/bin/env python3
"""Validates the NDJSON telemetry produced by the firmware.

Checks:
  1. every line is valid JSON
  2. required fields present on every consumer record
  3. P == V * I * PF  (within rounding tolerance)
  4. energy is monotonically non-decreasing per consumer
  5. feeder balance: input_energy - sum_meter_energy == unaccounted_energy
  6. NORMAL mode unaccounted% stays near the technical-loss baseline
  7. NETWORK_IMBALANCE mode drives unaccounted% clearly above baseline
  8. anomaly modes actually flag SURGE/DIP statuses
  9. values stay in physically sane ranges (no NaN/inf, V in 200..250)
"""
import json, math, sys
from collections import defaultdict

path = sys.argv[1]
rows, bad = [], []
with open(path) as f:
    for n, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            bad.append((n, str(e), line[:120]))

fails = []
def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)

print(f"\nParsed {len(rows)} records from {path}")
print("-" * 68)

check(not bad, f"1. all {len(rows)} lines are valid JSON (bad lines: {len(bad)})")
for b in bad[:5]:
    print("        ", b)

cons = [r for r in rows if r.get("record_type") == "consumer"]
zones = [r for r in rows if r.get("record_type") == "zone"]
feed = [r for r in rows if r.get("record_type") == "feeder"]

REQ = ["consumer_id", "zone_id", "feeder_id", "voltage", "current", "power",
       "energy", "power_factor", "timestamp", "status", "mode"]
missing = {k for r in cons for k in REQ if k not in r}
check(not missing, f"2. every consumer record has all required fields (missing: {missing or 'none'})")

worst = 0.0
for r in cons:
    p = r["voltage"] * r["current"] * r["power_factor"]
    if r["power"] > 1.0:
        worst = max(worst, abs(p - r["power"]) / r["power"])
check(worst < 0.005, f"3. P = V*I*PF holds on all {len(cons)} records (worst rel. error {worst*100:.4f} %)")

last = {}
mono_ok = True
for r in cons:
    cid = r["consumer_id"]
    if cid in last and r["energy"] < last[cid] - 1e-9:
        mono_ok = False
    last[cid] = r["energy"]
check(mono_ok, "4. per-consumer energy is monotonically non-decreasing")

bal = max(abs(r["input_energy"] - r["sum_meter_energy"] - r["unaccounted_energy"]) for r in feed)
check(bal < 1e-3, f"4b/5. feeder balance closes: in - metered - unaccounted = 0 (max residual {bal:.2e} kWh)")

# The INSTANTANEOUS gap is what these checks use. The cumulative
# `unaccounted_pct` is diluted by run history, so its value depends on how long
# each mode ran - useless as a pass/fail signal and a poor model feature.
KEY = "unaccounted_power_pct"
missing_key = [r for r in feed if KEY not in r]
check(not missing_key, f"6a. every feeder record carries the instantaneous `{KEY}`")

by_mode = defaultdict(list)
for r in feed:
    by_mode[r["mode"]].append(r.get(KEY, float("nan")))

def avg(m):
    v = by_mode.get(m) or [float("nan")]
    return sum(v) / len(v)

norm_avg = avg("NORMAL")
imb_avg = avg("NETWORK_IMBALANCE")
pers_avg = avg("PERSISTENT_ANOMALY")
abn_avg = avg("ABNORMAL_CONSUMPTION")

check(2.0 < norm_avg < 4.5,
      f"6b. NORMAL instantaneous gap ~ technical loss ({norm_avg:.2f} % of feeder input)")
check(imb_avg > norm_avg + 8.0,
      f"7a. NETWORK_IMBALANCE lifts the gap to {imb_avg:.2f} % (+{imb_avg-norm_avg:.1f} pts)")
check(pers_avg > norm_avg + 3.0,
      f"7b. PERSISTENT_ANOMALY under-registration widens it to {pers_avg:.2f} % (+{pers_avg-norm_avg:.1f} pts)")
check(abn_avg > norm_avg,
      f"7c. ABNORMAL_CONSUMPTION also shows a raised gap ({abn_avg:.2f} %)")

# run-length independence: the metric must not drift as history accumulates
nv = by_mode.get("NORMAL", [])
if len(nv) >= 20:
    first, last = nv[:len(nv)//4], nv[-len(nv)//4:]
    drift = abs(sum(first)/len(first) - sum(last)/len(last))
    check(drift < 0.5,
          f"7d. gap metric is history-free: NORMAL drifts only {drift:.3f} pts first-quarter vs last")

flag = defaultdict(set)
for r in cons:
    if r["status"] != "OK":
        flag[r["mode"]].add(r["consumer_id"])
check(len(flag["ABNORMAL_CONSUMPTION"]) >= 2,
      f"8. ABNORMAL mode raises SURGE/DIP on {sorted(flag['ABNORMAL_CONSUMPTION'])}")
check(not flag["NORMAL"] or len(flag["NORMAL"]) <= 5,
      f"8b. NORMAL mode stays mostly quiet (flagged: {sorted(flag['NORMAL']) or 'none'})")

sane = True
vmin, vmax = 999, 0
for r in cons:
    for k in ("voltage", "current", "power", "energy", "power_factor"):
        v = r[k]
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v) or v < 0:
            sane = False
    vmin, vmax = min(vmin, r["voltage"]), max(vmax, r["voltage"])
check(sane and 200 <= vmin and vmax <= 250,
      f"9. no NaN/inf/negatives; voltage range {vmin:.1f}..{vmax:.1f} V")

nrec = len(cons) + len(zones) + len(feed)
check(len(zones) == 2 * len(feed) and len(cons) == 5 * len(feed),
      f"10. frame integrity: {len(feed)} frames x (5 consumer + 2 zone + 1 feeder) = {nrec}")

print("-" * 68)
print(("ALL CHECKS PASSED" if not fails else f"{len(fails)} CHECK(S) FAILED"))
sys.exit(1 if fails else 0)
