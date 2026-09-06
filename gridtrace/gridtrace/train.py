"""
Training pipeline: collect normal data -> fit both models -> calibrate -> save.

    python -m gridtrace.train                  # collect fresh normal data + train
    python -m gridtrace.train --collect-only   # just write data/train_normal.ndjson
    python -m gridtrace.train --no-collect     # retrain on whatever data is already there
    python -m gridtrace.train --append         # add more normal data, then retrain on all of it

Both models are trained on NORMAL data only. That is what makes this an anomaly
detector rather than a classifier: nothing at training time has ever seen an
anomaly, so the models cannot have memorised the specific ones the simulator
injects - they can only report "this does not look like what I learned".
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from gridtrace import config
from gridtrace.autoencoder import Autoencoder
from gridtrace.features import Baselines, frames_to_matrix
from gridtrace.hybrid import Calibrator
from gridtrace.simulator_core import Simulator


def collect_normal(ticks: int, seeds=(0xC0FFEE, 0xBEEF01, 0x5EED42), loads=(0.85, 1.0, 1.15)):
    """Run the simulator in NORMAL mode only, across a few seeds and demand levels."""
    records = []
    for seed in seeds:
        for load in loads:
            sim = Simulator(seed=seed, load=load)
            for _ in range(ticks):
                records.extend(sim.step(0))          # mode 0 = NORMAL
    return records


def load_ndjson(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                              # a corrupt line must not kill training
    return out


def write_ndjson(path, records, append=False):
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def train(records, verbose=True):
    if verbose:
        print(f"[1/6] {len(records):,} raw records; learning per-consumer baselines")
    consumer_rows = [r for r in records if r.get("record_type") == "consumer"]
    baselines = Baselines().fit(consumer_rows)
    if verbose:
        for cid in sorted(baselines.mean_kw):
            print(f"      {cid}  mean {baselines.mean_kw[cid]*1000:7.0f} W   "
                  f"sd {baselines.std_kw[cid]*1000:6.0f} W")

    if verbose:
        print("[2/6] building consumer-relative feature matrix")
    X, ctxs = frames_to_matrix(records, baselines)
    X = np.asarray(X, dtype=float)
    if len(X) < 200:
        raise SystemExit(f"only {len(X)} usable feature rows - collect more normal data")
    if verbose:
        print(f"      {X.shape[0]:,} samples x {X.shape[1]} features")

    if verbose:
        print("[3/6] scaling (StandardScaler fitted on normal data only)")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if verbose:
        print("[4/6] training Isolation Forest")
    iforest = IsolationForest(**config.IFOREST_PARAMS).fit(Xs)
    s_if_normal = -iforest.score_samples(Xs)

    if verbose:
        print("[5/6] training autoencoder "
              f"({config.FEATURE_NAMES.__len__()}-{'-'.join(map(str, config.AE_HIDDEN))}"
              f"-{'-'.join(map(str, reversed(config.AE_HIDDEN[:-1])))}-{len(config.FEATURE_NAMES)})")
    ae = Autoencoder(Xs.shape[1], config.AE_HIDDEN, config.AE_LR, config.RANDOM_SEED)
    ae.fit(Xs, epochs=config.AE_EPOCHS, batch=config.AE_BATCH,
           val_fraction=config.AE_VAL_FRACTION, patience=config.AE_PATIENCE, verbose=verbose)
    err_normal = ae.reconstruction_error(Xs)

    if verbose:
        print("[6/6] calibrating both score distributions to percentiles")
    cal_if = Calibrator(s_if_normal)
    cal_ae = Calibrator(err_normal)

    import sklearn
    bundle = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "feature_names": config.FEATURE_NAMES,
        "n_train": int(X.shape[0]),
        "baselines": baselines.to_dict(),
        "scaler": scaler,
        "iforest": iforest,
        "autoencoder": ae.to_dict(),
        "cal_if": cal_if.to_dict(),
        "cal_ae": cal_ae.to_dict(),
        "ae_val_mse": float(getattr(ae, "best_val", 0.0)),
        "normal_if_p99": float(np.percentile(s_if_normal, 99)),
        "normal_ae_p99": float(np.percentile(err_normal, 99)),
    }
    return bundle


def save(bundle, path=None):
    path = path or config.BUNDLE_PATH
    joblib.dump(bundle, path)
    return path


def load(path=None):
    """Load a trained bundle and return ready-to-use objects."""
    from gridtrace.hybrid import HybridScorer
    import sklearn
    path = path or config.BUNDLE_PATH
    b = joblib.load(path)

    trained_with = b.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        print(
            f"\n  !! This model bundle was trained with scikit-learn {trained_with}, "
            f"but you are running {sklearn.__version__}.\n"
            f"     Unpickling estimators across versions can give WRONG results silently.\n"
            f"     Fix it in 10 seconds:   python -m gridtrace.train\n",
            file=sys.stderr)

    ae = Autoencoder.from_dict(b["autoencoder"])
    scorer = HybridScorer(b["iforest"], b["scaler"], ae,
                          Calibrator.from_dict(b["cal_if"]),
                          Calibrator.from_dict(b["cal_ae"]))
    baselines = Baselines.from_dict(b["baselines"])
    return scorer, baselines, b


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train GridTrace AI on normal telemetry.")
    ap.add_argument("--ticks", type=int, default=400,
                    help="normal frames per (seed, load) combination. Default 400.")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--no-collect", action="store_true", help="retrain on existing data file")
    ap.add_argument("--append", action="store_true", help="add new normal data to the file")
    a = ap.parse_args(argv)

    if a.no_collect:
        records = load_ndjson(config.TRAIN_NDJSON)
        print(f"loaded {len(records):,} records from {config.TRAIN_NDJSON}")
    else:
        print(f"collecting normal data ({a.ticks} frames x 3 seeds x 3 demand levels)...")
        fresh = collect_normal(a.ticks)
        write_ndjson(config.TRAIN_NDJSON, fresh, append=a.append)
        records = load_ndjson(config.TRAIN_NDJSON) if a.append else fresh
        print(f"wrote {config.TRAIN_NDJSON}  ({len(records):,} records)")

    if a.collect_only:
        return 0

    bundle = train(records)
    p = save(bundle)
    print(f"\nsaved model bundle -> {p}")
    print(f"  autoencoder val MSE : {bundle['ae_val_mse']:.6f}")
    print(f"  normal IF  p99      : {bundle['normal_if_p99']:.4f}")
    print(f"  normal AE  p99      : {bundle['normal_ae_p99']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
