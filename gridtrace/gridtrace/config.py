"""Central configuration for GridTrace AI. Everything the prototype needs, in one place."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models_store"
for _d in (DATA_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_NDJSON = DATA_DIR / "train_normal.ndjson"
DB_PATH = DATA_DIR / "gridtrace.db"
BUNDLE_PATH = MODEL_DIR / "gridtrace_bundle.joblib"

# ---------------------------------------------------------------- mqtt
MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
TOPIC_CONSUMER = "gridtrace/telemetry/consumer"
TOPIC_ZONE = "gridtrace/telemetry/zone"
TOPIC_FEEDER = "gridtrace/telemetry/feeder"
TOPIC_EVENT = "gridtrace/telemetry/event"
TOPIC_ALERT = "gridtrace/alert"
TOPIC_ALL = "gridtrace/telemetry/#"

# ---------------------------------------------------------------- network topology
FEEDER_ID = "F01"
ZONES = ("Z01", "Z02")
CONSUMER_IDS = ("C001", "C002", "C003", "C004", "C005")

# Baseline resistive loss the network always has. Anything above this is what the
# system has to explain; it is NOT an anomaly on its own.
TECHNICAL_LOSS_PCT = 3.10

# ---------------------------------------------------------------- feature engineering
ROLL_WINDOW = 15          # frames of per-consumer history (= 15 simulated minutes)
PERSISTENCE_WINDOW = 12   # frames used for the "is this sustained?" signal
MIN_HISTORY = 5           # frames needed before a consumer is scored at all

FEATURE_NAMES = [
    "voltage",
    "current",
    "power_kw",
    "power_factor",
    "energy_delta_kwh",
    "dev_from_own_baseline",   # consumer-specific z-score
    "roll_mean_ratio",         # rolling mean / consumer's training mean
    "roll_cv",                 # rolling coefficient of variation
    "hourly_profile_ratio",    # power / consumer's own mean for this hour-of-day
]
# NOTE: the feeder imbalance is deliberately NOT in this vector - see hybrid.py.
# It is shared by all five consumers, so including it makes every meter look
# anomalous during a purely network-level event and destroys localization.
# It enters the risk score as separate *supporting evidence* instead.

# ---------------------------------------------------------------- models
IFOREST_PARAMS = dict(n_estimators=200, max_samples=256, contamination=0.02, random_state=42)
AE_HIDDEN = (6, 3)        # encoder widths; decoder mirrors it. 9 -> 6 -> 3 -> 6 -> 9
AE_EPOCHS = 300
AE_BATCH = 64
AE_LR = 0.004
AE_VAL_FRACTION = 0.15
AE_PATIENCE = 30
RANDOM_SEED = 42

# ---------------------------------------------------------------- hybrid scoring
W_MEAN = 0.5              # agreement term vs sensitivity term (see hybrid.py)
SEVERITY_NINES = 3.0      # percentile -> severity: 99.9th pct of normal maps to 1.0
W_NETWORK = 12.0          # max risk points contributed by feeder-level evidence
W_PERSISTENCE = 12.0      # max risk points contributed by a sustained pattern

RISK_BANDS = [
    (0, 30, "NORMAL"),
    (31, 60, "SUSPICIOUS"),
    (61, 80, "HIGH RISK"),
    (81, 100, "CRITICAL"),
]

# ---------------------------------------------------------------- dashboard
DASH_HOST = "127.0.0.1"
DASH_PORT = 8000


def risk_band(score: float) -> str:
    """Map a 0-100 risk score onto the investigation bands."""
    s = max(0.0, min(100.0, float(score)))
    for lo, hi, name in RISK_BANDS:
        if s <= hi:
            return name
    return "CRITICAL"
