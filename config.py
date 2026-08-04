"""
Single source of truth for every tunable in the pipeline.

Nothing downstream should hardcode a business constant. If a number appears in
the report, it traces to this file or to the data.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"          # raw csv + intermediate artifacts
OUT  = ROOT / "outputs"       # tables and report fragments
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)

RAW_TXN = DATA / "train_transaction.csv"
RAW_ID  = DATA / "train_identity.csv"

# ----------------------------------------------------------------- seeds
SEED           = 42     # model fitting
SEED_LABEL_SIM = 7      # chargeback-lag simulation in 06

# ------------------------------------------------------- temporal split
# Day index is (TransactionDT - min) // 86400. Data spans days 0..181.
TRAIN_END = 120   # train  = day <  120
CALIB_END = 150   # calib  = 120 <= day < 150 ;  test = day >= 150

# ------------------------------------------------- model hyperparameters
MODEL = dict(
    max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=100,
    l2_regularization=1.0, max_features=0.6, categorical_features="from_dtype",
    early_stopping=True, validation_fraction=0.12, n_iter_no_change=25,
    random_state=SEED,
)
MAX_CAT_LEVELS = 120   # HistGBM caps native categoricals at 255; tail -> 'other'

# ====================================================================
# BUSINESS CONSTANTS
# Change these and rerun 05 + 06 to regenerate every table in the report.
# ====================================================================

# --- unit economics of the merchant -----------------------------------
COGS    = 0.70   # cost of goods sold as a fraction of price
MARGIN  = 0.30   # contribution margin (= 1 - COGS)
DISCOUNT = 0.85  # one-year discount / survival haircut on forward spend

# --- cost of a false negative:  L = Amount*COGS + CB_FEE + OPS_DISPUTE --
CB_FEE      = 25.0   # network + acquirer chargeback fee
OPS_DISPUTE = 12.0   # internal dispute handling per chargeback

# --- cost of a false positive:  M = Amount*MARGIN + relationship_cost --
# relationship_cost is P(churn|declined) * residual LTV, both per tenure bucket.
# residual LTV is MEASURED in 04_ltv.py. P(churn|declined) is the one input
# with no data support in this dataset -- graded by tenure from published
# false-decline research. Treat it as the softest assumption in the model.
P_CHURN_ON_DECLINE = {
    "new(0d)":   0.42,
    "1-7d":      0.38,
    "8-30d":     0.30,
    "31-90d":    0.22,
    "91-180d":   0.16,
    "181-400d":  0.12,
    "400d+":     0.09,
}
TENURE_EDGES  = [-1, 0, 7, 30, 90, 180, 400, 1e9]
TENURE_LABELS = ["new(0d)", "1-7d", "8-30d", "31-90d", "91-180d", "181-400d", "400d+"]

# --- intervention parameters ------------------------------------------
F_PASS    = 0.11   # fraud pass-through rate of the step-up challenge (vendor number)
A_ABANDON = 0.07   # good-customer abandonment rate at challenge
Q_ANALYST = 0.91   # analyst agreement with the eventual outcome
C_REVIEW  = 6.77   # analyst cost per case ($58/hr loaded, 7 min handling)
D_DELAY   = 1.20   # delay cost per reviewed case

# --- operational constraints ------------------------------------------
DAILY_REVIEW_SLOTS = 60       # ~1 analyst FTE at 64 cases/day
COST_PER_DECISION  = 0.0008   # infrastructure, per scored event

# --- label latency (used only by the 06 simulation) --------------------
CHARGEBACK_MEDIAN_DAYS = 34.0
CHARGEBACK_LOG_SIGMA   = 0.85

# --- sensitivity grid --------------------------------------------------
SENS_F  = [0.05, 0.11, 0.20, 0.35, 0.50]
SENS_CD = [2.0, 4.0, 7.97, 15.0]
