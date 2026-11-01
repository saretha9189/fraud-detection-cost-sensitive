from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "creditcard.csv"
OUT_DIR = ROOT / "outputs"
ARTIFACT_DIR = ROOT / "artifacts"
SEED = 42

# Time-based split: train on the earliest 60% of transactions, tune on the next 20%, test once on the final 20%.
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2

# Business assumptions (they are assumptions, not facts about any bank: change them in one place).
REVIEW_COST = 5.0        # cost of a analyst manually reviewing one flagged transaction
REVIEW_BUDGET = 0.005    # analysts can review the top 0.5% most suspicious transactions
