"""python run_all.py          # uses data/creditcard.csv
   python run_all.py --demo   # synthetic stand-in, writes to outputs_demo/ so real results are never overwritten"""
import argparse

from src import config
from src.data import load, synthetic
from src.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        run(synthetic(), label="synthetic demo data", out_dir=config.ROOT / "outputs_demo", art_dir=config.ROOT / "artifacts_demo")
    else:
        run(load())
