#!/usr/bin/env python
"""Standalone reproducibility check.

    python verify_against_notebook.py [--model PATH] [--ignore-hash]

Loads the model, hashes it, compares with sample_data/model_hash.json, runs every bundled sample through
the app's exact pipeline and compares the result with the notebook's frozen predictions.
Exit code 0 = PASS, 1 = FAIL.  A hash mismatch counts as FAIL unless --ignore-hash is given
(the per-sample prediction check is reported either way).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="path to the .keras file (default: model/bt_ssscnet_best_val_loss.keras)")
    ap.add_argument("--ignore-hash", action="store_true", help="do not fail on a hash mismatch")
    args = ap.parse_args()
    sd = APP_DIR / "sample_data"

    try:
        import pandas as pd
        import preprocessing as P
        import model_utils as M
    except Exception as exc:
        print(f"FAIL: could not import dependencies ({type(exc).__name__}: {exc}). Run: pip install -r requirements.txt")
        return 1

    missing = [f for f in ("samples_index.json", "norm_stats.npz") if not (sd / f).is_file()]
    if missing:
        print(f"FAIL: sample_data is missing {missing} (unzip the notebook's BT_SSSCNet_Streamlit_samples.zip into {sd}).")
        return 1

    model_path = Path(args.model) if args.model else None
    if model_path is None:
        cands = M.find_model_candidates(APP_DIR)
        if not cands:
            print(f"FAIL: no .keras model found. Put {M.MODEL_FILENAME} in {APP_DIR / 'model'} or pass --model.")
            return 1
        model_path = cands[0]
    print(f"Model file : {model_path}")
    print(f"TensorFlow : {M.tf.__version__} | Keras {M.keras.__version__}")

    hs = M.hash_status(model_path, sd)
    print(f"SHA-256    : {hs['sha256']}")
    print(f"Recorded   : {hs['recorded']}  (sample_data/model_hash.json, file {hs['recorded_file']})")
    print(f"Brief hash : {hs['reported']}  -> {'matches' if hs['reported_match'] else 'does NOT match'} the loaded file")
    print(f"Hash status: {hs['status'].upper()}")

    try:
        model = M.load_bt_ssscnet(model_path)
    except M.ModelLoadError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"Loaded model with {model.count_params():,} parameters; inputs/outputs validated.")

    try:
        mean, std = P.load_norm(sd / "norm_stats.npz")
        cfg, _ = M.read_json(sd / "preprocessing_config.json")
        clip = float((cfg or {}).get("norm_clip", P.DEFAULT_CLIP))
        df = M.check_all_samples(model, sd, mean, std, clip)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", lambda v: f"{v:.6g}"):
        print("\n" + df.to_string(index=False))
    samples_ok = bool((df["result"] == "PASS").all()) and len(df) > 0
    hash_ok = hs["status"] == "match" or args.ignore_hash
    print(f"\nSamples: {int((df['result'] == 'PASS').sum())}/{len(df)} PASS (max|dp| < 1e-3 and >=99.9% identical argmax)")
    if hs["status"] != "match":
        print(f"WARNING: model hash is {hs['status']}: this file is not confirmed to be the one that produced the frozen predictions.")
    ok = samples_ok and hash_ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    print("This confirms the app reproduces the notebook's inference for these samples. It does not validate model accuracy.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())