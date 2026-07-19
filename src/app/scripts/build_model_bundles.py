"""
Build inference bundles from existing plain RandomForest .joblib files.

Run once after training (or if you only have ALL_REGIONS_*_RandomForest.joblib):
    cd D:\\INCOIS\\Agro_project\\app
    python scripts/build_model_bundles.py

Creates:
    trained_models/ALL_REGIONS_temp_qc_bundle.joblib
    trained_models/ALL_REGIONS_psal_qc_bundle.joblib
"""
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from backend.ml.predictor import build_bundle_for_target, models_status
from backend.config import TARGET_COLS


def main():
    for target in TARGET_COLS:
        print(f"Building bundle for {target}…")
        try:
            bundle = build_bundle_for_target(target)
            print(f"  OK — {len(bundle['feature_cols'])} features, "
                  f"{len(bundle['target_encoder'].classes_)} classes")
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")

    print("\nModel status:")
    for row in models_status():
        print(f"  {row['target']}: loaded={row['bundle_loaded']}, "
              f"plain_exists={row['plain_exists']}")


if __name__ == "__main__":
    main()
