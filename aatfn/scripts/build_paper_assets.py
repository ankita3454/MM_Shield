"""
Collects the final figures/tables referenced in the writeup into one clean
aatfn/paper_assets/ folder, renamed to what you'd actually cite in the
paper. Pure file copying + one static table (the run 1-5 ablation
comparison, transcribed from pipeline.md) -- no model inference, so this
doesn't need torch and is safe to re-run any time after evaluate_fusion.py.

Usage:
    cd aatfn/scripts
    python3 evaluate_fusion.py       # regenerate results/ first (adds the
                                      # combined roc_combined.pdf/pr_combined.pdf
                                      # this script expects)
    python3 build_paper_assets.py
"""
import csv
import shutil

from feature_extractors.paths import AATFN_DIR

RESULTS_DIR = AATFN_DIR / "results"
ASSETS_DIR = AATFN_DIR / "paper_assets"

# (source filename in results/, destination filename in paper_assets/)
FILE_MAP = [
    ("confusion_typo.pdf", "fig_confusion_typo.pdf"),
    ("confusion_patch.pdf", "fig_confusion_patch.pdf"),
    ("confusion_stego.pdf", "fig_confusion_stego.pdf"),
    ("roc_combined.pdf", "fig_roc.pdf"),
    ("pr_combined.pdf", "fig_pr.pdf"),
    ("calibration.pdf", "fig_calibration.pdf"),
    ("main_results.csv", "table_main_results.csv"),
    ("final_metrics_per_head.csv", "table_per_head.csv"),
]

# Transcribed from pipeline.md section 5c -- static, doesn't need a model run.
ABLATION_ROWS = [
    {"run": 1, "config": "weak stego attack (136-bit localized payload), wd=1e-4",
     "typo_f1": 0.68, "patch_f1": 0.88, "stego_f1": 0.66, "exact_match": 0.27,
     "note": "stego F1 inflated -- base-rate guessing, no real signal"},
    {"run": 2, "config": "fixed stego attack (15% scattered LSB), wd=1e-4",
     "typo_f1": 0.61, "patch_f1": 0.86, "stego_f1": 0.64, "exact_match": 0.28, "note": ""},
    {"run": 3, "config": "fixed stego, + regularization (patch width 128, added dropout), wd=1e-3",
     "typo_f1": 0.59, "patch_f1": 0.82, "stego_f1": 0.56, "exact_match": 0.26,
     "note": "over-regularized, underfit"},
    {"run": 4, "config": "fixed stego, + regularization, wd=3e-4",
     "typo_f1": 0.65, "patch_f1": 0.875, "stego_f1": 0.54, "exact_match": 0.29,
     "note": "best optimization-only config"},
    {"run": 5, "config": "wd=3e-4, SAA branch widened (3-layer) + dropout 0.05 (FROZEN, final model)",
     "typo_f1": 0.69, "patch_f1": 0.868, "stego_f1": 0.62, "exact_match": 0.31,
     "note": "single-variable SAA-branch ablation"},
]


def main():
    if not RESULTS_DIR.exists():
        raise SystemExit(f"{RESULTS_DIR} not found -- run evaluate_fusion.py first")

    ASSETS_DIR.mkdir(exist_ok=True)

    missing = []
    for src_name, dst_name in FILE_MAP:
        src = RESULTS_DIR / src_name
        if not src.exists():
            missing.append(src_name)
            continue
        shutil.copy2(src, ASSETS_DIR / dst_name)
        print(f"copied {src_name} -> paper_assets/{dst_name}")

    if missing:
        print(f"\nWARNING: missing from results/, not copied: {missing}")
        if "roc_combined.pdf" in missing or "pr_combined.pdf" in missing:
            print("  -> re-run evaluate_fusion.py (it was updated to also produce "
                  "roc_combined.pdf / pr_combined.pdf) before running this script again.")

    ablation_path = ASSETS_DIR / "table_ablation.csv"
    with open(ablation_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ABLATION_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ABLATION_ROWS)
    print(f"wrote table_ablation.csv -> {ablation_path}")

    print(f"\npaper_assets/ ready -> {ASSETS_DIR}")


if __name__ == "__main__":
    main()
