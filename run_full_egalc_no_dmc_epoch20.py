"""
Last optional controlled training for the paper:
Full EG-GM-PUNet w/o DMC, fixed 20-epoch budget.

Purpose
-------
This is NOT a new model search. It only removes DMC from the exact strict
Full EG-GM-PUNet configuration already trained today.

Scientific constraints
----------------------
1. Read the exact strict Full epoch-20 checkpoint ONLY to recover its embedded config.
   Its WEIGHTS are never used as initialization.
2. Force initialization back to the common epoch-0 state:
   common_v2b_seed2026_a2_architecture_epoch000.pth
3. Keep MS-GMM, MS-DEH, EG-ALC, seed, data split, optimizer/LR schedule,
   feedback schedule and all other settings unchanged.
4. Change only DMC: enabled -> false, weight -> 0.
5. Train exactly 20 epochs.
6. Run smoke test first, then formal training, then optional test evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml


PROJECT = Path(r"E:\workspace\EG-GM-PUNet")
PYTHON = Path(r"E:\anaconda\envs\torch221\python.exe")

TRAINER = PROJECT / "train" / "train_eg_gm_punet_v3_a2_direct_diversity_match.py"
EVALUATOR = PROJECT / "evaluate" / "evaluate_eg_gm_punet_v3_a2.py"

STRICT_ROOT = PROJECT / "result" / "component_ablation_4_3_1_strict"
SOURCE_FULL_E20 = STRICT_ROOT / "full_egalc" / "formal" / "epoch_020_eg_gm_punet_v3_a2.pth"
COMMON_INIT = STRICT_ROOT / "common_init" / "common_v2b_seed2026_a2_architecture_epoch000.pth"

OUT_ROOT = STRICT_ROOT / "full_egalc_no_dmc"
FORMAL_DIR = OUT_ROOT / "formal"
SMOKE_DIR = OUT_ROOT / "smoke"

FORMAL_CFG = PROJECT / "configs" / "component_431_full_egalc_no_dmc_epoch20.yaml"
SMOKE_CFG = PROJECT / "configs" / "component_431_full_egalc_no_dmc_epoch20_smoke.yaml"

EXPECTED_COMMON_SHA256 = "429e7bf253cef7e94e0f43b3278f0746e65ffc24fa1e9daf206f3e091624a2d8"


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_stream(cmd, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 118)
    print("RUN :", " ".join(str(x) for x in cmd))
    print("LOG :", log_path)
    print("=" * 118)

    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(PROJECT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return p.wait()


def set_dmc_off(cfg: dict) -> None:
    tr = cfg.setdefault("training", {})

    # Current LIDC A2 schema.
    dmc = tr.get("direct_diversity_matching")
    if isinstance(dmc, dict):
        dmc["enabled"] = False
        dmc["weight"] = 0.0

    # Compatibility with variants that also expose a flat flag.
    if "direct_diversity_matching_enabled" in tr:
        tr["direct_diversity_matching_enabled"] = False

    # Do not touch feedback/evidence.
    tr["feedback_enabled"] = True

    model = cfg.setdefault("model", {})
    feedback = model.get("feedback")
    if isinstance(feedback, dict):
        feedback["enabled"] = True


def force_common_initialization(cfg: dict) -> None:
    cfg["resume"] = None

    init = cfg.get("initialization")
    if not isinstance(init, dict):
        raise RuntimeError("Embedded strict config has no initialization dict.")

    # Preserve the strict-suite initialization mode, but force the exact path.
    init["checkpoint"] = str(COMMON_INIT)

    # Explicitly reject leakage paths.
    bad_tokens = ("a1a", "a1final", "direct_diversity_match_k5", "full_egalc/formal")
    init_text = json.dumps(init, ensure_ascii=False).lower().replace("\\\\", "/")
    for token in bad_tokens:
        if token in init_text and str(COMMON_INIT).lower().replace("\\", "/") not in init_text:
            raise RuntimeError(f"Potential initialization leakage detected: {token}")


def make_configs() -> tuple[dict, dict]:
    if not SOURCE_FULL_E20.exists():
        raise FileNotFoundError(f"Missing source strict Full epoch20 checkpoint:\n{SOURCE_FULL_E20}")
    if not COMMON_INIT.exists():
        raise FileNotFoundError(f"Missing strict common initialization:\n{COMMON_INIT}")

    actual_sha = sha256_file(COMMON_INIT)
    print("[CHECK] Common-init SHA256:", actual_sha)
    if actual_sha != EXPECTED_COMMON_SHA256:
        raise RuntimeError(
            "Common-init SHA256 mismatch.\n"
            f"Expected: {EXPECTED_COMMON_SHA256}\n"
            f"Actual  : {actual_sha}\n"
            "Stop to avoid using a different initialization."
        )

    # Config extraction only. Source Full weights are NOT loaded into the new model.
    obj = torch.load(SOURCE_FULL_E20, map_location="cpu")
    embedded = obj.get("config")
    if not isinstance(embedded, dict):
        raise RuntimeError("The strict Full epoch20 checkpoint does not contain an embedded config.")

    formal = copy.deepcopy(embedded)
    formal["experiment_name"] = "component_431_full_egalc_no_dmc_epoch20"
    formal["save_dir"] = str(FORMAL_DIR)
    formal["seed"] = 2026

    formal.setdefault("training", {})["epochs"] = 20
    force_common_initialization(formal)
    set_dmc_off(formal)

    # Safety checks: only DMC is removed; evidence and feedback stay active.
    if not bool(formal.get("training", {}).get("feedback_enabled", False)):
        raise RuntimeError("EG-ALC feedback unexpectedly disabled.")
    if float(formal.get("training", {}).get("evidence_weight", 0.0)) <= 0:
        raise RuntimeError("MS-DEH evidence training unexpectedly disabled.")

    dmc = formal.get("training", {}).get("direct_diversity_matching", {})
    if isinstance(dmc, dict) and bool(dmc.get("enabled", True)):
        raise RuntimeError("DMC is still enabled after config modification.")

    smoke = copy.deepcopy(formal)
    smoke["experiment_name"] = "component_431_full_egalc_no_dmc_epoch20_smoke"
    smoke["save_dir"] = str(SMOKE_DIR)

    FORMAL_CFG.parent.mkdir(parents=True, exist_ok=True)
    with FORMAL_CFG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(formal, f, sort_keys=False, allow_unicode=True)
    with SMOKE_CFG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(smoke, f, sort_keys=False, allow_unicode=True)

    manifest = {
        "purpose": "Full EG-GM-PUNet w/o DMC, strict matched 20-epoch controlled run",
        "source_checkpoint_used_for_config_only": str(SOURCE_FULL_E20),
        "source_weights_used_for_initialization": False,
        "common_initialization": str(COMMON_INIT),
        "common_init_sha256": actual_sha,
        "epochs": 20,
        "seed": 2026,
        "MS_GMM": True,
        "MS_DEH": True,
        "EG_ALC": True,
        "DMC": False,
        "formal_config": str(FORMAL_CFG),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "preflight_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n[PASS] Preflight")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return formal, smoke


def find_epoch20_checkpoint() -> Path:
    patterns = [
        "epoch_020_eg_gm_punet_v3_a2.pth",
        "epoch_020*.pth",
        "*epoch*020*.pth",
    ]
    for pat in patterns:
        hits = sorted(FORMAL_DIR.glob(pat))
        if hits:
            return hits[-1]
    raise FileNotFoundError(f"Could not find epoch-20 checkpoint in {FORMAL_DIR}")


def parse_epoch20_validation() -> dict:
    hist = FORMAL_DIR / "history.csv"
    if not hist.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(hist)
    row = df.loc[df["epoch"] == 20]
    if row.empty:
        return {}
    r = row.iloc[-1]
    out = {}
    mapping = {
        "Val_GED2@4": "val_prior_ged2",
        "Val_CrossDice": "val_prior_cross_dice",
        "Val_BestDice": "val_prior_best_dice",
        "Val_Diversity": "val_prior_sample_diversity",
        "Val_ECE": "val_evidence_ece",
        "Val_ErrorAUROC": "val_evidence_error_detection_auroc",
        "Val_DisagreementAUROC": "val_evidence_rater_disagreement_auroc",
        "Val_DemandPearson": "val_demand_annotation_pearson",
        "Val_DemandSpearman": "val_demand_annotation_spearman",
    }
    for k, col in mapping.items():
        if col in r.index:
            try:
                out[k] = float(r[col])
            except Exception:
                pass
    return out


def newest_summary(root: Path) -> Path | None:
    hits = list(root.rglob("summary.json"))
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def parse_test_summary(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    p16 = obj["probabilistic_summary"]["16"]
    evi = obj.get("evidence_calibration", {})
    return {
        "Test_GED2@16": float(p16["ged2"]["mean"]),
        "Test_CrossDice@16": float(p16["cross_dice"]["mean"]),
        "Test_BestDice@16": float(p16["best_sample_dice_to_raters"]["mean"]),
        "Test_Diversity@16": float(p16["sample_diversity_distance"]["mean"]),
        "Test_ECE": evi.get("ece"),
        "Test_ErrorAUROC": evi.get("error_detection_auroc"),
        "Test_DisagreementAUROC": evi.get("rater_disagreement_auroc"),
    }


def evaluate_epoch20(checkpoint: Path) -> dict:
    cmd = [
        str(PYTHON), str(EVALUATOR),
        "--config", str(FORMAL_CFG),
        "--checkpoint", str(checkpoint),
        "--split", "test",
        "--num-samples", "16",
        "--sample-counts", "1", "4", "8", "16",
    ]
    code = run_stream(cmd, OUT_ROOT / "_logs" / "test_epoch20.log")
    if code != 0:
        raise SystemExit(f"Test evaluation failed with exit code {code}")

    summary = newest_summary(FORMAL_DIR / "evaluation")
    if summary is None:
        raise FileNotFoundError("Evaluation finished but no summary.json was found.")
    print("[TEST SUMMARY]", summary)
    return parse_test_summary(summary)


def write_result_row(val: dict, test: dict) -> None:
    row = {
        "Variant": "EG-GM-PUNet w/o DMC",
        "Epoch": 20,
        "DMC": "OFF",
        **val,
        **test,
    }
    out_csv = OUT_ROOT / "full_no_dmc_epoch20_result.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    (OUT_ROOT / "full_no_dmc_epoch20_result.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + "=" * 118)
    print("FINAL RESULT")
    print("=" * 118)
    for k, v in row.items():
        print(f"{k}: {v}")
    print("CSV :", out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    make_configs()

    if not args.skip_smoke:
        smoke_cmd = [
            str(PYTHON), str(TRAINER),
            "--config", str(SMOKE_CFG),
            "--smoke-test",
        ]
        code = run_stream(smoke_cmd, OUT_ROOT / "_logs" / "smoke.log")
        if code != 0:
            raise SystemExit(
                f"Smoke test failed with exit code {code}. Formal run was NOT started."
            )
        print("\n[PASS] Smoke test. Starting formal 20-epoch run.")

    formal_cmd = [
        str(PYTHON), str(TRAINER),
        "--config", str(FORMAL_CFG),
    ]
    code = run_stream(formal_cmd, OUT_ROOT / "_logs" / "formal.log")
    if code != 0:
        raise SystemExit(f"Formal training failed with exit code {code}")

    val = parse_epoch20_validation()
    test = {}

    if not args.skip_eval:
        ckpt = find_epoch20_checkpoint()
        print("\n[INFO] Evaluating epoch20 checkpoint:", ckpt)
        test = evaluate_epoch20(ckpt)

    write_result_row(val, test)


if __name__ == "__main__":
    main()
