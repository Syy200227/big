from __future__ import annotations
import csv, json, subprocess
from pathlib import Path
import torch, yaml

PROJECT = Path(r"E:\workspace\EG-GM-PUNet")
PYTHON = Path(r"E:\anaconda\envs\torch221\python.exe")
EVALUATOR = PROJECT / "evaluate" / "evaluate_eg_gm_punet_v3_a2.py"
BASE = PROJECT / "result" / "component_ablation_4_3_1_strict"
OUT = BASE / "epoch20_matched_evaluation"
CFG_DIR = OUT / "_configs"
LOG_DIR = OUT / "_logs"

VARIANTS = {
    "ms_gmm": BASE / "ms_gmm" / "formal" / "epoch_020_eg_gm_punet_v3_a2.pth",
    "ms_gmm_deh": BASE / "ms_gmm_deh" / "formal" / "epoch_020_eg_gm_punet_v3_a2.pth",
    "full_egalc": BASE / "full_egalc" / "formal" / "epoch_020_eg_gm_punet_v3_a2.pth",
}
LABELS = {
    "ms_gmm": "MS-GMM",
    "ms_gmm_deh": "MS-GMM + MS-DEH",
    "full_egalc": "EG-GM-PUNet",
}

def run_stream(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 116)
    print("RUN:", " ".join(str(x) for x in cmd))
    print("=" * 116)
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            [str(x) for x in cmd], cwd=str(PROJECT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line); log.flush()
        return p.wait()

def newest_summary(save_dir: Path) -> Path:
    hits = list(save_dir.rglob("summary.json"))
    if not hits:
        raise FileNotFoundError(f"No summary.json found below {save_dir}")
    return max(hits, key=lambda p: p.stat().st_mtime)

def parse_summary(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    p16 = obj["probabilistic_summary"]["16"]
    return {
        "GED2@16": float(p16["ged2"]["mean"]),
        "CrossDice@16": float(p16["cross_dice"]["mean"]),
        "BestDice@16": float(p16["best_sample_dice_to_raters"]["mean"]),
        "Diversity@16": float(p16["sample_diversity_distance"]["mean"]),
        "NumCases": int(obj["num_cases"]),
        "CheckpointEpoch": int(obj.get("checkpoint_epoch", 20)),
        "SummaryPath": str(path),
    }

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not EVALUATOR.exists():
        raise FileNotFoundError(EVALUATOR)

    rows = []
    for key, checkpoint in VARIANTS.items():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        obj = torch.load(checkpoint, map_location="cpu")
        cfg = obj.get("config")
        if not isinstance(cfg, dict):
            raise RuntimeError(f"No embedded config in {checkpoint}")
        eval_save_dir = OUT / key
        cfg = dict(cfg)
        cfg["save_dir"] = str(eval_save_dir)
        cfg["experiment_name"] = f"component_431_epoch20_eval_{key}"
        cfg["resume"] = None
        cfg_path = CFG_DIR / f"{key}_epoch20_eval.yaml"
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        cmd = [
            str(PYTHON), str(EVALUATOR),
            "--config", str(cfg_path),
            "--checkpoint", str(checkpoint),
            "--split", "test",
            "--num-samples", "16",
            "--sample-counts", "1", "4", "8", "16",
        ]
        code = run_stream(cmd, LOG_DIR / f"{key}.log")
        if code != 0:
            raise SystemExit(f"{key} failed with exit code {code}")
        summary = newest_summary(eval_save_dir)
        m = parse_summary(summary)
        rows.append({
            "Variant": LABELS[key],
            "MS-GMM": "✓",
            "MS-DEH": "✓" if key in ("ms_gmm_deh","full_egalc") else "",
            "EG-ALC": "✓" if key == "full_egalc" else "",
            **m
        })

    csv_path = OUT / "epoch20_ablation_table.csv"
    json_path = OUT / "epoch20_ablation_table.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 116)
    print("EPOCH-20 MATCHED ABLATION TABLE")
    print("=" * 116)
    for r in rows:
        print(f"{r['Variant']:<28} GED2={r['GED2@16']:.6f}  CrossDice={r['CrossDice@16']:.6f}  BestDice={r['BestDice@16']:.6f}  Diversity={r['Diversity@16']:.6f}")
    print("CSV :", csv_path)
    print("JSON:", json_path)

if __name__ == "__main__":
    main()
