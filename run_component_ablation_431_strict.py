from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import torch
import yaml

BEIJING = ZoneInfo("Asia/Shanghai")
VARIANTS = ("ms_gmm", "ms_gmm_deh", "full_egalc")


def now() -> str:
    return datetime.now(BEIJING).isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return obj


def dump_yaml(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )


def resolve(project: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project / p


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_existing(project: Path, candidates: list[str], label: str) -> Path:
    hits = [resolve(project, x) for x in candidates if resolve(project, x).exists()]
    if not hits:
        raise FileNotFoundError(f"No {label} found. Checked: {candidates}")
    return hits[0]


def import_module_from_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def extract_data_root(cp_obj: Any) -> str | None:
    if not isinstance(cp_obj, dict):
        return None
    for key in ("config", "cfg"):
        cfg = cp_obj.get(key)
        if isinstance(cfg, dict) and cfg.get("data_root"):
            return str(cfg["data_root"])
    if cp_obj.get("data_root"):
        return str(cp_obj["data_root"])
    return None


def validate_base_config(base_cfg: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {"fatal": [], "warnings": []}
    data_root = str(base_cfg.get("data_root", ""))
    suffix = str(suite["expected_data_root_suffix"]).lower()
    if not data_root.replace("\\", "/").lower().endswith(suffix):
        report["fatal"].append(f"A2 template data_root mismatch: {data_root}")

    tr = base_cfg.get("training", {})
    dmc = tr.get("direct_diversity_matching")
    if not isinstance(dmc, dict):
        report["fatal"].append("A2 template has no training.direct_diversity_matching dict")
    else:
        if bool(suite["guardrails"]["require_dmc_enabled"]) and not bool(dmc.get("enabled")):
            report["fatal"].append("DMC must be ON in all three main-ablation rows")
        expected = float(suite["guardrails"]["expected_dmc_weight"])
        if abs(float(dmc.get("weight", -1.0)) - expected) > 1e-12:
            report["fatal"].append(
                f"DMC weight mismatch: expected {expected}, got {dmc.get('weight')}"
            )
    return report


def allowed_transfer_key(key: str) -> bool:
    # Only representation modules explicitly stated in the Methodology are transferred.
    allowed = (
        "image_encoder.",
        "prior.encoder.",
        "posterior.encoder.",
        "decoder.",
        "segmentation_head.",
    )
    k = key
    if k.startswith("probabilistic_model."):
        k = k[len("probabilistic_model."):]
    return any(k.startswith(prefix) for prefix in allowed)


def generic_safe_v2b_transfer(model: torch.nn.Module, cp_obj: dict[str, Any]) -> dict[str, Any]:
    src = cp_obj.get("model_state_dict", cp_obj)
    if not isinstance(src, dict):
        raise RuntimeError("V2-B checkpoint has no usable model_state_dict")

    dst = model.state_dict()
    loaded: list[str] = []
    skipped_shape: list[str] = []
    skipped_nontransfer: list[str] = []

    for sk, value in src.items():
        if not torch.is_tensor(value):
            continue
        if not allowed_transfer_key(sk):
            skipped_nontransfer.append(sk)
            continue

        candidates = [sk]
        if sk.startswith("probabilistic_model."):
            candidates.append(sk[len("probabilistic_model."):])
        else:
            candidates.append("probabilistic_model." + sk)

        matched = False
        for dk in candidates:
            if dk in dst:
                if tuple(dst[dk].shape) == tuple(value.shape):
                    dst[dk] = value.detach().clone()
                    loaded.append(dk)
                    matched = True
                    break
                skipped_shape.append(f"{sk} -> {dk}")
        if not matched:
            pass

    if len(loaded) < 50:
        raise RuntimeError(
            f"Only {len(loaded)} compatible representation tensors were transferred; "
            "refusing to create a possibly invalid common initialization."
        )

    model.load_state_dict(dst, strict=True)
    return {
        "mode": "generic_safe_representation_transfer",
        "loaded_key_count": len(loaded),
        "loaded_keys_preview": loaded[:30],
        "skipped_shape_count": len(skipped_shape),
        "skipped_nontransfer_count": len(skipped_nontransfer),
    }


def prepare_common_init(
    trainer_path: Path,
    base_cfg_path: Path,
    v2b_path: Path,
    common_path: Path,
    suite: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(int(suite["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(suite["seed"]))

    module = import_module_from_file(trainer_path, "_ablation_a2_trainer")
    cfg = load_yaml(base_cfg_path)

    if not hasattr(module, "build_model"):
        raise RuntimeError(
            "A2 trainer does not expose build_model(config). "
            "Please send the exact A2 trainer so the adapter can be updated safely."
        )
    model = module.build_model(cfg)

    cp_obj = torch.load(v2b_path, map_location="cpu")
    transfer_report = None

    # Prefer the project's own compatibility loader if available.
    for obj in (model, getattr(model, "probabilistic_model", None)):
        if obj is None:
            continue
        fn = getattr(obj, "load_compatible_v2b_checkpoint", None)
        if callable(fn):
            loaded = False
            last_exc = None
            for args, kwargs in (
                ((v2b_path,), {"map_location": "cpu"}),
                ((v2b_path,), {}),
                ((str(v2b_path),), {"map_location": "cpu"}),
                ((str(v2b_path),), {}),
            ):
                try:
                    transfer_report = fn(*args, **kwargs)
                    loaded = True
                    break
                except TypeError as exc:
                    last_exc = exc
            if loaded:
                if not isinstance(transfer_report, dict):
                    transfer_report = {"mode": "project_compatible_loader", "raw": str(transfer_report)}
                else:
                    transfer_report = {"mode": "project_compatible_loader", **transfer_report}
                break
            transfer_report = None

    if transfer_report is None:
        transfer_report = generic_safe_v2b_transfer(model, cp_obj)

    common_path.parent.mkdir(parents=True, exist_ok=True)
    common_obj = {
        "epoch": 0,
        "global_step": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "config": cfg,
        "best_records": {},
        "metrics": {"common_init_only": True},
        "ablation_common_initialization": {
            "source_v2b_checkpoint": str(v2b_path),
            "source_v2b_sha256": sha256(v2b_path),
            "seed": int(suite["seed"]),
            "transfer_report": transfer_report,
            "created_beijing": now(),
            "prohibited_sources": "No A1a/A1final/A2/full checkpoint used",
        },
    }
    torch.save(common_obj, common_path)

    return {
        "common_init": str(common_path),
        "common_init_sha256": sha256(common_path),
        "source_v2b": str(v2b_path),
        "source_v2b_sha256": sha256(v2b_path),
        "source_v2b_epoch": cp_obj.get("epoch") if isinstance(cp_obj, dict) else None,
        "source_v2b_data_root": extract_data_root(cp_obj),
        "transfer_report": transfer_report,
    }


def make_variant_cfg(
    base_cfg: dict[str, Any],
    suite: dict[str, Any],
    variant: str,
    mode: str,
    common_init: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    output_root = Path(suite["output_root"])
    cfg["save_dir"] = str(output_root / variant / mode)
    cfg["experiment_name"] = f"component431_strict_{variant}_{mode}"
    cfg["seed"] = int(suite["seed"])
    cfg["validation_seed"] = int(suite["validation_seed"])
    if "evaluation_seed" in cfg:
        cfg["evaluation_seed"] = int(suite["evaluation_seed"])
    cfg["resume"] = None

    # CRITICAL: every formal row loads exactly the same seed-fixed epoch-0 state.
    cfg["initialization"] = {
        "mode": "a1a_best_ged",
        "checkpoint": str(common_init),
        "expected_source_epoch": 0,
    }

    tr = cfg["training"]
    feedback = cfg["model"]["feedback"]

    # DMC remains exactly as in the A2 template.
    dmc = tr["direct_diversity_matching"]
    if not bool(dmc.get("enabled")):
        raise RuntimeError("DMC unexpectedly disabled in base A2 config")

    # Keep every non-tested setting identical. Only DEH/ALC switches change.
    if variant == "ms_gmm":
        tr["evidence_weight"] = 0.0
        if "evidence_learning_rate_values" in tr:
            tr["evidence_learning_rate_values"] = [0.0 for _ in tr["evidence_learning_rate_values"]]
        tr["feedback_enabled"] = False
        feedback["enabled"] = False
    elif variant == "ms_gmm_deh":
        tr["evidence_weight"] = float(base_cfg["training"]["evidence_weight"])
        tr["evidence_learning_rate_values"] = list(base_cfg["training"]["evidence_learning_rate_values"])
        tr["feedback_enabled"] = False
        feedback["enabled"] = False
    elif variant == "full_egalc":
        tr["evidence_weight"] = float(base_cfg["training"]["evidence_weight"])
        tr["evidence_learning_rate_values"] = list(base_cfg["training"]["evidence_learning_rate_values"])
        tr["feedback_enabled"] = True
        feedback["enabled"] = True
    else:
        raise ValueError(variant)

    if mode == "formal":
        tr["epochs"] = int(suite["formal"]["epochs"])
        feedback["start_epoch"] = int(suite["formal"]["feedback_start_epoch"])
        feedback["ramp_epochs"] = int(suite["formal"]["feedback_ramp_epochs"])
    elif mode == "quick":
        tr["epochs"] = int(suite["quick"]["epochs"])
        if variant == "full_egalc":
            feedback["start_epoch"] = int(suite["quick"]["force_egalc_start_epoch"])
            feedback["ramp_epochs"] = int(suite["quick"]["force_egalc_ramp_epochs"])
    elif mode == "smoke":
        # --smoke-test controls epoch/batch count. Force feedback path active only
        # in this non-paper sanity check.
        if variant == "full_egalc":
            feedback["start_epoch"] = int(suite["smoke"]["force_egalc_start_epoch"])
            feedback["ramp_epochs"] = int(suite["smoke"]["force_egalc_ramp_epochs"])
    else:
        raise ValueError(mode)

    return cfg


def run_stream(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 120)
    print("RUN:", " ".join(cmd))
    print("LOG:", log_path)
    print("=" * 120)
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
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
        return int(p.wait())


def parse_quick(save_dir: Path, suite: dict[str, Any], variant: str) -> dict[str, Any]:
    summary_path = save_dir / "training_summary.json"
    reasons: list[str] = []
    rec: dict[str, Any] = {}
    if summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("final_epoch", "last_epoch", "final_record"):
            if isinstance(obj.get(key), dict):
                rec = obj[key]
                break
    if not rec:
        # fallback history.csv
        hist = save_dir / "history.csv"
        if hist.exists():
            with hist.open("r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                rec = rows[-1]
    if not rec:
        reasons.append("no final record found")

    def num(name: str, default=float("nan")) -> float:
        try:
            return float(rec.get(name, default))
        except Exception:
            return default

    ged = num("val_prior_ged2")
    cross = num("val_prior_cross_dice")
    if not math.isfinite(ged):
        reasons.append("val_prior_ged2 non-finite/missing")
    elif ged > float(suite["quick"]["max_val_prior_ged2"]):
        reasons.append(f"val_prior_ged2 too high: {ged}")
    if not math.isfinite(cross):
        reasons.append("val_prior_cross_dice non-finite/missing")
    elif cross < float(suite["quick"]["min_val_prior_cross_dice"]):
        reasons.append(f"val_prior_cross_dice too low: {cross}")

    if variant == "full_egalc":
        betas = [num("train_coarse_beta", 1.0), num("train_middle_beta", 1.0), num("train_fine_beta", 1.0)]
        if all(math.isfinite(x) for x in betas) and max(abs(x - 1.0) for x in betas) < 1e-4:
            reasons.append("EG-ALC beta did not move during quick screen")

    return {
        "status": "PASS" if not reasons else "FAIL",
        "variant": variant,
        "reasons": reasons,
        "final_record": rec,
        "checked_beijing": now(),
    }


def resolve_best_ged_checkpoint(save_dir: Path) -> Path:
    records = save_dir / "best_records.json"
    if records.exists():
        obj = json.loads(records.read_text(encoding="utf-8"))
        rec = obj.get("val_prior_ged4")
        if isinstance(rec, dict) and rec.get("checkpoint"):
            p = save_dir / rec["checkpoint"]
            if p.exists():
                return p
    canonical = save_dir / "best_val_ged4_eg_gm_punet_v3_a2.pth"
    if canonical.exists():
        return canonical
    hits = list(save_dir.glob("*best*ged4*.pth"))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"Cannot resolve best GED4 checkpoint under {save_dir}")


def run_evaluator(
    evaluator: Path,
    cfg_path: Path,
    checkpoint: Path,
    suite: dict[str, Any],
    project: Path,
    python_exe: Path,
    log_path: Path,
) -> int:
    e = suite["evaluation"]
    cmd = [
        str(python_exe), str(evaluator),
        "--config", str(cfg_path),
        "--checkpoint", str(checkpoint),
        "--split", str(e["split"]),
        "--num-samples", str(int(e["num_samples"])),
        "--sample-counts", *[str(int(v)) for v in e["sample_counts"]],
    ]
    return run_stream(cmd, project, log_path)


def eval_summary_for(save_dir: Path, checkpoint: Path, num_samples: int) -> Path:
    direct = save_dir / "evaluation" / "test" / f"samples_{num_samples}" / checkpoint.stem / "summary.json"
    if direct.exists():
        return direct
    hits = list((save_dir / "evaluation").rglob("summary.json"))
    if not hits:
        raise FileNotFoundError(f"No evaluation summary under {save_dir}")
    return max(hits, key=lambda p: p.stat().st_mtime)


def metric_mean(block: dict[str, Any], key: str) -> float:
    item = block[key]
    if isinstance(item, dict):
        return float(item["mean"])
    return float(item)


def parse_eval_summary(path: Path, count: int = 16) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ps = obj["probabilistic_summary"][str(count)]
    return {
        "num_cases": int(obj["num_cases"]),
        "checkpoint_epoch": int(obj.get("checkpoint_epoch", -1)),
        "ged2": metric_mean(ps, "ged2"),
        "cross_dice": metric_mean(ps, "cross_dice"),
        "best_dice": metric_mean(ps, "best_sample_dice_to_raters"),
        "diversity": metric_mean(ps, "sample_diversity_distance"),
        "summary_path": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/component_ablation_431_strict.yaml")
    args = ap.parse_args()

    suite_path = Path(args.config)
    if not suite_path.is_absolute():
        project_guess = Path(__file__).resolve().parent
        suite_path = project_guess / suite_path
    suite = load_yaml(suite_path)

    project = Path(suite["project_root"])
    python_exe = Path(suite["python_exe"])
    output_root = Path(suite["output_root"])
    generated = Path(suite["generated_config_dir"])
    common_init = Path(suite["common_init_checkpoint"])
    logs = output_root / "_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)

    trainer = first_existing(project, suite["a2_trainer_candidates"], "A2 trainer")
    base_cfg_path = first_existing(project, suite["a2_config_candidates"], "A2 config")
    evaluator = first_existing(project, suite["evaluator_candidates"], "A2 evaluator")
    v2b = first_existing(project, suite["v2b_checkpoint_candidates"], "V2-B probability checkpoint")
    base_cfg = load_yaml(base_cfg_path)

    base_check = validate_base_config(base_cfg, suite)
    preflight = {
        "status": "PASS" if not base_check["fatal"] else "FAIL",
        "checked_beijing": now(),
        "trainer": str(trainer),
        "base_a2_config": str(base_cfg_path),
        "evaluator": str(evaluator),
        "v2b_checkpoint": str(v2b),
        "v2b_sha256": sha256(v2b),
        "base_checks": base_check,
        "note": "A1a/A1final/A2/full checkpoint is prohibited as formal initialization.",
    }
    write_json(output_root / "preflight_report.json", preflight)
    print(json.dumps(preflight, indent=2, ensure_ascii=False))
    if preflight["status"] != "PASS":
        raise SystemExit(2)

    # ------------------------------------------------------------------
    # Prepare ONE seed-fixed final-architecture initialization from V2-B.
    # ------------------------------------------------------------------
    if bool(suite["switches"]["prepare_common_init"]):
        common_report = prepare_common_init(
            trainer, base_cfg_path, v2b, common_init, suite
        )
        write_json(output_root / "common_init_report.json", common_report)
    elif not common_init.exists():
        raise FileNotFoundError(common_init)
    else:
        common_report = {
            "common_init": str(common_init),
            "common_init_sha256": sha256(common_init),
            "source_v2b": str(v2b),
            "source_v2b_sha256": sha256(v2b),
        }

    # Guard against accidentally using a rich-model checkpoint.
    low = str(common_report.get("source_v2b", "")).lower()
    for token in suite["guardrails"]["prohibit_init_path_tokens"]:
        if str(token).lower() in low:
            raise RuntimeError(
                f"Prohibited rich-model token {token!r} found in source init path: {low}"
            )

    # Generate all configs BEFORE running anything, then verify hashes/path.
    config_paths: dict[tuple[str, str], Path] = {}
    common_sha = sha256(common_init)
    for variant in VARIANTS:
        for mode in ("smoke", "quick", "formal"):
            cfg = make_variant_cfg(base_cfg, suite, variant, mode, common_init)
            p = generated / f"{variant}_{mode}.yaml"
            dump_yaml(p, cfg)
            config_paths[(variant, mode)] = p
            if Path(cfg["initialization"]["checkpoint"]) != common_init:
                raise RuntimeError("Generated variants do not share the same common init path")
    if bool(suite["guardrails"]["require_same_common_init_sha256"]):
        if sha256(common_init) != common_sha:
            raise RuntimeError("Common initialization changed during config generation")

    active: list[str] = []
    if bool(suite["switches"]["run_ms_gmm"]):
        active.append("ms_gmm")
    if bool(suite["switches"]["run_ms_gmm_deh"]):
        active.append("ms_gmm_deh")
    if bool(suite["switches"]["run_full_egalc"]):
        active.append("full_egalc")

    print("=" * 120)
    print("STRICT 4.3.1 MAIN COMPONENT ABLATION")
    print("Common V2-B source:", v2b)
    print("Common seed-fixed init:", common_init)
    print("Common init SHA256:", common_sha)
    print("Variants:", active)
    print("DMC: ON and identical")
    print("[START Beijing]", now())
    print("=" * 120)

    state: dict[str, str] = {}
    quick_pass: dict[str, bool] = {}

    # SMOKE
    if bool(suite["switches"]["run_smoke"]):
        for variant in active:
            code = run_stream(
                [str(python_exe), str(trainer), "--config", str(config_paths[(variant, "smoke")]), "--smoke-test"],
                project,
                logs / f"01_smoke_{variant}.log",
            )
            state[f"smoke:{variant}"] = "PASS" if code == 0 else f"FAIL({code})"
            if code != 0 and not bool(suite["switches"]["continue_after_variant_failure"]):
                raise SystemExit(code)

    # QUICK
    if bool(suite["switches"]["run_quick"]):
        for variant in active:
            if state.get(f"smoke:{variant}", "PASS") != "PASS":
                quick_pass[variant] = False
                continue
            qcfg = config_paths[(variant, "quick")]
            code = run_stream(
                [str(python_exe), str(trainer), "--config", str(qcfg)],
                project,
                logs / f"02_quick_{variant}.log",
            )
            report = parse_quick(Path(load_yaml(qcfg)["save_dir"]), suite, variant)
            if code != 0:
                report["status"] = "FAIL"
                report["reasons"].append(f"process_exit_code:{code}")
            write_json(output_root / variant / "quick_screen_report.json", report)
            quick_pass[variant] = report["status"] == "PASS"
            state[f"quick:{variant}"] = report["status"]

    # FORMAL -- every row starts again from common epoch-0 init.
    checkpoints: dict[str, Path] = {}
    if bool(suite["switches"]["run_formal"]):
        for variant in active:
            if bool(suite["switches"]["run_quick"]) and not quick_pass.get(variant, False):
                state[f"formal:{variant}"] = "SKIPPED_QUICK_FAIL"
                continue
            fcfg = config_paths[(variant, "formal")]
            code = run_stream(
                [str(python_exe), str(trainer), "--config", str(fcfg)],
                project,
                logs / f"03_formal_{variant}.log",
            )
            if code != 0:
                state[f"formal:{variant}"] = f"FAIL({code})"
                if not bool(suite["switches"]["continue_after_variant_failure"]):
                    raise SystemExit(code)
                continue
            save_dir = Path(load_yaml(fcfg)["save_dir"])
            checkpoints[variant] = resolve_best_ged_checkpoint(save_dir)
            state[f"formal:{variant}"] = "PASS"

    # TEST
    results: dict[str, dict[str, Any]] = {}
    if bool(suite["switches"]["run_test"]):
        for variant, cp in checkpoints.items():
            fcfg = config_paths[(variant, "formal")]
            code = run_evaluator(
                evaluator, fcfg, cp, suite, project, python_exe,
                logs / f"04_test_{variant}.log",
            )
            if code != 0:
                state[f"test:{variant}"] = f"FAIL({code})"
                continue
            save_dir = Path(load_yaml(fcfg)["save_dir"])
            summary = eval_summary_for(save_dir, cp, int(suite["evaluation"]["num_samples"]))
            results[variant] = parse_eval_summary(summary, 16)
            state[f"test:{variant}"] = "PASS"

    # AGGREGATE PAPER TABLE
    labels = {
        "ms_gmm": "MS-GMM",
        "ms_gmm_deh": "MS-GMM + MS-DEH",
        "full_egalc": "EG-GM-PUNet",
    }
    rows = []
    for variant in VARIANTS:
        if variant not in active:
            continue
        r = results.get(variant)
        rows.append({
            "Variant": labels[variant],
            "MS-GMM": "✓",
            "MS-DEH": "✓" if variant in ("ms_gmm_deh", "full_egalc") else "",
            "EG-ALC": "✓" if variant == "full_egalc" else "",
            "DMC(common)": "✓",
            "GED2@16": "" if r is None else r["ged2"],
            "CrossDice@16": "" if r is None else r["cross_dice"],
            "BestDice@16": "" if r is None else r["best_dice"],
            "Diversity@16": "" if r is None else r["diversity"],
            "CheckpointEpoch": "" if r is None else r["checkpoint_epoch"],
            "NumCases": "" if r is None else r["num_cases"],
            "CommonInitSHA256": common_sha,
            "Status": "OK" if r is not None else state.get(f"formal:{variant}", "MISSING"),
        })

    write_json(output_root / "ablation_table_431_strict.json", rows)
    if rows:
        with (output_root / "ablation_table_431_strict.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_json(output_root / "suite_summary.json", {
        "finished_beijing": now(),
        "state": state,
        "common_initialization": common_report,
        "formal_epochs_per_variant": int(suite["formal"]["epochs"]),
        "scientific_design": {
            "chain": ["MS-GMM", "MS-GMM + MS-DEH", "MS-GMM + MS-DEH + EG-ALC"],
            "all_rows_start_from_identical_epoch0_state": True,
            "A1a_A1final_A2_full_used_as_initialization": False,
            "DMC": "ON, weight fixed identically across all rows",
            "checkpoint_selection": "best validation GED^2@4",
            "test_prediction": "prior-only",
            "test_samples": suite["evaluation"]["sample_counts"],
        },
        "table": rows,
    })

    print("\n" + "=" * 120)
    print("STRICT ABLATION SUITE FINISHED")
    print("[END Beijing]", now())
    print("Morning table:", output_root / "ablation_table_431_strict.csv")
    print("=" * 120)


if __name__ == "__main__":
    main()
