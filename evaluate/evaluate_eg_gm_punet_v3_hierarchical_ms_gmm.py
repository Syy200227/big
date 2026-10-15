"""
EG-GM-PUNet v3 Stage A final evaluation.

输出：
- probabilistic_per_case_metrics.csv
- evidence_per_case_metrics.csv
- v3_mechanism_per_case.csv
- v3_mechanism_per_patient.csv
- v3_analysis_table.csv
- case_artifact_index.csv
- summary.json
- case_artifacts/*.npz

重要：
预测始终 prior-only；
posterior 仅用于离线 capacity diagnostic，绝不参与 test prediction。
component/basis statistics 只解释为 latent-density basis utilization，
不解释为真实 segmentation mode 数。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet_v3 import EGGMPUNetV3
from models.Evidential_Multi_scale_Gaussian_Mixture_prob_unet.evidential_multiscale_gmm_parts import (
    multi_rater_soft_target,
)
from utils.evidential_metrics import EvidentialMetricAccumulator
from utils.gmm_experiment_utils import prepare_device, set_seed
from utils.gmm_prior_evaluation import (
    evaluate_single_case,
    normalize_sample_counts,
    summarize_rows,
    write_csv,
)


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("YAML 必须为 dict。")
    return value


def build_model(cfg):
    m = cfg["model"]
    return EGGMPUNetV3(
        in_channels=int(m["in_channels"]),
        num_classes=int(m["num_classes"]),
        num_components={
            str(k): int(v)
            for k, v in m["num_components"].items()
        },
        latent_dims={
            str(k): int(v)
            for k, v in m["latent_dims"].items()
        },
        num_channels=tuple(int(v) for v in m["num_channels"]),
        group_feature_indices={
            str(k): int(v)
            for k, v in m["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(k): int(v)
            for k, v in m["group_decoder_indices"].items()
        },
        num_convs_per_block=int(m["num_convs_per_block"]),
        head_hidden_dim=int(m["head_hidden_dim"]),
        fusion_1x1_convs=int(m["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(
            m["segmentation_head_1x1_convs"]
        ),
        min_log_sigma=m.get("min_log_sigma"),
        max_log_sigma=m.get("max_log_sigma"),
        mixture_temperature=float(m["mixture_temperature"]),
        evidence_projection_channels=int(
            m["evidence_projection_channels"]
        ),
        evidence_fusion_channels=int(
            m["evidence_fusion_channels"]
        ),
        feedback_config=m["feedback"],
    )


def case_ids_from_batch(value, batch_size, fallback, split):
    if isinstance(value, str):
        return [value]
    if torch.is_tensor(value):
        x = value.detach().cpu().tolist()
        if not isinstance(x, list):
            x = [x]
        return [str(v) for v in x]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [
        f"{split}_{fallback+i:06d}"
        for i in range(batch_size)
    ]


def values_from_batch(value, batch_size, default=""):
    if isinstance(value, str):
        return [value] * batch_size
    if torch.is_tensor(value):
        x = value.detach().cpu().tolist()
        if not isinstance(x, list):
            x = [x]
        return [str(v) for v in x]
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [default] * batch_size


def patient_from_case(case_id):
    m = re.search(r"(LIDC-IDRI-\d{4})", str(case_id))
    return m.group(1) if m else str(case_id).split("_")[0]


def safe_filename(value):
    return "".join(
        c if c.isalnum() or c in "-_." else "_"
        for c in str(value)
    )


def annotation_diversity_iou(masks, eps=1e-7):
    x = masks.detach().float()
    if x.ndim == 4:
        # [R,H,W] or [R,1,H,W]
        if x.shape[1] == 1:
            x = x[:, 0]
    r = x.shape[0]
    vals = []
    for i in range(r):
        a = x[i].reshape(-1)
        for j in range(i+1, r):
            b = x[j].reshape(-1)
            inter = (a*b).sum()
            union = ((a+b)>0).float().sum()
            iou = (
                torch.tensor(1.0, device=x.device)
                if union <= eps
                else inter / union.clamp_min(eps)
            )
            vals.append(1.0-iou)
    return float(torch.stack(vals).mean().item())


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i+1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5*(i+j-1)+1.0
        i = j
    return ranks


def correlation(x, y, spearman=False):
    a=np.asarray(x,dtype=float); b=np.asarray(y,dtype=float)
    good=np.isfinite(a)&np.isfinite(b)
    a=a[good]; b=b[good]
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return None
    if spearman:
        a=rankdata(a); b=rankdata(b)
    return float(np.corrcoef(a,b)[0,1])


def aggregate_patient(rows):
    groups={}
    for row in rows:
        groups.setdefault(row["patient_id"],[]).append(row)

    output=[]
    for patient, items in sorted(groups.items()):
        result={
            "patient_id":patient,
            "num_cases":len(items),
        }
        numeric=[]
        for key in items[0]:
            if key in {"case_id","patient_id"}:
                continue
            vals=[]
            for item in items:
                try:
                    v=float(item[key])
                    if math.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if vals:
                result[key]=float(np.mean(vals))
        output.append(result)
    return output


def merge_analysis(mechanism_rows, probabilistic_rows, sample_counts):
    index={}
    for row in probabilistic_rows:
        index[(row["case_id"], int(row["num_samples"]))] = row

    out=[]
    for mech in mechanism_rows:
        row=dict(mech)
        case=mech["case_id"]
        for count in sample_counts:
            src=index.get((case,int(count)))
            if src is None:
                continue
            for key in (
                "ged2","cross_dice",
                "best_sample_dice_to_raters",
                "sample_diversity_distance",
                "mean_probability_mask_dice",
            ):
                if key in src:
                    row[f"{key}_{count}"]=src[key]
        out.append(row)
    return out


def mean_numeric(rows, key):
    vals=[]
    for row in rows:
        try:
            v=float(row[key])
            if math.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_v3_hierarchical_ms_gmm_lidc.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-counts", type=int, nargs="+", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args=parser.parse_args()

    config_path=Path(args.config)
    if not config_path.is_absolute():
        config_path=PROJECT_ROOT/config_path
    cfg=load_yaml(config_path)
    set_seed(int(cfg.get("validation_seed", cfg["seed"])))
    device=prepare_device(cfg.get("device","cuda"))

    save_dir=Path(cfg["save_dir"])
    checkpoint=Path(
        args.checkpoint
        or save_dir/"best_val_ged4_eg_gm_punet_v3.pth"
    )
    if not checkpoint.is_absolute():
        checkpoint=PROJECT_ROOT/checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    obj=torch.load(checkpoint,map_location=device)
    checkpoint_epoch=int(obj.get("epoch",0))

    model=build_model(cfg).to(device)
    model.load_state_dict(obj["model_state_dict"],strict=True)
    model.eval()

    ecfg=cfg["evaluation"]
    num_samples=int(
        args.num_samples
        or ecfg["final_prior_samples"]
    )
    sample_counts=normalize_sample_counts(
        args.sample_counts or ecfg["sample_counts"],
        num_samples,
    )

    output_dir=(
        save_dir/"evaluation"/args.split
        /f"samples_{num_samples}"
        /checkpoint.stem
    )
    if args.smoke_test:
        output_dir=output_dir/"smoke_test"
    output_dir.mkdir(parents=True,exist_ok=True)
    artifact_dir=output_dir/"case_artifacts"
    if bool(ecfg.get("save_all_case_artifacts",True)):
        artifact_dir.mkdir(parents=True,exist_ok=True)

    loader=build_lidc_full_dataloader(
        root=cfg["data_root"],
        split=args.split,
        batch_size=1 if args.smoke_test else int(cfg["training"]["batch_size"]),
        num_workers=int(cfg["training"]["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(cfg["validation"]["rater_index"]),
        return_consensus=True,
        return_meta=True,
    )

    evidence_metrics=EvidentialMetricAccumulator(
        num_bins=int(cfg["validation"]["ece_bins"]),
        ranking_bins=int(cfg["validation"]["ranking_bins"]),
    )

    probabilistic_rows=[]
    evidence_rows=[]
    mechanism_rows=[]
    artifact_index=[]
    fallback=0

    max_batches=2 if args.smoke_test else None
    iterator=loader if max_batches is None else islice(loader, max_batches)

    progress=tqdm(iterator,desc="v3 evaluate",ncols=180)
    for batch in progress:
        images=batch["image"].to(device=device,dtype=torch.float32)
        segmentation=batch["mask"].to(device=device,dtype=torch.float32)
        all_seg=batch["masks"].to(device=device,dtype=torch.float32)

        case_ids=case_ids_from_batch(
            batch.get("case_id"),images.shape[0],fallback,args.split
        )
        patient_ids=values_from_batch(
            batch.get("patient_id"),images.shape[0],""
        )
        patient_ids=[
            p if p else patient_from_case(c)
            for p,c in zip(patient_ids,case_ids)
        ]

        # posterior features are used only for capacity diagnosis.
        state=model(
            image=images,
            segmentation=segmentation,
            is_training=True,
        )

        with torch.no_grad():
            _, feedback_probe=model.deterministic_feedback_probe(state)

            controls=model.feedback_controller(
                evidence=feedback_probe,
                state=state,
                base_beta_groups={
                    str(k):float(v)
                    for k,v in cfg["training"]["base_beta_groups"].items()
                },
                current_epoch=checkpoint_epoch,
                feedback_enabled=bool(cfg["training"]["feedback_enabled"]),
                update_stats=False,
            )

            samples=model.sample_with_evidence(
                state=state,
                num_samples=num_samples,
            )

        p_logits=samples.probabilistic_logits
        p_masks=p_logits.argmax(dim=2).float()
        p_fg=torch.softmax(p_logits,dim=2)[:,:,1]

        e_prob=samples.evidence_probabilities
        e_masks=e_prob.argmax(dim=2).float()
        e_fg=e_prob[:,:,1]

        # Evidence calibration: 与旧 evaluator 保持同一口径，
        # 对每个 prior sample 的 evidence prediction 都累计一次。
        soft_target=multi_rater_soft_target(
            segmentation,
            all_seg,
            num_classes=model.num_classes,
        ).to(device=device,dtype=e_prob.dtype)

        for evidence in samples.evidence_outputs:
            evidence_metrics.update(
                evidence_output=evidence.fused,
                evidence_summary=evidence.summaries["fused"],
                segmentation=segmentation,
                all_segmentations=all_seg,
                soft_target=soft_target,
            )

        for bi in range(images.shape[0]):
            cid=case_ids[bi]
            pid=patient_ids[bi]

            pr=evaluate_single_case(
                sample_masks=p_masks[:,bi],
                foreground_probabilities=p_fg[:,bi],
                rater_masks=all_seg[bi],
                sample_counts=sample_counts,
            )
            er=evaluate_single_case(
                sample_masks=e_masks[:,bi],
                foreground_probabilities=e_fg[:,bi],
                rater_masks=all_seg[bi],
                sample_counts=sample_counts,
            )
            for row in pr:
                row.update({
                    "case_id":cid,
                    "patient_id":pid,
                    "split":args.split,
                    "branch":"probabilistic",
                })
            for row in er:
                row.update({
                    "case_id":cid,
                    "patient_id":pid,
                    "split":args.split,
                    "branch":"evidence",
                })
            probabilistic_rows.extend(pr)
            evidence_rows.extend(er)

            mech={
                "case_id":cid,
                "patient_id":pid,
                "annotation_diversity_iou":annotation_diversity_iou(
                    all_seg[bi]
                ),
                "raw_ambiguity_demand":float(
                    controls.raw_demand[bi].item()
                ),
                "ambiguity_demand":float(
                    controls.ambiguity_demand[bi].item()
                ),
                "feedback_ramp":float(
                    controls.feedback_ramp.item()
                ),
            }

            # Per-scale demand-capacity and latent-density basis utilization.
            for g in model.group_names:
                mech[f"{g}_capacity"]=float(
                    controls.group_capacity[g][bi].item()
                )
                mech[f"{g}_error"]=float(
                    controls.group_error[g][bi].item()
                )
                mech[f"{g}_deficit"]=float(
                    controls.group_deficit[g][bi].item()
                )
                mech[f"{g}_surplus"]=float(
                    controls.group_surplus[g][bi].item()
                )
                mech[f"beta_{g}"]=float(
                    controls.beta_groups[g][bi].item()
                )
                mech[f"{g}_raw_joint_kl"]=float(
                    controls.raw_group_joint_kl[g][bi].item()
                )
                mech[f"{g}_raw_categorical_kl"]=float(
                    controls.raw_group_categorical_kl[g][bi].item()
                )
                mech[f"{g}_raw_gaussian_kl"]=float(
                    controls.raw_group_gaussian_kl[g][bi].item()
                )

                path_probs=(
                    samples.component_probabilities[g][:,bi]
                    .detach().cpu().numpy()
                ) # [S,K]
                marginal=path_probs.mean(axis=0)
                safe=np.clip(marginal,1e-8,1.0)
                keff=float(np.exp(-(safe*np.log(safe)).sum()))
                active=int(np.sum(
                    marginal>=float(ecfg["active_basis_threshold"])
                ))
                mech[f"{g}_basis_keff"]=keff
                mech[f"{g}_active_basis_count"]=active
                for k,prob in enumerate(marginal):
                    mech[f"{g}_basis_probability_{k}"]=float(prob)

            mechanism_rows.append(mech)

            if bool(ecfg.get("save_all_case_artifacts",True)):
                artifact=artifact_dir/f"{safe_filename(cid)}.npz"
                payload={
                    "case_id":np.asarray(cid),
                    "patient_id":np.asarray(pid),
                    "image":images[bi].detach().cpu().numpy(),
                    "rater_masks":all_seg[bi].detach().cpu().numpy().astype(
                        np.uint8
                    ),
                    "generated_samples":p_masks[:,bi]
                        .detach().cpu().numpy().astype(np.uint8),
                    "ambiguity_map_fine":(
                        feedback_probe.scales["fine"].probabilities[
                            bi,1:2
                        ]
                        * feedback_probe.scales["fine"]
                        .expected_data_entropy[bi]
                    ).detach().cpu().numpy(),
                    "vacuity_map_fused":feedback_probe.fused.vacuity[
                        bi
                    ].detach().cpu().numpy(),
                    "expected_data_entropy_fused":(
                        feedback_probe.fused.expected_data_entropy[bi]
                        .detach().cpu().numpy()
                    ),
                }
                for g in model.group_names:
                    payload[f"component_indices_{g}"]=(
                        samples.component_indices[g][:,bi]
                        .detach().cpu().numpy()
                    )
                    payload[f"component_probabilities_{g}"]=(
                        samples.component_probabilities[g][:,bi]
                        .detach().cpu().numpy().astype(np.float16)
                    )
                np.savez_compressed(artifact,**payload)
                artifact_index.append({
                    "case_id":cid,
                    "patient_id":pid,
                    "artifact_path":str(artifact.resolve()),
                })

        fallback+=images.shape[0]

    write_csv(
        probabilistic_rows,
        output_dir/"probabilistic_per_case_metrics.csv",
    )
    write_csv(
        evidence_rows,
        output_dir/"evidence_per_case_metrics.csv",
    )
    write_csv(
        mechanism_rows,
        output_dir/"v3_mechanism_per_case.csv",
    )
    patient_rows=aggregate_patient(mechanism_rows)
    write_csv(
        patient_rows,
        output_dir/"v3_mechanism_per_patient.csv",
    )
    analysis=merge_analysis(
        mechanism_rows,probabilistic_rows,sample_counts
    )
    write_csv(
        analysis,
        output_dir/"v3_analysis_table.csv",
    )
    write_csv(
        artifact_index,
        output_dir/"case_artifact_index.csv",
    )

    p_summary=summarize_rows(probabilistic_rows,sample_counts)
    e_summary=summarize_rows(evidence_rows,sample_counts)
    cal=evidence_metrics.compute()

    x=[r["annotation_diversity_iou"] for r in mechanism_rows]
    u=[r["raw_ambiguity_demand"] for r in mechanism_rows]
    patient_x=[
        r["annotation_diversity_iou"] for r in patient_rows
    ]
    patient_u=[
        r["raw_ambiguity_demand"] for r in patient_rows
    ]

    summary={
        "stage":"EG-GM-PUNet v3 Stage A — Hierarchical Scale-Decoupled MS-GMM",
        "checkpoint":str(checkpoint),
        "checkpoint_epoch":checkpoint_epoch,
        "split":args.split,
        "num_cases":len(mechanism_rows),
        "num_prior_samples":num_samples,
        "sample_counts":sample_counts,
        "probabilistic_summary":p_summary,
        "evidence_summary":e_summary,
        "evidence_calibration":cal,
        "demand_annotation_iou":{
            "slice_pearson":correlation(x,u,False),
            "slice_spearman":correlation(x,u,True),
            "patient_pearson":correlation(patient_x,patient_u,False),
            "patient_spearman":correlation(patient_x,patient_u,True),
        },
        "mechanism_means":{
            key:mean_numeric(mechanism_rows,key)
            for key in mechanism_rows[0]
            if key not in {"case_id","patient_id"}
        } if mechanism_rows else {},
        "architecture":{
            "top_level":"MS-GMM + MS-DEH + EG-ALC",
            "shared_semantic_k":False,
            "hierarchical_scale_decoupled_gmm":True,
            "component_semantics":"latent density basis only",
            "adaptive_categorical_beta":False,
            "adaptive_gumbel_temperature":False,
            "component_vitality":False,
            "scale_specific_demand":False,
            "demand_signal":"fine foreground-weighted expected-data entropy",
        },
        "artifact_schema":"EG_GM_PUNET_V3_HIERARCHICAL_MS_GMM_V1",
    }
    (output_dir/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir/"artifact_schema.json").write_text(
        json.dumps({
            "schema":"EG_GM_PUNET_V3_HIERARCHICAL_MS_GMM_V1",
            "prediction_is_prior_only":True,
            "posterior_use":"capacity diagnostic only",
            "component_index_semantics":"internal GMM density basis index only",
            "arrays":[
                "image","rater_masks","generated_samples",
                "component_indices_coarse/middle/fine",
                "component_probabilities_coarse/middle/fine",
                "ambiguity_map_fine","vacuity_map_fused",
                "expected_data_entropy_fused",
            ],
        },indent=2,ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
