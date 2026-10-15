"""
EG-GM-PUNet v2 Stage 0.5b：Ambiguity Demand Robustness Confirmation 诊断脚本。

这个文件是干什么的：
1. 不训练任何新模型，不修改现有 checkpoint；
2. 读取当前第一版 Full EG-GM-PUNet 的 best GED checkpoint；
3. 使用 MS-DEH 输出的空间 evidence map，构造多种“病例级 ambiguity demand”候选信号；
4. 比较这些候选 demand 与 4 位医生真实 annotation diversity 的 Pearson / Spearman 相关性；
5. 同时重新统计 prior GMM 的有效 component 数、active component 数、全部 component diversity、
   active component diversity 和 probability-weighted component diversity；
6. 用于回答两个问题：
   (a) 当前 evidence map 本身是否有能力反映真实 ambiguity，只是旧的全局聚合方式不合适？
   (b) 哪一种 spatial-aware evidence aggregation 最适合进入 EG-GM-PUNet v2 的
       Evidence–Latent Demand–Capacity Controller？

为什么需要这个诊断：
Stage 0 已发现：
- K=5 时 prior/posterior K_eff 约为 2；
- 高 annotation diversity 病例没有使用更多离散 component；
- 旧版 prior_probe_uncertainty 与 annotation diversity 反而负相关。

因此这里不能直接把旧 uncertainty 做 EMA normalization，而必须先找到真正代表
“该病例需要多少 stochastic latent capacity”的 demand signal。

本脚本只使用预测得到的 evidence / probability 构造 demand，4 位医生标注仅用于
离线诊断相关性，不会泄漏到未来推理阶段的 demand 计算中。

默认读取：
    configs/eg_gm_punet_lidc.yaml
    result/eg_gm_punet_k5_full/best_val_ged4_eg_gm_punet.pth

Smoke test：
    python evaluate/diagnose_eg_gm_punet_v2_ambiguity_demand.py --smoke-test

正式诊断：
    python evaluate/diagnose_eg_gm_punet_v2_ambiguity_demand.py

默认输出：
    result/eg_gm_punet_v2_diagnostics/ambiguity_demand/test/
        ambiguity_demand_per_case.csv
        ambiguity_demand_summary.json

说明：
- 本脚本属于“诊断实验”，不参与训练、不反向传播；
- 所有模型 forward 都在 torch.no_grad() 下运行；
- posterior probe 使用当前配置中的固定 rater，与第一版训练时的 posterior feedback 路径保持一致；
- prior probe 不依赖医生标注，更接近部署/推理时可获得的 uncertainty signal。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from itertools import combinations, islice
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from tqdm import tqdm


# =============================================================================
# 项目路径与项目内部导入
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.lidc_dataset_full import build_lidc_full_dataloader
from models.EG_GM_PUNet import EGGMPUNet
from models.Multi_scale_Gaussian_Mixture_prob_unet.multiscale_grouped_gmm_model import (
    MultiScaleGroupedGMMState,
)
from utils.gmm_experiment_utils import prepare_device, set_seed


# =============================================================================
# 通用工具
# =============================================================================

def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置，并保证顶层对象是字典。"""
    with open(path, "r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层必须为字典：{path}")
    return value


def build_legacy_model(config: dict[str, Any]) -> EGGMPUNet:
    """
    根据第一版 EG-GM-PUNet 配置构建模型。

    这里故意使用 models/EG_GM_PUNet 中的旧模型，而不是 v2 模型，
    因为 Stage 0.5 的任务是诊断“已经训练完成的第一版模型”。
    """
    item = config["model"]
    return EGGMPUNet(
        in_channels=int(item["in_channels"]),
        num_classes=int(item["num_classes"]),
        num_components=int(item["num_components"]),
        latent_dims={
            str(name): int(value)
            for name, value in item["latent_dims"].items()
        },
        num_channels=tuple(int(value) for value in item["num_channels"]),
        group_feature_indices={
            str(name): int(value)
            for name, value in item["group_feature_indices"].items()
        },
        group_decoder_indices={
            str(name): int(value)
            for name, value in item["group_decoder_indices"].items()
        },
        num_convs_per_block=int(item["num_convs_per_block"]),
        fusion_1x1_convs=int(item["fusion_1x1_convs"]),
        segmentation_head_1x1_convs=int(item["segmentation_head_1x1_convs"]),
        min_log_sigma=item.get("min_log_sigma"),
        max_log_sigma=item.get("max_log_sigma"),
        evidence_projection_channels=int(item["evidence_projection_channels"]),
        evidence_fusion_channels=int(item["evidence_fusion_channels"]),
        feedback_config=item.get("feedback"),
    )


def ids_from_batch(
    value: Any,
    batch_size: int,
    prefix: str,
    offset: int,
) -> list[str]:
    """把 DataLoader 中可能为 list/tensor/string 的 id 统一转成字符串列表。"""
    if value is None:
        return [f"{prefix}_{offset + index:06d}" for index in range(batch_size)]
    if isinstance(value, str):
        return [value] * batch_size
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)] * batch_size


def finite_float(value: Any) -> float | None:
    """安全转换为有限浮点数，NaN/Inf 返回 None。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


# =============================================================================
# Annotation ambiguity：只用于离线诊断目标，不进入 demand 计算
# =============================================================================

def binary_iou(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """二值 mask 的 IoU；若两者均为空，按完全一致返回 1。"""
    a = mask_a.float().reshape(-1)
    b = mask_b.float().reshape(-1)
    intersection = (a * b).sum()
    union = ((a + b) > 0).float().sum()
    if float(union.item()) <= eps:
        return a.new_tensor(1.0)
    return intersection / (union + eps)


def binary_dice(
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """二值 mask 的 Dice；若两者均为空，按完全一致返回 1。"""
    a = mask_a.float().reshape(-1)
    b = mask_b.float().reshape(-1)
    denominator = a.sum() + b.sum()
    if float(denominator.item()) <= eps:
        return a.new_tensor(1.0)
    return (2.0 * (a * b).sum() + eps) / (denominator + eps)


def annotation_diversity(masks: torch.Tensor) -> float:
    """
    计算一个病例多位医生标注之间的平均 1-IoU。

    输入：
        masks: [R,H,W] 或 [R,1,H,W]

    输出：
        case-level annotation diversity。

    这个值是 Stage 0/0.5 的主要离线 ground-truth ambiguity 指标。
    """
    values: list[float] = []
    for first, second in combinations(range(masks.shape[0]), 2):
        values.append(
            1.0 - float(binary_iou(masks[first], masks[second]).item())
        )
    return float(np.mean(values)) if values else 0.0


def annotation_spatial_ambiguity(masks: torch.Tensor) -> tuple[float, float]:
    """
    根据多医生标注构造空间 ambiguity map：A=4*p*(1-p)。

    返回两个辅助诊断指标：
    1. union ROI 内的平均 ambiguity；
    2. union ROI 内存在分歧像素的比例。

    它们只用于帮助理解 D_ann，不参与模型 demand 构造。
    """
    value = masks.float()
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    mean_mask = value.mean(dim=0)
    ambiguity = 4.0 * mean_mask * (1.0 - mean_mask)
    union = (value.sum(dim=0) > 0).float()
    denominator = union.sum().clamp_min(1.0)
    ambiguity_roi_mean = float((ambiguity * union).sum().item() / denominator.item())
    disagreement = ((mean_mask > 0.0) & (mean_mask < 1.0)).float()
    disagreement_fraction = float((disagreement * union).sum().item() / denominator.item())
    return ambiguity_roi_mean, disagreement_fraction


# =============================================================================
# GMM component capacity / diversity 诊断
# =============================================================================

def distribution_stats(
    probabilities: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 categorical entropy、归一化 entropy 和 effective component count。

    K_eff = exp(H(pi))：
    - one-hot routing 时接近 1；
    - 完全均匀 K components 时等于 K。
    """
    safe = probabilities.clamp_min(eps)
    entropy = -(safe * safe.log()).sum(dim=1)
    normalized_entropy = entropy / math.log(float(probabilities.shape[1]))
    k_eff = torch.exp(entropy)
    return entropy, normalized_entropy, k_eff


def component_mean_outputs(
    model: EGGMPUNet,
    state: MultiScaleGroupedGMMState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    分别取每个 prior component 的 Gaussian mean 解码。

    输入：
        state.prior 中包含 [B,K,D_g] 的各尺度 Gaussian mean。

    输出：
        component_masks: [B,K,H,W]
        component_foreground_probabilities: [B,K,H,W]

    目的：
        区分“component 概率被使用”与“component 在输出空间是否真的不同”。
    """
    mixture_probs = state.prior.mixture_probs
    batch_size, num_components = mixture_probs.shape
    masks: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []

    for component_index in range(num_components):
        z_groups = {
            name: state.prior.group_mu[name][:, component_index]
            for name in model.group_names
        }
        indices = torch.full(
            (batch_size,),
            component_index,
            device=mixture_probs.device,
            dtype=torch.long,
        )
        decode = model.probabilistic_model.decode_from_groups(
            state=state,
            z_groups=z_groups,
            component_indices=indices,
        )
        foreground = torch.softmax(decode.logits, dim=1)[:, 1]
        probabilities.append(foreground)
        masks.append((foreground >= 0.5).float())

    return torch.stack(masks, dim=1), torch.stack(probabilities, dim=1)


def mean_pairwise_iou_distance(
    masks: torch.Tensor,
    indices: Iterable[int] | None = None,
) -> float:
    """计算指定 component 集合的平均 pairwise 1-IoU。"""
    if indices is None:
        selected = list(range(masks.shape[0]))
    else:
        selected = [int(value) for value in indices]

    if len(selected) < 2:
        return 0.0

    values: list[float] = []
    for first, second in combinations(selected, 2):
        values.append(
            1.0 - float(binary_iou(masks[first], masks[second]).item())
        )
    return float(np.mean(values)) if values else 0.0


def mean_pairwise_dice(
    masks: torch.Tensor,
    indices: Iterable[int] | None = None,
) -> float:
    """计算指定 component 集合的平均 pairwise Dice。"""
    if indices is None:
        selected = list(range(masks.shape[0]))
    else:
        selected = [int(value) for value in indices]

    if len(selected) < 2:
        return 1.0

    values: list[float] = []
    for first, second in combinations(selected, 2):
        values.append(
            float(binary_dice(masks[first], masks[second]).item())
        )
    return float(np.mean(values)) if values else 1.0


def weighted_pairwise_iou_distance(
    masks: torch.Tensor,
    mixture_probs: torch.Tensor,
) -> float:
    """
    计算 probability-weighted component diversity。

    数学上对应：
        sum_{k,l} pi_k*pi_l*d(mask_k, mask_l)

    它比“把所有 K 个 component 等权两两比较”更接近真实 prior sampling
    能观察到的离散 component diversity，因为死亡 component 的权重自然接近 0。
    """
    num_components = masks.shape[0]
    total = 0.0
    for first in range(num_components):
        for second in range(num_components):
            if first == second:
                continue
            distance = 1.0 - float(
                binary_iou(masks[first], masks[second]).item()
            )
            total += (
                float(mixture_probs[first].item())
                * float(mixture_probs[second].item())
                * distance
            )
    return float(total)


def weighted_component_variance(
    component_probabilities: torch.Tensor,
    mixture_probs: torch.Tensor,
) -> float:
    """
    计算 component foreground probability 的 mixture-weighted variance。

    输出乘 4 后大致归一化到更易解释的范围。
    """
    mean_probability = (
        mixture_probs[:, None, None] * component_probabilities
    ).sum(dim=0)
    variance = (
        mixture_probs[:, None, None]
        * (component_probabilities - mean_probability[None]).square()
    ).sum(dim=0)
    return float((4.0 * variance).mean().item())


# =============================================================================
# Spatial-aware Evidence Demand 候选信号
# =============================================================================

def weighted_spatial_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    对 [B,1,H,W] 的 value 做 soft spatial weighted mean。

    返回 shape [B]。
    """
    if value.ndim != 4 or weight.ndim != 4:
        raise ValueError(
            f"value/weight 必须为 [B,1,H,W]，实际 {value.shape}/{weight.shape}"
        )
    dimensions = tuple(range(2, value.ndim))
    numerator = (value * weight).sum(dim=dimensions)[:, 0]
    denominator = weight.sum(dim=dimensions)[:, 0].clamp_min(eps)
    return numerator / denominator


def global_spatial_mean(value: torch.Tensor) -> torch.Tensor:
    """对 [B,1,H,W] uncertainty map 做全图平均，返回 [B]。"""
    dimensions = tuple(range(2, value.ndim))
    return value.mean(dim=dimensions)[:, 0]


def top_fraction_mean(
    value: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """
    计算全图 uncertainty 最高 fraction 比例像素的均值。

    该指标用于检查“小范围高不确定区域被全图平均淹没”是否是旧 feedback 的问题。
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction 必须位于 (0,1]。")
    flattened = value[:, 0].flatten(start_dim=1)
    count = max(1, int(math.ceil(flattened.shape[1] * fraction)))
    return torch.topk(flattened, k=count, dim=1).values.mean(dim=1)


def roi_top_fraction_mean(
    value: torch.Tensor,
    foreground: torch.Tensor,
    fraction: float,
    roi_threshold: float,
) -> torch.Tensor:
    """
    在预测 lesion-support ROI 内计算 uncertainty top-q 均值。

    ROI 完全由模型自己的 foreground probability 定义，不使用医生标注：
        ROI = p_fg >= roi_threshold

    若某个病例 ROI 为空，则回退到 foreground probability 最高的 1% 像素，
    避免得到 NaN。
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction 必须位于 (0,1]。")

    outputs: list[torch.Tensor] = []
    for batch_index in range(value.shape[0]):
        current_value = value[batch_index, 0].reshape(-1)
        current_foreground = foreground[batch_index, 0].reshape(-1)
        roi = current_foreground >= float(roi_threshold)

        if int(roi.sum().item()) == 0:
            fallback_count = max(1, int(math.ceil(current_foreground.numel() * 0.01)))
            fallback_indices = torch.topk(
                current_foreground,
                k=fallback_count,
                dim=0,
            ).indices
            selected_values = current_value[fallback_indices]
        else:
            selected_values = current_value[roi]

        count = max(1, int(math.ceil(selected_values.numel() * fraction)))
        outputs.append(torch.topk(selected_values, k=count, dim=0).values.mean())

    return torch.stack(outputs, dim=0)


def spatial_evidence_demands(
    evidence,
    *,
    roi_threshold: float,
    top_fractions: tuple[float, float],
) -> dict[str, torch.Tensor]:
    """
    从 MultiScaleEvidenceOutput 构造一组 ambiguity-demand 候选信号。

    核心原则：
    1. 不使用医生标注，因此未来可以真正用于训练/推理 feedback；
    2. 同时比较 global / foreground / predicted-boundary / top-q；
    3. 同时拆开 vacuity、predictive entropy、mutual information，避免 composite
       掩盖某一种 uncertainty 的有效性；
    4. fused 分支额外计算 top-10% / top-20% 和 ROI-top-q；
    5. coarse/middle/fine 分支也计算 global/foreground/boundary composite，
       为后续 scale-specific latent-capacity control 提供诊断依据。

    返回：
        每个 value 都是 shape [B] 的 tensor。
    """
    results: dict[str, torch.Tensor] = {}

    branches: dict[str, Any] = {"fused": evidence.fused}
    for name, output in evidence.scales.items():
        branches[str(name)] = output

    for branch_name, output in branches.items():
        foreground = output.probabilities[:, 1:2]
        boundary = 4.0 * foreground * (1.0 - foreground)

        vacuity = output.vacuity
        entropy = output.predictive_entropy
        expected_data_entropy = output.expected_data_entropy
        mutual_information = output.mutual_information
        composite = (
            0.60 * vacuity
            + 0.20 * entropy
            + 0.20 * mutual_information
        )

        # Stage 0.5b 关键新增：
        # expected_data_entropy 更接近数据本身/多标注带来的 aleatoric ambiguity。
        # Stage 0.5 已发现 fine + foreground 是最有效的尺度/空间聚合方式，
        # 因此这里显式加入 expected-data entropy 进行正式比较。
        maps = {
            "vacuity": vacuity,
            "entropy": entropy,
            "expected_data_entropy": expected_data_entropy,
            "mi": mutual_information,
            "composite": composite,
        }

        for uncertainty_name, uncertainty_map in maps.items():
            results[f"{branch_name}_global_{uncertainty_name}"] = (
                global_spatial_mean(uncertainty_map)
            )
            results[f"{branch_name}_foreground_{uncertainty_name}"] = (
                weighted_spatial_mean(uncertainty_map, foreground)
            )
            results[f"{branch_name}_boundary_{uncertainty_name}"] = (
                weighted_spatial_mean(uncertainty_map, boundary)
            )

        # 额外保留第一版 feedback 中使用的 summary 指标，便于直接比较。
        summary = evidence.summaries[branch_name]
        results[f"{branch_name}_legacy_mean_vacuity"] = summary.mean_vacuity
        results[f"{branch_name}_legacy_foreground_vacuity"] = (
            summary.foreground_weighted_vacuity
        )
        results[f"{branch_name}_legacy_ambiguity_vacuity"] = (
            summary.ambiguity_weighted_vacuity
        )
        results[f"{branch_name}_legacy_mean_entropy"] = (
            summary.mean_predictive_entropy
        )
        results[f"{branch_name}_legacy_mean_mi"] = (
            summary.mean_mutual_information
        )
        # 完整复现第一版 feedback controller 使用的 scalar 组合，
        # 便于直接验证旧 signal 与新 spatial-aware candidate 的差异。
        results[f"{branch_name}_legacy_feedback_composite"] = (
            0.60 * summary.ambiguity_weighted_vacuity
            + 0.20 * summary.mean_predictive_entropy
            + 0.20 * summary.mean_mutual_information
        )

        # top-q 只在 fused 分支计算，避免输出列过度膨胀。
        if branch_name == "fused":
            for fraction in top_fractions:
                percentage = int(round(100 * fraction))
                results[f"fused_top{percentage}_composite"] = (
                    top_fraction_mean(composite, fraction)
                )
                results[f"fused_roi_top{percentage}_composite"] = (
                    roi_top_fraction_mean(
                        composite,
                        foreground,
                        fraction,
                        roi_threshold,
                    )
                )
                results[f"fused_top{percentage}_vacuity"] = (
                    top_fraction_mean(vacuity, fraction)
                )
                results[f"fused_roi_top{percentage}_vacuity"] = (
                    roi_top_fraction_mean(
                        vacuity,
                        foreground,
                        fraction,
                        roi_threshold,
                    )
                )
                results[f"fused_top{percentage}_expected_data_entropy"] = (
                    top_fraction_mean(expected_data_entropy, fraction)
                )
                results[f"fused_roi_top{percentage}_expected_data_entropy"] = (
                    roi_top_fraction_mean(
                        expected_data_entropy,
                        foreground,
                        fraction,
                        roi_threshold,
                    )
                )

    return results


# =============================================================================
# 相关性统计：不依赖 scipy，保证项目环境直接运行
# =============================================================================

def rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩实现，用于 Spearman correlation；支持并列值。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        next_index = index + 1
        while (
            next_index < len(values)
            and values[order[next_index]] == values[order[index]]
        ):
            next_index += 1
        rank = 0.5 * (index + next_index - 1) + 1.0
        ranks[order[index:next_index]] = rank
        index = next_index
    return ranks


def correlation(
    first: Iterable[float],
    second: Iterable[float],
    *,
    spearman: bool = False,
) -> float | None:
    """计算 Pearson 或 Spearman；常数列/样本不足时返回 None。"""
    first_array = np.asarray(list(first), dtype=float)
    second_array = np.asarray(list(second), dtype=float)
    valid = np.isfinite(first_array) & np.isfinite(second_array)
    first_array = first_array[valid]
    second_array = second_array[valid]

    if (
        len(first_array) < 3
        or np.std(first_array) < 1e-12
        or np.std(second_array) < 1e-12
    ):
        return None

    if spearman:
        first_array = rankdata(first_array)
        second_array = rankdata(second_array)

    return float(np.corrcoef(first_array, second_array)[0, 1])


def demand_columns(rows: list[dict[str, Any]], prefix: str) -> list[str]:
    """从 CSV row 自动找出 prior_/posterior_ 开头的 evidence demand 列。"""
    # 这里只排除 latent distribution 自身的统计列，不能按 ``_entropy`` 后缀排除，
    # 因为 fused_global_entropy 等正是我们需要比较的 uncertainty candidate。
    excluded_exact = {
        f"{prefix}entropy",
        f"{prefix}normalized_entropy",
        f"{prefix}k_eff",
        f"{prefix}active_component_count",
        f"{prefix}top2_probability_mass",
    }
    columns: list[str] = []
    for key in rows[0].keys():
        if not key.startswith(prefix):
            continue
        if key in excluded_exact:
            continue
        if "component_" in key:
            continue
        # 只有真正 evidence demand 列才包含 fused/coarse/middle/fine。
        if not any(
            marker in key
            for marker in ("fused_", "coarse_", "middle_", "fine_")
        ):
            continue
        columns.append(key)
    return columns


def compute_demand_correlations(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> list[dict[str, Any]]:
    """
    对所有 demand candidate 计算其与 annotation diversity 的 Pearson/Spearman。

    排序优先使用 Spearman，因为我们更关心 ambiguity 排序是否方向正确；
    Spearman 相同/缺失时再参考 Pearson。
    """
    target = [float(row["annotation_diversity"]) for row in rows]
    outputs: list[dict[str, Any]] = []

    for column in columns:
        values = [float(row[column]) for row in rows]
        pearson = correlation(target, values, spearman=False)
        spearman_value = correlation(target, values, spearman=True)
        outputs.append(
            {
                "signal": column,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "pearson": pearson,
                "spearman": spearman_value,
            }
        )

    def score(item: dict[str, Any]) -> tuple[float, float]:
        spearman_value = item.get("spearman")
        pearson_value = item.get("pearson")
        return (
            -float("inf") if spearman_value is None else float(spearman_value),
            -float("inf") if pearson_value is None else float(pearson_value),
        )

    outputs.sort(key=score, reverse=True)
    return outputs


def aggregate_rows_by_patient(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> list[dict[str, Any]]:
    """
    按 patient_id 聚合 slice-level 结果，避免把同一患者的相邻切片当作完全独立样本。

    对每位患者：
    - annotation_diversity 取其所有切片平均；
    - 每个 demand candidate 取其所有切片平均；
    - 输出 patient-level rows，用于更严格的相关性复核。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        patient_id = str(row["patient_id"])
        grouped.setdefault(patient_id, []).append(row)

    patient_rows: list[dict[str, Any]] = []
    for patient_id, items in grouped.items():
        current: dict[str, Any] = {
            "patient_id": patient_id,
            "num_slices": len(items),
            "annotation_diversity": float(
                np.mean([float(item["annotation_diversity"]) for item in items])
            ),
        }
        for column in columns:
            current[column] = float(
                np.mean([float(item[column]) for item in items])
            )
        patient_rows.append(current)
    return patient_rows


def compute_patient_level_correlations(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> list[dict[str, Any]]:
    """计算 patient-level Pearson / Spearman，复核 slice-level 结论。"""
    patient_rows = aggregate_rows_by_patient(rows, columns)
    target = [float(row["annotation_diversity"]) for row in patient_rows]
    outputs: list[dict[str, Any]] = []
    for column in columns:
        values = [float(row[column]) for row in patient_rows]
        outputs.append({
            "signal": column,
            "num_patients": len(patient_rows),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "pearson": correlation(target, values, spearman=False),
            "spearman": correlation(target, values, spearman=True),
        })

    def score(item: dict[str, Any]) -> tuple[float, float]:
        s = item.get("spearman")
        p = item.get("pearson")
        return (
            -float("inf") if s is None else float(s),
            -float("inf") if p is None else float(p),
        )

    outputs.sort(key=score, reverse=True)
    return outputs


def find_signal(
    items: list[dict[str, Any]],
    signal: str,
) -> dict[str, Any] | None:
    """按 signal 名称查找一项相关性结果。"""
    for item in items:
        if str(item.get("signal")) == signal:
            return item
    return None


def demand_acceptance(
    *,
    slice_correlations: list[dict[str, Any]],
    patient_correlations: list[dict[str, Any]],
    preferred_signal: str,
    slice_threshold: float,
    patient_threshold: float,
) -> dict[str, Any]:
    """
    自动判断 preferred ambiguity demand 是否通过 Stage 0.5b。

    默认要求：
    - slice-level Spearman >= 0.50
    - patient-level Spearman >= 0.40
    """
    slice_item = find_signal(slice_correlations, preferred_signal)
    patient_item = find_signal(patient_correlations, preferred_signal)
    slice_rho = None if slice_item is None else finite_float(slice_item.get("spearman"))
    patient_rho = None if patient_item is None else finite_float(patient_item.get("spearman"))
    passed = bool(
        slice_rho is not None
        and patient_rho is not None
        and slice_rho >= float(slice_threshold)
        and patient_rho >= float(patient_threshold)
    )
    return {
        "preferred_signal": preferred_signal,
        "slice_level": slice_item,
        "patient_level": patient_item,
        "slice_spearman_threshold": float(slice_threshold),
        "patient_spearman_threshold": float(patient_threshold),
        "passed": passed,
        "decision": (
            "PASS_STAGE_0_5B_ENTER_STAGE_1"
            if passed
            else "FAIL_OR_RECHECK_DEMAND_BEFORE_STAGE_1"
        ),
    }


def quartile_summary(
    rows: list[dict[str, Any]],
    selected_demand_columns: list[str],
) -> list[dict[str, Any]]:
    """
    按 annotation diversity 四分位分组，观察 demand 与 latent capacity 的单调趋势。
    """
    annotation_values = np.asarray(
        [float(row["annotation_diversity"]) for row in rows],
        dtype=float,
    )
    quantiles = np.quantile(annotation_values, [0.25, 0.50, 0.75])
    bounds = [-np.inf, *quantiles.tolist(), np.inf]
    outputs: list[dict[str, Any]] = []

    for quartile_index in range(4):
        selector = (
            (annotation_values > bounds[quartile_index])
            & (annotation_values <= bounds[quartile_index + 1])
        )
        if quartile_index == 0:
            selector = annotation_values <= bounds[quartile_index + 1]

        selected_indices = np.where(selector)[0]
        item: dict[str, Any] = {
            "quartile": quartile_index + 1,
            "num_cases": int(selector.sum()),
            "annotation_diversity_mean": float(annotation_values[selector].mean()),
            "prior_k_eff_mean": float(
                np.mean([float(rows[i]["prior_k_eff"]) for i in selected_indices])
            ),
            "prior_active_component_count_mean": float(
                np.mean(
                    [
                        float(rows[i]["prior_active_component_count"])
                        for i in selected_indices
                    ]
                )
            ),
            "all_component_diversity_mean": float(
                np.mean(
                    [
                        float(rows[i]["all_component_iou_distance"])
                        for i in selected_indices
                    ]
                )
            ),
            "active_component_diversity_mean": float(
                np.mean(
                    [
                        float(rows[i]["active_component_iou_distance"])
                        for i in selected_indices
                    ]
                )
            ),
            "weighted_component_diversity_mean": float(
                np.mean(
                    [
                        float(rows[i]["weighted_component_iou_distance"])
                        for i in selected_indices
                    ]
                )
            ),
        }

        for column in selected_demand_columns:
            item[f"{column}_mean"] = float(
                np.mean([float(rows[i][column]) for i in selected_indices])
            )

        outputs.append(item)

    return outputs


def summarize(
    rows: list[dict[str, Any]],
    *,
    active_threshold: float,
    roi_threshold: float,
) -> dict[str, Any]:
    """生成 Stage 0.5 的完整 summary。"""
    annotation_values = [float(row["annotation_diversity"]) for row in rows]

    prior_columns = demand_columns(rows, "prior_")
    posterior_columns = demand_columns(rows, "posterior_")

    prior_correlations = compute_demand_correlations(rows, prior_columns)
    posterior_correlations = compute_demand_correlations(rows, posterior_columns)

    # Stage 0.5b 新增：患者级相关性复核。
    prior_patient_correlations = compute_patient_level_correlations(rows, prior_columns)
    posterior_patient_correlations = compute_patient_level_correlations(rows, posterior_columns)

    # Stage 0.5 结果显示 fine + foreground 最可靠；这里优先验证 expected-data entropy。
    preferred_prior_signal = "prior_fine_foreground_expected_data_entropy"
    preferred_posterior_signal = "posterior_fine_foreground_expected_data_entropy"
    prior_acceptance = demand_acceptance(
        slice_correlations=prior_correlations,
        patient_correlations=prior_patient_correlations,
        preferred_signal=preferred_prior_signal,
        slice_threshold=0.50,
        patient_threshold=0.40,
    )
    posterior_acceptance = demand_acceptance(
        slice_correlations=posterior_correlations,
        patient_correlations=posterior_patient_correlations,
        preferred_signal=preferred_posterior_signal,
        slice_threshold=0.50,
        patient_threshold=0.40,
    )

    # 输出 top-10，既方便人工查看，也避免 summary.json 过度庞大；
    # full correlations 仍全部写入 JSON，便于后续分析。
    best_prior = prior_correlations[0] if prior_correlations else None
    best_posterior = posterior_correlations[0] if posterior_correlations else None

    selected_for_quartile: list[str] = []
    if best_prior is not None:
        selected_for_quartile.append(str(best_prior["signal"]))
    if best_posterior is not None:
        selected_for_quartile.append(str(best_posterior["signal"]))

    # 同时固定保留几个我们设计阶段最关心的 fused candidate。
    preferred_columns = [
        "prior_fused_global_composite",
        "prior_fused_foreground_composite",
        "prior_fused_foreground_expected_data_entropy",
        "prior_fine_foreground_expected_data_entropy",
        "prior_middle_foreground_expected_data_entropy",
        "prior_coarse_foreground_expected_data_entropy",
        "prior_fused_boundary_composite",
        "prior_fused_top10_composite",
        "prior_fused_top20_composite",
        "prior_fused_roi_top10_composite",
        "prior_fused_roi_top20_composite",
        "posterior_fused_global_composite",
        "posterior_fused_foreground_composite",
        "posterior_fused_foreground_expected_data_entropy",
        "posterior_fine_foreground_expected_data_entropy",
        "posterior_middle_foreground_expected_data_entropy",
        "posterior_coarse_foreground_expected_data_entropy",
        "posterior_fused_boundary_composite",
    ]
    for column in preferred_columns:
        if column in rows[0] and column not in selected_for_quartile:
            selected_for_quartile.append(column)

    component_metrics = {
        "corr_annotation_diversity_prior_k_eff_pearson": correlation(
            annotation_values,
            [float(row["prior_k_eff"]) for row in rows],
        ),
        "corr_annotation_diversity_prior_k_eff_spearman": correlation(
            annotation_values,
            [float(row["prior_k_eff"]) for row in rows],
            spearman=True,
        ),
        "corr_annotation_diversity_active_component_count_spearman": correlation(
            annotation_values,
            [float(row["prior_active_component_count"]) for row in rows],
            spearman=True,
        ),
        "corr_annotation_diversity_all_component_diversity_pearson": correlation(
            annotation_values,
            [float(row["all_component_iou_distance"]) for row in rows],
        ),
        "corr_annotation_diversity_active_component_diversity_pearson": correlation(
            annotation_values,
            [float(row["active_component_iou_distance"]) for row in rows],
        ),
        "corr_annotation_diversity_weighted_component_diversity_pearson": correlation(
            annotation_values,
            [float(row["weighted_component_iou_distance"]) for row in rows],
        ),
        "corr_annotation_diversity_component_weighted_variance_pearson": correlation(
            annotation_values,
            [float(row["component_weighted_variance"]) for row in rows],
        ),
    }

    return {
        "num_cases": len(rows),
        "annotation_diversity_mean": float(np.mean(annotation_values)),
        "annotation_diversity_std": float(np.std(annotation_values)),
        "annotation_spatial_ambiguity_roi_mean": float(
            np.mean([float(row["annotation_spatial_ambiguity_roi"]) for row in rows])
        ),
        "annotation_disagreement_fraction_mean": float(
            np.mean([float(row["annotation_disagreement_fraction"]) for row in rows])
        ),
        "prior_k_eff_mean": float(
            np.mean([float(row["prior_k_eff"]) for row in rows])
        ),
        "posterior_k_eff_mean": float(
            np.mean([float(row["posterior_k_eff"]) for row in rows])
        ),
        "prior_active_component_count_mean": float(
            np.mean([float(row["prior_active_component_count"]) for row in rows])
        ),
        "active_component_threshold": float(active_threshold),
        "predicted_roi_threshold": float(roi_threshold),
        "component_diagnostics": component_metrics,
        "best_prior_demand_by_spearman": best_prior,
        "best_posterior_demand_by_spearman": best_posterior,
        "prior_demand_correlations": prior_correlations,
        "posterior_demand_correlations": posterior_correlations,
        "prior_patient_level_correlations": prior_patient_correlations,
        "posterior_patient_level_correlations": posterior_patient_correlations,
        "stage_0_5b_prior_acceptance": prior_acceptance,
        "stage_0_5b_posterior_acceptance": posterior_acceptance,
        "stage_0_5b_overall_passed": bool(prior_acceptance["passed"]),
        "stage_0_5b_next_step": (
            "ENTER_STAGE_1_RELATIVE_AMBIGUITY_DEMAND_FEEDBACK"
            if prior_acceptance["passed"]
            else "DO_NOT_TRAIN_STAGE_1_RECHECK_AMBIGUITY_DEMAND"
        ),
        "quartiles": quartile_summary(rows, selected_for_quartile),
    }


# =============================================================================
# 主诊断流程
# =============================================================================

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EG-GM-PUNet v2 Stage 0.5 ambiguity-demand diagnosis"
    )
    parser.add_argument(
        "--config",
        default="configs/eg_gm_punet_lidc.yaml",
        help="第一版 EG-GM-PUNet 配置文件。",
    )
    parser.add_argument(
        "--checkpoint",
        default="result/eg_gm_punet_k5_full/best_val_ged4_eg_gm_punet.pth",
        help="第一版 Full EG-GM-PUNet checkpoint。",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
        help="诊断的数据划分。Stage 0.5b 默认使用 val，避免继续用 test 做机制选择。",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="只运行前 2 个 batch，用于检查接口和输出。",
    )
    parser.add_argument(
        "--active-threshold",
        type=float,
        default=0.02,
        help="prior component probability >= 此阈值才视为 active。默认 0.02。",
    )
    parser.add_argument(
        "--roi-threshold",
        type=float,
        default=0.05,
        help="predicted foreground probability >= 此值进入 top-q lesion-support ROI。",
    )
    parser.add_argument(
        "--top10-fraction",
        type=float,
        default=0.10,
        help="Top-10%% demand 使用的比例。",
    )
    parser.add_argument(
        "--top20-fraction",
        type=float,
        default=0.20,
        help="Top-20%% demand 使用的比例。",
    )
    args = parser.parse_args()

    if not (0.0 <= args.active_threshold <= 1.0):
        raise ValueError("--active-threshold 必须位于 [0,1]。")
    if not (0.0 <= args.roi_threshold <= 1.0):
        raise ValueError("--roi-threshold 必须位于 [0,1]。")
    for fraction in (args.top10_fraction, args.top20_fraction):
        if not (0.0 < fraction <= 1.0):
            raise ValueError("top fraction 必须位于 (0,1]。")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 不存在：{checkpoint_path}")

    config = load_yaml(config_path)
    set_seed(int(config["validation_seed"]))
    device = prepare_device(config.get("device", "cuda"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    model = build_legacy_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    loader = build_lidc_full_dataloader(
        root=config["data_root"],
        split=args.split,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=False,
        random_rater=False,
        rater_index=int(config["validation"]["rater_index"]),
    )

    output_directory = (
        PROJECT_ROOT
        / "result"
        / "eg_gm_punet_v2_diagnostics"
        / "ambiguity_demand_stage05b"
        / args.split
    )
    if args.smoke_test:
        output_directory = output_directory / "smoke_test"
    output_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    fallback_offset = 0

    iterable = islice(loader, 2) if args.smoke_test else loader
    total_batches = min(len(loader), 2) if args.smoke_test else len(loader)

    print("=" * 108)
    print("EG-GM-PUNet v2 Stage 0.5b: Ambiguity Demand Robustness Confirmation")
    print("=" * 108)
    print(f"[INFO] Config          : {config_path}")
    print(f"[INFO] Checkpoint      : {checkpoint_path}")
    print(f"[INFO] Epoch           : {checkpoint.get('epoch')}")
    print(f"[INFO] Device          : {device}")
    print(f"[INFO] Split           : {args.split}")
    print(f"[INFO] Cases           : {len(loader.dataset)}")
    print(f"[INFO] Active threshold: {args.active_threshold}")
    print(f"[INFO] ROI threshold   : {args.roi_threshold}")
    print(f"[INFO] Output          : {output_directory}")

    progress = tqdm(
        iterable,
        total=total_batches,
        desc="v2 ambiguity demand diagnose",
        ncols=150,
    )

    for batch in progress:
        image = batch["image"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        selected_segmentation = batch["mask"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )
        all_segmentations = batch["masks"].to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        batch_size = image.shape[0]
        case_ids = ids_from_batch(
            batch.get("case_id"),
            batch_size,
            args.split,
            fallback_offset,
        )
        patient_ids = ids_from_batch(
            batch.get("patient_id"),
            batch_size,
            "patient",
            fallback_offset,
        )

        # ---------------------------------------------------------------------
        # 1. 构建 prior + posterior state。
        #    selected_segmentation 使用当前 validation.rater_index 指定的固定医生，
        #    与第一版训练/验证中的 posterior 路径保持一致。
        # ---------------------------------------------------------------------
        state = model(
            image=image,
            segmentation=selected_segmentation,
            is_training=True,
        )

        prior_entropy, prior_normalized_entropy, prior_k_eff = distribution_stats(
            state.prior.mixture_probs
        )
        posterior_entropy, posterior_normalized_entropy, posterior_k_eff = (
            distribution_stats(state.posterior.mixture_probs)
        )

        # ---------------------------------------------------------------------
        # 2. posterior probe：对应旧版训练 feedback 实际使用的 deterministic probe。
        # ---------------------------------------------------------------------
        posterior_evidence = model.feedback_probe(state)
        posterior_demands = spatial_evidence_demands(
            posterior_evidence,
            roi_threshold=float(args.roi_threshold),
            top_fractions=(
                float(args.top10_fraction),
                float(args.top20_fraction),
            ),
        )

        # ---------------------------------------------------------------------
        # 3. prior probe：去掉 posterior，完全由 image-conditioned prior 决定。
        #    这是无需医生标注也能获得的部署可用 demand signal。
        # ---------------------------------------------------------------------
        prior_only_state = MultiScaleGroupedGMMState(
            image_encoder_features=state.image_encoder_features,
            prior=state.prior,
            posterior=None,
        )
        prior_evidence = model.feedback_probe(prior_only_state)
        prior_demands = spatial_evidence_demands(
            prior_evidence,
            roi_threshold=float(args.roi_threshold),
            top_fractions=(
                float(args.top10_fraction),
                float(args.top20_fraction),
            ),
        )

        # ---------------------------------------------------------------------
        # 4. 对每个 prior component mean 解码，重新分析 component collapse。
        # ---------------------------------------------------------------------
        component_masks, component_probabilities = component_mean_outputs(
            model,
            prior_only_state,
        )

        for batch_index in range(batch_size):
            current_all_masks = all_segmentations[batch_index]
            current_prior_probs = state.prior.mixture_probs[batch_index]

            current_annotation_diversity = annotation_diversity(
                current_all_masks
            )
            (
                current_spatial_ambiguity,
                current_disagreement_fraction,
            ) = annotation_spatial_ambiguity(current_all_masks)

            active_indices = torch.where(
                current_prior_probs >= float(args.active_threshold)
            )[0].detach().cpu().tolist()

            all_component_distance = mean_pairwise_iou_distance(
                component_masks[batch_index]
            )
            active_component_distance = mean_pairwise_iou_distance(
                component_masks[batch_index],
                active_indices,
            )
            all_component_dice = mean_pairwise_dice(
                component_masks[batch_index]
            )
            active_component_dice = mean_pairwise_dice(
                component_masks[batch_index],
                active_indices,
            )
            weighted_component_distance = weighted_pairwise_iou_distance(
                component_masks[batch_index],
                current_prior_probs,
            )
            component_variance = weighted_component_variance(
                component_probabilities[batch_index],
                current_prior_probs,
            )

            sorted_probs = torch.sort(
                current_prior_probs,
                descending=True,
            ).values
            top2_probability_mass = float(
                sorted_probs[: min(2, sorted_probs.numel())].sum().item()
            )

            row: dict[str, Any] = {
                "case_id": case_ids[batch_index],
                "patient_id": patient_ids[batch_index],
                "annotation_diversity": current_annotation_diversity,
                "annotation_spatial_ambiguity_roi": current_spatial_ambiguity,
                "annotation_disagreement_fraction": current_disagreement_fraction,
                "prior_entropy": float(prior_entropy[batch_index].item()),
                "prior_normalized_entropy": float(
                    prior_normalized_entropy[batch_index].item()
                ),
                "prior_k_eff": float(prior_k_eff[batch_index].item()),
                "posterior_entropy": float(posterior_entropy[batch_index].item()),
                "posterior_normalized_entropy": float(
                    posterior_normalized_entropy[batch_index].item()
                ),
                "posterior_k_eff": float(posterior_k_eff[batch_index].item()),
                "prior_active_component_count": int(len(active_indices)),
                "prior_top2_probability_mass": top2_probability_mass,
                "all_component_pairwise_dice": all_component_dice,
                "active_component_pairwise_dice": active_component_dice,
                "all_component_iou_distance": all_component_distance,
                "active_component_iou_distance": active_component_distance,
                "weighted_component_iou_distance": weighted_component_distance,
                "component_weighted_variance": component_variance,
            }

            # 保存 K 个 component 的 prior/posterior probability，便于后续画图。
            for component_index in range(model.num_components):
                row[f"prior_component_{component_index}_prob"] = float(
                    state.prior.mixture_probs[
                        batch_index, component_index
                    ].item()
                )
                row[f"posterior_component_{component_index}_prob"] = float(
                    state.posterior.mixture_probs[
                        batch_index, component_index
                    ].item()
                )

            # 保存所有 prior spatial-aware demand candidates。
            for name, value in prior_demands.items():
                row[f"prior_{name}"] = float(value[batch_index].item())

            # 保存所有 posterior spatial-aware demand candidates。
            for name, value in posterior_demands.items():
                row[f"posterior_{name}"] = float(value[batch_index].item())

            rows.append(row)

        fallback_offset += batch_size

    if not rows:
        raise RuntimeError("没有生成任何 ambiguity-demand 诊断结果。")

    # =========================================================================
    # 保存逐病例 CSV
    # =========================================================================
    csv_path = output_directory / "ambiguity_demand_stage05b_per_case.csv"
    with open(
        csv_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # =========================================================================
    # 汇总相关性、quartile、component diagnostics
    # =========================================================================
    summary = summarize(
        rows,
        active_threshold=float(args.active_threshold),
        roi_threshold=float(args.roi_threshold),
    )
    summary.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "config": str(config_path),
            "split": args.split,
            "smoke_test": bool(args.smoke_test),
        }
    )

    summary_path = output_directory / "ambiguity_demand_stage05b_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # =========================================================================
    # 控制台打印最关键结论
    # =========================================================================
    print("\n" + "=" * 108)
    print("[SUMMARY] Stage 0.5b Ambiguity Demand Robustness Confirmation")
    print("=" * 108)
    print(f"Cases                     : {summary['num_cases']}")
    print(f"Annotation diversity mean : {summary['annotation_diversity_mean']:.6f}")
    print(f"Prior K_eff mean          : {summary['prior_k_eff_mean']:.6f}")
    print(f"Posterior K_eff mean      : {summary['posterior_k_eff_mean']:.6f}")
    print(
        "Active components mean     : "
        f"{summary['prior_active_component_count_mean']:.6f}"
    )

    component_summary = summary["component_diagnostics"]
    print("\n[COMPONENT DIAGNOSTICS]")
    for key, value in component_summary.items():
        rendered = "N/A" if value is None else f"{float(value):.6f}"
        print(f"  {key:66s}: {rendered}")

    print("\n[TOP PRIOR DEMAND SIGNALS | sorted by Spearman]")
    for index, item in enumerate(summary["prior_demand_correlations"][:10], start=1):
        pearson = item.get("pearson")
        spearman_value = item.get("spearman")
        print(
            f"  {index:2d}. {item['signal']:<48s} "
            f"Pearson={str(None if pearson is None else round(float(pearson), 6)):>10s}  "
            f"Spearman={str(None if spearman_value is None else round(float(spearman_value), 6)):>10s}"
        )

    print("\n[TOP POSTERIOR DEMAND SIGNALS | sorted by Spearman]")
    for index, item in enumerate(
        summary["posterior_demand_correlations"][:10],
        start=1,
    ):
        pearson = item.get("pearson")
        spearman_value = item.get("spearman")
        print(
            f"  {index:2d}. {item['signal']:<48s} "
            f"Pearson={str(None if pearson is None else round(float(pearson), 6)):>10s}  "
            f"Spearman={str(None if spearman_value is None else round(float(spearman_value), 6)):>10s}"
        )

    best_prior = summary.get("best_prior_demand_by_spearman")
    best_posterior = summary.get("best_posterior_demand_by_spearman")

    print("\n[PATIENT-LEVEL TOP PRIOR DEMAND SIGNALS | sorted by Spearman]")
    for index, item in enumerate(summary["prior_patient_level_correlations"][:10], start=1):
        pearson = item.get("pearson")
        spearman_value = item.get("spearman")
        print(
            f"  {index:2d}. {item['signal']:<56s} "
            f"Pearson={str(None if pearson is None else round(float(pearson), 6)):>10s}  "
            f"Spearman={str(None if spearman_value is None else round(float(spearman_value), 6)):>10s}"
        )

    print("\n[STAGE 0.5b ACCEPTANCE]")
    prior_acceptance = summary["stage_0_5b_prior_acceptance"]
    prior_slice = prior_acceptance.get("slice_level") or {}
    prior_patient = prior_acceptance.get("patient_level") or {}
    print(f"  Preferred signal        : {prior_acceptance['preferred_signal']}")
    print(f"  Slice-level Spearman    : {prior_slice.get('spearman')}")
    print(f"  Patient-level Spearman  : {prior_patient.get('spearman')}")
    print(f"  Passed                  : {prior_acceptance['passed']}")
    print(f"  Decision                : {prior_acceptance['decision']}")
    print(f"  Next step               : {summary['stage_0_5b_next_step']}")

    print("\n[RECOMMENDATION CANDIDATES]")
    if isinstance(best_prior, dict):
        print(
            "  Best prior signal     : "
            f"{best_prior['signal']} | "
            f"Pearson={best_prior.get('pearson')} | "
            f"Spearman={best_prior.get('spearman')}"
        )
    if isinstance(best_posterior, dict):
        print(
            "  Best posterior signal : "
            f"{best_posterior['signal']} | "
            f"Pearson={best_posterior.get('pearson')} | "
            f"Spearman={best_posterior.get('spearman')}"
        )

    print("\n[QUARTILES]")
    for item in summary["quartiles"]:
        text = (
            f"  Q{item['quartile']}: n={item['num_cases']}, "
            f"D_ann={item['annotation_diversity_mean']:.4f}, "
            f"K_eff={item['prior_k_eff_mean']:.4f}, "
            f"activeK={item['prior_active_component_count_mean']:.3f}, "
            f"D_active={item['active_component_diversity_mean']:.4f}, "
            f"D_weighted={item['weighted_component_diversity_mean']:.4f}"
        )
        print(text)

    print("\n" + "=" * 108)
    print(f"[DONE] CSV     : {csv_path}")
    print(f"[DONE] Summary : {summary_path}")
    print("=" * 108)


if __name__ == "__main__":
    main()
