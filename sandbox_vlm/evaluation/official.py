"""VSI-Bench 官方评测口径 (与 thinking-in-space 的 lmms_eval 实现一致)。

- MCA(选项题): fuzzy_matching(pred)=pred.split(' ')[0].rstrip('.').strip(); exact_match。
- NA(数值题): MRA = mean over conf in linspace(.5,.95,11) of (|p-g|/|g| <= 1-conf);
  预测无法解析为 float 时记 0。
- 聚合: 各题型取均值; rel_direction 的 easy/medium/hard 先各取均值再平均为一项;
  overall = 8 类任务指标的均值。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard", "object_rel_distance",
    "route_planning", "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance", "object_counting",
    "object_size_estimation", "room_size_estimation",
]
TASK_GROUPS = [
    "object_counting", "object_abs_distance", "object_size_estimation",
    "room_size_estimation", "object_rel_distance", "object_rel_direction",
    "route_planning", "obj_appearance_order",
]


def fuzzy_matching(pred: str) -> str:
    return str(pred).split(" ")[0].rstrip(".").strip()


def exact_match(pred: str, target: str) -> float:
    return 1.0 if str(pred).lower() == str(target).lower() else 0.0


def abs_dist_norm(pred: float, target: float) -> float:
    return abs(pred - target) / target if target != 0 else (0.0 if pred == 0 else 1.0)


def mean_relative_accuracy(pred: float, target: float,
                           start: float = 0.5, end: float = 0.95,
                           interval: float = 0.05) -> float:
    num_pts = (end - start) / interval + 2
    conf = np.linspace(start, end, int(num_pts))
    return float((abs_dist_norm(pred, target) <= 1 - conf).mean())


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def score_one(q_type: str, pred_raw: str, gt: str):
    """返回 (metric_name, value); metric_name in {'accuracy','MRA'}。"""
    if q_type in MCA_QUESTION_TYPES:
        return "accuracy", exact_match(fuzzy_matching(pred_raw), gt)
    if q_type in NA_QUESTION_TYPES:
        p = _to_float(fuzzy_matching(pred_raw))
        g = _to_float(gt)
        if p is None or g is None:
            return "MRA", 0.0
        return "MRA", mean_relative_accuracy(p, g)
    return None, None


def aggregate_official(records: list[dict]) -> dict:
    """records: [{q_type, pred_raw, gt}] -> 官方风格指标(×100)。"""
    per = defaultdict(list)
    for r in records:
        m, v = score_one(r["q_type"], r.get("pred_raw", ""), r["gt"])
        if m:
            per[r["q_type"]].append(v)
    type_mean = {t: float(np.mean(v)) for t, v in per.items() if v}

    out = {}
    for t in ["object_counting", "object_abs_distance", "object_size_estimation",
              "room_size_estimation"]:
        if t in type_mean:
            out[f"{t}_MRA"] = type_mean[t] * 100
    for t in ["object_rel_distance", "route_planning", "obj_appearance_order"]:
        if t in type_mean:
            out[f"{t}_accuracy"] = type_mean[t] * 100
    rd = [type_mean.get("object_rel_direction_easy"),
          type_mean.get("object_rel_direction_medium"),
          type_mean.get("object_rel_direction_hard")]
    rd = [x for x in rd if x is not None]
    if rd:
        out["object_rel_direction_accuracy"] = float(np.mean(rd)) * 100

    group_vals = []
    for g in TASK_GROUPS:
        for k in (f"{g}_accuracy", f"{g}_MRA"):
            if k in out:
                group_vals.append(out[k])
                break
    out["overall"] = float(np.mean(group_vals)) if group_vals else 0.0
    out["n_records"] = len(records)
    out["n_by_type"] = {t: len(v) for t, v in per.items()}
    return out
