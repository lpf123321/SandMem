"""配置加载与项目路径工具。

约定: 不在代码中硬编码个人绝对路径; 路径、模型 ID、dtype、采样与阈值均来自 configs/*.yaml。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置; 默认 configs/default.yaml。"""
    path = Path(path) if path else (PROJECT_ROOT / "configs" / "default.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path: str | Path, base: Path | None = None) -> Path:
    """把相对路径解析为基于项目根目录(或给定 base)的绝对路径。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or PROJECT_ROOT) / p


def work_dir(cfg: dict[str, Any]) -> Path:
    d = resolve(cfg.get("paths", {}).get("work_dir", "work"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    """创建并返回配置中声明的中间产物目录。"""
    base = work_dir(cfg)
    out: dict[str, Path] = {}
    for name in cfg.get("paths", {}).get("dirs", []):
        p = base / name
        p.mkdir(parents=True, exist_ok=True)
        out[name] = p
    return out


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
