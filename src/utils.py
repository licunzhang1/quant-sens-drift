"""通用工具：可序列化转换、随机种子、配置读取。"""

from __future__ import annotations

import json
import math
import pathlib
import random
from typing import Any, Dict

import numpy as np

__all__ = ["jsonable", "dump_json", "seed_everything", "load_config"]


def jsonable(obj: Any) -> Any:
    """递归把对象转成严格可 JSON 序列化的形式。

    处理三件 numpy 常见的坑：
      * ``np.bool_`` / ``np.int64`` / ``np.float64`` 不是 JSON 原生类型
      * ``NaN`` / ``Inf`` 不是合法 JSON（``allow_nan=False`` 会直接报错）
      * ``np.ndarray`` 需要转 list

    非有限浮点会被转成 ``None``，而不是写出非法的 ``NaN`` 字面量。
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return str(obj)


def dump_json(path: str | pathlib.Path, obj: Any, indent: int = 2) -> pathlib.Path:
    """严格 JSON 落盘（不允许 NaN/Inf 字面量）。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(obj), indent=indent, ensure_ascii=False, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return path


def seed_everything(seed: int = 42) -> None:
    """固定所有随机源。**每一步实验前都要调**，否则结果不可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch 是可选的
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        pass


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    """读 YAML 配置。若没装 pyyaml，退回到一个极简解析（只支持本项目用到的子集）。"""
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def _mini_yaml(text: str) -> Dict[str, Any]:
    """极简 YAML 子集解析（仅用于 pyyaml 缺失时的兜底）。"""
    import re

    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        rest = rest.split("#")[0].strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce(rest)
    return root


def _split_top(s: str, sep: str = ",") -> list:
    """按 ``sep`` 切分，但**忽略括号内的分隔符**。

    这是 mini-YAML 回退解析器的关键：``{snr_db: [-2, 6], cfo: 0.02}`` 里的
    第一个逗号在方括号内，不能当作外层分隔符。朴素 split(",") 会把它切成
    ``snr_db: [-2`` 和 ``6]`` —— 静默产生错误配置，比直接报错更危险。
    """
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "[{(":
            depth += 1
            cur.append(ch)
        elif ch in "]})":
            depth -= 1
            cur.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _coerce(s: str) -> Any:
    if s in ("null", "~", "None", ""):
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_coerce(v.strip()) for v in _split_top(inner) if v.strip()]
    if s.startswith("{") and s.endswith("}"):
        inner = s[1:-1].strip()
        out: Dict[str, Any] = {}
        if not inner:
            return out
        for part in _split_top(inner):
            if ":" in part:
                k, _, v = part.partition(":")
                out[k.strip()] = _coerce(v.strip())
        return out
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip("'\"")
