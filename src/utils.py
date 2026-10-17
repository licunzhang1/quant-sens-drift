"""Shared utilities: JSON-safe conversion, random seeding, config loading."""

from __future__ import annotations

import json
import math
import pathlib
import random
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "jsonable",
    "dump_json",
    "seed_everything",
    "load_config",
    "discover_seed_roots",
    "find_result_file",
]


def discover_seed_roots(results_dir: "pathlib.Path") -> Dict[int, "pathlib.Path"]:
    """Map a seed number to its results root.

    Two layouts are supported, because the project used both:

    * ``results/seed{N}/`` -- the multi-seed layout, preferred;
    * ``results/`` -- the earlier flat single-seed layout. It is reported
      under seed ``0``, since the seed cannot be recovered from the path.

    Callers that only need to *find* a file should use :func:`find_result_file`
    rather than walking this mapping themselves.
    """
    results_dir = pathlib.Path(results_dir)
    out: Dict[int, pathlib.Path] = {}
    for d in sorted(results_dir.glob("seed*")):
        if d.is_dir() and d.name[4:].isdigit():
            out[int(d.name[4:])] = d
    if (results_dir / "sensitivity").is_dir() or (results_dir / "equal_error").is_dir():
        out.setdefault(0, results_dir)
    return out


def find_result_file(
    results_dir: "pathlib.Path",
    sub: str,
    filename: str,
    seed: Optional[int] = None,
) -> Optional["pathlib.Path"]:
    """Locate ``<sub>/<filename>`` inside a results tree.

    Search order: the requested seed directory, then every discovered seed
    directory in ascending order, then the flat layout. Returns ``None`` when
    the file does not exist anywhere.

    This exists because the project moved from a flat ``results/sensitivity/``
    layout to ``results/seed{N}/sensitivity/``; scripts that hard-coded the
    flat path silently stopped finding their inputs after the move.
    """
    results_dir = pathlib.Path(results_dir)
    if seed is not None:
        p = results_dir / f"seed{seed}" / sub / filename
        if p.exists():
            return p
    for _seed, root in discover_seed_roots(results_dir).items():
        p = root / sub / filename
        if p.exists():
            return p
    p = results_dir / sub / filename
    return p if p.exists() else None


def jsonable(obj: Any) -> Any:
    """Recursively convert an object into a strictly JSON-serialisable form.

    Handles three common numpy pitfalls:
      * ``np.bool_`` / ``np.int64`` / ``np.float64`` are not native JSON types
      * ``NaN`` / ``Inf`` are not valid JSON (``allow_nan=False`` raises)
      * ``np.ndarray`` has to be turned into a list

    Non-finite floats become ``None`` rather than an illegal ``NaN`` literal.
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
    """Write strict JSON to disk (no NaN/Inf literals allowed)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(obj), indent=indent, ensure_ascii=False, allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return path


def seed_everything(seed: int = 42) -> None:
    """Seed every random source. **Call it before each experiment step**, or the
    results will not be reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        pass


def load_config(path: str | pathlib.Path) -> Dict[str, Any]:
    """Read a YAML config. If pyyaml is missing, fall back to a minimal parser
    (only the subset used by this project is supported)."""
    path = pathlib.Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml(text)


def _mini_yaml(text: str) -> Dict[str, Any]:
    """Minimal YAML subset parser (fallback only for when pyyaml is missing)."""
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
    """Split on ``sep`` but **ignore separators inside brackets**.

    This is the crux of the mini-YAML fallback parser: the first comma in
    ``{snr_db: [-2, 6], cfo: 0.02}`` sits inside square brackets and must not be
    taken as an outer separator. A naive split(",") would cut it into
    ``snr_db: [-2`` and ``6]`` -- silently producing a wrong config, which is
    more dangerous than raising an error outright.
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
