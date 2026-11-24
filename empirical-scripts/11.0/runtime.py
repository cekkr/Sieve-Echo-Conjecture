"""Run infrastructure for Sieve Echo v11: paths, Evolvo bootstrap, git and package info, JSON IO.

Owns nothing scientific. Every stage writes through these helpers so that each result file can be
traced back to the code (root + Evolvo SHAs), the configuration and the seed that produced it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

V11_DIR = Path(__file__).resolve().parent
REPO_ROOT = V11_DIR.parents[1]
EVOLVO_ROOT = REPO_ROOT / "libs" / "evolvo"
EVOLVO_SRC = EVOLVO_ROOT / "src"
RESULTS_DIR = V11_DIR / "results"

TRACKED_PACKAGES = ("numpy", "scipy", "sklearn", "sympy", "torch", "matplotlib", "psutil")


class EvolvoMissingError(RuntimeError):
    pass


def bootstrap_evolvo():
    """Import `evolvo` from the libs/evolvo submodule and nowhere else (AGENTS.md directive D2)."""
    if not (EVOLVO_SRC / "evolvo" / "__init__.py").is_file():
        raise EvolvoMissingError(
            f"Evolvo submodule not found at {EVOLVO_SRC}. Run `git submodule update --init --recursive` "
            "from the repository root. v11 never falls back to the historical evolvo*.py copies."
        )
    src = str(EVOLVO_SRC)
    if src in sys.path:
        sys.path.remove(src)
    sys.path.insert(0, src)
    evolvo = importlib.import_module("evolvo")
    origin = Path(evolvo.__file__).resolve()
    if EVOLVO_SRC.resolve() not in origin.parents:
        raise EvolvoMissingError(f"`evolvo` resolved to {origin}, not to the submodule at {EVOLVO_SRC}.")
    return evolvo


def _git(cwd: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(path: Path) -> dict[str, Any]:
    """HEAD SHA plus the paths that differ from it (tracked changes and untracked files).

    Run outputs under `empirical-scripts/11.0/results/` and `.DS_Store` files are not counted.
    """
    sha = _git(path, "rev-parse", "HEAD")
    status = _git(path, "status", "--porcelain")
    if status is None:
        return {"sha": sha, "dirty": None, "changed_paths": None}
    changed = []
    for line in status.splitlines():
        rel = line[3:].strip().strip('"')
        if rel.startswith("empirical-scripts/11.0/results/") or rel.endswith(".DS_Store"):
            continue
        changed.append(line)
    return {"sha": sha, "dirty": bool(changed), "changed_paths": changed[:50]}


def code_sha256() -> str:
    """Hash of the v11 source files, so a run of uncommitted code is still identifiable."""
    h = hashlib.sha256()
    for f in sorted(V11_DIR.rglob("*.py")):
        if "results" in f.relative_to(V11_DIR).parts:
            continue
        h.update(str(f.relative_to(V11_DIR)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def repo_relative(path: Path | str) -> str:
    """Path relative to the repository root when inside it (keeps home directories out of results)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return p.name


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except ImportError:
            versions[name] = None
    return versions


def torch_devices() -> dict[str, bool]:
    try:
        import torch
    except ImportError:
        return {"torch": False}
    return {
        "torch": True,
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": repo_relative(sys.executable),
        "platform": platform.platform(),
        "packages": package_versions(),
        "devices": torch_devices(),
        "git_root": git_info(REPO_ROOT),
        "git_evolvo": git_info(EVOLVO_ROOT),
        "v11_code_sha256": code_sha256(),
    }


def _json_default(obj: Any):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return _clean_float(float(obj))
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [_json_default(x) if not isinstance(x, (int, str)) else x for x in obj.tolist()]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _clean_float(x: float):
    return None if (math.isnan(x) or math.isinf(x)) else x


def _clean(obj: Any):
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically; NaN/inf become null so files stay standard JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(data), indent=2, default=_json_default, sort_keys=False) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(_clean(obj), sort_keys=True, default=_json_default).encode())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
