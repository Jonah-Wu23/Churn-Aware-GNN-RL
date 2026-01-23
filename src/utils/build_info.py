"""Build/version identification utilities."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import subprocess
from typing import Iterable, Optional


DEFAULT_BUILD_FILES = [
    Path("src/env/gym_env.py"),
    Path("src/train/dqn.py"),
    Path("scripts/run_realtime_viz.py"),
    Path("src/utils/reward_hack_alerts.py"),
    Path("configs/manhattan_curriculum_v13.yaml"),
]


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def _hash_files(repo_root: Path, files: Iterable[Path]) -> str:
    hasher = sha256()
    for rel_path in files:
        path = (repo_root / rel_path).resolve()
        hasher.update(str(rel_path).encode("ascii", "ignore"))
        if not path.exists():
            hasher.update(b"MISSING")
            continue
        data = path.read_bytes()
        hasher.update(data)
    return hasher.hexdigest()


def get_build_id(repo_root: Optional[Path] = None, files: Optional[Iterable[Path]] = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[2]
    commit = _git_commit(root)
    if commit:
        return commit
    file_list = list(files) if files is not None else DEFAULT_BUILD_FILES
    return _hash_files(root, file_list)


def write_build_id(path: Path, build_id: Optional[str] = None) -> Path:
    target = Path(path)
    build_id = build_id or get_build_id()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{build_id}\n", encoding="ascii")
    return target
