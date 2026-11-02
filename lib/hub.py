"""Parallel model download via aria2c into a plain directory."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def hf_token() -> str | None:
    """Token from the environment, else the file the CLI writes on login."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    if path.exists():
        tok = path.read_text().strip()
        return tok or None
    return None


def _wanted(name: str) -> bool:
    if any(bad in name for bad in config.DOWNLOAD_DENY_SUBSTRINGS):
        return False
    return name.endswith(config.DOWNLOAD_ALLOW_SUFFIXES)


@dataclass
class RemoteFile:
    name: str
    size: int | None


@dataclass
class Plan:
    repo_id: str
    revision: str
    dest: Path
    todo: list[RemoteFile]
    skipped: list[RemoteFile]

    @property
    def bytes_todo(self) -> int:
        return sum(f.size or 0 for f in self.todo)


def plan_download(
    repo_id: str, dest: Path | None = None, revision: str = "main"
) -> Plan:
    """List the files we need and drop the ones already complete on disk."""
    api = HfApi(token=hf_token())
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    dest = dest or config.MODELS_DIR / repo_id.split("/")[-1]

    todo: list[RemoteFile] = []
    skipped: list[RemoteFile] = []
    for sib in info.siblings:
        if not _wanted(sib.rfilename):
            continue
        rf = RemoteFile(sib.rfilename, sib.size)
        local = dest / sib.rfilename
        done = (
            local.exists()
            and rf.size is not None
            and local.stat().st_size == rf.size
            and not local.with_suffix(local.suffix + ".aria2").exists()
        )
        (skipped if done else todo).append(rf)

    return Plan(repo_id, info.sha or revision, dest, todo, skipped)


def _write_aria_input(plan: Plan, token: str | None, workdir: Path) -> Path:
    """aria2c input file; the auth header goes here rather than in argv."""
    lines: list[str] = []
    for rf in plan.todo:
        url = f"https://huggingface.co/{plan.repo_id}/resolve/{plan.revision}/{rf.name}"
        target = plan.dest / rf.name
        lines.append(url)
        lines.append(f"  dir={target.parent}")
        lines.append(f"  out={target.name}")
        if token:
            lines.append(f"  header=Authorization: Bearer {token}")

    path = workdir / f"aria2-{plan.repo_id.replace('/', '_')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)
    return path


def assert_access(repo_id: str, revision: str, probe_file: str) -> None:
    """Fail fast if the token cannot actually fetch file *content*."""
    token = hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{probe_file}"
    resp = requests.head(url, headers=headers, allow_redirects=False, timeout=20)
    if resp.status_code in (401, 403):
        raise PermissionError(
            f"{repo_id}: HTTP {resp.status_code} fetching {probe_file}.\n"
            f"The repo is gated and this token has no grant "
            f"({'no token found' if not token else 'token present but not authorised'})"
            f".\nAccept the license at https://huggingface.co/{repo_id} "
            f"or point config.py at an ungated mirror."
        )


def download(
    repo_id: str,
    dest: Path | None = None,
    revision: str = "main",
    workdir: Path | None = None,
) -> Path:
    """Fetch a repo's config/tokenizer/safetensors into ``dest``. Idempotent."""
    if shutil.which("aria2c") is None:
        raise RuntimeError("aria2c not found on PATH (brew install aria2)")

    plan = plan_download(repo_id, dest, revision)
    plan.dest.mkdir(parents=True, exist_ok=True)

    if plan.todo:
        weights = [f for f in plan.todo if f.name.endswith(".safetensors")]
        assert_access(repo_id, plan.revision, (weights or plan.todo)[0].name)

    if not plan.todo:
        print(f"[hub] {repo_id}: already complete ({len(plan.skipped)} files)")
        _write_meta(plan)
        return plan.dest

    gb = plan.bytes_todo / 1e9
    print(
        f"[hub] {repo_id} @ {plan.revision[:8]}\n"
        f"      {len(plan.todo)} files, {gb:.2f} GB to fetch"
        f" ({len(plan.skipped)} already present) -> {plan.dest}"
    )

    workdir = workdir or plan.dest / ".aria2"
    input_file = _write_aria_input(plan, hf_token(), workdir)
    try:
        cmd = [
            "aria2c",
            f"--input-file={input_file}",
            f"--max-connection-per-server={config.ARIA2_CONNECTIONS_PER_FILE}",
            f"--split={config.ARIA2_CONNECTIONS_PER_FILE}",
            f"--min-split-size={config.ARIA2_MIN_SPLIT}",
            f"--max-concurrent-downloads={config.ARIA2_CONCURRENT_FILES}",
            "--continue=true",
            "--auto-file-renaming=false",
            "--allow-overwrite=true",
            "--file-allocation=none",
            "--console-log-level=warn",
            "--summary-interval=15",
            "--retry-wait=2",
            "--max-tries=5",
        ]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"aria2c exited {proc.returncode} for {repo_id}. "
                "A 401/403 here means the gated-repo grant is not live for this token."
            )
    finally:
        input_file.unlink(missing_ok=True)
        try:
            workdir.rmdir()
        except OSError:
            pass

    remaining = plan_download(repo_id, plan.dest, revision).todo
    if remaining:
        raise RuntimeError(
            f"{repo_id}: {len(remaining)} files still incomplete: "
            + ", ".join(f.name for f in remaining[:5])
        )

    _write_meta(plan)
    print(f"[hub] {repo_id}: done")
    return plan.dest


def _write_meta(plan: Plan) -> None:
    (plan.dest / ".download_meta.json").write_text(
        json.dumps({"repo_id": plan.repo_id, "revision": plan.revision}, indent=2)
    )


def local_path(repo_id: str) -> Path:
    """Where ``repo_id`` lives locally, whether or not it is downloaded yet."""
    return config.MODELS_DIR / repo_id.split("/")[-1]


def is_complete(repo_id: str) -> tuple[bool, str]:
    """Whether the local copy is safe to load."""
    p = local_path(repo_id)
    if not (p / "config.json").exists():
        return False, "no local copy"
    partial = sorted(f.name for f in p.rglob("*.aria2") if f.is_file())
    if partial:
        return False, f"download in progress: {', '.join(partial[:3])}"
    if not (p / ".download_meta.json").exists():
        return False, "no .download_meta.json; the download never completed"
    return True, "complete"


def resolve(repo_id: str) -> str:
    """Path to a verified-complete local copy, else the hub id."""
    p = local_path(repo_id)
    if not (p / "config.json").exists():
        return repo_id
    ok, why = is_complete(repo_id)
    if not ok:
        raise RuntimeError(
            f"{repo_id}: local copy at {p} is not usable ({why}). "
            f"Wait for scripts/00_download.py to finish, or delete the directory "
            f"to force a clean fetch."
        )
    return str(p)
