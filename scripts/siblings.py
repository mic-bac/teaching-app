"""
Fetch and inspect the sibling lesson repos listed in ``siblings.toml``.

    python scripts/siblings.py fetch    clone missing siblings on the matching branch
    python scripts/siblings.py data     download the public datasets listed there
    python scripts/siblings.py status   show each local sibling's branch vs. expected

Branch rule (siblings are maintained first): the app's ``develop`` runs against
each sibling's ``develop``, ``main`` against ``main``. The app branch is taken
from, in order: ``$SIBLING_BRANCH``, ``$RAILWAY_GIT_BRANCH`` (set by Railway),
the local git checkout, else the ``default`` in siblings.toml.

Standard library only — it runs before (and independently of) the app's env.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "siblings.toml"


def load() -> dict:
    return tomllib.loads(MANIFEST.read_text())


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def app_branch() -> str | None:
    """Which branch of the app is running — env first (deployments), then git."""
    for variable in ("SIBLING_BRANCH", "RAILWAY_GIT_BRANCH"):
        if os.environ.get(variable):
            return os.environ[variable]
    try:
        return git("rev-parse", "--abbrev-ref", "HEAD", cwd=ROOT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None  # no .git in a container build context


def sibling_branch(spec: dict, branch: str | None, mapping: dict) -> str:
    """The branch a sibling should be on for the given app branch."""
    if "only_branch" in spec:
        return spec["only_branch"]
    return mapping.get(branch or "", mapping["default"])


def fetch() -> None:
    """Clone every sibling that is not checked out yet, on the matching branch.

    A failed clone is reported but not fatal: the affected page degrades (it
    says the sibling is unavailable) instead of the whole app refusing to start.
    """
    manifest = load()
    branch = app_branch()
    print(f"  app branch: {branch or 'unknown'}")
    for name, spec in manifest["siblings"].items():
        target = ROOT / name
        if target.exists():
            print(f"  = {name}: already present, left as is")
            continue
        wanted = sibling_branch(spec, branch, manifest["branches"])
        try:
            git("clone", "-q", "--depth", "1", "--branch", wanted, spec["url"], str(target))
            print(f"  + {name}: {wanted} @ {git('rev-parse', '--short', 'HEAD', cwd=target)}")
        except subprocess.CalledProcessError as exc:
            print(f"  ! {name}: could not clone branch {wanted!r} — {exc.stderr.strip()}", file=sys.stderr)


def data() -> None:
    """Download the public datasets, skipping any file that already exists."""
    for item in load().get("data", []):
        wanted = {member: ROOT / dest for member, dest in item["extract"].items()}
        if all(path.exists() for path in wanted.values()):
            print(f"  = {item['name']}: already present")
            continue
        siblings = {ROOT / Path(dest).parts[0] for dest in item["extract"].values()}
        if not all(sibling.exists() for sibling in siblings):
            print(f"  ! {item['name']}: its sibling is missing, skipped", file=sys.stderr)
            continue
        try:
            with urllib.request.urlopen(item["url"], timeout=120) as response:
                archive = zipfile.ZipFile(io.BytesIO(response.read()))
            for member, path in wanted.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(member))
            print(f"  + {item['name']}")
        except OSError as exc:
            print(f"  ! {item['name']}: download failed — {exc}", file=sys.stderr)


def status() -> None:
    """Local siblings vs. the branch they should be on for the app's branch."""
    manifest = load()
    branch = app_branch()
    print(f"  app branch: {branch or 'unknown'}")
    for name, spec in manifest["siblings"].items():
        repo = ROOT / name
        wanted = sibling_branch(spec, branch, manifest["branches"])
        if not (repo / ".git").exists():
            print(f"  ✗ {name:<16} missing (expected {wanted}) — run `make siblings`")
            continue
        local = git("branch", "--show-current", cwd=repo) or "detached"
        dirty = ", uncommitted changes" if git("status", "--porcelain", cwd=repo) else ""
        mark = "✓" if local == wanted else "≠"
        print(f"  {mark} {name:<16} on {local:<8} expected {wanted}{dirty}")


if __name__ == "__main__":
    commands = {"fetch": fetch, "data": data, "status": status}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()
