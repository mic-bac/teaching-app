"""
The deployment fetches siblings from ``siblings.toml`` by branch — so every
sibling the app reads from must be listed there, or that page silently breaks
on Railway; and the develop->develop / main->main rule must resolve correctly.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOT_SIBLINGS = {"data"}  # the app's own folder


def manifest() -> dict:
    return tomllib.loads((ROOT / "siblings.toml").read_text())


def script():
    spec = importlib.util.spec_from_file_location("siblings_script", ROOT / "scripts" / "siblings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_sibling_the_app_reads_is_listed():
    referenced = set()
    for path in [*ROOT.glob("pages/*.py"), *ROOT.glob("utils/*.py"), ROOT / "Home.py"]:
        text = path.read_text()
        referenced |= set(re.findall(r'(?:parents\[1\]|ROOT) / "([a-z-]+)"', text))
        referenced |= set(re.findall(r'load_sibling_(?:module|package)\(\s*"([a-z-]+)"', text))
    missing = referenced - NOT_SIBLINGS - set(manifest()["siblings"])
    assert not missing, f"add these to siblings.toml, or the deployed app lacks them: {missing}"


def test_urls_are_https_so_the_container_needs_no_ssh_key():
    for name, spec in manifest()["siblings"].items():
        assert spec["url"].startswith("https://"), name


def test_develop_tracks_develop_and_main_tracks_main():
    siblings, mapping = manifest()["siblings"], manifest()["branches"]
    resolve = script().sibling_branch
    assert resolve(siblings["predictions"], "develop", mapping) == "develop"
    assert resolve(siblings["predictions"], "main", mapping) == "main"
    assert resolve(siblings["predictions"], "feature/xyz", mapping) == mapping["default"]
    assert resolve(siblings["predictions"], None, mapping) == mapping["default"]
    # A single-branch sibling stays on that branch whatever the app is on.
    assert resolve(siblings["postgresql"], "develop", mapping) == "main"


def test_branch_comes_from_the_environment_first(monkeypatch):
    module = script()
    monkeypatch.setenv("RAILWAY_GIT_BRANCH", "develop")
    assert module.app_branch() == "develop"
    monkeypatch.setenv("SIBLING_BRANCH", "main")  # explicit override wins
    assert module.app_branch() == "main"


def test_siblings_are_git_ignored_so_they_never_land_in_this_repo():
    ignored = (ROOT / ".gitignore").read_text()
    docker_ignored = (ROOT / ".dockerignore").read_text()
    for name in manifest()["siblings"]:
        assert f"/{name}/" in ignored
        assert f"/{name}/" in docker_ignored  # the container fetches, never copies a local checkout


def test_ci_tests_every_branch_but_deploys_only_from_main():
    """The pipeline must test both branches against matching siblings and deploy only main."""
    workflow = (ROOT / ".github" / "workflows" / "ci-cd.yml").read_text()
    assert "branches: [develop, main]" in workflow
    assert "scripts/siblings.py fetch" in workflow  # CI uses the same sibling rule as the deployment
    assert "SIBLING_BRANCH: ${{ github.base_ref || github.ref_name }}" in workflow
    deploy = workflow[workflow.index("  deploy:"):]
    assert "needs: [test, docker]" in deploy
    assert "if: github.event_name == 'push' && github.ref == 'refs/heads/main'" in deploy
