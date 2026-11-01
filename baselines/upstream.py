"""Loading the pinned upstream clones under `vendor/` (or `$HARNESS_MAP_VENDOR`); none is edited."""
from __future__ import annotations

import contextlib
import importlib
import os
import subprocess
import sys
import types

from pathlib import Path

_HERE = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("HARNESS_MAP_VENDOR") or _HERE / "vendor")

# name -> (url, directory, pinned commit). The pins are the commits the paper's runs used.
REPOS = {
    "gepa": ("https://github.com/gepa-ai/gepa.git", VENDOR / "gepa",
             "d26029fee83f44099f89fac1a450acd028bcbc90"),
    "dgm": ("https://github.com/jennyzzt/dgm.git", VENDOR / "dgm",
            "a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2"),
    "meta-harness": ("https://github.com/stanford-iris-lab/meta-harness.git", VENDOR / "meta-harness",
                     "44b9942127847f7421db70d8c7e48407f09a3c70"),
    "openevolve": ("https://github.com/algorithmicsuperintelligence/openevolve.git",
                   VENDOR / "openevolve", "411fb59c886c18704caaffb611e17cf9e7d824d2"),
    "shinkaevolve": ("https://github.com/SakanaAI/shinkaevolve.git", VENDOR / "shinkaevolve",
                     "2bf8cfeb6fd39c79555cd94a8f395d64e740aae8"),
}

# ShinkaEvolve's dependencies pin packages such as pydantic, so they live in a sidecar venv
# (clone.sh) appended with site.addsitedir: packages already installed keep precedence.
SHINKA_VENV = Path(os.environ.get("SHINKA_VENV") or _HERE / ".venv_shinka")

META_HARNESS_EXAMPLE = VENDOR / "meta-harness" / "reference_examples" / "terminal_bench_2"


class VendorMissing(RuntimeError):
    """Raised with the command that fixes it, rather than an ImportError from four frames in."""


def _require(name: str) -> Path:
    url, path, pin = REPOS[name]
    if not (path / ".git").exists():
        raise VendorMissing(
            f"vendored upstream {name!r} is missing at {path}.\n"
            f"  fix: bash {_HERE / 'clone.sh'}\n"
            f"  or:  git clone {url} {path} && git -C {path} checkout {pin}")
    return path


@contextlib.contextmanager
def _on_path(p: Path):
    """Put `p` on sys.path only during an import; dgm/ and the TB2 example ship colliding names."""
    s = str(p)
    sys.path.insert(0, s)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(s)


# ── gepa: the whole engine, no stubs ───────────────────────────────────────────────────────────
def load_gepa():
    """Import the upstream `gepa` package from its `src/` layout (a unique name, left on path)."""
    src = str(_require("gepa") / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import gepa
    return gepa


# ── dgm: selection + archive real, the Docker self-improve step stubbed ───────────────────────
_DGM_STUBS = {
    # DGM_outer imports `self_improve` at module level; importing it for real pulls docker,
    # anthropic, openai and the SWE-bench harness. The arm never calls it (our testbed replaces it).
    "self_improve_step": {"self_improve": "DGM's Docker/SWE-bench self-improve step is replaced "
                                          "by the testbed adapter; see dgm.py"},
    # Only `setup_logger` is imported from this, and only by DGM_outer.main (which we do not call).
    "utils.docker_utils": {"setup_logger": "DGM's docker logger is not used; the arm passes its "
                                           "own logger into filter_compiled"},
}


def _stub(name: str, attrs: dict) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__doc__ = f"stub for {name} (baselines.upstream)"
    for attr, why in attrs.items():
        def _raise(*a, _why=why, _n=f"{name}.{attr}", **k):
            raise NotImplementedError(f"{_n} is stubbed: {_why}")
        setattr(m, attr, _raise)
    return m


def load_dgm():
    """Import upstream `DGM_outer` with its two Docker-bound modules stubbed."""
    root = _require("dgm")
    saved = {k: sys.modules.get(k) for k in _DGM_STUBS}
    for k, attrs in _DGM_STUBS.items():
        sys.modules[k] = _stub(k, attrs)
    try:
        with _on_path(root):
            import DGM_outer
            importlib.import_module("utils.evo_utils")        # real, no docker
        return DGM_outer
    finally:
        for k, old in saved.items():                          # leave sys.modules as we found it
            if old is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = old


# ── openevolve: the whole library, in-process ──────────────────────────────────────────────────
def load_openevolve():
    """Import the upstream `openevolve` package (needs `dacite` + `flask`)."""
    root = _require("openevolve")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import openevolve
    return openevolve


# ── shinkaevolve: the library, via the sidecar venv ────────────────────────────────────────────
def load_shinka():
    """Import upstream `shinka` with its dependency sidecar on the path (see `SHINKA_VENV`)."""
    _require("shinkaevolve")
    site_pkgs = sorted(SHINKA_VENV.glob("lib/python*/site-packages"))
    if not site_pkgs:
        raise VendorMissing(
            f"the ShinkaEvolve dependency sidecar is missing at {SHINKA_VENV}.\n"
            f"  fix: bash {_HERE / 'clone.sh'}  (or: python -m venv {SHINKA_VENV} && "
            f"{SHINKA_VENV}/bin/python -m pip install -e {REPOS['shinkaevolve'][1]})")
    import site
    sp = str(site_pkgs[0])
    if sp not in sys.path:
        site.addsitedir(sp)              # APPEND + process the editable .pth; installed packages win
    import shinka.core
    return shinka.core


# ── meta-harness: the TB2 reference example, imported whole ─────────────────────────────────────
def load_meta_harness():
    """Import upstream `meta_harness` (the Terminal-Bench 2 reference example); no stubs needed."""
    _require("meta-harness")
    with _on_path(META_HARNESS_EXAMPLE):
        import meta_harness
    return meta_harness


# ── provenance ─────────────────────────────────────────────────────────────────────────────────
def pins() -> dict:
    """The pinned commit of each clone and what is actually checked out, for a results table."""
    out = {}
    for name, (url, path, pin) in REPOS.items():
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True,
                                 text=True, timeout=30).stdout.strip()
            date = subprocess.run(["git", "log", "-1", "--format=%ci"], cwd=path,
                                  capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:                                # noqa: BLE001 — a missing clone is reported
            sha, date = f"<unavailable: {e}>", ""
        out[name] = {"url": url, "path": str(path), "pinned": pin, "commit": sha, "date": date,
                     "at_pin": sha == pin}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(pins(), indent=2))
