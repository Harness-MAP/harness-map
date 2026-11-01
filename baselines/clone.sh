#!/usr/bin/env bash
# Fetch the upstream baseline repositories into vendor/ at the pinned commits (see upstream.py),
# install OpenEvolve's two extra packages, and build ShinkaEvolve's dependency sidecar.
#
#   bash baselines/clone.sh            # clone (or verify) all five
#   HARNESS_MAP_VENDOR=/shared/vendor bash baselines/clone.sh   # elsewhere; upstream.py reads the same variable
set -euo pipefail
cd "$(dirname "$0")"
VENDOR="${HARNESS_MAP_VENDOR:-$PWD/vendor}"
SIDECAR="${SHINKA_VENV:-$PWD/.venv_shinka}"
PY="${PYTHON:-python3}"
mkdir -p "$VENDOR"

clone() {                       # clone <url> <dir> <commit>
  local url="$1" dir="$VENDOR/$2" pin="$3"
  if [ ! -d "$dir/.git" ]; then
    echo "== $2: cloning $url"
    git clone --quiet "$url" "$dir"
  fi
  if [ "$(git -C "$dir" rev-parse HEAD)" != "$pin" ]; then
    git -C "$dir" fetch --quiet origin "$pin" 2>/dev/null || git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$pin"
  fi
  printf '   %-14s %s  %s\n' "$2" "$(git -C "$dir" rev-parse HEAD)" "$(git -C "$dir" log -1 --format=%ci)"
}

clone https://github.com/gepa-ai/gepa.git                           gepa          d26029fee83f44099f89fac1a450acd028bcbc90
clone https://github.com/jennyzzt/dgm.git                           dgm           a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2
clone https://github.com/stanford-iris-lab/meta-harness.git         meta-harness  44b9942127847f7421db70d8c7e48407f09a3c70
clone https://github.com/algorithmicsuperintelligence/openevolve.git openevolve   411fb59c886c18704caaffb611e17cf9e7d824d2
clone https://github.com/SakanaAI/shinkaevolve.git                  shinkaevolve  2bf8cfeb6fd39c79555cd94a8f395d64e740aae8

# OpenEvolve runs in-process and needs two pure-Python packages.
"$PY" -m pip install -q dacite flask || echo "   (install dacite + flask for openevolve)"

# ShinkaEvolve's dependencies go in a sidecar virtualenv built from this interpreter; upstream.py
# appends its site-packages, so the packages already installed keep precedence.
if [ ! -d "$SIDECAR" ]; then
  echo "== shinkaevolve: building the dependency sidecar at $SIDECAR"
  "$PY" -m venv "$SIDECAR"
  "$SIDECAR/bin/python" -m pip install -q --upgrade pip
  "$SIDECAR/bin/python" -m pip install -q -e "$VENDOR/shinkaevolve"
fi

echo
echo "pins:"
HARNESS_MAP_VENDOR="$VENDOR" SHINKA_VENV="$SIDECAR" "$PY" "$PWD/upstream.py"
