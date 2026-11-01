"""The descriptor φ of a harness directory, from its source files alone (WL, AST, numeric, doc)."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np


# Path parts that are not part of the genome: bytecode caches, `_donor` copies of other harnesses
# and (by prefix) evaluation outputs.
DEFAULT_SKIP = ("__pycache__", "_donor")


DEFAULT_SKIP_PREFIXES = ("rep_", "ep_seed", "val_", "traces")


# Reports written by the meta-agent after evaluation; excluded so they never move φ.
DEFAULT_SKIP_ARTIFACTS = ("diagnosis.md", "retrospective.md", "summary.md", "judge.md",
                          "review.md", "eval_summary.md")


# (role, keywords) pairs for `component_role`; `main.role_rules` prepends a benchmark's own roles.
DEFAULT_ROLE_RULES = [
    ("controller", ("run_episode", "main", "loop", "rollout", "episode_step")),
    ("memory",     ("memory", "history", "recall", "remember")),
    ("prompt",     ("prompt", "template", "instruction", "compose_system", "build_correction", "system")),
    ("perception", ("capture", "view", "render", "frame", "mask", "rgb", "chip", "box", "segment",
                    "compon", "dilate", "head", "letter", "visible", "estimate", "gold")),
    ("toolcall",   ("_ask", "vlm", "invoke", "query")),
    ("policy",     ("pick", "choose", "steer", "momentum", "decide", "move", "action", "select", "key")),
    ("validity",   ("legal", "reverse", "valid", "adjacent", "collide", "safe")),
    ("recovery",   ("fallback", "correct", "retry", "honest", "recover", "repair")),
    ("parse",      ("parse", "_read", "resolve", "extract", "decode", "split")),
    ("spatial",    ("vector", "compass", "direction", "distance", "dist", "toward", "food",
                    "waypoint", "arrival", "heading")),
    ("data",       ("result", "state", "config")),
]


_CF_TYPES = ["If", "For", "AsyncFor", "While", "Try", "With", "AsyncWith",
             "comprehension", "BoolOp", "IfExp", "Break", "Continue"]


def harness_pyfiles(cdir, skip=DEFAULT_SKIP, skip_prefixes=DEFAULT_SKIP_PREFIXES):
    """The .py files that constitute the genome (eval artifacts excluded)."""
    cdir = Path(cdir)
    out = []
    for p in sorted(cdir.rglob("*.py")):
        parts = p.relative_to(cdir).parts
        if any(part in skip or any(part.startswith(pre) for pre in skip_prefixes)
               for part in parts):
            continue
        out.append((str(p.relative_to(cdir)), p))
    return out


def _toplevel(tree):
    return [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def component_role(name: str, rules=DEFAULT_ROLE_RULES) -> str:
    """Role of a component: the first rule with a keyword in its lowercased name, else 'logic'."""
    n = name.lower()
    for role, kws in rules:
        if any(k in n for k in kws):
            return role
    return "logic"


def roles(cdir, rules=DEFAULT_ROLE_RULES) -> dict:
    """Role of each top-level component; every module but the main one is a 'tool'."""
    per_mod, out = {}, {}
    for rel, p in harness_pyfiles(cdir):
        mod = Path(rel).stem
        if mod == "__init__":
            continue
        try:
            t = ast.parse(p.read_text())
        except Exception:
            continue
        names = [n.name for n in _toplevel(t)]
        per_mod[mod] = names
        for name in names:
            out[name] = component_role(name, rules)
    if per_mod:
        main_mod = max(per_mod, key=lambda m: len(per_mod[m]))
        for mod in per_mod:
            if mod != main_mod:
                out[mod] = "tool"
    return out


def component_graph(cdir):
    """(nodes, edges) of top-level components and modules, linked by calls, names and imports."""
    comp2mod, trees = {}, {}
    for rel, p in harness_pyfiles(cdir):
        mod = Path(rel).stem
        try:
            t = ast.parse(p.read_text())
        except Exception:
            continue
        trees[mod] = t
        for n in _toplevel(t):
            comp2mod[n.name] = mod
    nodes, edges = set(comp2mod) | set(trees), set()
    for mod, t in trees.items():
        for n in _toplevel(t):
            for sub in ast.walk(n):
                nm = None
                if isinstance(sub, ast.Call):
                    f = sub.func
                    nm = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                elif isinstance(sub, ast.Name):
                    nm = sub.id
                if nm and nm in comp2mod and nm != n.name:
                    edges.add((n.name, nm))
        for n in ast.walk(t):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                base = getattr(n, "module", None) or ""
                for a in n.names:
                    tmod = (a.name or base).split(".")[-1]
                    if tmod in trees and tmod != mod:
                        edges.add((mod, tmod))
    return nodes, edges


def wl_features(cdir, h: int = 2, rules=DEFAULT_ROLE_RULES) -> dict:
    """Weisfeiler-Lehman label histogram (depth h) of the role-labelled component graph."""
    rl = roles(cdir, rules)
    N, E = component_graph(cdir)
    adj = {n: set() for n in N}
    for a, b in E:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    labels = {n: rl.get(n, "logic") for n in N}
    hist = Counter(f"0:{v}" for v in labels.values())
    for it in range(1, h + 1):
        new = {}
        for n in N:
            sig = labels[n] + "|" + ",".join(sorted(labels[m] for m in adj[n]))
            new[n] = f"{it}:" + hashlib.md5(sig.encode()).hexdigest()[:10]
        labels = new
        hist.update(labels.values())
    tot = max(len(N), 1)
    return {k: v / tot for k, v in hist.items()}


def _component_srcs(cdir):
    out = []
    for _rel, p in harness_pyfiles(cdir):
        try:
            src = p.read_text(); tree = ast.parse(src)
        except Exception:
            continue
        for n in _toplevel(tree):
            seg = ast.get_source_segment(src, n)
            if seg:
                out.append((n.name, seg))
    return out


def f_AST(cdir) -> dict:
    """Control-flow statistics per component, and the share of each node type."""
    comps = _component_srcs(cdir)
    per, types = [], {t: 0 for t in _CF_TYPES}
    for _name, seg in comps:
        try:
            t = ast.parse(seg)
        except Exception:
            continue
        c = 0
        for node in ast.walk(t):
            nm = type(node).__name__
            if nm in types:
                types[nm] += 1; c += 1
        per.append(c)
    n = max(len(per), 1)
    tot = max(sum(types.values()), 1)
    mu = sum(per) / n
    out = {"cf_mean": mu, "cf_max": float(max(per) if per else 0),
           "cf_std": (sum((x - mu) ** 2 for x in per) / n) ** 0.5}
    out.update({f"p_{t}": v / tot for t, v in types.items()})
    return out


def f_num(cdir) -> dict:
    """Numeric-constant signature: constants per component and mean/std of log1p(|value|)."""
    nums = []
    ncomp = 0
    for _name, seg in _component_srcs(cdir):
        ncomp += 1
        try:
            for node in ast.walk(ast.parse(seg)):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                        and not isinstance(node.value, bool):
                    nums.append(abs(float(node.value)))
        except Exception:
            continue
    logs = [math.log1p(x) for x in nums]
    n = max(len(logs), 1)
    mu = sum(logs) / n
    return {"num_per_comp": len(nums) / max(ncomp, 1), "logmag_mean": mu,
            "logmag_std": (sum((x - mu) ** 2 for x in logs) / n) ** 0.5}


# Encoder for the doc-text block. The empty-text fallback in f_doc is 768 zeros, so a model set
# through META_EMB_MODEL should also be 768-d.
EMB_MODEL = os.environ.get("META_EMB_MODEL", "nomic-ai/CodeRankEmbed")


_ENC = None


def get_encoder():
    """Lazily load the `EMB_MODEL` sentence-transformers encoder (CPU)."""
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer(EMB_MODEL, trust_remote_code=True, device="cpu")
    return _ENC


def _doc_text(cd) -> str:
    import ast as _ast
    parts = []
    for m in sorted(Path(cd).rglob("*.md")):
        rel = m.relative_to(cd)
        if rel.name in DEFAULT_SKIP_ARTIFACTS:
            continue
        if any(pp in DEFAULT_SKIP or any(pp.startswith(pre) for pre in DEFAULT_SKIP_PREFIXES)
               for pp in rel.parts):
            continue
        try:
            parts.append(m.read_text(errors="replace"))
        except Exception:
            pass
    for _rel, p in harness_pyfiles(cd):
        try:
            src = p.read_text()
        except Exception:
            continue
        try:
            for n in _ast.walk(_ast.parse(src)):
                if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef, _ast.Module)):
                    g = _ast.get_docstring(n)
                    if g:
                        parts.append(g)
        except Exception:
            pass
        for ln in src.splitlines():
            s = ln.strip()
            if s.startswith("#"):
                parts.append(s.lstrip("#").strip())
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()


def f_doc(cd, cache_dir) -> np.ndarray:
    """Unit-norm embedding of the documentation text (zeros if none), cached in `cache_dir`."""
    t = _doc_text(cd)
    if not t:
        return np.zeros(768)
    cp = Path(cache_dir) / "doc_emb_cache.json"
    # An unreadable cache is moved aside and rebuilt instead of failing every later run.
    cache = {}
    if cp.exists():
        try:
            cache = json.loads(cp.read_text())
        except Exception as e:                           # noqa: BLE001 — truncated/corrupt cache
            bad = cp.with_suffix(".json.corrupt")
            try:
                cp.replace(bad)
            except Exception:                            # noqa: BLE001
                pass
            print(f"[descriptors] doc_emb_cache unreadable ({e!r}); quarantined to {bad.name} and "
                  f"rebuilding (check free disk space)", flush=True)
    k = hashlib.md5(("doc|" + t[:12000]).encode()).hexdigest()
    if k not in cache:
        cache[k] = [round(float(x), 5) for x in get_encoder().encode(t[:12000],
                                                                     normalize_embeddings=True)]
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f".json.tmp{os.getpid()}")
        try:
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, cp)                          # atomic within the same filesystem
        except Exception as e:                           # noqa: BLE001 — cache write is optional
            try:
                tmp.unlink(missing_ok=True)
            except Exception:                            # noqa: BLE001
                pass
            print(f"[descriptors] doc_emb_cache NOT written ({e!r}) — continuing uncached", flush=True)
    return np.asarray(cache[k], float)


def desc_blocks(cd, rules, cache_dir):
    """Ordered (name, block) list: the "wl", "ast" and "num" feature dicts and the "doc" vector."""
    return [("wl", wl_features(cd, h=2, rules=rules)), ("ast", f_AST(cd)), ("num", f_num(cd)),
            ("doc", f_doc(cd, cache_dir))]


def _block_matrix(blocks_per_dir, i):
    b0 = blocks_per_dir[0][i][1]
    if isinstance(b0, dict):
        keys = sorted({k for bl in blocks_per_dir for k in bl[i][1]})
        M = np.array([[bl[i][1].get(k, 0.0) for k in keys] for bl in blocks_per_dir], float)
        return M, keys
    M = np.array([np.asarray(bl[i][1], float) for bl in blocks_per_dir], float)
    return M, []
