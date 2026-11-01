"""Doc-block sweep: clone a base candidate, overwrite named files, report the rung-0 cell.

Usage: python sweep.py <base_dir> <variant_dir> [...]
Each <variant_dir> holds files named like `AGENT.md`, `NOTES.md`, `skills__x.md`
(`__` -> `/`); a zero-byte file means DELETE that path from the clone.
"""
import shutil, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/azureuser/workspace/AutoSpatial")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phi_probe import phi, Z, META
from harness_map.qd import descriptors as D

C = Z["Z"]
off = 0; per = {}
for s in META["blocks"]:
    per[s["name"]] = (off, off + s["dim"]); off += s["dim"]


def measure(cdir, want=21):
    p, _c, _r = phi(cdir)
    d = np.linalg.norm(C - p[None, :], axis=1)
    order = np.argsort(d)
    rival = min((j for j in range(len(C)) if j != want - 1), key=lambda j: d[j])
    sub = {nm: float(np.linalg.norm(C[want-1][a:b] - p[a:b])) for nm, (a, b) in per.items()}
    return dict(cell=int(order[0]) + 1, d=float(d[want-1]),
                margin=float(d[rival] - d[want-1]), rival=int(rival) + 1,
                doclen=len(D._doc_text(cdir)), **{k: round(v, 4) for k, v in sub.items()})


if __name__ == "__main__":
    base = Path(sys.argv[1])
    print(f"{'variant':<26} {'cell':>4} {'d':>8} {'margin':>8} {'rival':>5} "
          f"{'doclen':>7} {'doc':>7} {'ast':>7}")
    r = measure(base)
    print(f"{'(base)':<26} {r['cell']:>4} {r['d']:8.4f} {r['margin']:+8.4f} {r['rival']:>5} "
          f"{r['doclen']:>7} {r['doc']:7.4f} {r['ast']:7.4f}")
    for vd in sys.argv[2:]:
        vd = Path(vd)
        tmp = Path(tempfile.mkdtemp(prefix="sw_"))
        clone = tmp / base.name
        shutil.copytree(base, clone, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in sorted(vd.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(vd).as_posix().replace("__", "/")
            tgt = clone / rel
            if f.stat().st_size == 0:
                tgt.unlink(missing_ok=True)
            else:
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, tgt)
        r = measure(clone)
        print(f"{vd.name:<26} {r['cell']:>4} {r['d']:8.4f} {r['margin']:+8.4f} {r['rival']:>5} "
              f"{r['doclen']:>7} {r['doc']:7.4f} {r['ast']:7.4f}")
        shutil.rmtree(tmp, ignore_errors=True)
