"""Five published search methods and two descriptor-swap ablations, as arms over one adapter."""
from .common import BudgetExhausted, Cand, Ledger, RunLog, per_instance

__all__ = ["BASELINES", "run_baseline", "BudgetExhausted", "Cand", "Ledger", "RunLog", "per_instance"]
BASELINES = ("meta-harness", "gepa", "dgm", "openevolve", "shinka",
             "openevolve_cvt", "open_harness")


def run_baseline(name: str, adapter, **kw) -> dict:
    """Dispatch to one arm's `run(adapter, ...)`; each arm takes its knobs and ignores the rest."""
    if name == "meta-harness":
        from . import meta_harness
        return meta_harness.run(adapter, **kw)
    if name == "gepa":
        from . import gepa
        return gepa.run(adapter, **kw)
    if name == "dgm":
        from . import dgm
        return dgm.run(adapter, **kw)
    if name == "openevolve":
        from . import openevolve
        return openevolve.run(adapter, **kw)
    if name == "shinka":
        from . import shinka_arm
        return shinka_arm.run(adapter, **kw)
    if name == "openevolve_cvt":
        from . import openevolve_cvt
        return openevolve_cvt.run(adapter, **kw)
    if name == "open_harness":
        from . import open_harness
        return open_harness.run(adapter, **kw)
    raise KeyError(f"unknown baseline {name!r}; known: {BASELINES}")
