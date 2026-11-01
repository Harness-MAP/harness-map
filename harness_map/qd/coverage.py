"""Coverage-gated initialization: cover ⌈m·k0⌉ cells with founders first, evaluate second."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from harness_map.main import BatchContractError, Node, _evaluate_all
from harness_map.qd import archive as _arch
from harness_map.qd.partition import ladder_cell_index


class CoverageError(RuntimeError):
    """Strict mode: initialization covered fewer than ceil(random_init * k0) cells."""


@dataclass
class _PhiProbe:
    """Node stand-in carrying only the genome, so φ can place a candidate before evaluation."""
    genome: object


def nearest_cell(fr, p):
    """Active cell whose centroid is nearest to `p` (ties go to the first in `fr.cells`)."""
    return min(fr.cells, key=lambda cc: float(((p - cc["z"]) ** 2).sum()))


def locate_cell(fr, p):
    """Cell of descriptor `p`, by the same rule as `archive.place`."""
    if getattr(fr.ladder, "parents", None) is not None:
        return fr.cells[ladder_cell_index(fr.ladder, p, fr.level)]
    return nearest_cell(fr, p)


def phi_of_genome(fr, genome):
    """φ of an unevaluated genome (not cached: `fr._phi_cache` is keyed by node index)."""
    return np.asarray(fr.phi_fn(_PhiProbe(genome)), float)


def coverage_directive(*, occupied, k0, target, exemplars, aim=None):
    """Proposer directive aimed at an empty cell (target centroid, dominant axis of the move)."""
    head = (f"ARCHIVE-COVERAGE STEER — the archive occupies {occupied} of {k0} behaviour cells and "
            f"is initializing toward {target}. ")
    screen_truth = (
        "Novelty of MECHANISM is the objective here, not score: NOTHING proposed in this phase is "
        "evaluated until the whole coverage target is reached — this pass only LOCALIZES what you "
        "return — and a candidate that lands in a cell already occupied, or already claimed by an "
        "earlier candidate, is discarded UNEVALUATED. Keep the public entry point intact.")
    if not aim:
        return (head + f"Propose a harness that is STRUCTURALLY UNLIKE every one of "
                       f"{list(exemplars)}: change the component graph (which modules exist, what "
                       f"calls what, where the decision is made), not just wording or constants. "
                + screen_truth)
    _origin = (f"The nearest OCCUPIED cell is {aim['from']}, at descriptor distance "
               f"{aim['dist']:.3f}"
               if aim.get("from_is_nearest", True) else
               f"You are breeding FROM the elite of cell {aim['from']}, at descriptor distance "
               f"{aim['dist']:.3f} (the occupied cell nearest the target is "
               f"{aim.get('nearest_occupied')}, but it is not the harness you were handed)")
    lines = [
        head.rstrip(),
        "ARCHITECTURAL STEERING DIRECTIVE (aim the NEW candidates at a different design niche; "
        "keep them functional and honest):",
        f"- TARGET: behaviour cell {aim['target']} of K0={k0} is EMPTY. Its descriptor CENTROID is "
        f"{aim['centroid']}. {_origin}; the move asked for is {aim['from']} -> {aim['target']}. "
        f"No exemplar "
        f"harness is named for this niche because an empty cell HAS none — this is a direction "
        f"stated in COORDINATES, not a harness to imitate.",
        f"- DOMINANT AXIS: {aim['axis']} (‖Δstructure‖={aim['d_struct']:.3f} vs "
        f"‖Δtoken‖={aim['d_token']:.3f}).",
    ]
    if aim["axis"].startswith("structure"):
        lines += [
            "- STRUCTURE: restructure the COMPONENT GRAPH substantially — a different "
            "decomposition (split/merge components, change who calls whom), keeping behavior "
            "honest. Change the COMPONENT GRAPH (which boxes exist / who calls whom), not just "
            "wording.",
            "- IMPLEMENTATION: keep per-component code style/complexity similar."]
    else:
        lines += [
            "- STRUCTURE: keep the component graph essentially UNCHANGED (same components and "
            "wiring).",
            "- IMPLEMENTATION: move the TOKEN-LEVEL content — control-flow complexity per "
            "component, the count and magnitude of numeric constants/thresholds, the wording of "
            "the instructions — i.e. change code INSIDE components, keeping the component graph "
            "fixed."]
    lines += [
        "This is an exploration round: a mechanically DIFFERENT design in the stated direction is "
        "worth more than a small tweak to the parent.",
        screen_truth]
    return "\n".join(lines)


def cover_aim(fr, claimed, n, srcs=None):
    """One aim per parent slot at an empty, unclaimed cell nearest an occupied one, or None."""
    B = fr.block_split
    occ = [c for c in fr.cells if c["mem"]]
    empt = [c for c in fr.cells if not c["mem"] and c["cid"] not in claimed]
    if B is None or not occ or not empt:
        return [None] * n
    ranked = []
    for e in empt:
        src = min(occ, key=lambda c: (float(((c["z"] - e["z"]) ** 2).sum()), c["cid"]))
        ranked.append((float(((src["z"] - e["z"]) ** 2).sum()) ** 0.5, e["cid"], e, src))
    ranked.sort(key=lambda r: (r[0], r[1]))
    srcs = list(srcs or ())
    aims = []
    for i in range(max(0, int(n))):
        _d_near, cid, e, near = ranked[i % len(ranked)]
        src = srcs[i] if i < len(srcs) and srcs[i] is not None else near
        z, zs = e["z"], src["z"]
        if not (0 < B < len(z)):
            aims.append(None)
            continue
        ds = float(((z[:B] - zs[:B]) ** 2).sum()) ** 0.5
        dt = float(((z[B:] - zs[B:]) ** 2).sum()) ** 0.5
        aims.append({"target": cid, "from": src["cid"],
                     "dist": float(((z - zs) ** 2).sum()) ** 0.5,
                     "axis": ("structure/wiring (component graph)" if ds >= dt
                              else "token-level implementation"),
                     "d_struct": ds, "d_token": dt,
                     "nearest_occupied": near["cid"],
                     "from_is_nearest": bool(src["cid"] == near["cid"]),
                     "centroid": [round(float(x), 4) for x in z]})
    return aims


def phase_label(it, depth=None):
    """Log label: coverage passes (negative `it`) are labelled as such, generations as rounds."""
    if isinstance(it, int) and it < 0:
        return (f"INIT coverage pass {-it} (pre-generation"
                + (f", founder depth={depth}" if depth is not None else "") + ")")
    return f"round {it if depth is None else depth}"


@dataclass
class _CoverView:
    """Archive interface used by `coverage_init`; built by `archive.frontier_cover_view`."""
    k0: int                                            # coarse cell count, fixed for the phase
    tag: str                                           # log prefix
    occupied: Callable                                 # () -> set of occupied cell keys
    key_of_genome: Callable                            # (genome) -> cell key, without evaluation
    select: Callable                                   # (p,b,claimed) -> (parents,meta)|(None,)*2
    land: Callable                                     # (meta, units, p) -> gen row for pass_rows
    empty_cells: Callable                              # () -> still-empty cell keys, for reports


# Correction for non-uniform landing of proposals (see `coverage_proposal_allocation`).
COVERAGE_NONUNIFORMITY_DEFAULT = 1.4


# The stage-A proposal cap is ceil(safety * R): headroom above R, which is only a mean.
COVERAGE_SAFETY_DEFAULT = float(os.environ.get("COVERAGE_SAFETY", "3.0"))

# A spent cap is extended while coverage still grows (`cap_extension`): stage A costs no
# evaluations, so reaching the cap does not show that the target is unreachable.
COVERAGE_AUTO_EXTEND_DEFAULT = os.environ.get("COVERAGE_AUTO_EXTEND", "1").strip().lower() not in (
    "0", "false", "no", "off", "")
COVERAGE_MAX_EXTENSIONS_DEFAULT = max(0, int(os.environ.get("COVERAGE_MAX_EXTENSIONS", "8")))


def cap_extension(*, proposals, cap, covered_now, covered_at_last_extension, extensions,
                  max_extensions, block):
    """Next stage-A proposal cap and its reason (None: stop); extensions need rising coverage."""
    if proposals < cap:
        return cap, "under_cap"
    if max_extensions is not None and extensions >= max_extensions:
        return None, "extension_limit"
    if covered_at_last_extension is not None and covered_now <= covered_at_last_extension:
        return None, "no_progress_since_last_extension"
    return cap + max(1, int(math.ceil(block))), "extended"


def coverage_proposal_allocation(k0, occupied, target, nonuniformity=None):
    """Expected proposals to raise coverage from `occupied` to `target` cells (coupon-collector)."""
    k0 = int(k0)
    if k0 <= 0:
        return 0.0
    nu = float(nonuniformity or 0.0) or COVERAGE_NONUNIFORMITY_DEFAULT
    if nu <= 0.0:
        raise ValueError(f"nonuniformity multiplies the uniform coupon-collector expectation "
                         f"R_uniform to give the phase's planned proposal count, so it must be "
                         f"> 0; got {nonuniformity!r}. Pass 1.0 for the uncorrected uniform model.")
    e = max(0, min(int(occupied), k0))
    t = max(0, min(int(target), k0))
    if t <= e:
        return 0.0
    return nu * float(sum(k0 / (k0 - j) for j in range(e, t)))


def _coverage_validate(random_init, random_init_batch, max_proposals, max_localizations, stall,
                       phase_max_evals=0, safety=0, nonuniformity=0):
    ri = float(random_init or 0.0)
    if not (0.0 <= ri <= 1.0):
        raise ValueError(f"random_init is a FRACTION of k0 (0.0-1.0), got {random_init!r}; it is "
                         f"not a cell COUNT. random_init=0.5 means 'cover half of K0'.")
    sf = float(safety or 0.0) or COVERAGE_SAFETY_DEFAULT
    if sf < 1.0:
        raise ValueError(
            f"random_init_safety multiplies the occupancy model's expected proposal count R to get "
            f"the phase's HARD proposal cap, so it must be >= 1.0; got {safety!r}. A value below 1 "
            f"caps the phase BELOW its own planned first tranche. Pass an explicit "
            f"random_init_max_proposals if you want a specific smaller number.")
    nu = float(nonuniformity or 0.0) or COVERAGE_NONUNIFORMITY_DEFAULT
    if nu <= 0.0:
        raise ValueError(
            f"random_init_nonuniformity multiplies the UNIFORM coupon-collector expectation "
            f"R_uniform to give the phase's planned proposal count R, so it must be > 0; got "
            f"{nonuniformity!r}. Pass 1.0 for the uncorrected uniform model, or the factor your "
            f"own testbed measured.")
    return (ri, max(1, int(random_init_batch)), max(0, int(max_proposals)),
            max(0, int(max_localizations)), max(0, int(stall)), max(0, int(phase_max_evals or 0)),
            sf, nu)


def _shortfall_cost_clause(tel):
    n = int(tel.get("evaluations") or 0)
    if n == 0:
        return ("a shortfall whose stop reason is stage A's costs ZERO evaluations, and none were "
                "spent here")
    src = []
    if int(tel.get("repair_cycles") or 0):
        src.append(f"{tel['repair_cycles']} earlier repair cycle(s), each of which ran a PAID "
                   f"stage B")
    if int(tel.get("localization_errors") or 0):
        src.append(f"the descriptor fail-open ({tel['localization_errors']} localization error(s), "
                   f"which are evaluated unscreened inside stage A)")
    return ("a shortfall whose stop reason is stage A's costs ZERO ADDITIONAL evaluations — but "
            f"{n} were ALREADY charged to N before it, by "
            + (" and ".join(src) or "an earlier cover-evaluate cycle")
            + ", so this shortfall did NOT come free; read the count, not the guarantee")


def coverage_init(ctx, view, *, random_init, batch, screen_on, strict,
                  max_proposals, max_localizations, stall, max_evals, n_workers,
                  phase_max_evals=0, max_repairs=0, safety=0, nonuniformity=0,
                  auto_extend=None, max_extensions=None):
    """Propose founders until ceil(random_init * k0) cells are covered; returns the telemetry."""
    import time as _time
    t_start = _time.time()
    k0 = int(view.k0)
    target = min(k0, max(0, math.ceil(random_init * k0)))
    entry = len(view.occupied())
    target_new = max(0, target - entry)
    # safety >= 1 keeps the hard cap at or above the first tranche.
    safety = max(1.0, float(safety or COVERAGE_SAFETY_DEFAULT))
    nonuniformity = float(nonuniformity or COVERAGE_NONUNIFORMITY_DEFAULT)
    planned = coverage_proposal_allocation(k0, entry, target, nonuniformity=nonuniformity)
    max_proposals_src = ("caller" if max_proposals else
                         f"auto ceil({safety:g} * R) where R={planned:.2f}")
    max_proposals = max_proposals or max(1, math.ceil(safety * planned))
    # Only the automatic cap may be extended (`cap_extension`); an explicit cap is final.
    auto_extend = (COVERAGE_AUTO_EXTEND_DEFAULT if auto_extend is None else bool(auto_extend))
    auto_extend = auto_extend and max_proposals_src != "caller"
    max_extensions = (COVERAGE_MAX_EXTENSIONS_DEFAULT if max_extensions is None
                      else max(0, int(max_extensions)))
    # Localization cap, generous: placing a candidate by φ is cheap compared with proposing one.
    max_localizations = max_localizations or max(256, 128 * target_new)
    # Evaluation cap, tight (two per new cell), so repeated invalid evaluations stop early.
    phase_cap = phase_max_evals or max(1, 2 * target_new)
    # Repair cap: how many more cover-then-evaluate cycles (stage C) may follow the first.
    repair_cap = max_repairs or max(2, target_new)

    tel = {"enabled": True, "counted_in_generations": False,
           "random_init": random_init, "random_init_batch": batch,
           "random_init_screen": bool(screen_on), "random_init_strict": bool(strict),
           "max_proposals": max_proposals, "max_localizations": max_localizations,
           "max_proposals_source": max_proposals_src,
           "max_proposals_initial": max_proposals, "cap_auto_extend": bool(auto_extend),
           "cap_max_extensions": max_extensions, "cap_extensions": 0, "cap_extension_log": [],
           "cap_extension_stop": None,
           # `planned_proposals` is R at entry; `proposals_per_planned` (actual / R) well above 1
           # means that proposals concentrate in few cells.
           "proposal_model": ("R = nonuniformity * K0 * (H_{K0-e} - H_{K0-t})  "
                              "[coupon-collector occupancy, MEASURED non-uniformity correction]"),
           "planned_proposals": round(planned, 3),
           "planned_proposals_uniform": round(
               coverage_proposal_allocation(k0, entry, target, nonuniformity=1.0), 3),
           "planned_proposals_total": 0.0,
           "proposals_per_planned": None,
           "proposal_safety": safety,
           "proposal_nonuniformity": nonuniformity,
           "top_up_tranches": 0, "tranches": [],
           "max_evals_phase": phase_cap,
           "max_evals_phase_source": ("caller" if phase_max_evals else
                                      "auto max(1, 2*target_new)"),
           "max_repair_cycles": repair_cap,
           "stall_limit": stall,
           "k0": k0, "coverage_target_frac": random_init, "target_cells": target,
           "occupied_at_entry": entry, "target_new_cells": target_new,
           "coverage_at_entry": round(entry / k0, 4) if k0 else None,
           "order": ("cover_then_evaluate" if screen_on else "interleaved_unscreened_control"),
           "passes": 0, "cover_passes": 0, "cycles": 0, "repair_cycles": 0,
           "phi_covered": False, "cells_pooled": 0, "uncovered_after_evaluation": [],
           "proposals": 0, "localizations": 0, "evaluations": 0,
           "evaluations_valid": 0, "screened_out": 0,
           "screened_out_by_reason": {"occupied": 0, "claimed": 0, "target_met": 0,
                                      "localization_budget": 0, "eval_budget": 0,
                                      "phase_eval_budget": 0},
           "localization_errors": 0, "screen_errors": 0,
           "propose_errors": 0, "eval_errors": 0,
           "truncated_children": 0,
           "localization_wall_s": 0.0, "pass_rows": [], "localization_log": []}

    if target_new == 0:
        tel.update(status="already_covered", covered=True, occupied_at_exit=entry,
                   coverage_at_exit=round(entry / k0, 4) if k0 else None,
                   cells_newly_covered=0, evals_per_new_cell=None,
                   evals_saved_vs_unscreened=0, rejection_rate=0.0,
                   invalid_evals=0, phi_covered=True,
                   empty_cells=[str(c) for c in view.empty_cells()],
                   wall_s=round(_time.time() - t_start, 2))
        ctx.log(f"[{view.tag}] coverage-init: already covered — {entry}/{k0} cells occupied "
                f"(target {target}); the proposer authors no founders")
        return tel

    _screen_note = ("on" if screen_on else
                    "OFF (UNSCREENED CONTROL ARM: every candidate evaluated, as pymap_elites does)")
    ctx.log(f"[{view.tag}] coverage-init: target {target}/{k0} cells ({random_init:.0%} of K0), "
            f"{entry} occupied at entry -> {target_new} founder cell(s) for the proposer to reach. "
            f"ORDER={tel['order']} (stage A COVER is free; stage B EVALUATE runs only after "
            f"coverage passes; stage C REPAIRs cells an invalid evaluation left empty, "
            f"<= {repair_cap} cycle(s)). "
            f"PLAN: the occupancy model R = {nonuniformity:g}*K0*(H_(K0-e) - H_(K0-t)) "
            f"(coupon-collector, times the MEASURED non-uniformity correction) expects "
            f"{planned:.1f} "
            f"proposal(s) for this coverage — that is the FIRST TRANCHE, not a limit; if it is not "
            f"enough the phase KEEPS PROPOSING under recomputed top-up tranches. "
            f"Caps: {max_proposals} proposals ({max_proposals_src}) / {max_localizations} "
            f"localizations (WALL-CLOCK guards, not evaluation budget) / {phase_cap} EVALUATIONS "
            f"({tel['max_evals_phase_source']} — the phase-local leak-stopper, which IS the "
            f"evaluation budget) | screen={_screen_note} | strict={bool(strict)}")

    counters = {"loc": 0, "phi_s": 0.0}
    passes = {}                    # p -> {"meta", "parents", "rids"}, to land pooled genomes in B
    pool = []                      # stage-A (genome, cell) claims, in proposal order
    p = barren = 0
    status = "covered"
    # Current tranche: its planned size R and the proposal count when it opened.
    tranche = {"n": 0, "start": 0, "plan": 0.0}
    # Cells covered when the last cap extension was granted (None: no extension yet).
    ext = {"covered_at_last": None}

    def _row(u, i, key, dt, decision, reason, **kw):
        r = {"pass": p, "slot": u.slot, "rid": u.rid, "cand": i,
             "label": getattr(u.cands[i], "label", None), "cell": (None if key is None
                                                                   else str(key)),
             "decision": decision, "reason": reason,
             "localization_ms": (None if dt is None else int(dt * 1000))}
        r.update(kw)
        tel["localization_log"].append(r)
        if decision == "reject":
            tel["screened_out"] += 1
            tel["screened_out_by_reason"][reason] = \
                tel["screened_out_by_reason"].get(reason, 0) + 1
        return r

    def _phi_occupied():
        return len(set(view.occupied()) | {e["cell"] for e in pool})

    def _phi_covered():
        return _phi_occupied() >= target

    def _maybe_extend_cap(where):
        nonlocal max_proposals
        if not auto_extend:
            return False
        covered_now = _phi_occupied()
        new_cap, reason = cap_extension(
            proposals=tel["proposals"], cap=max_proposals, covered_now=covered_now,
            covered_at_last_extension=ext["covered_at_last"], extensions=tel["cap_extensions"],
            max_extensions=max_extensions,
            # Floored at the first tranche: near the target R shrinks to a single term, which would
            # leave the last (often hardest) cell the smallest window.
            block=max(coverage_proposal_allocation(k0, covered_now, target,
                                                   nonuniformity=nonuniformity), planned))
        if new_cap is None:
            tel["cap_extension_stop"] = reason
            ctx.log(f"[{view.tag}] INIT stage A cap NOT extended ({reason}) at {tel['proposals']} "
                    f"proposal(s), {covered_now}/{target} cell(s) covered"
                    + (" — the last extension bought no new cell, which is the signature of a "
                       "descriptor that cannot separate this proposer's output, and more proposals "
                       "cannot fix a distribution problem."
                       if reason == "no_progress_since_last_extension" else
                       f" — {tel['cap_extensions']} extension(s) already granted "
                       f"(COVERAGE_MAX_EXTENSIONS={max_extensions})."))
            return False
        prev, max_proposals = max_proposals, new_cap
        tel["cap_extensions"] += 1
        tel["max_proposals"] = max_proposals
        tel["cap_extension_log"].append({"extension": tel["cap_extensions"], "at": where,
                                         "proposals": tel["proposals"], "from": prev,
                                         "to": max_proposals, "covered": covered_now,
                                         "target": target, "cycle": tel["cycles"]})
        ext["covered_at_last"] = covered_now
        ctx.log(f"[{view.tag}] INIT stage A CAP EXTENSION {tel['cap_extensions']} of "
                f"{max_extensions}: the {prev}-proposal cap is spent with {covered_now}/{target} "
                f"cell(s) of K0={k0} covered, and the phase is STILL COVERING (it has gained a cell "
                f"since the last extension), so the cap grows to {max_proposals}. Stage A evaluates "
                f"nothing, so this spends wall clock and tokens, never N. The next extension is "
                f"granted only if another cell is covered before this cap is spent too.")
        return True

    def _open_tranche(kind):
        occ_now = _phi_occupied()
        r = coverage_proposal_allocation(k0, occ_now, target, nonuniformity=nonuniformity)
        tranche["n"] += 1
        tranche["start"] = tel["proposals"]
        tranche["plan"] = r
        if kind == "top_up":
            tel["top_up_tranches"] += 1
        tel["tranches"].append({"tranche": tranche["n"], "kind": kind, "cycle": tel["cycles"],
                                "occupied_at_open": occ_now, "target_cells": target,
                                "planned_proposals": round(r, 3),
                                "proposals_at_open": tel["proposals"]})
        return r

    def _tally(units, p_i):
        tel["passes"] = tel["cover_passes"] = p_i
        tel["proposals"] += sum(u.proposed for u in units)
        tel["truncated_children"] += sum(u.truncated for u in units)
        tel["evaluations"] += sum(u.evaluated for u in units)
        tel["evaluations_valid"] += sum(len(u.children) for u in units)
        tel["propose_errors"] += sum(1 for u in units if u.propose_error)
        tel["eval_errors"] += sum(u.eval_errors for u in units)
        # If `_screen_cover` raises, `breed_batch` evaluates the whole pass unscreened and uncapped.
        se = sum(1 for u in units if u.screen_error)
        if se:
            tel["screen_errors"] += se
            err = next(u.screen_error for u in units if u.screen_error)
            ctx.log(f"[{view.tag}] INIT coverage pass {p_i}: THE SCREEN ITSELF RAISED on {se} "
                    f"slot(s) ({err}) — breed_batch failed OPEN and evaluated those slots' "
                    f"candidates UNSCREENED and OUTSIDE both evaluation caps: "
                    f"{sum(u.evaluated for u in units)} evaluation(s) this pass are charged to N")

    # ── Stage A: cover (no evaluations) ──────────────────────────────────────────────────────
    def _screen_cover(units):
        """Stage-A screen: pool the first candidate per new cell, evaluating none (fail-open)."""
        keep = {}
        live = set(view.occupied())
        claimed = {e["cell"]: (e["slot"], e["cand"]) for e in pool}
        for u in units:
            keep[u.slot] = set()
            for i, g in enumerate(u.cands):
                if counters["loc"] >= max_localizations:
                    _row(u, i, None, None, "reject", "localization_budget"); continue
                if len(live | set(claimed)) >= target:
                    _row(u, i, None, None, "reject", "target_met"); continue
                t0 = _time.time()
                try:
                    key = view.key_of_genome(g)
                    # An unhashable key becomes a counted `localization_error` here instead of
                    # failing the whole screen.
                    hash(key)
                except Exception as e:                  # noqa: BLE001 — fail open, counted
                    tel["localization_errors"] += 1
                    # The counters update at pass boundaries, so fail-open can overshoot by a pass.
                    if max_evals is not None and ctx._evals >= max_evals:
                        _row(u, i, None, None, "reject", "eval_budget"); continue
                    if tel["evaluations"] >= phase_cap:
                        _row(u, i, None, None, "reject", "phase_eval_budget"); continue
                    ctx.log(f"[{view.tag}] INIT coverage pass {p} slot {u.slot} cand {i}: "
                            f"localization failed ({e!r}) — evaluating UNSCREENED")
                    keep[u.slot].add(i)
                    _row(u, i, None, None, "evaluate", "localization_error")
                    continue
                dt = _time.time() - t0
                counters["loc"] += 1; counters["phi_s"] += dt
                if key in live:
                    _row(u, i, key, dt, "reject", "occupied")
                elif key in claimed:
                    _row(u, i, key, dt, "reject", "claimed", claimed_by=list(claimed[key]))
                else:
                    claimed[key] = (u.slot, i)
                    row = _row(u, i, key, dt, "pool", "new_cell", evaluated=False)
                    pool.append({"genome": g, "cell": key, "pass": p, "slot": u.slot,
                                 "cand": i, "rid": u.rid, "row": row})
                    tel["cells_pooled"] += 1
        return keep

    def _stage_a():
        """Propose and pool until covered; returns "covered_in_phi" or the stop reason."""
        nonlocal p
        dry, dry_cap = 0, max(2, int(stall or 0))
        r0 = _open_tranche("first" if tel["cycles"] <= 1 else "repair")
        ctx.log(f"[{view.tag}] INIT stage A tranche {tranche['n']} "
                f"({tel['tranches'][-1]['kind']}): occupancy model "
                f"R = {nonuniformity:g}*K0*(H_(K0-e) - H_(K0-t)) "
                f"plans {r0:.1f} proposal(s) to carry "
                f"{tel['tranches'][-1]['occupied_at_open']}/{target} cell(s) of K0={k0} to the "
                f"target. An ALLOCATION, not a limit.")
        while True:
            if _phi_covered():
                return "covered_in_phi"
            if tel["proposals"] >= max_proposals and not _maybe_extend_cap("stage_A"):
                return "proposal_budget"
            # A spent tranche is not a cap: plan the next one from the current occupancy.
            if tel["proposals"] - tranche["start"] >= tranche["plan"]:
                spent, prev = tel["proposals"] - tranche["start"], tranche["plan"]
                r = _open_tranche("top_up")
                ctx.log(f"[{view.tag}] INIT stage A TOP-UP {tel['top_up_tranches']}: the planned "
                        f"tranche of {prev:.1f} proposal(s) is spent ({spent} used) and only "
                        f"{_phi_occupied()}/{target} cell(s) of K0={k0} are covered — the plan is a "
                        f"FIRST TRANCHE, not a cap, so the phase KEEPS PROPOSING: tranche "
                        f"{tranche['n']} plans {r:.1f} more. Total spent {tel['proposals']} of the "
                        f"{max_proposals}-proposal WALL-CLOCK hard cap ({max_proposals_src}). "
                        f"A run that keeps topping up is telling you proposals are not landing "
                        f"UNIFORMLY over the K0 cells — read proposals_per_planned, and suspect the "
                        f"CVT was fitted on a different population than the proposer samples from.")
            if counters["loc"] >= max_localizations:
                return "localization_budget"
            if max_evals is not None and ctx._evals >= max_evals:
                return "eval_budget"
            if tel["evaluations"] >= phase_cap:
                return "phase_eval_budget"
            if ctx.should_stop and ctx.should_stop():
                return "stopped"
            p += 1
            parents, meta = view.select(p, batch, {e["cell"] for e in pool})
            if not parents:
                return "no_parent"
            n_pool0, n_loc0, n_prop0 = len(pool), counters["loc"], tel["proposals"]
            # Not a generation, so C (children per parent) does not apply.
            units = _arch.breed_batch(ctx, parents, directives=meta["directives"], n_workers=n_workers,
                                gen=-p, tag=f"{view.tag}-cover",
                                screen=_screen_cover, n_children=meta.get("n_children"))
            passes[p] = {"meta": meta, "parents": parents, "rids": [u.rid for u in units]}
            _tally(units, p)
            if any(u.children for u in units):          # only fail-open candidates are evaluated
                row = view.land(meta, units, p)
                row["stage"] = "A_cover_fail_open"
                tel["pass_rows"].append(row)
            ev = sum(u.evaluated for u in units)
            ctx.log(f"[{view.tag}] INIT stage A (COVER — FREE) pass {p}: "
                    f"{sum(u.proposed for u in units)} proposed, "
                    f"{counters['loc'] - n_loc0} localized, +{len(pool) - n_pool0} pooled -> the "
                    f"pool phi-covers "
                    f"{len(set(view.occupied()) | {e['cell'] for e in pool})}/{target} of "
                    f"K0={k0} | {ev} evaluated (stage A evaluates NOTHING; a non-zero count here "
                    f"is the fail-open localization error)")
            # No proposal and no localization this pass: counts toward the `no_proposals` exit.
            if tel["proposals"] == n_prop0 and counters["loc"] == n_loc0:
                dry += 1
                if dry >= dry_cap:
                    ctx.log(f"[{view.tag}] INIT stage A: {dry} consecutive pass(es) produced NO "
                            f"CANDIDATES — neither a proposal nor a localization was spent, so no "
                            f"proposal/localization/evaluation cap can ever fire and the phase "
                            f"would propose forever. Stopping at {_phi_occupied()}/{target} cell(s)"
                            f" of K0={k0}. This is the PROPOSER returning an empty candidate list "
                            f"(or raising, which breed_batch's isolated worker reports as one) — "
                            f"raising random_init_max_proposals cannot help.")
                    return "no_proposals"
            else:
                dry = 0

    # ── Stage B: evaluate ──────────────────────────────────────────────────────────────────────
    def _eval_pooled(units):
        """Evaluate pooled genomes (`evaluate_batch` only if the adapter is `EVAL_SUBSET_SAFE`)."""
        import concurrent.futures as _futures
        subset_ok = bool(getattr(ctx.adapter, "EVAL_SUBSET_SAFE", False))
        hook = getattr(ctx.adapter, "evaluate_batch", None)
        w = _arch.effective_workers(ctx.adapter, n_workers, log=lambda *a, **k: None, stage="eval")
        for u in units:
            u.evaluated = len(u.cands)
            u.fits = [None] * len(u.cands)
        if hook is not None and subset_ok:
            for u in units:
                if not u.cands:
                    continue
                try:
                    u.fits = _evaluate_all(ctx.adapter, u.cands, u.rid, check_contract=True)
                except BatchContractError:
                    raise                                # adapter bug, not a transient failure
                except Exception as e:                  # noqa: BLE001
                    u.fits = [None] * len(u.cands)
                    u.eval_errors += len(u.cands)
                    ctx.log(f"[{view.tag}] INIT stage B slot {u.slot}: evaluate_batch failed for "
                            f"all {len(u.cands)} pooled candidate(s): {e}")
            return
        if hook is not None and not _eval_pooled.noted:
            _eval_pooled.noted = True
            ctx.log(f"[{view.tag}] {type(ctx.adapter).__name__} owns evaluate_batch but does not "
                    f"declare EVAL_SUBSET_SAFE — stage B evaluates the POOLED genomes with "
                    f"per-genome adapter.evaluate() (its propose-time stash is stale by "
                    f"construction here)")
        flat = [(u, i, g) for u in units for i, g in enumerate(u.cands)]
        if w > 1 and len(flat) > 1:
            with _futures.ThreadPoolExecutor(max_workers=w) as ex:
                futs = [ex.submit(ctx.adapter.evaluate, g, u.rid) for u, _, g in flat]
            pairs = list(zip(flat, futs))
            getters = [(u, i, f.result) for (u, i, _), f in pairs]
        else:
            getters = [(u, i, (lambda u=u, g=g: ctx.adapter.evaluate(g, u.rid)))
                       for u, i, g in flat]
        for u, i, get in getters:
            try:
                u.fits[i] = get()
            except Exception as e:                       # noqa: BLE001 — drop the candidate
                u.fits[i] = None
                u.eval_errors += 1
                ctx.log(f"[{view.tag}] INIT stage B slot {u.slot} cand {i}: evaluate failed: {e}")

    _eval_pooled.noted = False

    def _stage_b():
        """Evaluate one pooled genome per still-empty cell; returns (evaluations, cells gained)."""
        live = set(view.occupied())
        room = max(0, target - len(live))
        chosen, seen = [], set()
        for e in pool:
            row, cell = e["row"], e["cell"]
            if cell in live or cell in seen:
                row.update(evaluated=False, skipped="occupied"); continue
            if len(chosen) >= room:
                row.update(evaluated=False, skipped="target_met"); continue
            if max_evals is not None and ctx._evals + len(chosen) >= max_evals:
                row.update(evaluated=False, skipped="eval_budget"); continue
            if tel["evaluations"] + len(chosen) >= phase_cap:
                row.update(evaluated=False, skipped="phase_eval_budget"); continue
            seen.add(cell); chosen.append(e)
        occ0 = len(live)
        by_pass = {}
        for e in chosen:
            by_pass.setdefault(e["pass"], []).append(e)
        n_ev = 0
        for pp in sorted(by_pass):
            info = passes[pp]
            dirs = info["meta"].get("directives") or [None] * len(info["parents"])
            units = [_arch.BredUnit(parent=info["parents"][j], slot=j, rid=info["rids"][j],
                              directive=dirs[j]) for j in range(len(info["parents"]))]
            for e in by_pass[pp]:
                units[e["slot"]].cands.append(e["genome"])
            _eval_pooled(units)
            for u in units:
                ctx.spend(u.evaluated)                   # charge the run's budget N
                u.children = [Node(ctx.next_idx(), g, f)
                              for g, f in zip(u.cands, u.fits)
                              if f is not None and getattr(f, "valid", False)]
                n_ev += u.evaluated
            valid = {id(g) for u in units for g, f in zip(u.cands, u.fits)
                     if f is not None and getattr(f, "valid", False)}
            for e in by_pass[pp]:
                e["row"].update(evaluated=True, valid=(id(e["genome"]) in valid),
                                eval_cycle=tel["cycles"])
            tel["evaluations"] += sum(u.evaluated for u in units)
            tel["evaluations_valid"] += sum(len(u.children) for u in units)
            tel["eval_errors"] += sum(u.eval_errors for u in units)
            row = view.land(info["meta"], units, pp)
            row["stage"] = "B_evaluate"
            tel["pass_rows"].append(row)
        gained = len(view.occupied()) - occ0
        ctx.log(f"[{view.tag}] INIT stage B (EVALUATE — PAID, cycle {tel['cycles']}): "
                f"{n_ev} evaluation(s) for the {len(chosen)} cell(s) the pool covered "
                f"(one genome per cell, first-localized in proposal order) -> archive occupies "
                f"{len(view.occupied())}/{target} of K0={k0}, +{gained} this stage")
        return n_ev, gained

    # ── Cover-evaluate loop ──────────────────────────────────────────────────────────────────
    def _cover_then_evaluate():
        """Screened path, A -> B -> (C -> A -> B)*; a failed stage A sets `status` to its reason."""
        nonlocal status, barren
        while True:
            n_occ = len(view.occupied())
            if n_occ >= target:
                status = "covered"; break
            if tel["proposals"] >= max_proposals and not _maybe_extend_cap("cycle_boundary"):
                status = "proposal_budget"; break
            if counters["loc"] >= max_localizations:
                status = "localization_budget"; break
            if max_evals is not None and ctx._evals >= max_evals:
                status = "eval_budget"; break
            if tel["evaluations"] >= phase_cap:
                status = "phase_eval_budget"; break
            if stall and barren >= stall:
                status = "stalled"; break
            if ctx.should_stop and ctx.should_stop():
                status = "stopped"; break
            if tel["cycles"] > repair_cap:           # cycle 1 + `repair_cap` repair cycles
                status = "repair_budget"; break
            tel["cycles"] += 1
            tel["repair_cycles"] = tel["cycles"] - 1
            a = _stage_a()                                # stage A: cover
            tel["phi_covered"] = (a == "covered_in_phi")
            if not tel["phi_covered"]:
                status = a
                ctx.log(f"[{view.tag}] INIT stage A did NOT reach the target (stop reason={a}) — "
                        f"stage B is NOT entered, so the shortfall cost "
                        f"{tel['evaluations']} evaluation(s)")
                break
            _n_ev, gained = _stage_b()                    # stage B: evaluate
            pooled_cells = [e["cell"] for e in pool]
            pool.clear()
            barren = 0 if gained > 0 else barren + 1
            if len(view.occupied()) < target:             # stage C: repair
                _occ = set(view.occupied())
                tel["uncovered_after_evaluation"] = [str(c) for c in pooled_cells if c not in _occ]
                ctx.log(f"[{view.tag}] INIT stage C (REPAIR): phi-coverage was reached but "
                        f"{target - len(view.occupied())} cell(s) came back empty (an evaluation "
                        f"returned None/invalid/taint-FAIL, so a cell the POOL covered the archive "
                        f"does not) — re-entering stage A with the REMAINING caps as repair cycle "
                        f"{tel['cycles']} of {repair_cap}")

    def _interleaved_unscreened():
        """Unscreened baseline: evaluate every proposed candidate, as in standard MAP-Elites."""
        nonlocal status, p, barren
        while True:
            n_occ = len(view.occupied())
            if n_occ >= target:
                status = "covered"; break
            if tel["proposals"] >= max_proposals:
                status = "proposal_budget"; break
            if max_evals is not None and ctx._evals >= max_evals:
                status = "eval_budget"; break
            if tel["evaluations"] >= phase_cap:
                status = "phase_eval_budget"; break
            if stall and barren >= stall:
                status = "stalled"; break
            if ctx.should_stop and ctx.should_stop():
                status = "stopped"; break
            p += 1
            tel["cycles"] = p
            parents, meta = view.select(p, batch, set())
            if not parents:
                status = "no_parent"; break
            units = _arch.breed_batch(ctx, parents, directives=meta["directives"], n_workers=n_workers,
                                gen=-p, tag=f"{view.tag}-cover", screen=None)
            _tally(units, p)
            row = view.land(meta, units, p)
            row["stage"] = "unscreened_interleaved"
            tel["pass_rows"].append(row)
            gained = len(view.occupied()) - n_occ
            barren = 0 if gained > 0 else barren + 1
            ctx.log(f"[{view.tag}] INIT coverage pass {p} (UNSCREENED control arm — propose and "
                    f"evaluate INTERLEAVED): +{gained} cell(s) -> {len(view.occupied())}/{target} "
                    f"of K0={k0} | this pass: {sum(u.proposed for u in units)} proposed, "
                    f"{sum(u.evaluated for u in units)} evaluated")

    (_cover_then_evaluate if screen_on else _interleaved_unscreened)()

    exit_occ = len(view.occupied())
    gained = exit_occ - entry
    tel["localizations"] = counters["loc"]
    tel["planned_proposals_total"] = round(sum(t["planned_proposals"] for t in tel["tranches"]), 3)
    tel["proposals_per_planned"] = (round(tel["proposals"] / planned, 3) if planned > 0 else None)
    tel["uniform_occupancy_consistent"] = (None if not planned or not tel["proposals"]
                                           else bool(tel["proposals"] <= math.ceil(planned)))
    tel["localization_wall_s"] = round(counters["phi_s"], 3)
    tel["localization_ms_mean"] = (int(1000 * counters["phi_s"] / counters["loc"])
                                   if counters["loc"] else None)
    tel.update(status=status, covered=(exit_occ >= target), occupied_at_exit=exit_occ,
               coverage_at_exit=round(exit_occ / k0, 4) if k0 else None,
               cells_newly_covered=gained,
               evals_per_new_cell=(round(tel["evaluations"] / gained, 3) if gained else None),
               evals_saved_vs_unscreened=tel["proposals"] - tel["evaluations"],
               rejection_rate=(round(tel["screened_out"] / counters["loc"], 3)
                               if counters["loc"] else 0.0),
               invalid_evals=tel["evaluations"] - tel["evaluations_valid"],
               empty_cells=[str(c) for c in view.empty_cells()],
               wall_s=round(_time.time() - t_start, 2))

    _plan_clause = (
        f"PLAN vs ACTUAL: planned R={planned:.1f} proposal(s) (first tranche; "
        f"{tel['planned_proposals_total']:.1f} summed over {len(tel['tranches'])} tranche(s), "
        f"{tel['top_up_tranches']} of them TOP-UPs), spent {tel['proposals']} -> ratio "
        f"{tel['proposals_per_planned']} against a hard cap of {max_proposals} "
        f"({max_proposals_src}). A ratio persistently above 1 means proposals are NOT landing "
        f"uniformly over the K0 cells — the model's one assumption — which is what a CVT fitted on "
        f"a different population than the proposer samples from looks like from here.")

    if tel["covered"]:
        ctx.log(f"[{view.tag}] coverage-init COVERED: {exit_occ}/{k0} cells "
                f"({exit_occ / k0:.1%}, target {target}) in {p} pass(es) / {tel['cycles']} "
                f"cover-evaluate cycle(s) ({tel['repair_cycles']} repair) — "
                f"{tel['proposals']} proposals / {counters['loc']} localizations / "
                f"{tel['evaluations']} EVALUATIONS ({tel['screened_out']} screened out unevaluated)"
                f" | {_plan_clause}")
    else:
        ctx.log(f"[{view.tag}] coverage-init SHORTFALL: {exit_occ}/{k0} cells "
                f"({exit_occ / k0:.1%}) against a target of {target} ({random_init:.1%} of K0) — "
                f"stop reason={status}. Spent {tel['proposals']} proposals and "
                f"{counters['loc']} localizations (the WALL-CLOCK currency) but only "
                f"{tel['evaluations']} evaluations of a phase cap of {phase_cap} (the BUDGET "
                f"currency) — " + _shortfall_cost_clause(tel) + ". "
                f"Still-empty cells: {tel['empty_cells']}. Rejection rate "
                f"{tel['rejection_rate']} — near 1.0 means the proposer is not producing structural "
                f"novelty, so the DIRECTIVE needs work, not the mechanism. " + _plan_clause)
    return tel


def _coverage_raise(tel, mode):
    """Return (not raise) the `CoverageError` for a shortfall; strict callers raise it."""
    return CoverageError(
        f"{mode}: coverage-gated initialization did NOT reach its target. "
        f"ACHIEVED {tel['occupied_at_exit']}/{tel['k0']} cells "
        f"({tel['coverage_at_exit']:.1%}); TARGET {tel['target_cells']}/{tel['k0']} "
        f"({tel['coverage_target_frac']:.1%} of K0). Stop reason={tel['status']}. "
        f"SPENT — proposals {tel['proposals']}, localizations {tel['localizations']} "
        f"(wall-clock/token currency); evaluations {tel['evaluations']} of a phase cap of "
        f"{tel['max_evals_phase']} ({tel['max_evals_phase_source']}), of which "
        f"{tel['invalid_evals']} invalid (the N currency — the phase-local cap is what keeps a "
        f"shortfall from costing N, which is why the PROPOSAL caps are generous and this is an "
        f"ERROR rather than a warning). ORDER={tel['order']}: stage A (COVER) is free and stage B "
        f"(EVALUATE) is entered only once coverage is reached, so "
        + _shortfall_cost_clause(tel) +
        f"; {tel['repair_cycles']} of {tel['max_repair_cycles']} "
        f"repair cycle(s) ran (stage C, re-covering cells an invalid evaluation left empty). "
        f"STILL-EMPTY CELLS: {tel['empty_cells']}. "
        f"Rejection rate {tel['rejection_rate']}. "
        f"THE PLAN: the occupancy model "
        f"R = {tel['proposal_nonuniformity']:g}*K0*(H_(K0-e) - H_(K0-t)) allocated "
        f"{tel['planned_proposals']} proposal(s) as the FIRST TRANCHE; "
        f"{tel['top_up_tranches']} TOP-UP tranche(s) then ran (the plan is not a limit — the phase "
        f"keeps proposing past it), for {tel['proposals']} spent against the "
        f"{tel['max_proposals']}-proposal hard cap ({tel['max_proposals_source']}"
        + (f", AUTO-EXTENDED {tel['cap_extensions']}x from {tel['max_proposals_initial']} and then "
           f"stopped on '{tel['cap_extension_stop']}'" if tel["cap_extensions"] else
           (f", auto-extension refused at the first cap on '{tel['cap_extension_stop']}'"
            if tel.get("cap_extension_stop") else ", auto-extension OFF")) + f"), a "
        f"plan/actual ratio of {tel['proposals_per_planned']} (the plan ALREADY carries the "
        f"{tel['proposal_nonuniformity']:g}x non-uniformity correction — the uncorrected uniform "
        f"expectation was {tel['planned_proposals_uniform']} — so this ratio is measured against "
        f"a corrected mean and ~1 is its healthy value). A ratio well above 1 says the "
        f"proposals are landing even less evenly than that correction assumes, so raising the "
        f"cap chases a "
        f"distribution problem with wall clock: check the CVT was fitted on the population the "
        f"proposer actually samples from (a mismatched fit has been measured putting 100% of "
        f"proposals into ONE cell of six). "
        f"Proceeding would enter the generations having covered less of K0 than the run promised, "
        f"which invalidates the K0 that N = M + G*P*C is stated against. Either raise "
        f"random_init_safety / random_init_max_proposals (stop reason proposal_budget) or "
        f"random_init_max_evals (stop "
        f"reason phase_eval_budget — but read that one as 'the proposer is burning EVALUATIONS on "
        f"candidates that come back invalid', which raising the cap only makes more expensive), "
        f"improve the proposer's structural novelty, lower random_init, "
        f"or pass random_init_strict=False to accept partial coverage explicitly.")
