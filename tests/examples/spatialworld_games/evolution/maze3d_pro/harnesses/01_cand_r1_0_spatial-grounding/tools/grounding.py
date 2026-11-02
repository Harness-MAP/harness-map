"""Frame-differencing PROPRIOCEPTION for maze3d_pro: turn the rendered frame into self-derived
spatial state.

WHY THIS EXISTS (all numbers below are MEASURED on this build, see NOTES.md):
the maze3d_pro frame carries NO HUD — the whole 800x600 render uses 7 colours with a max channel
of 147 and not a single bright text pixel, so there is no POS/FLOOR/DIR readout and no
"[Q] UP"/"[E] DOWN" stair prompt. The agent therefore has NO channel that tells it whether the
action it just took did anything. Replaying the parent's own recorded answers through the real
evaluator, 48-100% of every episode's actions produced a BYTE-IDENTICAL frame (level 10 and
level 18: 99 of 99 transitions, one single distinct view for the whole 100-step episode).

The cure is proprioception, not prose: an engine action that leaves the frame byte-identical
changed NOTHING (verified across move_forward / move_backward / turn / climb_up / climb_down —
`frame_data.image` and `evaluator.capture_game_screenshot()` move in exact lockstep). That one
bit — "did the view change?" — is enough to dead-reckon heading, floor and a local wall map
entirely from the observation, with no privileged state of any kind.

ANTI-CHEAT: every field here is derived from the pixels the model is shown plus the action the
model itself chose. Nothing reads `evaluator.game`, the layout file, the exit, or the true pose.
"""
from collections import Counter
from hashlib import blake2b

# --- the action algebra, in step with tools/action.py and rules.md -------------------
TURN_DELTA = {"turn_left": -1, "turn_right": 1}
TRANSLATION = {"move_forward": 1, "move_backward": -1}
CLIMB_DELTA = {"climb_up": 1, "climb_down": -1}
GRAN_CELLS = {"small": 1, "medium": 2, "large": 3}
# bearings are RELATIVE to the heading the episode started with: 0 = the start facing.
BEARING_VEC = ((0, 1), (1, 0), (0, -1), (-1, 0))
BEARING_NAME = ("start-facing", "right-of-start", "behind-start", "left-of-start")
# which turn sequence aims the body at a bearing offset from the current heading
AIM = {0: "already facing it", 1: "turn_right", 2: "turn_right twice", 3: "turn_left"}


def _strip_comments(text: str) -> str:
    """Drop the <!-- ... --> author notes so only the scratchpad itself reaches the agent."""
    out, rest = [], text or ""
    while "<!--" in rest:
        head, _, rest = rest.partition("<!--")
        out.append(head)
        _, _, rest = rest.partition("-->")
    out.append(rest)
    return "".join(out).strip()


def view_digest(image) -> str:
    """A collision-free id for the rendered view. The frame is a deterministic pure function of
    the game state, so equal digests on consecutive steps means the state did not move."""
    try:
        import numpy as np
        buf = np.ascontiguousarray(image).tobytes()
    except Exception:                      # noqa: BLE001 — a str/bytes frame still hashes
        buf = image if isinstance(image, bytes) else str(image).encode()
    return blake2b(buf, digest_size=10).hexdigest()


class _Region:
    """A dead-reckoned local map. One region per floor-visit: a climb lands the body somewhere
    unknown, so rather than fake a coordinate we simply start a fresh local frame there."""

    __slots__ = ("cell", "walls", "open_edges", "visited", "exact")

    def __init__(self):
        self.cell = (0, 0)
        self.walls = set()          # (cell, bearing) proven to be a wall
        self.open_edges = set()     # (cell, bearing) proven traversable
        self.visited = Counter({(0, 0): 1})
        self.exact = True           # False once a medium/large move landed an unknown distance


class Grounding:
    """Per-episode proprioceptive state. `observe` closes out the PREVIOUS action by diffing the
    frame; `commit` records the action about to be executed."""

    MAX_TRACE = 6

    def __init__(self, template: str = ""):
        # memory.md documents itself in HTML comments; those are for the proposer, not the agent
        self.template = _strip_comments(template)
        self.step = 0
        self.heading = 0            # exact: a turn is never blocked
        self.floor = 0              # exact: net SUCCESSFUL climbs
        self.regions = [_Region()]
        self.prev_digest = None
        self.pending = None         # (action_name, granularity) issued last step
        self.stay_nulls = set()     # actions PROVEN null at the current, unchanged state
        self.stay_len = 0           # steps spent on the current view
        self.view_seen = Counter()
        self.view_nulls = {}        # digest -> set(actions) — advisory across the episode
        self.n_actions = 0
        self.n_nulls = 0
        self.trace = []             # (action, "moved"|"no-op")
        self.vetoes = 0
        self.substitutions = 0
        self.sight_lines = []
        self.sight_flags = {}

    # ── the map of the floor the body is standing on ────────────────────────────────
    @property
    def region(self) -> _Region:
        return self.regions[-1]

    def _ahead(self, bearing_offset: int = 0):
        return (self.heading + bearing_offset) % 4

    def _neighbour(self, bearing: int):
        x, y = self.region.cell
        dx, dy = BEARING_VEC[bearing]
        return (x + dx, y + dy)

    # ── perception ──────────────────────────────────────────────────────────────────
    def observe(self, image) -> str:
        """Diff this frame against the previous one and resolve the pending action."""
        digest = view_digest(image)
        self.sight_lines, self.sight_flags = scan_tiles(image)
        if self.pending is not None:
            self._resolve(self.pending, changed=(digest != self.prev_digest))
            self.pending = None            # resolved exactly once, never re-applied
        if self.prev_digest is not None and digest == self.prev_digest:
            self.stay_len += 1
        else:
            self.stay_nulls = set()
            self.stay_len = 0
        self.prev_digest = digest
        self.view_seen[digest] += 1
        self.step += 1
        return digest

    def _resolve(self, pending, changed: bool):
        name, gran = pending
        self.n_actions += 1
        if not changed:
            self.n_nulls += 1
            self.stay_nulls.add(name)
            self.view_nulls.setdefault(self.prev_digest, set()).add(name)
        self.trace.append((name, "moved" if changed else "NO-OP"))
        del self.trace[:-self.MAX_TRACE]

        if name in TURN_DELTA:
            # a turn always rotates the body, whether or not the pixels happen to differ
            self.heading = (self.heading + TURN_DELTA[name]) % 4
            return

        if name in TRANSLATION:
            bearing = self._ahead(0 if TRANSLATION[name] > 0 else 2)
            reg = self.region
            if not changed:
                if reg.exact:
                    reg.walls.add((reg.cell, bearing))
                return
            if GRAN_CELLS.get(gran, 1) == 1:
                if reg.exact:
                    reg.open_edges.add((reg.cell, bearing))
                    reg.open_edges.add((self._neighbour(bearing), (bearing + 2) % 4))
                    reg.cell = self._neighbour(bearing)
                    reg.visited[reg.cell] += 1
            else:
                # a medium/large move stops early at a wall: the distance covered is unknowable
                reg.exact = False
            return

        if name in CLIMB_DELTA:
            if changed:
                self.floor += CLIMB_DELTA[name]
                self.regions.append(_Region())   # new floor, fresh local frame
            elif self.region.exact:
                self.region.walls.add((self.region.cell, "stair:" + name))

    def commit(self, name: str, granularity: str = "small") -> None:
        """Record the action that is about to be executed, so the NEXT frame can resolve it."""
        self.pending = (name, (granularity or "small").strip().lower())

    def desync(self) -> None:
        """Called when a step bypassed this tool (the harness fell back to the pristine VLM step
        after an error). An unseen action may have turned or moved the body, so stop asserting a
        map that can no longer be trusted — heading and position are abandoned, and only the
        airtight "same view, same action, same nothing" rule keeps working."""
        self.pending = None
        self.stay_nulls = set()
        self.regions.append(_Region())
        self.region.exact = False

    # ── what the mechanism refuses ──────────────────────────────────────────────────
    def proven_null(self) -> set:
        """Actions that CANNOT do anything from here. Two airtight sources:
        (1) the action was tried since the last time the view changed — the state is provably
            unchanged, so repeating it must produce the same nothing;
        (2) the local map (built only while dead reckoning is exact) already proves a wall on
            the bearing the action would travel.
        Turns are NEVER refused: a turn is the one action that always re-aims the body, and it
        is the escape hatch out of every fixed point."""
        out = {a for a in self.stay_nulls if a not in TURN_DELTA}
        reg = self.region
        if reg.exact:
            if (reg.cell, self._ahead(0)) in reg.walls:
                out.add("move_forward")
            if (reg.cell, self._ahead(2)) in reg.walls:
                out.add("move_backward")
            for climb in CLIMB_DELTA:
                if (reg.cell, "stair:" + climb) in reg.walls:
                    out.add(climb)
        return out

    def rescue(self) -> str:
        """The most productive action the frame has NOT already refuted. Walking is preferred to
        turning — a rescue only fires after the model picked something provably inert, so the
        point is to spend the step on ground that can actually change. A turn is the fallback
        because a turn can never be refused by the frame, which makes this terminate."""
        dead = self.proven_null()
        if "move_forward" not in dead:
            return "move_forward"
        reg = self.region
        right, left = self._ahead(1), self._ahead(3)
        for bearing, turn in ((right, "turn_right"), (left, "turn_left")):
            if reg.exact and (reg.cell, bearing) not in reg.walls \
                    and reg.visited.get(self._neighbour(bearing), 0) == 0:
                return turn                      # aim at ground never walked
        for bearing, turn in ((right, "turn_right"), (left, "turn_left")):
            if not reg.exact or (reg.cell, bearing) not in reg.walls:
                return turn
        if "move_backward" not in dead:
            return "move_backward"
        return "turn_right"

    # ── the scratchpad the agent is shown ───────────────────────────────────────────
    def _bearing_table(self) -> str:
        reg = self.region
        rows = []
        for off in (0, 1, 2, 3):
            bearing = self._ahead(off)
            if not reg.exact:
                status = "unknown (dead reckoning lost)"
            elif (reg.cell, bearing) in reg.walls:
                status = "WALL — proven, never retry it"
            elif (reg.cell, bearing) in reg.open_edges:
                seen = reg.visited.get(self._neighbour(bearing), 0)
                status = f"OPEN — already walked ({seen} visit(s) to that cell)"
            else:
                status = "UNEXPLORED — this is where new maze is"
            label = ("ahead", "to your right", "behind you", "to your left")[off]
            rows.append(f"  - {label:<14} ({BEARING_NAME[bearing]:<14}): {status}"
                        f"   [aim: {AIM[off]}]")
        return "\n".join(rows)

    def block(self, refused: str = None) -> str:
        reg = self.region
        last = f"{self.trace[-1][0]} -> {self.trace[-1][1]}" if self.trace else "none yet"
        waste = (100.0 * self.n_nulls / self.n_actions) if self.n_actions else 0.0
        pos = f"({reg.cell[0]}, {reg.cell[1]}) cells from where this floor began" if reg.exact \
            else "UNKNOWN — a medium/large move covered an unknowable distance"
        null_now = sorted(self.proven_null())
        values = {
            "STEP": str(self.step),
            "LAST": last,
            "STUCK": (f"{self.stay_len} step(s) in a row have left the view BYTE-IDENTICAL"
                      if self.stay_len else
                      ("the view changed on your last action" if self.trace
                       else "no action taken yet this episode")),
            "HEADING": f"{BEARING_NAME[self.heading]} (quarter-turns from the start facing: {self.heading})",
            "FLOOR": f"{self.floor:+d} relative to the floor you started on"
                     f" ({len(self.regions) - 1} successful climb(s))",
            "POS": pos,
            "BEARINGS": self._bearing_table(),
            "DEAD": (", ".join(null_now) if null_now else "none — every action can still do something"),
            "REVISIT": (f"this exact view has now been rendered {self.view_seen[self.prev_digest]} time(s)"
                        if self.prev_digest else "first frame"),
            "SIGHT": "\n".join("  - " + ln for ln in self.sight_lines),
            "TRACE": ", ".join(f"{a}:{o}" for a, o in self.trace) or "none yet",
            "WASTE": f"{self.n_nulls}/{self.n_actions} actions so far did literally nothing ({waste:.0f}%)",
            "REFUSED": ("" if not refused else
                        f"\nYOUR LAST ANSWER WAS REFUSED: `{refused}` is one of the dead actions above. "
                        f"It cannot move you. Pick a DIFFERENT action — a turn is always available."),
        }
        text = self.template or _FALLBACK_TEMPLATE
        for key, val in values.items():
            text = text.replace("{{" + key + "}}", val)
        return text


_FALLBACK_TEMPLATE = """## grounded state (measured from the frames, not guessed)
- what is on this frame right now:
{{SIGHT}}
- step: {{STEP}}
- last action: {{LAST}}
- {{STUCK}}
- facing: {{HEADING}}
- floor: {{FLOOR}}
- position: {{POS}}
- neighbours:
{{BEARINGS}}
- dead actions right now: {{DEAD}}
- {{REVISIT}}
- recent: {{TRACE}}
- {{WASTE}}{{REFUSED}}
"""


# ── salient-tile scanner ────────────────────────────────────────────────────────────
# MEASURED on this build by replaying the parent's two WINNING tapes (levels 6 and 17) and
# sweeping 25 levels with a frame-diff wall follower: the maze renders in 7 brown/grey shades
# with a max channel of 147, and the ONLY saturated colours that ever appear are
#   exit  = RGB(255,200,0) filled, RGB(255,255,255) outline
#   stair = RGB(0,204,204)  border, RGB(0,76,91)/RGB(0,52,62) fill
# The corridor is drawn as concentric rectangles centred on (col 400, row 300) whose width is
# 600/depth (measured rings: 597,297,197,147,119,99,83 px -> depths 1..7), so a face-on tile's
# pixel width gives its distance in CELLS. Reading these is reading the frame the model is
# already shown — the same photons, counted instead of guessed.
EXIT_RGB = (255, 200, 0)
EXIT_OUTLINE_RGB = (255, 255, 255)
STAIR_RGB = (0, 204, 204)
CENTRE_COL = 400
RING_WIDTH_AT_DEPTH_1 = 600.0
RING_ASPECT = 0.670          # ring height / ring width: measured 0.655-0.688, 46/46
ASPECT_TOLERANCE = 0.05      # anything outside this is a clipped / occluded tile
LANE_TOLERANCE = 25          # px from centre that still counts as "dead ahead" (all 46
                             # measured sightings sat at exactly 400.0)


def _blob(arr, rgb):
    """Bounding box + pixel count of one exact colour, or None when it is absent."""
    mask = (arr[:, :, 0] == rgb[0]) & (arr[:, :, 1] == rgb[1]) & (arr[:, :, 2] == rgb[2])
    count = int(mask.sum())
    if not count:
        return None
    rows, cols = mask.nonzero()
    x0, x1, y0, y1 = int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())
    width = x1 - x0 + 1
    return {"n": count, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "w": width, "h": y1 - y0 + 1, "cx": (x0 + x1) / 2.0}


def _describe(tag, blob):
    """One grounded sentence about a visible tile: how far, and whether it is in your lane.
    Across a 25-level sweep every single gold/cyan sighting (46 of 46) was centred at column
    400.0 with aspect 0.655-0.688 and width 600/depth, so a sighting IS a line-of-sight fact."""
    if blob is None:
        return None
    offset = blob["cx"] - CENTRE_COL
    if abs(offset) > LANE_TOLERANCE:
        side = "LEFT" if offset < 0 else "RIGHT"
        return (f"{tag} is ON SCREEN but off to the {side} (columns {blob['x0']}-{blob['x1']}, "
                f"{blob['n']} px) — it is not in your lane; reach the opening first, then turn "
                f"{side.lower()} toward it.")
    aspect = blob["h"] / float(max(blob["w"], 1))
    if blob["w"] < 20 or abs(aspect - RING_ASPECT) > ASPECT_TOLERANCE:
        return (f"{tag} IS DEAD AHEAD in your corridor ({blob['n']} px, columns "
                f"{blob['x0']}-{blob['x1']}) but partly occluded, so its exact distance is "
                f"unclear. Keep moving forward along this corridor.")
    depth = max(1, min(int(round(RING_WIDTH_AT_DEPTH_1 / blob["w"])), 12))
    reach = ("it is the VERY NEXT CELL — move_forward now and you are on it"
             if depth == 1 else
             f"the corridor between you and it is clear, so {depth} `small` move_forward steps "
             f"put you on it. Do not turn away from it")
    return (f"{tag} IS DEAD AHEAD, exactly {depth} cell(s) in front of you "
            f"({blob['n']} px, columns {blob['x0']}-{blob['x1']}). {reach}.")


def scan_tiles(image):
    """What the frame says about the two tiles that matter. Returns (lines, flags)."""
    try:
        import numpy as np
        arr = np.ascontiguousarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return [], {}
    except Exception:                                  # noqa: BLE001
        return [], {}
    exit_blob = _blob(arr, EXIT_RGB)
    stair_blob = _blob(arr, STAIR_RGB)
    lines, flags = [], {}
    for tag, blob, key in (("THE GOLD EXIT TILE", exit_blob, "exit"),
                           ("A CYAN STAIR TILE", stair_blob, "stair")):
        line = _describe(tag, blob)
        if line:
            lines.append(line)
            flags[key] = blob
    if not lines:
        lines.append("Neither the gold exit nor a cyan stair is anywhere on this frame — "
                     "every coloured surface is corridor wall. Head for UNEXPLORED ground.")
    return lines, flags
