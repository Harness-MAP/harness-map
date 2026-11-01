"""Price the candidate's REAL tools/vlm.py:decide against the adapter replica.

Drives the candidate's own `decide` with a stand-in for the frozen model (the memoryless
argmax-P(pick | last applied) policy the diagnostician fitted from this lane's own 67 emissions,
plus a uniform control), over a 45-line replica of snake_adapter._try_move/_step. Reports the
paper primary min(meals/4, 1), survival, honour rate and the death mix per arm.
Usage: python price.py <candidate_dir> [...]
"""
import asyncio, collections, importlib, random, sys, types
from dataclasses import dataclass, field
from pathlib import Path

G, MAX_STEPS, EARLY = 4, 100, 4
DV = {"move_right": (1, 0, 0), "move_left": (-1, 0, 0), "move_up": (0, -1, 0),
      "move_down": (0, 1, 0), "move_forward": (0, 0, -1), "move_backward": (0, 0, 1)}
KEY = {"move_right": "d", "move_left": "a", "move_up": "w",
       "move_down": "s", "move_forward": "q", "move_backward": "e"}
NAMES = list(DV)
IV = {v: k for k, v in DV.items()}
# the diagnostician's fitted cycler: argmax P(pick | last applied) over this lane's 67 emissions
CYCLER = {"move_right": "move_forward", "move_forward": "move_forward",
          "move_down": "move_forward", "move_left": "move_forward",
          "move_up": "move_forward", "move_backward": "move_forward"}


@dataclass
class Action:
    type: str
    key: str = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FrameData:
    frame_number: int


class Replica:
    def __init__(self, seed):
        self.r = random.Random(); self.r.seed(seed)
        self.snake = [(2, 3, 2), (1, 3, 2)]
        self.last = (1, 0, 0)
        self.food = self._food()
        self.steps = self.score = 0
        self.over = False
        self.death = None

    def _food(self):
        while True:
            c = (self.r.randint(0, G - 1), self.r.randint(0, G - 1), self.r.randint(0, G - 1))
            if c not in self.snake:
                return c

    def press(self, key):
        name = {v: k for k, v in KEY.items()}[key]
        d = DV[name]
        if all(a + b == 0 for a, b in zip(d, self.last)):
            return "refused"                                    # no step, no move, no last_dir
        self.last = d
        nh = tuple(h + e for h, e in zip(self.snake[0], d))
        self.steps += 1
        if not all(0 <= c < G for c in nh):
            self.over = True; self.death = "WALL"; return "dead"
        if nh in self.snake[:-1]:
            self.over = True; self.death = "BODY"; return "dead"
        self.snake.insert(0, nh)
        if nh == self.food:
            self.score += 1
            self.food = self._food()
        else:
            self.snake.pop()
        if self.steps >= MAX_STEPS:
            self.over = True
        return "ok"


def episode(decide, seed, standin, rp, honour):
    env = Replica(seed)
    last_applied = "move_right"
    ev = types.SimpleNamespace()

    async def baseline_step(fd):
        pick = CYCLER.get(last_applied, "move_forward") if standin == "cycler" else rp.choice(NAMES)
        return Action(type="key_press", key=KEY[pick],
                      metadata={"function_called": pick, "frame_number": fd.frame_number})
    ev.baseline_step = baseline_step
    ev.openai_ai_model = baseline_step
    n = 0
    while not env.over and env.score < EARLY and n < MAX_STEPS:
        act = asyncio.get_event_loop().run_until_complete(decide(ev, FrameData(frame_number=n)))
        asked = act.metadata.get("model_action", act.metadata.get("function_called"))
        shipped = act.metadata.get("function_called")
        honour[0] += 1
        honour[1] += int(asked == shipped)
        r = env.press(act.key)
        if r == "ok":
            last_applied = shipped
        n += 1
    return env


def run(cdir, standin):
    sys.path.insert(0, str(cdir))
    for m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
        sys.modules.pop(m, None)
    meals = steps = 0
    deaths = collections.Counter()
    honour = [0, 0]
    prim = []
    for seed in range(1, 21):
        for m in [m for m in list(sys.modules) if m == "tools" or m.startswith("tools.")]:
            sys.modules.pop(m, None)
        vlm = importlib.import_module("tools.vlm")          # fresh module -> fresh per-episode ledger
        env = episode(vlm.decide, seed, standin, random.Random(seed), honour)
        meals += env.score; steps += env.steps
        deaths[env.death or ("early" if env.score >= EARLY else "budget")] += 1
        prim.append(min(env.score / EARLY, 1.0))
    sys.path.remove(str(cdir))
    return dict(primary=round(sum(prim) / len(prim), 4), meals=meals,
                steps_per_ep=round(steps / 20, 1), deaths=dict(deaths),
                honour=round(honour[1] / max(honour[0], 1), 3))


if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    for cd in sys.argv[1:]:
        print(f"\n### {Path(cd).name}")
        for standin in ("cycler", "uniform"):
            print(f"  {standin:8s} {run(Path(cd), standin)}")
