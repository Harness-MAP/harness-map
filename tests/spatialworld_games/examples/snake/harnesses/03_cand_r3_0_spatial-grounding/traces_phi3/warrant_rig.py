"""warrant_rig.py — offline validation of tools/vlm.py against the REAL renderer + adapter.

Runs the candidate's own `decide` coroutine, unmodified, behind a stand-in evaluator whose
`baseline_step` captures the live frame, appends a user+assistant turn exactly as the engine's
request builder does, and answers from a configurable ask model. So the appeal plumbing (history
rollback, the system-text splice and its restore, the second call) is exercised for real; only the
language model is a stand-in.

Reported per ask model x appeal arm: deaths, pellets, decisions, honour rate, and the three
exactness checks (reckoned head vs the engine's head, HUD digit vs the engine's score, decoded
pellet vs the engine's pellet, bucketed by head-pellet L1).

    python traces_phi3/warrant_rig.py            # 20 seeds, every arm
"""
import asyncio, os, random, sys
from collections import Counter

SW = "/home/azureuser/workspace/SpatialWorld"
sys.path.insert(0, SW)
sys.path.insert(0, SW + "/mllm_base_agent/environments/game")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
CAND = os.environ.get("CAND_DIR", str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, CAND)

import numpy as np
import pygame
from games.Snake.snake_adapter import SnakeGameAdapter
from tools import vlm
from tools.action import KEY_MAPPING

ACTION_FOR_KEY = {v: k for k, v in KEY_MAPPING.items()}
MENU = list(vlm.MENU)
# the live ask mix measured over 1953 clamped decisions (diagnosis 4b)
MIX = (["move_forward"] * 700 + ["move_backward"] * 650 + ["move_down"] * 291
       + ["move_right"] * 253 + ["move_up"] * 55 + ["move_left"] * 4)


class Act:
    def __init__(self, key, name):
        self.type = "key_press"
        self.key = key
        self.metadata = {"function_called": name, "parsed_from_json": True}


class Frame:
    def __init__(self, image, n):
        self.image = image
        self.frame_number = n


class Stand:
    """A stand-in evaluator: the engine surface `decide` is allowed to touch, and nothing else."""

    def __init__(self, game, asker, heed):
        self.game = game
        self.asker = asker
        self.heed = heed                      # does the stand-in read the spliced appeal?
        self.conversation_history = []
        self.current_prompt = "SYSTEM PROMPT (stand-in)"
        self.api_fatal_failure = False
        self.model_responses = []
        self.calls = 0
        self.second_calls = 0
        self.card_seen = 0

    def shot(self):
        self.game.render()
        return pygame.surfarray.array3d(self.game.screen).swapaxes(0, 1)

    async def baseline_step(self, frame_data):
        self.calls += 1
        first = not self.conversation_history
        if first:
            text = self.current_prompt
        else:
            text = "Continue based on previous conversation history."
        base = (self.conversation_history[0]["content"][0]["text"]
                if self.conversation_history else self.current_prompt)
        name = self.asker(base, self.heed)
        if first:
            self.conversation_history.append(
                {"role": "user", "content": [{"type": "input_text", "text": self.current_prompt},
                                             {"type": "input_image", "image_url": "..."}]})
        else:
            self.conversation_history.append(
                {"role": "user", "content": [{"type": "input_text", "text": text},
                                             {"type": "input_image", "image_url": "..."}]})
        self.conversation_history.append(
            {"role": "assistant", "content": [{"type": "output_text",
                                               "text": '{"action": "%s"}' % name}]})
        self.model_responses.append({"action_taken": name})
        if vlm.MARK in base:
            self.card_seen += 1
        return Act(KEY_MAPPING.get(name), name)


def make_asker(kind, rng):
    def ask(base, heed):
        if heed and vlm.MARK in base:
            card = base.split(vlm.MARK)[1]
            for line in card.splitlines():
                if line.startswith("- of those, the ones that shorten"):
                    opts = [w.strip() for w in line.split(":", 1)[1].split(",") if w.strip() in MENU]
                    if opts:
                        return opts[0]
                if line.startswith("- moves the engine would actually execute"):
                    opts = [w.strip() for w in line.split(":", 1)[1].split(",") if w.strip() in MENU]
                    if opts:
                        return opts[0]
        if kind == "mix":
            return rng.choice(MIX)
        if kind == "uniform":
            return rng.choice(MENU)
        if kind == "latch":
            return "move_forward"
        if kind == "junk":
            return rng.choice(MENU + ["spin", "", "move_sideways"])
        return "move_forward"
    return ask


def run(seed, kind, heed, stats):
    random.seed(seed)
    g = SnakeGameAdapter(grid_size=4, max_steps=100)
    g.init()
    rng = random.Random(seed * 7919 + (1 if heed else 0))
    ev = Stand(g, make_asker(kind, rng), heed)
    loop = asyncio.new_event_loop()
    decisions = honoured = 0
    try:
        for n in range(100):
            if g.game_over:
                break
            img = ev.shot()
            truth_head = (g.snake[0].x, g.snake[0].y, g.snake[0].z)
            truth_food = (g.food.x, g.food.y, g.food.z)
            truth_score = g.score
            led = getattr(ev, vlm.LEDGER_ATTR, None)
            reck = led["head"] if led else vlm.OPENING_HEAD
            stats["head_n"] += 1
            stats["head_ok"] += (reck == truth_head)
            stats["hud_n"] += 1
            stats["hud_ok"] += (vlm._tally_glyph(img, None) == truth_score)
            got = vlm._pellet_cell(img)
            d = min(sum(abs(a - b) for a, b in zip(truth_head, truth_food)), 4)
            stats["food_n"][d] += 1
            stats["food_ok"][d] += (got == truth_food)
            act = loop.run_until_complete(vlm.decide(ev, Frame(img, n)))
            decisions += 1
            stats["empty_warrant"] += int((act.metadata or {}).get("harness_pressed") is None)
            led = getattr(ev, vlm.LEDGER_ATTR)
            name = ACTION_FOR_KEY.get(act.key)
            if name:
                getattr(g, name)()
            if g.score >= 4:
                break
        honoured = led["honoured"]
    finally:
        loop.close()
    out_of_budget = decisions >= 100 and g.score < 4
    return {"seed": seed, "score": g.score, "budget": out_of_budget,
            "dead": bool(g.game_over) and g.score < 4 and not out_of_budget,
            "decisions": decisions, "honoured": honoured, "appeals": led["appeals"],
            "won_back": led["won_back"], "calls": ev.calls, "card_seen": ev.card_seen}


def main():
    seeds = range(1, 21)
    grand = {"head_n": 0, "head_ok": 0, "hud_n": 0, "hud_ok": 0, "empty_warrant": 0,
             "food_n": Counter(), "food_ok": Counter(), "dec": 0, "hon": 0, "app": 0, "won": 0}
    for kind in ("mix", "uniform", "latch", "junk"):
        for heed in (False, True):
            stats = {"head_n": 0, "head_ok": 0, "hud_n": 0, "hud_ok": 0, "empty_warrant": 0,
                     "food_n": Counter(), "food_ok": Counter()}
            rows = [run(s, kind, heed, stats) for s in seeds]
            dead = sum(r["dead"] for r in rows)
            pellets = sum(r["score"] for r in rows)
            prim = sum(min(r["score"] / 4.0, 1.0) for r in rows) / len(rows)
            dec = sum(r["decisions"] for r in rows)
            hon = sum(r["honoured"] for r in rows)
            app = sum(r["appeals"] for r in rows)
            won = sum(r["won_back"] for r in rows)
            seen = sum(r["card_seen"] for r in rows)
            print(f"ask={kind:<8} appeal_heeded={str(heed):<5} primary={prim:.4f} "
                  f"deaths={dead:>2}/20 budget_out={sum(r['budget'] for r in rows):>2}/20 "
                  f"empty_warrant={stats['empty_warrant']} pellets={pellets:>3} decisions={dec:>4} "
                  f"honour={hon}/{dec}={hon/max(dec,1):.3f} appeals={app} won_back={won} "
                  f"cards_delivered={seen}")
            print(f"            reckoned head == engine head: {stats['head_ok']}/{stats['head_n']}"
                  f"   HUD digit == engine score: {stats['hud_ok']}/{stats['hud_n']}")
            fn, fo = stats["food_n"], stats["food_ok"]
            print("            pellet decode by head-pellet L1: " + "  ".join(
                f"d{k}:{fo[k]}/{fn[k]}" for k in sorted(fn)) +
                f"   ALL:{sum(fo.values())}/{sum(fn.values())}")
            for k in ("head_n", "head_ok", "hud_n", "hud_ok", "empty_warrant"):
                grand[k] += stats[k]
            grand["food_n"] += fn; grand["food_ok"] += fo
            grand["dec"] += dec; grand["hon"] += hon; grand["app"] += app; grand["won"] += won
    print("\n==== GRAND TOTAL over all 8 arms ====")
    print(f"decisions={grand['dec']}  honoured={grand['hon']}  appeals={grand['app']}  "
          f"won_back={grand['won']}  empty_warrant={grand['empty_warrant']}")
    print(f"reckoned head == engine head: {grand['head_ok']}/{grand['head_n']}"
          f"   HUD digit == engine score: {grand['hud_ok']}/{grand['hud_n']}")
    fn, fo = grand["food_n"], grand["food_ok"]
    print("pellet decode: " + "  ".join(
        f"d{k}:{fo[k]}/{fn[k]}={fo[k]/fn[k]:.4f}" for k in sorted(fn))
        + f"   ALL:{sum(fo.values())}/{sum(fn.values())}={sum(fo.values())/sum(fn.values()):.4f}")


if __name__ == "__main__":
    main()
