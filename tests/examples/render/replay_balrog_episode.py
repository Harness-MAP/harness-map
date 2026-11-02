#!/usr/bin/env python
"""Replay a BALROG-MiniHack episode in the game engine and render it as a video of the game's tiles.

usage: replay_balrog_episode.py EPISODE.csv.gz EPISODE.json OUT.mp4 [--title TEXT] [--fps 2] [--ffmpeg BIN]
                                [--balrog-repo DIR] [--check] [--crop X0,Y0,X1,Y1] [--box-out FILE] [--box-only]
The episode's json (written by BALROG next to the CSV) gives the task and the seed; the CSV gives
the action the agent produced at every step. The environment is rebuilt exactly as BALROG built
it (same wrappers and kwargs, `env.reset(seed=seed)`), the recorded actions are replayed through
`check_action_validity` + `step` as the evaluator did, and MiniHack's tile rendering (`pixel`)
is captured after every step. With `--check`, the text observation of every replayed step is
compared with the one the agent saw (the CSV's Observation column); the replay is aborted on the
first mismatch, so a rendered video is guaranteed to be the episode the agent played.
The map is cropped to the region drawn during the episode; `--box-out` writes that region to a
file and `--crop` imposes one, so that several episodes of the same level share one framing.
Needs the BALROG environment (nle, minihack, balrog) plus Pillow and an ffmpeg binary.
"""
import argparse, csv, gzip, importlib, io, json, os, random, shutil, subprocess, sys, tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SIDE, PAD, TEXT_H, STATUS_H = 440, 10, 380, 42


def read_rows(path):
    raw = (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")).read()
    text = raw.decode("utf-8", "replace").replace("\x00", "")
    csv.field_size_limit(10 ** 8)
    rd = csv.reader(io.StringIO(text))
    hdr = next(rd)
    idx = {h: i for i, h in enumerate(hdr)}
    rows = []
    for r in rd:
        if len(r) < 6:
            continue
        rows.append(dict(step=int(r[idx["Step"]]), action=r[idx["Action"]], reasoning=r[idx["Reasoning"]],
                         obs=r[idx["Observation"]], reward=float(r[idx["Reward"]] or 0), done=r[idx["Done"]]))
    return rows


def make_env(task, repo):
    """BALROG's own make_env (same wrappers, same kwargs) with `pixel` added to the observation keys."""
    sys.path.insert(0, repo)
    import gym
    importlib.import_module("minihack")  # registers the MiniHack environments
    from omegaconf import OmegaConf
    from balrog.environments import make_env as balrog_make_env
    cfg = OmegaConf.load(os.path.join(repo, "balrog", "config", "config.yaml"))
    orig = gym.make

    def make_with_pixel(*args, **kw):
        keys = list(kw.get("observation_keys") or [])
        if keys and "pixel" not in keys:
            kw["observation_keys"] = keys + ["pixel"]
        return orig(*args, **kw)
    gym.make = make_with_pixel
    try:
        return balrog_make_env("minihack", task, cfg)
    finally:
        gym.make = orig


def norm_obs(s):
    s = s.replace("\r", "").replace("\x00", "")
    if "Observation:\n" in s[:200]:
        s = s.split("Observation:\n", 1)[1]
    return "\n".join(l.rstrip() for l in s.strip("\n").split("\n"))


def wrap_px(text, font, width):
    """Greedy word wrap by rendered width."""
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if font.getlength(t) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def replay(a):
    """Rebuild the environment and replay the logged actions; returns the tile frames and per-step infos."""
    rows = read_rows(a.episode_csv)
    meta = json.load(open(a.episode_json))
    task, seed = meta["task"], int(meta["seed"])
    env = make_env(task, a.balrog_repo)
    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    frames, texts, infos = [obs["obs"]["pixel"].copy()], [obs["text"]["long_term_context"]], []
    ret, mismatch = 0.0, None
    for r in rows:
        action = env.check_action_validity(r["action"])
        obs, reward, term, trunc, info = env.step(action)
        ret += reward
        frames.append(obs["obs"]["pixel"].copy())
        texts.append(obs["text"]["long_term_context"])
        infos.append(dict(step=r["step"], action=action, raw=r["action"], reasoning=r["reasoning"], reward=reward,
                          ret=ret, done=bool(term or trunc)))
        if a.check and norm_obs(obs["text"]["long_term_context"]) != norm_obs(r["obs"]):
            mismatch = r["step"]
            break
        if term or trunc:
            break
    env.close()
    if mismatch is not None:
        sys.exit(f"REPLAY MISMATCH at step {mismatch} ({task}, seed {seed}): the replayed observation differs from the log")
    if len(infos) != len(rows):
        print(f"warning: replayed {len(infos)} of {len(rows)} logged steps", file=sys.stderr)
    return task, seed, frames, texts, infos


def auto_box(frames):
    """The region of the tile map drawn at some point of the episode, with a margin: [x0, y0, x1, y1]."""
    stack = np.stack(frames)
    mask = stack.max(axis=(0, 3)) > 0
    ys, xs = np.where(mask)
    return [max(int(xs.min()) - 16, 0), max(int(ys.min()) - 16, 0),
            min(int(xs.max()) + 17, stack.shape[2]), min(int(ys.max()) + 17, stack.shape[1])]


def message_and_status(text):
    """The game's message line and the two status lines of a text observation."""
    m = text.split("\n")
    message = ""
    if m and m[0].startswith("message:"):
        message = " ".join(l.strip() for l in m[1:4] if l.strip() and not l.startswith("language observation"))
    status = [l.strip() for l in m if l.startswith("Agent") or l.startswith("Dlvl")][:2]
    return message, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_csv"); ap.add_argument("episode_json"); ap.add_argument("out")
    ap.add_argument("--title", default=""); ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    ap.add_argument("--balrog-repo", default=os.environ.get("BALROG_REPO", os.path.expanduser("~/workspace/BALROG")))
    ap.add_argument("--check", action="store_true", help="abort if the replayed observations differ from the log")
    ap.add_argument("--crop", default="", help="X0,Y0,X1,Y1 region of the tile map to show (default: the drawn region)")
    ap.add_argument("--box-out", default="", help="write the drawn region of this episode to this json file")
    ap.add_argument("--box-only", action="store_true", help="only replay and write --box-out, render nothing")
    ap.add_argument("--hold-last", type=float, default=2.0)
    ap.add_argument("--scale", type=int, default=2, help="tile upscaling factor")
    a = ap.parse_args()
    task, seed, frames, texts, infos = replay(a)
    box = auto_box(frames)
    if a.box_out:
        json.dump(dict(task=task, seed=seed, steps=len(infos), box=box), open(a.box_out, "w"))
    if a.box_only:
        print(f"{task} seed {seed}: {len(infos)} steps replayed{' and verified' if a.check else ''}, drawn region {box}")
        return
    if a.crop:
        box = [int(v) for v in a.crop.split(",")]
    x0, y0, x1, y1 = box
    font, bold = ImageFont.truetype(FONT, 14), ImageFont.truetype(BOLD, 15)
    title_lines = wrap_px(a.title or f"{task} seed {seed}", bold, SIDE)
    n_steps = len(infos)
    tmp = tempfile.mkdtemp(prefix="replay_")
    for i, px in enumerate(frames):
        tile = Image.fromarray(px[y0:y1, x0:x1])
        tile = tile.resize((tile.width * a.scale, tile.height * a.scale), Image.NEAREST)
        top_h = max(tile.height, TEXT_H)
        W, H = PAD + tile.width + 15 + SIDE + PAD, PAD + top_h + 12 + STATUS_H + PAD
        im = Image.new("RGB", (W, H), (16, 16, 20))
        im.paste(tile, (PAD, PAD))
        d = ImageDraw.Draw(im)
        x, y = PAD + tile.width + 15, PAD
        for line in title_lines:
            d.text((x, y), line, font=bold, fill=(255, 220, 120)); y += 20
        y += 8
        if i == 0:
            d.text((x, y), f"start (step 0 / {n_steps})", font=bold, fill=(230, 230, 230)); y += 20
        else:
            inf = infos[i - 1]
            d.text((x, y), f"after step {inf['step'] + 1} / {n_steps}", font=bold, fill=(230, 230, 230)); y += 20
            act = inf["action"] if inf["action"] == inf["raw"] else f"{inf['action']}  (agent wrote: {inf['raw'][:30]})"
            for line in wrap_px(f"action: {act}", bold, SIDE)[:2]:
                d.text((x, y), line, font=bold, fill=(120, 220, 255)); y += 20
            d.text((x, y), f"reward: {inf['reward']:+.3f}   return: {inf['ret']:+.3f}", font=font, fill=(200, 200, 200)); y += 20
            d.text((x, y), f"done: {inf['done']}", font=font, fill=(200, 200, 200)); y += 28
            d.text((x, y), "reasoning:", font=bold, fill=(230, 230, 230)); y += 20
            for line in wrap_px(inf["reasoning"] or "(none)", font, SIDE)[:6]:
                d.text((x, y), line, font=font, fill=(190, 190, 190)); y += 18
            y += 10
        message, status = message_and_status(texts[i])
        d.text((x, y), "message:", font=bold, fill=(230, 230, 230)); y += 20
        for line in wrap_px(message or "(none)", font, SIDE)[:4]:
            d.text((x, y), line, font=font, fill=(190, 190, 190)); y += 18
        # the game's status lines, in their own strip under the map
        ys_ = PAD + top_h + 12
        for line in status:
            d.text((PAD, ys_), line, font=font, fill=(200, 200, 160)); ys_ += 18
        im.save(f"{tmp}/f{i:05d}.png")
    n = len(frames)
    for j in range(int(round(a.hold_last * a.fps))):
        shutil.copy(f"{tmp}/f{n-1:05d}.png", f"{tmp}/f{n+j:05d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "24", a.out], check=True)
    shutil.rmtree(tmp)
    print(f"{a.out}: {n_steps} steps replayed{' and verified' if a.check else ''}, {os.path.getsize(a.out) // 1024} KB")


if __name__ == "__main__":
    main()
