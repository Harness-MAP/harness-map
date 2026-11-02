#!/usr/bin/env python
"""Render a BALROG-MiniHack episode log (<task>_run_NN.csv[.gz]) as a video of the text frames the agent saw.

usage: render_balrog_episode.py EPISODE.csv.gz OUT.mp4 [--title TEXT] [--fps 2] [--ffmpeg BIN] [--gif]
Each row of BALROG's per-step CSV (Step, Action, Reasoning, Observation, Reward, Done) becomes one
frame: the observation text (message, language observation, cursor, map, status lines) on the left
and the step, the action taken from that observation, the agent's reasoning and the running return
on the right. Needs Pillow and an ffmpeg binary (`--ffmpeg`, or `ffmpeg` on PATH).
"""
import argparse, csv, gzip, io, os, shutil, subprocess, sys, tempfile, textwrap
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                   "/usr/share/fonts/dejavu/DejaVuSansMono.ttf", "DejaVuSansMono.ttf"]
BOLD_CANDIDATES = [p.replace("Mono.ttf", "Mono-Bold.ttf") for p in FONT_CANDIDATES]


def load_font(cands, size):
    for p in cands:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


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
        rows.append(dict(step=r[idx["Step"]], action=r[idx["Action"]], reasoning=r[idx["Reasoning"]],
                         obs=r[idx["Observation"]], reward=float(r[idx["Reward"]] or 0), done=r[idx["Done"]]))
    return rows


def render_frame(row, title, cum, n_steps, font, bold, cw, lh, obs_lines=50, side=46):
    left_w = 84 * cw
    W, H = left_w + side * cw + 3 * cw, (obs_lines + 1) * lh + 8
    im = Image.new("RGB", (W, H), (16, 16, 20))
    d = ImageDraw.Draw(im)
    y = 4
    for line in row["obs"].split("\n")[:obs_lines]:
        d.text((cw, y), line.rstrip("\n")[:84], font=font, fill=(214, 214, 200))
        y += lh
    x = left_w + 2 * cw
    d.line([(left_w + cw, 0), (left_w + cw, H)], fill=(60, 60, 70))
    y = 4
    for line in textwrap.wrap(title, side):
        d.text((x, y), line, font=bold, fill=(255, 220, 120))
        y += lh
    y += lh // 2
    d.text((x, y), f"step {row['step']} / {n_steps}", font=bold, fill=(230, 230, 230)); y += lh
    d.text((x, y), f"action: {row['action']}", font=bold, fill=(120, 220, 255)); y += lh
    d.text((x, y), f"reward: {row['reward']:+.3f}   return: {cum:+.3f}", font=font, fill=(200, 200, 200)); y += lh
    d.text((x, y), f"done: {row['done']}", font=font, fill=(200, 200, 200)); y += 2 * lh
    d.text((x, y), "reasoning:", font=bold, fill=(230, 230, 230)); y += lh
    for line in textwrap.wrap(row["reasoning"] or "(none)", side)[: obs_lines - 12]:
        d.text((x, y), line, font=font, fill=(190, 190, 190)); y += lh
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode"); ap.add_argument("out")
    ap.add_argument("--title", default=""); ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg")); ap.add_argument("--gif", action="store_true")
    ap.add_argument("--hold-last", type=float, default=2.0, help="seconds to hold the final frame")
    a = ap.parse_args()
    rows = read_rows(a.episode)
    if not rows:
        sys.exit(f"no rows in {a.episode}")
    font, bold = load_font(FONT_CANDIDATES, 13), load_font(BOLD_CANDIDATES, 13)
    cw = font.getbbox("M")[2]; lh = font.getbbox("Mg")[3] + 2
    tmp = tempfile.mkdtemp(prefix="balrog_frames_")
    cum = 0.0
    title = a.title or os.path.basename(a.episode).replace(".csv.gz", "").replace(".csv", "")
    for i, row in enumerate(rows):
        cum += row["reward"]
        render_frame(row, title, cum, len(rows), font, bold, cw, lh).save(f"{tmp}/f{i:05d}.png")
    extra = int(round(a.hold_last * a.fps))
    for j in range(extra):  # hold the last frame
        shutil.copy(f"{tmp}/f{len(rows)-1:05d}.png", f"{tmp}/f{len(rows)+j:05d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cmd = [a.ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png"]
    if a.gif or a.out.endswith(".gif"):
        cmd += ["-vf", "split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse", a.out]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "28", a.out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp)
    print(f"{a.out}: {len(rows)} steps, {os.path.getsize(a.out)//1024} KB")


if __name__ == "__main__":
    main()
