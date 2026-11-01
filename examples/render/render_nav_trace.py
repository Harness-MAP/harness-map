#!/usr/bin/env python
"""Render a traced SimWorld-MMNav episode (the evaluator's traces/ep_*/ directory) as a video.

usage: render_nav_trace.py EP_DIR OUT.mp4 [--title TEXT] [--fps 2] [--ffmpeg BIN]
The episode directory holds env_step_log.jsonl (agent and target location per step), actions.jsonl,
vlm_trajectory.jsonl (every model call with the frames it saw under frames/), gt.json (destination
and ideal path) and verdict.json (per-subtask success). Each step becomes one frame with three
panels: the most recent camera image the model was shown, the expected view at the end of the
current subtask (the second image the model is shown) with the step, action, distance and
per-subtask verdict under it, and a top-down map with the ideal path, the destination and the
robot's track so far; the title runs along the bottom. Needs Pillow and an ffmpeg binary
(`--ffmpeg`, or `ffmpeg` on PATH).
"""
import argparse, json, os, shutil, subprocess, tempfile

from PIL import Image, ImageDraw, ImageFont

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]
PAD, FW, FH, IW, MW = 10, 720, 600, 320, 420


def font(size, bold=False):
    for p in FONTS:
        try:
            return ImageFont.truetype(p.replace(".ttf", "-Bold.ttf") if bold else p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def wrap_px(text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if f.getlength(t) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ep_dir"); ap.add_argument("out")
    ap.add_argument("--title", default=""); ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    ap.add_argument("--hold-last", type=float, default=2.0)
    a = ap.parse_args()
    ep = a.ep_dir.rstrip("/")
    steps = jsonl(f"{ep}/env_step_log.jsonl")
    acts = {r["step"]: r for r in jsonl(f"{ep}/actions.jsonl")}
    calls = jsonl(f"{ep}/vlm_trajectory.jsonl")
    gt = json.load(open(f"{ep}/gt.json")) if os.path.exists(f"{ep}/gt.json") else {}
    verdict = json.load(open(f"{ep}/verdict.json")) if os.path.exists(f"{ep}/verdict.json") else {}
    traj = json.load(open(f"{ep}/trajectory.json")) if os.path.exists(f"{ep}/trajectory.json") else {}
    frame_at, hint_at = {}, {}
    for c in calls:  # the perceiver sends [egocentric view, expected view at the end of the subtask, segmentation]
        imgs = c.get("images") or []
        if imgs:
            frame_at[int(c["step"])] = imgs[0]
            if len(imgs) > 1:
                hint_at[int(c["step"])] = imgs[1]
    initial = f"{ep}/observations/step_000_initial.png"
    pts = [(s["agent_location"][0], s["agent_location"][1]) for s in steps if s.get("agent_location")]
    ideal = [tuple(p[:2]) for p in gt.get("ideal_path", [])]
    dest = gt.get("destination") or {}
    dest_xy = (dest["x"], dest["y"]) if dest else (steps[0]["target_location"][:2] if steps else None)
    allpts = pts + ideal + ([dest_xy] if dest_xy else [])
    xs, ys = [p[0] for p in allpts], [p[1] for p in allpts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 0.08 * max(x1 - x0, y1 - y0, 1.0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    scale = min(MW / max(x1 - x0, 1e-6), MW / max(y1 - y0, 1e-6))

    def m(p):  # world -> map pixels
        return (int((p[0] - x0) * scale), int(MW - (p[1] - y0) * scale))

    f_small, f_mid, f_bold, f_title = font(14), font(16), font(16, True), font(18, True)
    X_INSET, X_MAP = PAD + FW + PAD, PAD + FW + PAD + IW + PAD
    W = X_MAP + MW + PAD
    title_lines = wrap_px(a.title, f_title, W - 2 * PAD) if a.title else []
    TOP = PAD + 20  # captions sit in a row above the panels; nothing is drawn over a picture
    H = TOP + FH + PAD + (len(title_lines) * 24 + PAD if title_lines else 0)
    tmp = tempfile.mkdtemp(prefix="nav_frames_")
    sub = verdict.get("subtask_success") or traj.get("subtask_success") or []
    labels = traj.get("rubric_labels") or []
    names = labels if len(labels) == len(sub) else [f"subtask {k + 1}" for k in range(len(sub))]
    initial_dist = None
    last_img = last_hint = None
    n = len(steps)
    for i, s in enumerate(steps):
        im = Image.new("RGB", (W, H), (18, 18, 22))
        d = ImageDraw.Draw(im)
        ref = frame_at.get(i)
        if ref and os.path.exists(f"{ep}/{ref}"):
            last_img = f"{ep}/{ref}"
        elif last_img is None and os.path.exists(initial):
            last_img = initial
        d.text((PAD, PAD), "camera view the model was last shown", font=f_small, fill=(200, 200, 200))
        if last_img:
            im.paste(Image.open(last_img).convert("RGB").resize((FW, FH)), (PAD, TOP))
        # expected view and the step's facts, in their own column
        hint = hint_at.get(i)
        if hint and os.path.exists(f"{ep}/{hint}"):
            last_hint = f"{ep}/{hint}"
        d.text((X_INSET, PAD), "expected view at subtask end", font=f_small, fill=(200, 200, 200))
        y = TOP
        ih = round(IW * FH / FW)
        if last_hint:
            im.paste(Image.open(last_hint).convert("RGB").resize((IW, ih)), (X_INSET, y))
        d.rectangle([X_INSET, y, X_INSET + IW - 1, y + ih - 1], outline=(120, 120, 130))
        y += ih + 16
        act = acts.get(i, {})
        dist = s.get("dist_to_target")
        if dist is None and s.get("agent_location") and s.get("target_location"):
            ax, ay = s["agent_location"][:2]; tx, ty = s["target_location"][:2]
            dist = ((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5
        if initial_dist is None and dist is not None:
            initial_dist = dist
        d.text((X_INSET, y), f"step {i + 1} / {n}", font=f_bold, fill=(230, 230, 230)); y += 24
        for line in wrap_px(f"action: {act.get('action_name', act.get('action', '?'))}", f_mid, IW):
            d.text((X_INSET, y), line, font=f_mid, fill=(120, 220, 255)); y += 22
        if dist is not None:
            d.text((X_INSET, y), f"distance to target: {dist / 100:.0f} m", font=f_mid, fill=(230, 230, 230)); y += 22
            if initial_dist:
                d.text((X_INSET, y), f"progress: {100 * (1 - dist / initial_dist):.0f} % of the initial distance", font=f_mid, fill=(200, 200, 200)); y += 22
        if sub:
            y += 8
            d.text((X_INSET, y), "verdict at the end of the episode", font=f_small, fill=(200, 200, 200)); y += 20
            for nm, v in zip(names, sub):
                d.text((X_INSET, y), f"{nm}: {'ok' if v else 'fail'}", font=f_mid, fill=(130, 230, 130) if v else (240, 120, 120)); y += 22
        # map
        mx, my = X_MAP, TOP
        d.text((mx, PAD), "top-down map", font=f_small, fill=(200, 200, 200))
        d.rectangle([mx, my, mx + MW, my + MW], fill=(30, 32, 40), outline=(90, 90, 100))
        if len(ideal) > 1:
            d.line([(mx + m(p)[0], my + m(p)[1]) for p in ideal], fill=(90, 140, 220), width=3)
        if dest_xy:
            px, py = m(dest_xy); d.ellipse([mx + px - 7, my + py - 7, mx + px + 7, my + py + 7], fill=(240, 80, 80))
        track = pts[: i + 1]
        if len(track) > 1:
            d.line([(mx + m(p)[0], my + m(p)[1]) for p in track], fill=(250, 210, 60), width=3)
        if track:
            px, py = m(track[-1]); d.ellipse([mx + px - 6, my + py - 6, mx + px + 6, my + py + 6], fill=(250, 250, 250))
        d.text((mx, my + MW + 8), "blue: ideal path   red: destination", font=f_small, fill=(170, 170, 170))
        d.text((mx, my + MW + 26), "yellow: robot track   white: robot now", font=f_small, fill=(170, 170, 170))
        ty = TOP + FH + PAD
        for line in title_lines:
            d.text((PAD, ty), line, font=f_title, fill=(255, 220, 120)); ty += 24
        im.save(f"{tmp}/f{i:05d}.png")
    for j in range(int(round(a.hold_last * a.fps))):
        shutil.copy(f"{tmp}/f{n - 1:05d}.png", f"{tmp}/f{n + j:05d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "26", a.out], check=True)
    shutil.rmtree(tmp)
    print(f"{a.out}: {n} steps, {os.path.getsize(a.out) // 1024} KB")


if __name__ == "__main__":
    main()
