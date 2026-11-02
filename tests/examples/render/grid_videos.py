#!/usr/bin/env python
"""Tile several videos into one grid video, every panel letterboxed into the same cell.

usage: grid_videos.py OUT.mp4 VIDEO [VIDEO ...] [--cols 2] [--scale 1.0] [--fps 2] [--gap 6] [--ffmpeg BIN]
The cell is the largest panel (after `--scale`), smaller panels are centred in it, and a panel
that ends before the others holds its last frame, so panels never overlap whatever their sizes
and lengths. Needs OpenCV (decoding), Pillow and an ffmpeg binary (`--ffmpeg`, or `ffmpeg` on PATH).
"""
import argparse, os, shutil, subprocess, tempfile

import cv2
from PIL import Image


def frames_of(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not out:
        raise SystemExit(f"could not decode {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("videos", nargs="+")
    ap.add_argument("--cols", type=int, default=2); ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--gap", type=int, default=6)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg")); ap.add_argument("--crf", type=int, default=26)
    a = ap.parse_args()
    panels = [frames_of(v) for v in a.videos]
    if a.scale != 1.0:
        panels = [[f.resize((max(1, round(f.width * a.scale)), max(1, round(f.height * a.scale))), Image.LANCZOS)
                   for f in p] for p in panels]
    cw = max(f.width for p in panels for f in p)
    ch = max(f.height for p in panels for f in p)
    cols = max(1, min(a.cols, len(panels)))
    rows = -(-len(panels) // cols)
    W, H = cols * cw + (cols + 1) * a.gap, rows * ch + (rows + 1) * a.gap
    n = max(len(p) for p in panels)
    tmp = tempfile.mkdtemp(prefix="grid_")
    for k in range(n):
        im = Image.new("RGB", (W, H), (40, 40, 40))
        for j, p in enumerate(panels):
            f = p[min(k, len(p) - 1)]
            x = a.gap + (j % cols) * (cw + a.gap) + (cw - f.width) // 2
            y = a.gap + (j // cols) * (ch + a.gap) + (ch - f.height) // 2
            im.paste(f, (x, y))
        im.save(f"{tmp}/f{k:05d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", str(a.crf), a.out],
                   check=True)
    shutil.rmtree(tmp)
    print(f"{a.out}: {len(panels)} panels in {cw}x{ch} cells, {n} frames, {os.path.getsize(a.out) // 1024} KB")


if __name__ == "__main__":
    main()
