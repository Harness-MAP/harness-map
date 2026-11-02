"""decodecmp.py — four pellet decoders on the same frames, bucketed by head-pellet L1.

  A  whole-mask bbox + ONE-SIDED gate   <- what tools/vlm.py ships
  B  whole-mask bbox + SYMMETRIC gate   (round 2's rule, on the shipped front end)
  C  largest blob (minarea 40) + ONE-SIDED gate   (the diagnosis's L2 decoder)
  D  largest blob (minarea 40) + SYMMETRIC gate   (the decoder the diagnosis refutes)
Frames come from driving the real adapter with the shipped policy, 20 seeds.
"""
import os, random, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                    # noqa: E402
import warrant_rig as R                               # noqa: E402
from tools import vlm                                 # noqa: E402


def blobs(mask, minarea=40):
    seen = np.zeros_like(mask, dtype=bool); H, W = mask.shape; out = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        st = [(y0, x0)]; seen[y0, x0] = True; pts = []
        while st:
            y, x = st.pop(); pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; st.append((ny, nx))
        if len(pts) >= minarea:
            out.append((len(pts), np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts]),
                        max(p[0] for p in pts) - min(p[0] for p in pts) + 1,
                        max(p[1] for p in pts) - min(p[1] for p in pts) + 1))
    return sorted(out, reverse=True)


def snap(ax, ay, spread, onesided):
    fit = ((vlm.COVERED <= spread - vlm.DRAWN_D) & (spread - vlm.DRAWN_D <= vlm.SWOLLEN)
           if onesided else (np.abs(spread - vlm.DRAWN_D) <= vlm.SWOLLEN))
    gap = np.where(fit, vlm._OFFSET(ax, ay), np.inf)
    j = int(np.argmin(gap))
    return vlm.CELLS[j] if np.isfinite(gap[j]) else None


def four(img):
    a = np.asarray(img)
    m = vlm._WARM(a)
    rows, cols = np.nonzero(m)
    whole = None if not rows.size else (cols.mean(), rows.mean(), vlm._SPREAD(rows, cols))
    bs = blobs(m)
    big = None if not bs else (bs[0][2], bs[0][1], (bs[0][3] + bs[0][4]) / 2.0)
    out = []
    for src in (whole, whole, big, big):
        out.append(None if src is None else src)
    return [snap(*whole, True) if whole else None,
            snap(*whole, False) if whole else None,
            snap(*big, True) if big else None,
            snap(*big, False) if big else None]


def main():
    n = Counter(); ok = [Counter() for _ in range(4)]
    for s in range(1, 21):
        random.seed(s)
        g = R.SnakeGameAdapter(grid_size=4, max_steps=100); g.init()
        ev = R.Stand(g, R.make_asker("mix", random.Random(s)), False)
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            for i in range(100):
                if g.game_over or g.score >= 4:
                    break
                img = ev.shot()
                head = (g.snake[0].x, g.snake[0].y, g.snake[0].z)
                truth = (g.food.x, g.food.y, g.food.z)
                d = min(sum(abs(p - q) for p, q in zip(head, truth)), 4)
                n[d] += 1
                for k, got in enumerate(four(img)):
                    ok[k][d] += (got == truth)
                act = loop.run_until_complete(vlm.decide(ev, R.Frame(img, i)))
                name = R.ACTION_FOR_KEY.get(act.key)
                if name:
                    getattr(g, name)()
        finally:
            loop.close()
    lab = ["A whole-mask + one-sided (SHIPPED)", "B whole-mask + symmetric",
           "C largest-blob + one-sided", "D largest-blob + symmetric"]
    print(f"{'decoder':<36}" + "".join(f"  d={k}" + " " * 6 for k in sorted(n)) + "   ALL")
    print(f"{'(frames)':<36}" + "".join(f"  {n[k]:>4}      " for k in sorted(n))
          + f"  {sum(n.values()):>4}")
    for k, name in enumerate(lab):
        print(f"{name:<36}" + "".join(f"  {ok[k][j]/n[j]:.4f}    " for j in sorted(n))
              + f"  {sum(ok[k].values())/sum(n.values()):.4f}")


if __name__ == "__main__":
    main()
