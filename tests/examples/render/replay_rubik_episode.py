#!/usr/bin/env python
"""Replay a SpatialWorld Rubik's-cube episode in the game engine and render it as a video.

usage: replay_rubik_episode.py SESSION.log EVAL_RESULT.json OUT.mp4 [--spatialworld-repo DIR] [--check]
                               [--fps 2] [--hold-last 2] [--ffmpeg BIN]
The evaluation's `eval_result.json` gives the level (the scramble the engine applies) and the
outcome; the session log lists, in order, every face turn and every view turn the harness
executed (`Face rotation executed: FACE DIRECTION`, `View rotation executed: AXIS DIRECTION`).
The engine is started headlessly on the same level, the turns are replayed through the same
`rotate_face` / `rotate_cube` calls, and a 500x500 frame is captured at the start and after
every turn, so the video shows the cube as the engine drew it for the agent, from the agent's
viewpoint. With `--check`, the replay must execute exactly the logged number of steps and end in
the logged solved/unsolved state, and may not reach the solved state before its last turn.
Needs the SpatialWorld repository (its `rubik_cube` game and scramble data), pygame, Pillow and
an ffmpeg binary.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

LINE = re.compile(r"(Face|View) rotation executed: (\w+) (\w+)")


def read_actions(log_path):
    out = []
    for line in open(log_path, errors="replace"):
        m = LINE.search(line)
        if m:
            out.append((m.group(1).lower(), m.group(2), m.group(3)))
    return out


def load_engine(repo):
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "mllm_base_agent", "environments", "game", "games", "rubik_cube"))
    import pygame
    from rubik_cube_adapter import RubikCubeGame
    from main import is_cube_solved
    return pygame, RubikCubeGame, is_cube_solved


def settle(game):
    """Run the engine until the current turn's animation has finished, then draw."""
    n = 0
    while game.animation_in_progress:
        game.update()
        n += 1
        if n > 10000:
            raise SystemExit("the engine's animation never finished")
    game.update()
    game.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_log"); ap.add_argument("eval_result"); ap.add_argument("out")
    ap.add_argument("--spatialworld-repo", default=os.environ.get("SPATIALWORLD_REPO", os.path.expanduser("~/workspace/SpatialWorld")))
    ap.add_argument("--check", action="store_true", help="abort unless the replay reproduces the logged step count and outcome")
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--hold-last", type=float, default=2.0)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    a = ap.parse_args()
    res = json.load(open(a.eval_result))
    level = str(res.get("level") or res.get("config_name"))
    actions = read_actions(a.session_log)
    pygame, RubikCubeGame, is_cube_solved = load_engine(a.spatialworld_repo)
    game = RubikCubeGame(config_name=level)
    if not game.init():
        sys.exit("could not start the Rubik's-cube engine")
    from PIL import Image
    frames = []

    def capture():
        frames.append(Image.fromarray(pygame.surfarray.array3d(game.screen).transpose(1, 0, 2).copy()))

    settle(game)
    capture()
    solved_at = None
    for k, (kind, face, direction) in enumerate(actions):
        ok = game.rotate_face(face, direction) if kind == "face" else game.rotate_cube(face, direction)
        if not ok:
            sys.exit(f"the engine refused turn {k + 1}: {kind} {face} {direction}")
        settle(game)
        capture()
        if solved_at is None and is_cube_solved(game.rubik):
            solved_at = k + 1
    game.close()
    solved = solved_at is not None
    if a.check:
        want_steps, want_solved = int(res["steps"]), bool(res.get("cube_solved", res.get("score", 0) >= 1))
        if len(actions) != want_steps:
            sys.exit(f"REPLAY MISMATCH: {len(actions)} turns in the log, {want_steps} steps in the evaluation")
        if solved != want_solved or (solved and solved_at != len(actions)):
            sys.exit(f"REPLAY MISMATCH: replay {'solved at turn %d' % solved_at if solved else 'did not solve'}, "
                     f"evaluation {'solved' if want_solved else 'did not solve'} in {want_steps} steps")
    tmp = tempfile.mkdtemp(prefix="rubik_")
    for i, f in enumerate(frames):
        f.save(f"{tmp}/f{i:05d}.png")
    n = len(frames)
    for j in range(int(round(a.hold_last * a.fps))):
        shutil.copy(f"{tmp}/f{n-1:05d}.png", f"{tmp}/f{n+j:05d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    subprocess.run([a.ffmpeg, "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "24", a.out], check=True)
    shutil.rmtree(tmp)
    print(f"{a.out}: level {level}, {len(actions)} turns replayed{' and verified' if a.check else ''}, "
          f"{'solved at turn %d' % solved_at if solved else 'not solved'}, {os.path.getsize(a.out) // 1024} KB")


if __name__ == "__main__":
    main()
