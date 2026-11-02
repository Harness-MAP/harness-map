# Task examples, the harnesses evolved for them, and their videos

`tests/` is the hands-on part of the repository. For each benchmark of the paper it provides task
examples, every harness the paper's best run evolved on that benchmark, videos of the task examples
played by the anchor and by harnesses of the lineage, and the configuration and scripts to run the
search yourself.

| benchmark | task examples ([`tasks/`](tasks/README.md)) | evolved harnesses | videos | run the search |
|---|---|---|---|---|
| SpatialWorld 3D games | one level per game: maze level 6, block3d level 12, maze3d_pro level 5, rubik level 6, snake run 20 | [`examples/spatialworld_games/evolution/<game>/harnesses/`](examples/spatialworld_games/evolution/README.md), 4 to 13 per game | `examples/spatialworld_games/evolution/<game>/videos/` | [`examples/spatialworld_games/`](examples/spatialworld_games/README.md) |
| BALROG-MiniHack | four episodes: MazeWalk-15x15 run 00, Corridor-R3 run 04, Boxoban-Medium run 00, Quest-Easy run 00 | [`examples/balrog/evolution/harnesses/`](examples/balrog/evolution/README.md), 41 | `examples/balrog/evolution/videos/` | [`examples/balrog/`](examples/balrog/README.md) |
| SimWorld-MMNav, hard | map 20_10, task dist_21_0_1 | [`examples/simworld_nav/evolution/harnesses/`](examples/simworld_nav/evolution/README.md), 16 | `examples/simworld_nav/evolution/videos/` | [`examples/simworld_nav/`](examples/simworld_nav/README.md) |

## Layout

```
tests/
├── tasks/<benchmark>/            the task examples: one JSON record per task in the format the
│                                 evaluator takes (--tasks), a preview frame, and a README with the
│                                 task and the outcome of every shown harness on it
└── examples/
    ├── run_benchmark.py          the search driver the run scripts call
    ├── toy/run_toy.py            the method on a toy task, no model needed
    ├── render/                   the scripts that turned the episode logs into the videos
    └── <benchmark>/              config.json, run.sh (placeholders), README (setup, evaluator contract)
        └── evolution/            the paper's best run: README, lineage.json,
            ├── harnesses/<NN>_<label>/   every harness as evaluated, with NOTES.md and diagnosis.md
            └── videos/           the task examples played by four harnesses of the lineage, and a 2×2 grid
```

`lineage.json` lists the harnesses of a run with their scores, parents, cell depth and the size of
the diff against the parent; `NOTES.md` is the proposer's own account of a change and
`diagnosis.md` the diagnostician's report on the harness's evaluation.

## Replay a task example with an evolved harness

Each benchmark README under `examples/` specifies the evaluator contract
(`evaluator.py --harness DIR --tasks TASKS.json --out OUT`). Point `--harness` at a harness
directory and `--tasks` at a task-example file:

```bash
"$GAME_PYTHON" "$GAME_EVALUATOR" \
    --harness tests/examples/spatialworld_games/evolution/maze3d_pro/harnesses/12_cand_r6_1_action-planning \
    --tasks   tests/tasks/spatialworld_games/maze3d_pro_level_05.json \
    --out     /tmp/maze3d_pro_level_05
```

The policy model samples at every step, so a replay is a fresh draw of the same harness on the same
task rather than a frame-for-frame copy of the video; the BALROG episodes are fixed-seed, so the
level itself is identical. `examples/render/` turns the resulting episode logs into videos like the
shipped ones.
