# The paper's benchmarks, hands on

`tests/` holds, for each benchmark of the paper, the evaluator the paper's runs used, task examples
with the outcomes to expect on them, every harness the paper's best run evolved, videos of the task
examples played by the anchor and by harnesses of the lineage, and the configuration and scripts to
run the search yourself.

| benchmark | evaluator and task examples ([`tasks/`](tasks/README.md)) | evolved harnesses | videos | run the search |
|---|---|---|---|---|
| SpatialWorld 3D games | `tasks/spatialworld_games/evaluate.py`; one level per game: maze level 6, block3d level 12, maze3d_pro level 5, rubik level 6, snake run 20 | [`examples/spatialworld_games/evolution/<game>/harnesses/`](examples/spatialworld_games/evolution/README.md), 4 to 13 per game | `examples/spatialworld_games/evolution/<game>/videos/` | [`examples/spatialworld_games/`](examples/spatialworld_games/README.md) |
| BALROG-MiniHack | `tasks/balrog/evaluate.py`; four episodes: MazeWalk-15x15 run 00, Corridor-R3 run 04, Boxoban-Medium run 00, Quest-Easy run 00 | [`examples/balrog/evolution/harnesses/`](examples/balrog/evolution/README.md), 41 | `examples/balrog/evolution/videos/` | [`examples/balrog/`](examples/balrog/README.md) |
| SimWorld-MMNav, hard | `tasks/simworld_nav/evaluate.py`; map 20_10, task dist_21_0_1 | [`examples/simworld_nav/evolution/harnesses/`](examples/simworld_nav/evolution/README.md), 16 | `examples/simworld_nav/evolution/videos/` | [`examples/simworld_nav/`](examples/simworld_nav/README.md) |

## Layout

```
tests/
├── tasks/<benchmark>/            evaluate.py: the evaluator of the paper's runs, which drives the benchmark's
│                                 own runner (--harness DIR --tasks TASKS.json --out OUT); one JSON record per
│                                 task example, in the format --tasks takes; a preview frame per example;
│                                 expected.json: what every shipped harness scored on each example; README
└── examples/
    ├── run_benchmark.py          the search driver the run scripts call
    ├── adapter_example.py        a custom adapter skeleton, with the adapter protocol documented
    ├── render/                   the scripts that turned the episode logs into the videos
    └── <benchmark>/              config.json, run.sh, README (installing the benchmark and the base model)
        └── evolution/            the paper's best run: README, lineage.json,
            ├── harnesses/<NN>_<label>/   every harness as evaluated, with NOTES.md and diagnosis.md
            └── videos/           the task examples played by four harnesses of the lineage, and a 2×2 grid
```

`lineage.json` lists the harnesses of a run with their scores, parents, cell depth and the size of
the diff against the parent; `NOTES.md` is the proposer's own account of a change and
`diagnosis.md` the diagnostician's report on the harness's evaluation.

## Run a task example

Install the benchmark and serve the base model as the benchmark's README under `examples/`
describes, set the evaluator's environment variables ([`tasks/README.md`](tasks/README.md) lists
them per benchmark), then point the evaluator at a harness directory and a task-example record:

```bash
export SPATIALWORLD_REPO=/path/to/SpatialWorld GAME_API_BASE=http://127.0.0.1:8000/v1
"$GAME_PYTHON" tests/tasks/spatialworld_games/evaluate.py \
    --harness tests/examples/spatialworld_games/evolution/maze3d_pro/harnesses/12_cand_r6_1_action-planning \
    --tasks   tests/tasks/spatialworld_games/maze3d_pro_level_05.json \
    --out     /tmp/maze3d_pro_level_05
```

`OUT/score.json` holds the score and the per-episode rows; `tasks/<benchmark>/expected.json` gives
the outcome every shipped harness obtained on the same task. The base model samples at every step,
so a replay is a fresh draw of the same harness on the same task rather than a frame-for-frame copy
of the video; the BALROG episodes are fixed-seed, so the level itself is identical. The same
evaluators run the searches: each benchmark's `run.sh` passes them to `run_benchmark.py`.
`examples/render/` turns episode logs into videos like the shipped ones.
