# BALROG-MiniHack splits

Every episode's level is fixed by its seed,
`int(sha256(f"{seed_base}|{task}|{episode_idx}").hexdigest()[:8], 16)`
(`episode_seed` in `../evaluate.py`, installed over BALROG's own `envs.env_kwargs.seed`), so every
candidate plays the same levels and a difference between two harnesses is the harness, not the
draw. Each evaluation writes its `seed_base` and `seeds` into `score.json`.

| split | file | tasks | episodes | seed_base | disjoint from the search set |
|---|---|---|---|---|---|
| search | `evolve.json` | the official 8 | 8 × 5 = 40 | 0 | — |
| held-out levels | `eval_levels.json` | the same 8 | 8 × 5 = 40 | 1 | yes: no shared seed |
| held-out tasks | `eval_tasks.json` | 8 tasks outside the official set | 8 × 5 = 40 | 0 | yes: a different task string gives a different hash |

The held-out levels ask whether the evolved harness generalizes to new levels of the tasks it
searched on; the held-out tasks whether it generalizes to tasks it never saw: four unseen variants
of search-set families (Boxoban-Unfiltered, Corridor-R5, CorridorBattle LIT, Quest-Hard) and four
new families (Labyrinth, MultiRoom, River, LavaCrossing), all checked through BALROG's own
`make_minihack_env` (reset and step) before selection.

The records carry their `seed_base`, so a split runs as is (`--seed-base` overrides it):

    python tests/balrog/tasks/evaluate.py --harness <dir> --split eval_levels --out <dir>
    python tests/balrog/tasks/evaluate.py --harness <dir> --split eval_tasks  --out <dir>

The search never sees either held-out split. `run_benchmark.py` scores `holdout_levels` on the
selected harness when the search ends (`holdout_split` in `config.json`); `holdout_tasks` is
scored with the command above.
