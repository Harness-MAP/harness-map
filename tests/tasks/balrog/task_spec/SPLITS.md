# BALROG-MiniHack splits — search set and seeded holdouts

Every episode's level is fixed by
`seed = int(sha256(f"{seed_base}|{task}|{episode_idx}").hexdigest()[:8], 16)`
(`evaluate.py::episode_seed`, installed over BALROG's own
`envs.env_kwargs.seed` by `install_crn_seeding()`; the BALROG repo is never modified).
Same seeds for every arm = common random numbers, so an arm-to-arm difference is the harness,
not the draw. Each split's seeds are written into its score.json (`seed_base`, `seeds`).

| split | file | tasks | episodes | seed_base | disjoint from search |
|---|---|---|---|---|---|
| search | `evolve.json` | the OFFICIAL 8 | 8 x 5 = 40 | 0 | — |
| holdout A: unseen LEVELS | `eval_levels.json` | the same official 8 | 8 x 5 = 40 | 1 | yes (0 shared seeds) |
| holdout B: unseen TASKS | `eval_tasks.json` | 8 tasks NOT in the official 8 | 8 x 5 = 40 | 0 | yes (different task string -> different hash) |

Holdout A asks: does the evolved harness generalize to NEW LEVELS of the tasks it searched on?
Holdout B asks: does it generalize to TASKS it never saw — four unseen variants of search-set
families (Boxoban-Unfiltered, Corridor-R5, CorridorBattle LIT, Quest-Hard) and four new families
(Labyrinth, MultiRoom, River, LavaCrossing)? All twelve candidate tasks were smoke-tested through
BALROG's own `make_minihack_env` + `config.yaml` (reset + step) before selection.

Run either split with the evaluator's `--split` flag; holdout A additionally needs
`BALROG_SEED_BASE=1`:

    BALROG_SEED_BASE=1 python tests/tasks/balrog/evaluate.py --harness <dir> --split eval_levels --out <dir>
    BALROG_SEED_BASE=0 python tests/tasks/balrog/evaluate.py --harness <dir> --split eval_tasks  --out <dir>

NOTE the search has no in-loop holdout split on purpose: `evaluate_holdout` is the in-loop hook,
and no holdout may be visible to the search. The two files above are scored ONCE, after each arm finishes, on the
harness the search selected.
