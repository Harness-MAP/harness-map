## honesty and cost

- Every sheet field is self-derived: the digest comes from `frame_data.image`, the picture the model is shown, and the key comes from the `Action` this harness itself returned.
- `evaluator.game` is never read; the guard allows only `.screen`, and the ledger touches neither that nor the level files under `data/`.
- The sheet is labelled as the harness's own notebook rather than ground truth, so a wrong inference cannot be mistaken for an oracle reading.
- One `baseline_step` call per step, exactly as the anchor: the VLM-call counter stays non-zero and the decision is still the frozen model's.
- The frame-and-prompt path sits inside one `try`; a surprise there increments `slips` and the step proceeds exactly as the anchor's would.

Validation: compiled every harness module, imported `tools.vlm` under the episode interpreter, and drove `decide` against a stubbed evaluator with a scripted `baseline_step` and synthetic frames — a repeated frame raises the streak and fills the cue, a changed frame zeroes it, message 0 keeps its base prefix across steps, and step 1 is byte-identical to the anchor's.
