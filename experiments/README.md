# `experiments/` — experiment log

One markdown file per experiment. Filename pattern:

```
EXXX_short-slug.md        e.g. E003_mfcc-deltas.md
```

The numeric prefix is monotonically increasing. Never renumber or delete an
experiment — failed runs are data too.

## Workflow for every experiment

1. **Before running anything**, copy `_template.md` to `EXXX_slug.md` and fill in
   the *Hypothesis* and *Setup* sections. Writing the hypothesis first keeps you
   honest; writing it after the result is how post-hoc rationalization creeps
   in.
2. Run the experiment. Capture numbers (CV mean ± std of EER / min-DCF, plus
   any per-fold breakdown).
3. Fill in *Result*, *Interpretation*, *Next step*.
4. Append a one-line row to `index.md`.

## What counts as an experiment

A change you could describe in one sentence:

- "MFCC 13 vs 20 vs 40 coefficients" → one experiment, three configurations.
- "Add delta + delta-delta to the winning MFCC" → next experiment.
- "Swap GMM(16) for GMM(32)" → next experiment.

Don't bundle many axes into one experiment. If you change three things at once
and it gets better, you don't know which of the three mattered.

## Reproducibility checklist

Every experiment file must state:

- random seed,
- fold assignment source (commit hash or file path),
- exact command / notebook cell to reproduce,
- package versions if they differ from `uv.lock`.
