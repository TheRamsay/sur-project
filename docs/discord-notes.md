# SUR Discord log — key takeaways

Source: `~/discord_logs/VUT FIT - letni magistersky semestr 2 - sur-private [826109743680061530].html`
(6-year archive of the private course channel). Compiled 2026-04-19.

## Grading (25 pts total)

- **Tier-based rubric** from Burget's email: per-system thresholds (e.g. audio >55%,
  image >70%) give full points. Going 80→100% gains nothing.
- **Method & documentation matter more than accuracy** — "even if results are bad
  you can get full points." Multiple students got ~20/25 with broken systems but
  good writeups.
- **Oral defense is decisive**: if you can't explain what you wrote, he gives
  zero. He steers gently if you're wrong.
- Off-by-one in output files has cost −3 pts historically.
- Bonus points possible for attending final-lecture presentation ("winning team"
  award, not mandatory).

## What worked best

- **Audio**: MFCC + GMM is canonical, routinely 85–91% dev across years. UBM+MAP
  and SVC on averaged MFCCs also work. Our E008 UBM+MAP+aug is already past the
  full-point tier.
- **Image**: On this tiny dataset, **SVM/logreg on pixel-ish features beat small
  CNNs** — one team: CNN 30% → SVM >90%; another hit 100% with augmentation.
  Validates our PCA+LogReg+aug flagship. CNNs without pretraining peak ~40–70%.
- **Fusion**: Burget explicitly said in lecture — **map scores to (−∞, ∞) via
  logit first, then combine**. Don't multiply probabilities directly. Our
  Platt+weighted-sum is consistent with this.
- **Augmentation** is strongly encouraged by students and teacher across years.

## Pitfalls to avoid

- **No pretrained anything** — must be flagged in the report; Burget actively
  separates cheaters. Even Silero-VAD for silence trimming was contested.
- **Eval data shape surprises** — 2022 eval PNGs had an alpha channel vs RGB
  train. Validate eval format on day 1.
- **`glob` returns unordered paths** — several teams burned expecting sorted
  output. Always `sorted(glob(...))`.
- Eval accuracy is usually much worse than dev (distribution shift). One team
  had 40% dev / ~random on eval.
- Burget sometimes grades items "full or zero" — no partial credit on a given
  element.

## Submission gotchas

- **5 MB upload limit** on IS — surprisingly tight, plan ZIP size.
- `SRC/` directory inside the ZIP containing the code.
- Result files plain ASCII, self-describing names (e.g. `audio_gmm.txt`).
- Scores do not need to be sorted, but stem order must match what Burget expects.
- Documentation ~3 pages A4, explains *why* for each choice.
- Jupyter notebooks have been accepted as part of SRC in prior years.
- WIS/IS deadline and email deadline have historically disagreed by a day —
  follow the email.

## Deadline / eval-data flow

- Eval data released **morning of deadline day** (matches our 2026-05-03).
- Grading takes 2–3 weeks, often lands near the 2nd exam term — plan retakes
  accordingly.
- Brief changes yearly — don't over-trust old repos.

## Implications for us

Our flagships (E007 image 0.97%, E008 audio 4.21%, E009 fusion 3.75%) are well
above the full-point thresholds. **Remaining risk is the defense and the
report**:

- `dokumentace.pdf` must justify every choice (why PCA=50, why UBM=32, why the
  aug set, why fusion w=0.28) and we must defend each live.
- Add an explicit "no pretrained models used" statement to the report.
- Include ablation tables — they're what separates method-understanding from
  number-chasing in Burget's eyes.
- Self-test on a mini-eval before uploading — file-ordering bugs are the most
  common −points issue.
