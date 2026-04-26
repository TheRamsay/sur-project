# Excalidraw diagrams for the oral defense

Two diagrams. Goal: each one stands alone on a slide and an examiner gets the point in 10 seconds while you talk over it.

---

## Diagram 1 — End-to-end system architecture

**Story it tells.** Three parallel pipelines, each calibrated to a common scale, then weighted-summed into a single decision. The weights tell you which streams matter.

### Layout (left to right)

```
[ INPUT ]            [ FRONT-END ]              [ BACKBONE ]                [ CALIB ]               [ FUSION ]
                                                                                                    
.wav  ─────────► LPCC 13+Δ+ΔΔ + CMN ──► UBM-32 tied + MAP r=16 ──► Platt ──┐
        │                                                                  │
        └──────► MFCC 13+Δ+ΔΔ + CMN ──► UBM-32 diag + MAP r=16 ──► Platt ──┤
                                                                           │
                                                                           ├──►  weighted sum  ──►  score  ──► (≥ τ ?)  ──► 0/1
                                                                           │   w_lpcc = 0.34
                                                                           │   w_mfcc = 0.00
                                                                           │   w_image = 0.66
.png  ─────────► 80×80 grayscale + std + PCA-50 ──► LogReg C=1 ──► Platt ──┘
```

### How to draw it in Excalidraw

- **Shapes**: rounded rectangles for processing blocks, plain rectangles for inputs / outputs, a single **circle** for the weighted-sum node and another for the threshold node.
- **Three horizontal lanes**, one per stream. Vertical alignment is the readability win — do not make them weave.
- **Color the lane by modality**:
  - LPCC + tied lane → **blue (#3A7EC5)** outlines (audio)
  - MFCC lane → **light blue / muted** outlines (audio, secondary)
  - Image lane → **orange-red (#C84B2F)** outlines (image)
- **Weights on the edges going into the fusion node**, in monospace: `0.34`, `0.00`, `0.66`. The `0.00` is the punchline — make it a slightly faded grey to show MFCC dropped out.
- **Threshold τ box** small, just a circle with `≥ τ`.
- **Output**: tiny pill labelled `score (R)` and `decision (0 / 1)`.
- **Title at top**: `Trimodal target-speaker detector — E039 (CV: 0/222)`.
- **Tiny footnote bottom-left**: `Platt calibration fitted on OOF scores. Weights from 51×51 simplex grid that minimises OOF EER.`

### Things to point at while talking

1. "Three pipelines, totally independent — different feature spaces, different statistical models."
2. "Each one outputs a raw score on its own scale. Platt puts them all on a comparable logit axis."
3. "I grid-search the simplex of fusion weights, optimising directly for OOF EER. The grid converges to image 0.66, LPCC 0.34, MFCC 0.00."
4. "MFCC's weight collapses because it is correlated 0.84 with LPCC. I keep it in the optimisation in case LPCC ever weakens."

---

## Diagram 2 — Why fusion makes 0 errors

**Story it tells.** Each stream alone misses ~5–7 samples. They miss **different** samples. So a sum of the two scores rescues everything.

### Layout

Two number lines. Audio above, image below. Both go from "non-target" on the left to "target" on the right with a vertical dashed line in the middle (the EER threshold).

```
                       audio EER threshold
                                |
audio   non-target ─────────────|────────────► target
              ●●●●     ●● ✗     |       ●●●●●
                                |       ●●●●●
                                |
              ●●●●        ✗ ●●  |       ●●●●●
image   non-target ─────────────|────────────► target
                                |
                       image EER threshold
```

- ● = correctly placed sample
- ✗ = a target that landed on the wrong side of *that* stream's threshold

The **trick**: the ✗ on the audio line is at a *different x-position* than the ✗ on the image line. They're not the same sample.

### Add a vertical "join" line

Below both number lines, draw vertical dotted lines connecting each `✗` on audio to where that *same sample* lives on the image line — and show that on the image side it's far on the **target** side. Same in reverse for the image-only failures.

Annotate the connecting lines:

- Top arrow (audio failure): `audio says NO → image rescues (says YES, big margin)`
- Bottom arrow (image failure): `image says NO → audio rescues (says YES, big margin)`

### The headline numbers (top of the diagram)

```
audio fails on:   5  samples
image fails on:  15  samples
both fail on:     0  samples       ←  this is why fusion works
```

Make the `0` huge. Maybe 60-80pt. Same fusion-green colour as the dashboard hero.

### How to draw it in Excalidraw

- Two horizontal lines, ~600–800 px wide, stacked with ~200 px gap.
- Use the **arrow tool** for the dotted connector lines between stream failures (the rescue lines).
- Coloured circles: audio = blue, image = orange-red, fusion area = green tint.
- One sentence below the whole thing: `Disjoint failure modes → weighted sum lands every sample on the right side of the decision boundary.`

### Things to point at while talking

1. "Each stream alone gets ~5–7 wrong on the held-out folds."
2. "But look — when the audio gets it wrong, the image is *very* confident in the right direction, and vice versa."
3. "So when you weight-sum the calibrated scores, the strong stream rescues the weak one on every disputed sample."
4. "Zero errors out of 222 isn't because either stream is perfect. It's because the failure sets don't overlap."

---

## Practical tips

- **Use Excalidraw's library / Figma export, not screenshots.** Save as SVG/PNG, embed straight into the slides.
- **Avoid more than 2 colours per diagram beyond modality tints.** The hand-drawn feel is the appeal — over-styled excalidraw looks worse, not better.
- **Hand-drawn ≠ messy.** Snap to grid, align lanes vertically, give shapes consistent corner radii. Excalidraw's auto-grid is good.
- **One headline per diagram.** Diagram 1's headline is the weights. Diagram 2's is the `0`.

If you need a third diagram for any reason, the only one I'd add is a tiny **CV split visual** — three rows showing the LOSO partition for target sessions 01/02/03 (one row held out per fold). Quick to draw, useful if Burget asks "show me your validation". But default to two.
