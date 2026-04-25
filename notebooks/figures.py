# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (sur)
#     language: python
#     name: sur-venv
# ---

# %% [markdown]
# # SUR 2025/2026 — Publication figures
#
# All plots for `docs/draft.md`. Styled with seaborn (`whitegrid` / `paper`)
# for a clean, journal-ready look. Runs top-to-bottom; saves vector PDF +
# 300 DPI PNG into `docs/figures/`.

# %%
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm
from sklearn.metrics import roc_curve

REPO = Path.cwd() if Path.cwd().name == "project" else Path.cwd().parent
sys.path.insert(0, str(REPO))
FIG   = REPO / "docs" / "figures"
CACHE = REPO / "cache"
FIG.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Theme
#
# Seaborn `whitegrid` + `paper` context. Colorblind palette with three
# domain-specific accents used consistently across figures:
# audio = blue, image = orange, fusion = green. Baselines are rendered in
# neutral grey so the flagship bar visually pops.

# %%
sns.set_theme(
    style="whitegrid",
    context="paper",
    palette="colorblind",
    font="sans-serif",
    rc={
        "figure.dpi":        140,
        "savefig.dpi":       300,
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "font.size":         10,
        "axes.titlesize":    11.5,
        "axes.titleweight":  "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad":     10,
        "axes.labelsize":    10,
        "axes.labelweight":  "regular",
        "axes.edgecolor":    "#3A3F4B",
        "axes.linewidth":    0.8,
        "axes.grid.axis":    "y",
        "grid.color":        "#E4E7EC",
        "grid.linewidth":    0.7,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.frameon":    False,
        "lines.linewidth":   1.6,
        "lines.markersize":  5.5,
    },
)

# Explicit hex palette — colorblind-safe, each stream has its own identity.
# Avoids ambiguity from seaborn palette indexing.
C = {
    "audio":  "#1F6FB4",   # clear navy-blue  (Okabe-Ito blue)
    "image":  "#D7820E",   # warm burnt orange
    "fusion": "#2E8B57",   # forest green
    "muted":  "#CBD0D8",
    "grey":   "#7A7F8A",
    "ink":    "#1F2430",
    "accent": "#C44536",   # muted crimson, reserved for reference lines
}


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")


def annotate_bars(ax, fmt: str = "{:.2f}%", pad: float = 0.02, fontsize=8.5):
    """Label bar heights just above each bar."""
    ymax = ax.get_ylim()[1]
    offset = ymax * pad
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h) or h == 0:
            continue
        ax.text(p.get_x() + p.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=C["ink"])


# %% [markdown]
# ## Figure 1 — Audio: GMM covariance type ablation (E037)
#
# Tied covariance is the single biggest lever in the whole audio track:
# 6.3× lower EER than diagonal at a marginal parameter-count increase, and
# 32× cheaper than the over-fitting full-covariance variant.

# %%
cov_df = pd.DataFrame({
    "covariance": ["spherical", "diagonal", "tied", "full"],
    "EER":        [3.89, 4.35, 0.69, 1.48],
    "std":        [3.75, 4.40, 0.98, 0.92],
    "params":     [32, 1248, 1521, 48672],
})

fig, ax = plt.subplots(figsize=(5.6, 3.4))
palette = [C["muted"] if c != "tied" else C["audio"] for c in cov_df["covariance"]]
sns.barplot(data=cov_df, x="covariance", y="EER", hue="covariance",
            palette=palette, ax=ax, legend=False, width=0.62)
ax.errorbar(x=np.arange(len(cov_df)), y=cov_df["EER"], yerr=cov_df["std"],
            fmt="none", ecolor=C["grey"], elinewidth=0.8, capsize=3)
for i, (eer, n) in enumerate(zip(cov_df["EER"], cov_df["params"])):
    ax.text(i, eer + 0.25, f"{eer:.2f}%", ha="center", va="bottom",
            fontsize=9, color=C["ink"], fontweight="semibold")
    ax.text(i, -0.6, f"{n:,} params", ha="center", va="top",
            fontsize=8, color=C["grey"])
ax.set_ylabel("CV EER (%)")
ax.set_xlabel("")
ax.set_title("Audio: GMM covariance type ablation (E037)")
ax.set_ylim(0, 10.5)
ax.margins(x=0.05)
sns.despine(ax=ax)
save(fig, "fig1_covariance_ablation")
plt.show()


# %% [markdown]
# ## Figure 2 — Image: adversarial-rotation robustness (E033)
#
# Clean EER halves vs E007; rotation robustness improves 2.5–13× depending on
# the re-stress protocol (both reported to keep the comparison honest).

# %%
rot_df = pd.DataFrame({
    "condition": np.tile(["clean", "rot ±15°\n(E033 eval)", "rot ±15°\n(E051 re-stress)"], 2),
    "system":    ["E007 (+All aug)"] * 3 + ["E033 (+AdvRot)"] * 3,
    "EER":       [0.97, 13.70, 19.00, 0.51, 1.04, 7.59],
})

fig, ax = plt.subplots(figsize=(6.2, 3.4))
sns.barplot(data=rot_df, x="condition", y="EER", hue="system",
            palette=[C["muted"], C["image"]], ax=ax, width=0.7)
annotate_bars(ax, fontsize=8.5)
ax.set_ylabel("EER (%)")
ax.set_xlabel("")
ax.set_title("Image: adversarial rotation training (E033)")
ax.set_ylim(0, 22)
ax.legend(title="", loc="upper left")
sns.despine(ax=ax)
save(fig, "fig2_adv_rot_robustness")
plt.show()


# %% [markdown]
# ## Figure 3 — Audio: codec-bandwidth robustness (E052)
#
# Codec augmentation is the cleanly-additive case: zero clean-EER cost,
# −10 pp under bandwidth-limited stress.

# %%
codec_df = pd.DataFrame({
    "condition": np.tile(["clean", "codec 8 kHz"], 2),
    "system":    ["E042 (no codec aug)"] * 2 + ["E052 (+ codec aug)"] * 2,
    "EER":       [0.46, 13.33, 0.46, 3.33],
    "std":       [0.65,  3.79, 0.65, 4.14],
})

fig, ax = plt.subplots(figsize=(5.2, 3.4))
sns.barplot(data=codec_df, x="condition", y="EER", hue="system",
            palette=[C["muted"], C["audio"]], ax=ax, width=0.65, errorbar=None)
# Hand-draw error bars because seaborn's built-in stats recompute from data.
xs = []
for cont in ax.containers:
    for p in cont:
        xs.append(p.get_x() + p.get_width() / 2)
ax.errorbar(xs, codec_df["EER"], yerr=codec_df["std"], fmt="none",
            ecolor=C["grey"], elinewidth=0.8, capsize=3)
annotate_bars(ax, fontsize=8.5)
ax.set_ylabel("EER (%)")
ax.set_xlabel("")
ax.set_title("Audio: codec-bandwidth augmentation (E052)")
ax.set_ylim(0, 21)
ax.legend(title="", loc="upper left")
sns.despine(ax=ax)
save(fig, "fig3_codec_robustness")
plt.show()


# %% [markdown]
# ## Figure 4 — Experiment progression (flagship-moving steps only)
#
# One panel per stream. Only experiments that actually advanced the flagship
# are shown. Log-y axis; start/end EER called out. Audio: 39× reduction.
# Image: 9×. Fusion: 14×.

# %%
milestones = pd.DataFrame([
    ("audio",  1, "E001", "MFCC+GMM",              17.92),
    ("audio",  2, "E003", "+UBM/MAP",               7.45),
    ("audio",  3, "E008", "+aug",                   4.21),
    ("audio",  4, "E020", "LPCC",                   3.33),
    ("audio",  5, "E025", "+pitch",                 1.94),
    ("audio",  6, "E037", "+tied",                  0.69),
    ("audio",  7, "E052", "+codec",                 0.46),
    ("image",  1, "E004", "PCA+LogReg",             4.49),
    ("image",  2, "E007", "+aug",                   0.97),
    ("image",  3, "E033", "+adv-rot",               0.51),
    ("fusion", 1, "E009", "MFCC+img",               3.75),
    ("fusion", 2, "E023", "LPCC+img",               0.52),
    ("fusion", 3, "E027", "trimodal",               0.26),
    ("fusion", 4, "E039", "trimodal*",              0.26),
], columns=["stream", "step", "exp", "label", "EER"])

g = sns.relplot(
    data=milestones, x="step", y="EER", col="stream",
    kind="line", hue="stream", palette=[C["audio"], C["image"], C["fusion"]],
    marker="o", markersize=9, markeredgecolor="white", markeredgewidth=1.3,
    linewidth=1.8, legend=False,
    height=3.0, aspect=1.15, facet_kws={"sharex": False, "sharey": True},
)
g.set(yscale="log")
g.set_titles("{col_name}")
g.set_axis_labels("", "CV EER (%, log scale)")

for stream, ax in g.axes_dict.items():
    sub = milestones[milestones["stream"] == stream]
    ax.set_xticks(sub["step"])
    ax.set_xticklabels(sub["exp"], rotation=30, ha="right", fontsize=8.5)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylim(0.2, 30)
    ax.grid(which="both", axis="y", color="#E4E7EC", linewidth=0.6, alpha=0.8)
    # Bold final-EER callout, subtle start-EER note.
    last = sub.iloc[-1]; first = sub.iloc[0]
    ax.annotate(f"{last['EER']:.2f}%", (last["step"], last["EER"]),
                xytext=(-6, -14), textcoords="offset points",
                fontsize=10, color=C[stream], fontweight="bold", ha="right")
    ax.annotate(f"{first['EER']:.2f}%", (first["step"], first["EER"]),
                xytext=(8, 6), textcoords="offset points",
                fontsize=8.5, color=C[stream], alpha=0.9)

g.figure.suptitle("Experiment progression — flagship-moving steps only",
                  y=1.04, fontsize=12, fontweight="semibold", x=0.06, ha="left")
save(g.figure, "fig4_progression")
plt.show()


# %% [markdown]
# ## Figure 5 — Stress-test robustness (E028 / E051)
#
# Cross-section of both modalities under evaluation-time perturbations.
# Baselines (grey) vs flagships (coloured): photometric degradations are
# nearly free for image E033; codec is the audio system's main vulnerability
# but E052's codec aug fully closes that gap (see Fig. 3).

# %%
stress_df = pd.DataFrame([
    # image — E007 baseline vs E033 flagship on E051 re-stress protocol
    ("image", "clean",        "baseline",  0.97, "E007"),
    ("image", "JPEG q=15",    "baseline",  1.25, "E007"),
    ("image", "blur σ=3",     "baseline",  1.71, "E007"),
    ("image", "rot ±15°",     "baseline", 19.00, "E007"),
    ("image", "occlude 20²",  "baseline", 18.10, "E007"),
    ("image", "clean",        "flagship",  0.51, "E033"),
    ("image", "JPEG q=15",    "flagship",  0.51, "E033"),
    ("image", "blur σ=3",     "flagship",  0.51, "E033"),
    ("image", "rot ±15°",     "flagship",  7.59, "E033"),
    ("image", "occlude 20²",  "flagship", 11.06, "E033"),
    # audio — E042 (pre-codec-aug) vs E052 flagship on E051 stress conditions
    ("audio", "clean",        "baseline",  0.46, "E042"),
    ("audio", "speed ±10%",   "baseline",  0.74, "E042"),
    ("audio", "noise 20 dB",  "baseline",  4.35, "E042"),
    ("audio", "noise 10 dB",  "baseline",  6.85, "E042"),
    ("audio", "codec 8 kHz",  "baseline", 13.33, "E042"),
    ("audio", "clean",        "flagship",  0.46, "E052"),
    ("audio", "speed ±10%",   "flagship",  0.74, "E052"),
    ("audio", "noise 20 dB",  "flagship",  4.35, "E052"),
    ("audio", "noise 10 dB",  "flagship",  6.85, "E052"),
    ("audio", "codec 8 kHz",  "flagship",  3.33, "E052"),
], columns=["modality", "condition", "system", "EER", "exp"])

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
for ax, modality in zip(axes, ["image", "audio"]):
    sub = stress_df[stress_df["modality"] == modality]
    flagship_color = C[modality]
    sns.barplot(
        data=sub, x="condition", y="EER", hue="system",
        hue_order=["baseline", "flagship"],
        palette=[C["muted"], flagship_color],
        ax=ax, width=0.72, edgecolor="none",
    )
    ymax = stress_df["EER"].max() * 1.08
    ax.set_ylim(0, ymax)
    for p in ax.patches:
        h = p.get_height()
        if h and not np.isnan(h):
            ax.text(p.get_x() + p.get_width() / 2, h + ymax * 0.012,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=7.8, color=C["ink"])
    exps = sub.drop_duplicates("system").set_index("system")["exp"]
    handles = [
        mpl.patches.Patch(color=C["muted"],       label=f"{exps['baseline']} baseline"),
        mpl.patches.Patch(color=flagship_color,   label=f"{exps['flagship']} flagship"),
    ]
    ax.legend(handles=handles, title="", loc="upper left", fontsize=8.5)
    ax.set_title(modality)
    ax.set_xlabel(""); ax.set_ylabel("EER (%)" if modality == "image" else "")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(25); lbl.set_ha("right")
    sns.despine(ax=ax)

fig.suptitle("Stress-test robustness — baseline vs flagship",
             y=1.02, fontsize=12, fontweight="semibold", x=0.02, ha="left")
fig.tight_layout()
save(fig, "fig5_stress_robustness")
plt.show()


# %% [markdown]
# ## OOF pipeline — required for Figures 6 & 7
#
# Reuses `predict_fusion.py` helpers verbatim. Cached after first run.

# %%
OOF_CACHE = CACHE / "oof_scores.pkl"


def compute_oof_scores():
    import predict_fusion as pf
    from src.data.splits import load_manifest, iter_folds_loso

    manifest = load_manifest(REPO / "data")
    y_all    = manifest["label"].to_numpy()
    n        = len(manifest)
    oof_m    = np.full(n, np.nan)
    oof_l    = np.full(n, np.nan)
    oof_i    = np.full(n, np.nan)

    for fold_id, train_idx, val_idx in iter_folds_loso(manifest, seed=pf.SEED):
        seed_f   = pf.SEED + fold_id
        train_df = manifest.loc[train_idx]
        val_df   = manifest.loc[val_idx]
        print(f"  fold {fold_id}...")
        ubm_m, ad_m = pf._train_mfcc(train_df, REPO / "data", augment=True, seed=seed_f)
        ubm_l, ad_l = pf._train_lpcc(train_df, REPO / "data", augment=True, seed=seed_f)
        scaler, pca, clf = pf._train_image(train_df, REPO / "data", augment=True, seed=seed_f)
        for idx, row in val_df.iterrows():
            oof_m[idx] = pf._score_mfcc(pf._find_wav(row["stem"], REPO / "data"), ad_m, ubm_m)
            oof_l[idx] = pf._score_lpcc(pf._find_wav(row["stem"], REPO / "data"), ad_l, ubm_l)
            oof_i[idx] = pf._score_image(pf._find_png(row["stem"], REPO / "data"), scaler, pca, clf)

    cal_m = pf._fit_calibrator(oof_m, y_all)
    cal_l = pf._fit_calibrator(oof_l, y_all)
    cal_i = pf._fit_calibrator(oof_i, y_all)
    cal_mo = cal_m.decision_function(oof_m.reshape(-1, 1))
    cal_lo = cal_l.decision_function(oof_l.reshape(-1, 1))
    cal_io = cal_i.decision_function(oof_i.reshape(-1, 1))

    best = (np.inf, None)
    for w_m in np.linspace(0, 1, 51):
        for w_l in np.linspace(0, 1 - w_m, 51):
            w_i = 1 - w_m - w_l
            fused = w_m * cal_mo + w_l * cal_lo + w_i * cal_io
            eer, _ = pf.compute_eer(fused[y_all == 1], fused[y_all == 0])
            if eer < best[0]:
                best = (eer, (w_m, w_l, w_i))
    w_m, w_l, w_i = best[1]
    fused = w_m * cal_mo + w_l * cal_lo + w_i * cal_io
    print(f"  weights: mfcc={w_m:.2f}  lpcc={w_l:.2f}  image={w_i:.2f}  OOF EER={best[0]*100:.2f}%")
    return {"mfcc": cal_mo, "lpcc": cal_lo, "image": cal_io,
            "fusion": fused, "y": y_all, "weights": (w_m, w_l, w_i)}


if OOF_CACHE.exists():
    print(f"[oof] loading cached {OOF_CACHE}")
    scores = pickle.load(open(OOF_CACHE, "rb"))
else:
    print("[oof] no cache — training full OOF pipeline (~5 min)")
    scores = compute_oof_scores()
    pickle.dump(scores, open(OOF_CACHE, "wb"))

print(f"fusion weights: {scores['weights']}")


# %% [markdown]
# ## Figure 6 — DET curve (OOF)
#
# Standard speaker-verification visualization: miss rate vs false-alarm rate
# on probit-warped axes. Fusion achieves 0/222 errors (dot at corner,
# no curve is plottable — reported alongside the per-stream curves).

# %%
def det_axes(ax):
    ticks = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    labs  = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"]
    ax.set_xticks([norm.ppf(t) for t in ticks]); ax.set_xticklabels(labs)
    ax.set_yticks([norm.ppf(t) for t in ticks]); ax.set_yticklabels(labs)
    lo, hi = norm.ppf(0.0005), norm.ppf(0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], "--", color="#B0B4BC", linewidth=0.7, zorder=0)
    ax.set_xlabel("False-alarm rate (%)")
    ax.set_ylabel("Miss rate (%)")
    ax.grid(which="both", color="#E4E7EC", linewidth=0.5, alpha=0.9)


def det_points(s_tar, s_non):
    y   = np.r_[np.ones_like(s_tar), np.zeros_like(s_non)]
    s   = np.r_[s_tar, s_non]
    fpr, tpr, _ = roc_curve(y, s)
    far, frr    = fpr, 1.0 - tpr
    m = (far > 0) & (far < 1) & (frr > 0) & (frr < 1)
    return norm.ppf(far[m]), norm.ppf(frr[m])


streams = [
    ("MFCC",   scores["mfcc"],   C["grey"],   1.2, "-"),
    ("LPCC",   scores["lpcc"],   C["audio"],  1.5, "-"),
    ("Image",  scores["image"],  C["image"],  1.5, "-"),
    ("Fusion", scores["fusion"], C["fusion"], 2.2, "-"),
]
y = scores["y"]

fig, ax = plt.subplots(figsize=(5.4, 5.4))
det_axes(ax)
for name, s, c, lw, ls in streams:
    xx, yy = det_points(s[y == 1], s[y == 0])
    if len(xx) == 0:
        x0, y0 = norm.ppf(0.0015), norm.ppf(0.0015)
        ax.plot(x0, y0, marker="*", markersize=14, color=c,
                markeredgecolor="white", markeredgewidth=1.4,
                linestyle="None", label=f"{name}  (0/222 errors)", zorder=5)
        ax.annotate("0 errors",
                    xy=(x0, y0),
                    xytext=(norm.ppf(0.025), norm.ppf(0.0045)),
                    fontsize=9, color=c, fontweight="semibold",
                    arrowprops={"arrowstyle": "-", "color": c,
                                "linewidth": 0.9, "shrinkA": 4, "shrinkB": 6})
    else:
        ax.plot(xx, yy, color=c, linewidth=lw, linestyle=ls,
                label=name, solid_capstyle="round")
ax.set_title("DET curves — OOF scores (222 trials)")
ax.legend(loc="upper right", handlelength=2.2, borderaxespad=0.4)
save(fig, "fig6_det_curve")
plt.show()


# %% [markdown]
# ## Figure 7 — Fused OOF score distribution
#
# Calibrated log-odds of the trimodal fusion for all 222 OOF trials.
# Zero overlap at the Bayes threshold (log-odds = 0) — the visual proof of
# 0/222 errors. Densities (KDE) overlaid on top of the histograms.

# %%
fused = scores["fusion"]
y     = scores["y"]
sdf = pd.DataFrame({
    "score": fused,
    "class": np.where(y == 1, "target", "non-target"),
})
order   = ["non-target", "target"]
palette = {"non-target": C["muted"], "target": C["fusion"]}

fig, ax = plt.subplots(figsize=(7.2, 3.6))
sns.histplot(data=sdf, x="score", hue="class", hue_order=order, palette=palette,
             bins=42, stat="count", element="bars",
             edgecolor="white", linewidth=0.6, alpha=0.92, ax=ax)
# Soft KDE outlines sharing the primary y-axis via scale=count.
for cls in order:
    sns.kdeplot(data=sdf[sdf["class"] == cls], x="score", ax=ax,
                color=palette[cls], linewidth=1.4, bw_adjust=0.8,
                common_norm=False, weights=np.ones(int((sdf["class"] == cls).sum())))
# Reference threshold.
ax.axvline(0, color=C["accent"], linestyle=(0, (4, 3)), linewidth=1.2, zorder=0)
ymax = ax.get_ylim()[1]
ax.text(0.15, ymax * 0.96, "Bayes threshold\n(prior 0.5)",
        fontsize=8.5, color=C["accent"], va="top", ha="left")
# Class-count annotations at the margins.
ax.text(sdf[sdf["class"] == "non-target"]["score"].min() - 0.2, ymax * 0.85,
        f"non-target\nn={(y == 0).sum()}",
        color=C["grey"], fontsize=9, va="top", ha="left")
ax.text(sdf[sdf["class"] == "target"]["score"].max() + 0.2, ymax * 0.85,
        f"target\nn={(y == 1).sum()}",
        color=C["fusion"], fontsize=9, va="top", ha="right", fontweight="semibold")
ax.set_xlabel("Fused calibrated log-odds")
ax.set_ylabel("Count")
ax.set_title("OOF score distribution — E039 trimodal (0/222 errors)")
# Remove duplicate legend (class annotations cover it).
leg = ax.get_legend()
if leg is not None:
    leg.remove()
sns.despine(ax=ax)
save(fig, "fig7_score_histogram")
plt.show()


# %%
print("All figures written to", FIG)
