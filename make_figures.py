#!/usr/bin/env python3
"""Figures 1-3 for Artigo2_MAPAN (600 dpi PNG + TIFF), from gum_results.json and the source data."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gum_uncertainty_budget import RAW_GY, RAW_REF, COR_GY, COR_REF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "gum_results.json"), encoding="utf-8"))
S1, S2, DE, UC = R["stage1_raw"], R["stage2_corrected"], R["stage2_direct_error"], R["U_curve"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 600})
INK, ACC, GREY = "#1a1a1a", "#1f5aa6", "#8a8a8a"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".tif"), bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ---------------- Figure 1: calibration (a) and validation (b)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2))
xr = np.array(RAW_GY, float); yr = np.array(RAW_REF)
xs = np.linspace(0, 56, 50)
ax[0].plot(xr, yr, "o", ms=3.5, mfc="white", mec=INK, mew=0.8, label="Observations (n = 26)")
ax[0].plot(xs, S1["b0"] + S1["b1"] * xs, "-", color=ACC, lw=1.1,
           label=f"E_ref = {S1['b1']:.4f} r − {abs(S1['b0']):.4f}")
ax[0].plot([0, 45], [0, 45], ":", color=GREY, lw=0.9, label="Identity")
ax[0].set_xlabel("Raw GY-302 indication r (lx)"); ax[0].set_ylabel("Reference illuminance E_ref (lx)")
ax[0].set_title("(a) Stage 1 — calibration", fontsize=9, loc="left")
xc = np.array(COR_GY, float); yc = np.array(COR_REF)
xs2 = np.linspace(0, 45, 50)
ax[1].plot(xc, yc, "s", ms=3.2, mfc="white", mec=INK, mew=0.8, label="Observations (n = 22)")
ax[1].plot(xs2, S2["b0"] + S2["b1"] * xs2, "-", color=ACC, lw=1.1,
           label=f"E_ref = {S2['b1']:.4f} x + {S2['b0']:.4f}")
ax[1].plot([0, 45], [0, 45], ":", color=GREY, lw=0.9, label="Identity")
ax[1].set_xlabel("Corrected GY-302 indication x (lx)"); ax[1].set_ylabel("Reference illuminance E_ref (lx)")
ax[1].set_title("(b) Stage 2 — validation", fontsize=9, loc="left")
for a in ax:
    a.set_xlim(0, None); a.set_ylim(0, 46); a.legend(frameon=False, fontsize=7.5, loc="upper left")
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "Fig1_calibration")

# ---------------- Figure 2: measurement error of the corrected indication
e = np.array(DE["errors"]); etc = np.array(DE["errors_after_trunc_corr"])
fig, a = plt.subplots(figsize=(4.6, 3.3))
a.axhline(0, color=GREY, lw=0.7)
a.plot(yc, e, "o", ms=3.8, color=INK, label="e = x − E_ref (firmware truncation)")
a.plot(yc, etc, "^", ms=3.6, mfc="white", mec=ACC, mew=0.9, label="e + 0.5 lx (truncation corrected)")
a.axhline(DE["mean"], color=INK, lw=0.9, ls="--", label=f"Mean e = {DE['mean']:.2f} lx".replace("-", "−"))
a.axhline(DE["mean_after_trunc_corr"], color=ACC, lw=0.9, ls="--",
          label=f"Mean after correction = {DE['mean_after_trunc_corr']:.2f} lx".replace("-", "−"))
a.set_xlabel("Reference illuminance E_ref (lx)"); a.set_ylabel("Measurement error (lx)")
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
save(fig, "Fig2_error")

# ---------------- Figure 3: relative quantities vs the +/-20 % tolerance and U/E
rel = np.array(DE["rel_errors_pct"]); relres = np.array(R["stage2_residual_relative_pct"])
E = np.array(UC["E"]); UA = 100 * np.array(UC["U_A"]) / E; UCc = 100 * np.array(UC["U_C"]) / E
fig, a = plt.subplots(figsize=(4.6, 3.2))
a.axhspan(-20, 20, color="#e8eef7", lw=0)
a.axhline(20, color=ACC, lw=0.8); a.axhline(-20, color=ACC, lw=0.8)
a.text(43, 21.5, "±20 % tolerance [6]", ha="right", fontsize=7, color=ACC)
a.plot(E, UA, "-", color=INK, lw=1.0, label="±U/E, Scenario A (k = 2)")
a.plot(E, -UA, "-", color=INK, lw=1.0)
a.plot(E, UCc, "--", color=GREY, lw=0.9, label="±U/E, Scenario C (no reference terms)")
a.plot(E, -UCc, "--", color=GREY, lw=0.9)
a.plot(yc, rel, "o", ms=3.6, color=INK, label="Relative measurement error 100·e/E_ref")
a.plot(yc, relres, "x", ms=3.8, color=ACC, mew=0.9, label="Relative residual (Stage 2 line)")
a.set_ylim(-50, 35); a.set_xlim(0, 45)
a.set_xlabel("Reference illuminance E_ref (lx)"); a.set_ylabel("Relative value (%)")
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False, fontsize=6.8, loc="lower right")
fig.tight_layout()
save(fig, "Fig3_relative")
print("figures saved to", OUT)
