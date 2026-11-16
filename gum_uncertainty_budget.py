#!/usr/bin/env python3
"""
GUM (JCGM 100:2008) uncertainty evaluation of the corrected GY-302 (BH1750FVI)
illuminance sensor calibrated against an ITR-3000 luxmeter.

Data: Carvalho Neto (2021) dissertation, Tables 5 (n = 26, raw) and 6 (n = 22,
corrected).  Values are transcribed verbatim; nothing is estimated.

Firmware facts (dissertation, Appendix A):
  uint16_t lux = lightMeter.readLightLevel();   -> raw indication truncated to 1 lx
  int lux2 = 0.7777*lux - 0.4562;               -> corrected indication truncated to 1 lx
  Sensor operated in CONTINUOUS_HIGH_RES_MODE (datasheet resolution 1 lx).

Reference instrument (ITR-3000 manual, p. 13): accuracy +/-3 % of reading for
CIE illuminant A (2854 K); +/-6 % of reading for other visible light sources
(LED used here); valid for one year after calibration, 18-28 C, RH < 80 %.
Display resolution 0.01 lx (20 lx range) and 0.1 lx (200 lx range).
No calibration certificate is available.

Usage:  python gum_uncertainty_budget.py [output_dir]
"""
import json
import os
import sys

import numpy as np
from scipy import stats

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- data
RAW_GY = [54, 51, 49, 47, 44, 43, 41, 39, 37, 32, 29, 27, 24, 22, 19, 16, 12, 10, 8, 7, 6, 5, 5, 3, 3, 1]
RAW_REF = [41.5, 38.7, 37.2, 36.4, 33.9, 32.8, 31.6, 30.0, 28.3, 25.0, 23.2, 21.2, 18.8, 15.7,
           13.34, 11.3, 8.84, 6.71, 5.85, 5.1, 4.33, 3.6, 2.84, 2.08, 1.7, 1.2]
COR_GY = [43, 40, 39, 37, 35, 34, 31, 25, 22, 19, 18, 15, 14, 11, 9, 7, 6, 5, 4, 3, 2, 1]
COR_REF = [43.3, 40.9, 39.4, 37.7, 36.1, 35.1, 31.7, 26.4, 22.5, 20.5, 18.16, 15.34, 13.27,
           11.17, 8.85, 7.07, 6.13, 5.44, 4.55, 3.76, 2.31, 1.78]

# Values fixed by the authors (Reforco A) -- used only as an assertion check
FIXED = {
    "raw": dict(slope=0.7777, intercept=-0.4562, R2=0.9985, RMSE=0.517, MAE=0.402),
    "corrected": dict(slope=1.0123, intercept=0.2846, R2=0.9989, RMSE=0.470, MAE=0.360),
}

SETPOINTS = [10.0, 25.0, 33.0]          # lx, broiler programme levels (Cobb guide; dissertation)
EQ9_SLOPE = 0.7777                       # firmware correction slope
ITR_REL_ACC = 0.06                       # +/-6 % of reading, non-incandescent visible source
ITR_REL_ACC_INCAND = 0.03                # +/-3 %, illuminant A only (sensitivity scenario)
K = 2.0


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b0, b1 = beta
    yhat = X @ beta
    res = y - yhat
    nu = n - 2
    sse = float(res @ res)
    s = np.sqrt(sse / nu)                                  # residual standard deviation (GUM Type A)
    cov = s ** 2 * np.linalg.inv(X.T @ X)                  # covariance of (b0, b1)
    u_b0, u_b1 = np.sqrt(np.diag(cov))
    r01 = cov[0, 1] / (u_b0 * u_b1)
    t95 = stats.t.ppf(0.975, nu)
    sst = float(((y - y.mean()) ** 2).sum())
    xbar = float(x.mean())
    sxx = float(((x - xbar) ** 2).sum())
    # joint test H0: (b0, b1) = (0, 1)
    d = beta - np.array([0.0, 1.0])
    F = float(d @ (X.T @ X) @ d / (2 * s ** 2))
    pF = float(stats.f.sf(F, 2, nu))
    return dict(
        n=n, nu=nu, b0=float(b0), b1=float(b1), u_b0=float(u_b0), u_b1=float(u_b1),
        cov_b0b1=float(cov[0, 1]), r_b0b1=float(r01), s_res=float(s),
        RMSE=float(np.sqrt(sse / n)), MAE=float(np.mean(np.abs(res))),
        R2=1 - sse / sst, xbar=xbar, Sxx=sxx, t95=float(t95),
        CI_b1=[float(b1 - t95 * u_b1), float(b1 + t95 * u_b1)],
        CI_b0=[float(b0 - t95 * u_b0), float(b0 + t95 * u_b0)],
        joint_identity_F=F, joint_identity_p=pF,
        range_ref=[float(y.min()), float(y.max())],
        residuals=res.tolist(), fitted=yhat.tolist(),
    )


def welch_satterthwaite(contribs):
    """contribs: list of (u_i * c_i, nu_i) ; nu_i = inf for Type B."""
    uc2 = sum(c ** 2 for c, _ in contribs)
    den = sum((c ** 4) / nu for c, nu in contribs if np.isfinite(nu) and c > 0)
    return np.inf if den == 0 else uc2 ** 2 / den


def budget(fit, E_target, rel_acc_ref, include_ref=True):
    """Uncertainty budget for the illuminance estimated from one corrected indication.

    Measurement model (stage-2 calibration function):
        E = b0 + b1 * x + d_rep + d_q,raw + d_q,out + d_ref,acc + d_ref,res
    x      corrected GY-302 indication giving E_target on the calibration line
    d_rep  scatter of a single indication about the calibration line (s_res, Type A)
    d_q    quantisation of the raw (1 lx) and corrected (1 lx, truncated) indications
    d_ref  ITR-3000 accuracy and display resolution (manufacturer data, Type B)
    Quantisation and reference-resolution effects are already present in s_res;
    listing them separately is deliberately conservative (declared in the text).
    """
    b0, b1, n, nu = fit["b0"], fit["b1"], fit["n"], fit["nu"]
    x0 = (E_target - b0) / b1
    s = fit["s_res"]
    rows = []
    # Type A: calibration function (b0, b1 with covariance) at x0
    u_line = s * np.sqrt(1.0 / n + (x0 - fit["xbar"]) ** 2 / fit["Sxx"])
    rows.append(dict(source="Calibration function (b0, b1, incl. covariance)", type="A",
                     dist="t (normal)", value=f"s·√(1/n+(x0−x̄)²/Sxx)", u=u_line, c=1.0, nu=nu))
    rows.append(dict(source="Residual scatter of a single indication (s_res)", type="A",
                     dist="t (normal)", value=f"s = {s:.3f} lx", u=s, c=1.0, nu=nu))
    # Type B: sensor resolution (raw, BH1750FVI H-resolution mode, 1 lx; truncated)
    a_raw = 0.5
    rows.append(dict(source="BH1750FVI raw resolution, 1 lx (H-resolution mode; truncated)", type="B",
                     dist="rectangular", value="a = 0.5 lx", u=a_raw / np.sqrt(3), c=b1 * EQ9_SLOPE, nu=np.inf))
    # Type B: truncation of corrected output to integer lx
    rows.append(dict(source="Firmware truncation of corrected indication (int, 1 lx)", type="B",
                     dist="rectangular", value="a = 0.5 lx", u=0.5 / np.sqrt(3), c=b1, nu=np.inf))
    if include_ref:
        a_acc = rel_acc_ref * E_target
        rows.append(dict(source=f"ITR-3000 accuracy (±{rel_acc_ref*100:.0f} % of reading, manufacturer)",
                         type="B", dist="rectangular", value=f"a = {a_acc:.3f} lx",
                         u=a_acc / np.sqrt(3), c=1.0, nu=np.inf))
        res_ref = 0.01 if E_target < 20 else 0.1
        rows.append(dict(source=f"ITR-3000 display resolution ({res_ref} lx)", type="B",
                         dist="rectangular", value=f"a = {res_ref/2:.3f} lx",
                         u=(res_ref / 2) / np.sqrt(3), c=1.0, nu=np.inf))
    for r in rows:
        r["contrib"] = abs(r["c"]) * r["u"]
    uc = float(np.sqrt(sum(r["contrib"] ** 2 for r in rows)))
    nu_eff = welch_satterthwaite([(r["contrib"], r["nu"]) for r in rows])
    k95 = float(stats.t.ppf(0.97725, nu_eff)) if np.isfinite(nu_eff) else 2.0  # ~95.45 %
    U = K * uc
    for r in rows:
        r["share_pct"] = 100 * r["contrib"] ** 2 / uc ** 2
    return dict(E=E_target, x0=float(x0), rows=rows, uc=uc, nu_eff=float(nu_eff), k95=k95,
                U=float(U), U_rel_pct=float(100 * U / E_target), within_cobb_20pct=bool(U / E_target < 0.20))


def direct_errors(x, ref):
    x = np.asarray(x, float)
    ref = np.asarray(ref, float)
    e = x - ref                                     # VIM measurement error: indication - reference
    n = len(e)
    m, sd = e.mean(), e.std(ddof=1)
    t = m / (sd / np.sqrt(n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    ci = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    e_tc = e + 0.5                                  # after correcting the known truncation bias (+0.5 lx)
    t2 = e_tc.mean() / (e_tc.std(ddof=1) / np.sqrt(n))
    p2 = 2 * stats.t.sf(abs(t2), n - 1)
    rel = 100 * e / ref
    rel_tc = 100 * e_tc / ref
    ge5 = ref >= 5
    return dict(errors=e.tolist(), rel_errors_pct=rel.tolist(), mean=float(m), sd=float(sd),
                errors_after_trunc_corr=e_tc.tolist(), rel_errors_after_trunc_corr_pct=rel_tc.tolist(),
                sd_after_trunc_corr=float(e_tc.std(ddof=1)),
                n_within_20pct=int(np.sum(np.abs(rel) <= 20)),
                n_within_20pct_after_trunc_corr=int(np.sum(np.abs(rel_tc) <= 20)),
                max_abs_rel_ge5lx_pct=float(np.abs(rel[ge5]).max()),
                max_abs_rel_ge5lx_after_trunc_corr_pct=float(np.abs(rel_tc[ge5]).max()),
                mean_rel_ge5lx_after_trunc_corr_pct=float(np.mean(np.abs(rel_tc[ge5]))),
                points_outside_20pct=[(float(r_), float(v)) for r_, v in zip(ref, rel) if abs(v) > 20],
                ci95_mean=[float(m - ci), float(m + ci)], t=float(t), p=float(p),
                mean_after_trunc_corr=float(e_tc.mean()), p_after_trunc_corr=float(p2),
                max_abs=float(np.abs(e).max()),
                mean_rel_ge5lx_pct=float(np.mean(np.abs(100 * e / ref)[ref >= 5])))


def main():
    raw = ols(RAW_GY, RAW_REF)
    cor = ols(COR_GY, COR_REF)

    # ---- assertion check against the fixed numbers
    check = {}
    for name, fit in (("raw", raw), ("corrected", cor)):
        f = FIXED[name]
        ok = (round(fit["b1"], 4) == f["slope"] and round(fit["b0"], 4) == f["intercept"]
              and round(fit["R2"], 4) == f["R2"] and round(fit["RMSE"], 3) == f["RMSE"]
              and round(fit["MAE"], 3) == f["MAE"])
        check[name] = ok
        assert ok, f"Mismatch with fixed values for {name}"

    resid_rel = [100 * r / y for r, y in zip(cor["residuals"], COR_REF)]
    budgets = {
        "with_ITR_6pct": [budget(cor, E, ITR_REL_ACC) for E in SETPOINTS],
        "with_ITR_3pct_sensitivity": [budget(cor, E, ITR_REL_ACC_INCAND) for E in SETPOINTS],
        "without_ITR": [budget(cor, E, 0, include_ref=False) for E in SETPOINTS],
    }
    # U(E) curve over the validated range (Scenarios A and C) for the figures
    grid = np.round(np.arange(2.0, 43.01, 0.5), 2)
    curve = dict(E=grid.tolist(),
                 U_A=[budget(cor, E, ITR_REL_ACC)["U"] for E in grid],
                 U_C=[budget(cor, E, 0, include_ref=False)["U"] for E in grid])
    # crossover where the reference term starts to dominate (share > 50 %)
    share_ref = [budget(cor, E, ITR_REL_ACC)["rows"][4]["share_pct"] for E in grid]
    crossover = next((float(E) for E, sh in zip(grid, share_ref) if sh > 50), None)
    out = dict(check_against_fixed=check, stage1_raw=raw, stage2_corrected=cor,
               U_curve=curve, reference_dominates_above_lx=crossover,
               raw_sensitivity_ratio=1.0 / raw["b1"],
               stage2_residual_relative_pct=resid_rel,
               stage2_direct_error=direct_errors(COR_GY, COR_REF), budgets=budgets)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gum_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)

    # ---- console report
    p = print
    p("Assertion check vs fixed values:", check)
    for name, f in (("Stage 1 (raw, n=26)", raw), ("Stage 2 (corrected, n=22)", cor)):
        p(f"\n{name}: b1={f['b1']:.4f} u(b1)={f['u_b1']:.4f}  b0={f['b0']:.4f} u(b0)={f['u_b0']:.4f} "
          f"r(b0,b1)={f['r_b0b1']:.3f}  s_res={f['s_res']:.3f} (nu={f['nu']})  RMSE={f['RMSE']:.3f} "
          f"MAE={f['MAE']:.3f} R2={f['R2']:.4f}")
        p(f"   CI95 b1={np.round(f['CI_b1'],4)} b0={np.round(f['CI_b0'],3)}  xbar={f['xbar']:.3f} Sxx={f['Sxx']:.2f}"
          f"  joint H0(b0=0,b1=1): F={f['joint_identity_F']:.2f}, p={f['joint_identity_p']:.2e}")
    de = out["stage2_direct_error"]
    p(f"\nDirect error (indication - reference), n=22: mean={de['mean']:.3f} lx, SD={de['sd']:.3f}, "
      f"CI95 mean={np.round(de['ci95_mean'],3)}, p={de['p']:.2e}; max|e|={de['max_abs']:.2f} lx; "
      f"mean|e_rel| (E>=5 lx)={de['mean_rel_ge5lx_pct']:.1f} %")
    p(f"   after +0.5 lx truncation correction: mean={de['mean_after_trunc_corr']:.3f} lx, p={de['p_after_trunc_corr']:.2f}")
    p("   residual relative at 1.78 lx = %.1f %% ; direct error at 1.78 lx = %.1f %%"
      % (resid_rel[-1], de["rel_errors_pct"][-1]))
    for key, bl in budgets.items():
        p(f"\n=== Budget scenario: {key}")
        for b in bl:
            p(f" E = {b['E']:.0f} lx (x0 = {b['x0']:.2f}):")
            for r in b["rows"]:
                p(f"   {r['source'][:62]:62s} {r['type']} {r['dist'][:11]:11s} u={r['u']:.4f} c={r['c']:.4f} "
                  f"contr={r['contrib']:.4f} ({r['share_pct']:.1f} %)")
            p(f"   u_c={b['uc']:.3f} lx  nu_eff={b['nu_eff']:.0f}  k95={b['k95']:.2f}  "
              f"U(k=2)={b['U']:.2f} lx = {b['U_rel_pct']:.1f} %  (<20 %: {b['within_cobb_20pct']})")


if __name__ == "__main__":
    main()
