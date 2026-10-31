"""
Put the raw DFs in dimensionless, or reference-halo, units.

Only the ordinate is scaled, analytically (core.df_scaling): f_hat = f G^{3/2} L^3 rho^{1/2}.
Nothing is recomputed from particles. The abscissa stays eps_hat = E/Psi(0) for every halo.
Scales, with --scale-by:

  rs      (r_s, rho_s) of the NFW fit, or (r_-2, rho_-2) of Einasto (--shape)
  virial  (R_vir, rho(R_vir)): R_vir from the group catalogue, so this needs the snapshots

With --ref-halo the curves are also written in that halo's units.

Writes, under Halo_Results/Halo_XX/df/scaled/:
  df_dimensionless_<method>_<scale>.csv, df_ref<NN>_<method>_<scale>.csv,
  df_scaled_<method>_<scale>_meta.json
and overlay PNGs in Halo_Results/_overlays/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..core.density_fits import einasto_shape, nfw_shape
from ..core.df_scaling import df_to_reference_units, dimensionless_df, energy_scale
from ..io.auriga_root import resolve_auriga_data_root
from ..io.files import groupcat_files, snapshot_files
from ..io.header import read_header
from ..io.layout import resolve_data_dir_for_snap, resolve_halo_dir
from ..io.results_paths import density_dir, df_dir, df_raw_dir, overlays_dir
from ..io.virial import load_Rvir_kpc


def _load_curve(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.genfromtxt(p, delimiter=",", names=True)
    return np.asarray(d["eps_hat"], float), np.asarray(d["log10f"], float)


def _load_density_profile(hid: int, method: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.genfromtxt(density_dir(hid) / f"density_{method}.csv", delimiter=",", names=True)
    r = np.asarray(d["r_kpc"], float)
    rho = np.asarray(d["rho_msun_kpc3"], float)
    ok = np.isfinite(r) & (r > 0) & np.isfinite(rho) & (rho > 0)
    return r[ok], rho[ok]


def _rho_at(r_kpc: float, r: np.ndarray, rho: np.ndarray) -> float:
    """Measured density at r_kpc by log-log interpolation of the binned profile."""
    r_kpc = float(r_kpc)
    if not (r[0] <= r_kpc <= r[-1]):
        raise ValueError(
            f"r={r_kpc:.1f} kpc outside the measured profile [{r[0]:.2f}, {r[-1]:.1f}] kpc"
        )
    return float(10.0 ** np.interp(np.log10(r_kpc), np.log10(r), np.log10(rho)))


def _fit_rho_at(r_kpc: float, fit: dict, shape: str) -> float:
    """Same density from the fitted profile, for cross-checking the measured value."""
    if shape == "nfw":
        return float(fit["nfw"]["rhos"] * nfw_shape(np.array([r_kpc]), fit["nfw"]["rs"])[0])
    e = fit["einasto"]
    return float(e["rho2"] * einasto_shape(np.array([r_kpc]), e["r2"], e["alpha"])[0])


def _scales_rs(hid: int, method: str, shape: str) -> tuple[float, float, str, dict]:
    """(length, density, src, extra) from the stored density fit."""
    p = density_dir(hid) / f"density_fit_params_{method}.json"
    d = json.loads(p.read_text())
    if shape == "nfw":
        return float(d["nfw"]["rs"]), float(d["nfw"]["rhos"]), f"{p.name}:nfw", {}
    return float(d["einasto"]["r2"]), float(d["einasto"]["rho2"]), f"{p.name}:einasto", {}


def _scales_virial(hid: int, method: str, shape: str, snap: int, data_root: Path) -> tuple[float, float, str, dict]:
    """(R_vir, rho(R_vir), src, extra) — R_vir from the group catalog, density from the profile."""
    hdir = resolve_halo_dir(Path(data_root), hid)
    data_dir = resolve_data_dir_for_snap(hdir, int(snap))
    hdr = read_header(snapshot_files(data_dir, int(snap))[0])
    Rvir_kpc, rvir_src = load_Rvir_kpc(gfiles=groupcat_files(data_dir, int(snap)), a=float(hdr.a), h=float(hdr.h))

    r, rho = _load_density_profile(hid, method)
    rho_vir = _rho_at(Rvir_kpc, r, rho)
    fit = json.loads((density_dir(hid) / f"density_fit_params_{method}.json").read_text())
    extra = {
        "Rvir_src": str(rvir_src),
        "rho_at_Rvir_measured": float(rho_vir),
        "rho_at_Rvir_fit": float(_fit_rho_at(Rvir_kpc, fit, shape)),
        "a": float(hdr.a),
        "h": float(hdr.h),
    }
    return float(Rvir_kpc), float(rho_vir), f"groupcat:{rvir_src}+rho(Rvir)", extra


def _collapse_stats(
    curves: list[tuple[int, np.ndarray, np.ndarray]], n: int = 400
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (grid, median, residuals) for a bundle of curves on their common abscissa range.

    Each curve is first trimmed to where log10 f is finite — the Eddington inversion returns
    a non-positive f in the last few bins below eps_hat = 1 for about half the halos — and
    then interpolated onto a linear grid spanning the range every halo covers. The median is
    therefore taken over the same halos everywhere, and the residuals below it are the
    halo-to-halo scatter of the collapse, not an artefact of curves dropping in and out.
    """
    finite = []
    for hid, x, y in curves:
        m = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(m) < 2:
            raise RuntimeError(f"halo {hid}: fewer than 2 finite points to collapse")
        finite.append((hid, np.asarray(x, float)[m], np.asarray(y, float)[m]))
    lo = max(float(x[0]) for _, x, _ in finite)
    hi = min(float(x[-1]) for _, x, _ in finite)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise RuntimeError(f"curves share no common abscissa range: [{lo}, {hi}]")
    grid = np.linspace(lo, hi, int(n))
    stack = np.array([np.interp(grid, x, y) for _, x, y in finite])
    med = np.median(stack, axis=0)
    return grid, med, stack - med


def _collapse_figure(png, curves, *, title: str, ylabel: str, xlabel: str,
                     xlim: tuple[float, float] | None = None) -> float:
    """Curves + median on top, residual-to-median below. Returns the rms residual [dex]."""
    grid, med, resid = _collapse_stats(curves)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.0, 6.4), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0]}
    )
    for _, e, f in curves:
        m = np.isfinite(e) & np.isfinite(f)
        ax.plot(e[m], f[m], lw=1.0, alpha=0.75)
    ax.plot(grid, med, color="k", lw=1.4, label="median")
    for r in resid:
        axr.plot(grid, r, lw=0.8, alpha=0.75)
    axr.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    axr.set_ylabel("residual [dex]")
    axr.set_xlabel(xlabel)
    for a in (ax, axr):
        a.grid(alpha=0.25)
        if xlim is not None:
            a.set_xlim(*xlim)
    # hspace must be applied after tight_layout, not via gridspec_kw: passing it there
    # makes the axes incompatible with tight_layout ("results might be incorrect").
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.savefig(png, dpi=180)
    plt.close(fig)
    return float(np.sqrt(np.mean(resid**2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halos", type=int, nargs="+", required=True)
    ap.add_argument("--ref-halo", type=int, default=None, help="also write curves in this halo's units")
    ap.add_argument("--method", type=str, default="with_hubble_flow",
                    choices=["with_hubble_flow", "without_hubble_flow"])
    ap.add_argument("--scale-by", type=str, default="rs", choices=["rs", "virial"],
                    help="'rs': (r_s, rho_s) from the density fit. 'virial': (R_vir, rho(R_vir)), "
                         "R_vir read fresh from the group catalog (needs snapshot data).")
    ap.add_argument("--shape", type=str, default="nfw", choices=["nfw", "einasto"],
                    help="which fit supplies (r_s, rho_s) for --scale-by rs, and the fitted "
                         "rho(R_vir) cross-check for --scale-by virial")
    ap.add_argument("--snap", type=int, default=127)
    # The stored f(eps_hat) is built on a g(E) that carries the analytic Kepler tail,
    # which diverges as E^(-5/2) toward eps_hat -> 0. Scale only the reliable window.
    ap.add_argument("--eps-lo", type=float, default=0.03, help="Lower edge of the eps_hat window")
    ap.add_argument("--eps-hi", type=float, default=0.999, help="Upper edge of the eps_hat window")
    ap.add_argument("--raw-overlay", action="store_true",
                    help="also write the UNSCALED df_raw curves overlaid on the same window, as a "
                         "control: the collapse figure minus the (r_s, rho_s) factor")
    ap.add_argument("--ref-abscissa", type=str, default="eps_hat", choices=["eps_hat", "eps_hat_ref"],
                    help="abscissa of the --ref-halo overlay: each halo's own E/Psi_i(0) (default, "
                         "no per-halo stretch) or the same energies on the reference halo's Psi(0)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    halos = [int(h) for h in args.halos]
    method = str(args.method)
    shape = str(args.shape)
    scale_by = str(args.scale_by)
    ref_hid = None if args.ref_halo is None else int(args.ref_halo)
    ref_x = str(args.ref_abscissa)
    tag = shape if scale_by == "rs" else "virial"
    # Plot labels for the (length, density) pair this run scales by. They must move
    # together: R_vir pairs with rho_vir = rho(R_vir), never with rho_s.
    lab = r"r_s" if scale_by == "rs" else r"R_{\rm vir}"
    den = r"\rho_s" if scale_by == "rs" else r"\rho_{\rm vir}"
    eps_lo = float(args.eps_lo)
    eps_hi = float(args.eps_hi)
    if not (0.0 < eps_lo < eps_hi < 1.0):
        raise ValueError(f"Need 0 < eps_lo < eps_hi < 1, got [{eps_lo}, {eps_hi}]")

    data_root = Path(resolve_auriga_data_root()) if scale_by == "virial" else None

    def scales_for(hid: int) -> tuple[float, float, str, dict]:
        if scale_by == "rs":
            return _scales_rs(hid, method, shape)
        return _scales_virial(hid, method, shape, int(args.snap), data_root)

    ref_scales = None
    if ref_hid is not None:
        L_r, rho_r, _, _ = scales_for(ref_hid)
        meta_r = json.loads((df_raw_dir(ref_hid) / f"df_raw_{method}_meta.json").read_text())
        ref_scales = (L_r, rho_r, float(meta_r["Psi0"]))

    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    ref_curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    raw_curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for hid in halos:
        out_dir = df_dir(hid) / "scaled"
        out_dir.mkdir(parents=True, exist_ok=True)
        dimless_csv = out_dir / f"df_dimensionless_{method}_{tag}.csv"
        meta_path = out_dir / f"df_scaled_{method}_{tag}_meta.json"
        ref_csv = None if ref_hid is None else out_dir / f"df_ref{ref_hid:02d}_{method}_{tag}.csv"

        raw_csv = df_raw_dir(hid) / f"df_raw_{method}.csv"
        raw_meta_path = df_raw_dir(hid) / f"df_raw_{method}_meta.json"
        if not (raw_csv.exists() and raw_meta_path.exists()):
            print(f"[halo {hid}] skip: missing {raw_csv.name}")
            continue
        cached = dimless_csv.exists() and meta_path.exists() and (ref_csv is None or ref_csv.exists())
        if cached and not bool(args.force):
            # Stored CSVs may have been written with a wider window than the one requested
            # now (metas predating --eps-lo carry no window at all), so re-apply it here:
            # the figures must show the window the caller asked for, not the one on disk.
            d = np.genfromtxt(dimless_csv, delimiter=",", names=True)
            e = np.asarray(d["eps_hat"], float)
            keep = (e >= eps_lo) & (e <= eps_hi)
            if np.count_nonzero(keep) < 50:
                raise RuntimeError(f"[halo {hid}] only {np.count_nonzero(keep)} stored points in "
                                   f"[{eps_lo}, {eps_hi}]; rebuild with --force")
            curves.append((hid, e[keep], np.asarray(d["log10f_hat"], float)[keep]))
            if bool(args.raw_overlay):
                re_, rl = _load_curve(df_raw_dir(hid) / f"df_raw_{method}.csv")
                rkeep = (re_ >= eps_lo) & (re_ <= eps_hi)
                raw_curves.append((hid, re_[rkeep], rl[rkeep]))
            if ref_csv is not None:
                r = np.genfromtxt(ref_csv, delimiter=",", names=True)
                # window on eps_hat (the halo's own), whichever abscissa is being plotted
                rk = (np.asarray(r["eps_hat"], float) >= eps_lo) & (np.asarray(r["eps_hat"], float) <= eps_hi)
                ref_curves.append((hid, np.asarray(r[ref_x], float)[rk], np.asarray(r["log10f_ref"], float)[rk]))
            print(f"[halo {hid}] skip (exists; windowed to [{eps_lo}, {eps_hi}])")
            continue

        eps_hat, log10f = _load_curve(raw_csv)
        raw_meta = json.loads(raw_meta_path.read_text())
        if not bool(raw_meta.get("g_kepler_tail", False)):
            raise RuntimeError(
                f"[halo {hid}] {raw_csv.name} was built without the Kepler tail in g(E) "
                f"(meta has g_kepler_tail={raw_meta.get('g_kepler_tail')!r}). Rebuild it with "
                f"pipelines/build_df_raw.py --method {method} --force."
            )
        keep = (eps_hat >= eps_lo) & (eps_hat <= eps_hi)
        if np.count_nonzero(keep) < 50:
            raise RuntimeError(f"[halo {hid}] only {np.count_nonzero(keep)} points in [{eps_lo}, {eps_hi}]")
        eps_hat, log10f = eps_hat[keep], log10f[keep]
        Psi0 = float(raw_meta["Psi0"])
        L, rho_scale, src, extra = scales_for(hid)

        log10f_h = dimensionless_df(log10f, rs_kpc=L, rhos_msun_kpc3=rho_scale)
        # The abscissa is eps_hat = E/Psi_i(0) throughout; only the ordinate is scaled.
        with dimless_csv.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["eps_hat", "log10f_hat"])
            for a0, b in zip(eps_hat.tolist(), log10f_h.tolist()):
                w.writerow([float(a0), float(b) if np.isfinite(b) else "nan"])
        curves.append((hid, eps_hat, log10f_h))
        if bool(args.raw_overlay):
            raw_curves.append((hid, eps_hat, log10f))

        meta = {
            "halo": hid,
            "method": method,
            "scale_by": scale_by,
            "eps_lo": eps_lo,
            "eps_hi": eps_hi,
            "scale_src": src,
            "length_scale_kpc": float(L),
            "density_scale_msun_kpc3": float(rho_scale),
            "Psi0": Psi0,
            "E_s_kms2": float(energy_scale(L, rho_scale)),
            "Psi0_over_E_s": float(Psi0 / energy_scale(L, rho_scale)),
            "from": raw_csv.name,
            "particle_rescaling": False,
            **extra,
        }

        if ref_scales is not None:
            L_r, rho_r, Psi0_r = ref_scales
            eps_r, log10f_r = df_to_reference_units(
                eps_hat, log10f, Psi0=Psi0, rs_kpc=L, rhos_msun_kpc3=rho_scale,
                ref_rs_kpc=L_r, ref_rhos_msun_kpc3=rho_r, ref_Psi0=Psi0_r,
            )
            with ref_csv.open("w", newline="") as fp:
                w = csv.writer(fp)
                # eps_hat: this halo's own E/Psi_i(0), unchanged (no stretch)
                # eps_hat_ref: the same energies expressed on the reference halo's Psi(0)
                w.writerow(["eps_hat", "eps_hat_ref", "log10f_ref"])
                for a0, a, b in zip(eps_hat.tolist(), eps_r.tolist(), log10f_r.tolist()):
                    w.writerow([float(a0), float(a), float(b) if np.isfinite(b) else "nan"])
            ref_curves.append((hid, eps_hat if ref_x == "eps_hat" else eps_r, log10f_r))
            meta.update({"ref_halo": ref_hid, "ref_length_scale_kpc": L_r,
                         "ref_density_scale_msun_kpc3": rho_r, "ref_Psi0": Psi0_r})

        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        print(f"[halo {hid}] wrote {dimless_csv.name}  "
              f"(L={L:.2f} kpc, rho={rho_scale:.3e}, Psi0/E_s={Psi0/energy_scale(L, rho_scale):.2f})")

    if curves:
        ov = overlays_dir()
        ov.mkdir(parents=True, exist_ok=True)
        png = ov / f"df_dimensionless_overlay_{method}_{tag}.png"
        rms = _collapse_figure(
            png, curves,
            title=f"Dimensionless DFs ({method}, scale={tag}), {len(curves)} halos",
            ylabel=r"$\log_{10}\hat f$",
            xlabel=r"$\hat\varepsilon = E/\Psi(0)$",
            xlim=(0.0, 1.0),
        )
        print(f"wrote {png}  (halo-to-halo scatter about the median: {rms:.3f} dex)")

    if raw_curves:
        ov = overlays_dir()
        ov.mkdir(parents=True, exist_ok=True)
        png = ov / f"df_raw_overlay_{method}.png"
        rms = _collapse_figure(
            png, raw_curves,
            title=(rf"UNSCALED $f(E)$ — no $({lab},{den})$ factor"
                   f"  ({method}), {len(raw_curves)} halos"),
            ylabel=r"$\log_{10} f$  (each halo's own units)",
            xlabel=r"$\hat\varepsilon = E/\Psi_i(0)$",
        )
        print(f"wrote {png}  (halo-to-halo scatter about the median: {rms:.3f} dex)")

    if ref_curves and ref_hid is not None:
        ov = overlays_dir()
        ov.mkdir(parents=True, exist_ok=True)
        png = ov / f"df_ref{ref_hid:02d}_overlay_{method}_{tag}_{ref_x}.png"
        grid, med, resid = _collapse_stats(ref_curves)
        rms = float(np.sqrt(np.mean(resid**2)))

        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(7.0, 6.4), sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.0]},
        )
        for _, e, f in ref_curves:
            m = np.isfinite(e) & np.isfinite(f)
            ax.plot(e[m], f[m], lw=1.0, alpha=0.75)
        ax.plot(grid, med, color="k", lw=1.4, label="median")
        for r in resid:
            axr.plot(grid, r, lw=0.8, alpha=0.75)
        axr.axhline(0.0, color="k", lw=0.8)

        xlab = (r"$\hat\varepsilon = E/\Psi_i(0)$" if ref_x == "eps_hat"
                else rf"$\hat\varepsilon_{{{ref_hid}}} = E/\Psi_{{{ref_hid}}}(0)$")
        ax.set_ylabel(rf"$\log_{{10}} f_{{i\to{ref_hid}}}$  (halo-{ref_hid} units)")
        ax.set_title(
            rf"$f_{{i\to{ref_hid}}} = f_i\,({lab}^{{(i)}}/{lab}^{{({ref_hid})}})^3\,"
            rf"({den}^{{(i)}}/{den}^{{({ref_hid})}})^{{1/2}}$"
            f"  —  {method}, scale={tag}, {len(ref_curves)} halos"
        )
        ax.legend(frameon=False, loc="upper left")
        axr.set_ylabel("residual [dex]")
        axr.set_xlabel(xlab)
        for a in (ax, axr):
            a.grid(alpha=0.25)
        # hspace must be applied after tight_layout, not via gridspec_kw: passing it there
        # makes the axes incompatible with tight_layout ("results might be incorrect").
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.06)
        fig.savefig(png, dpi=180)
        plt.close(fig)
        print(f"wrote {png}  (halo-to-halo scatter about the median: {rms:.3f} dex)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
