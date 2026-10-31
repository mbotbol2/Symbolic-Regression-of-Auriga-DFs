"""
Fit one cross-validation fold with PySR, and optionally score it on the fold's held-out halos.

Target: the fold's training median in the reference frame, log10 f ~ g(u), u = transform(eps_hat).
The output is the full Pareto front, <cv>/fold<j>/pysr/<transform>/equations.csv. With --score,
each held-out curve is moved to the reference frame with its folds.json offset and compared,
on the same scale as the median baseline in cv_summary.csv.

Search settings as in the paper: unary log, exp, sqrt, asin, sinh (each may nest any other
once, never itself), binary +, -, *, / (--binary-operators), maxsize 40, 36 populations,
2000 iterations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..core.df_transforms import TRANSFORM_NAMES, apply as apply_transform

UNARY = ["log", "exp", "sqrt", "asin", "sinh"]


def nested_constraints(unary: list[str]) -> dict:
    """Any operator may nest any other at most once; never itself."""
    return {u: {v: (0 if v == u else 1) for v in unary} for u in unary}


def _read(path: Path, cols: tuple[str, ...]) -> dict:
    d = np.genfromtxt(path, delimiter=",", names=True)
    names = set(d.dtype.names or ())
    for c in cols:
        if c not in names:
            raise RuntimeError(f"{path}: missing column {c!r} (found {sorted(names)})")
    return {c: np.asarray(d[c], float) for c in cols}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=str, required=True, help="the cv_<...>_ref<NN> directory")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--transform", type=str, required=True, choices=TRANSFORM_NAMES)
    ap.add_argument("--niterations", type=int, default=2000)
    ap.add_argument("--populations", type=int, default=36)
    ap.add_argument("--maxsize", type=int, default=40)
    ap.add_argument("--binary-operators", type=str, nargs="+", default=["+", "-", "*", "/"])
    ap.add_argument("--unary-operators", type=str, nargs="+", default=UNARY)
    ap.add_argument("--procs", type=int, default=0, help="0 lets PySR choose")
    ap.add_argument("--seed", type=int, default=None,
                    help="only meaningful with --deterministic")
    ap.add_argument("--deterministic", action="store_true",
                    help="reproducible search; forces serial evaluation, so much slower")
    ap.add_argument("--timeout-s", type=int, default=None)
    ap.add_argument("--score", action="store_true",
                    help="also score the best expression on the held-out halos. Off by "
                         "default: the Pareto front is the product, and evaluation is a "
                         "separate step over all runs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the resolved config and exit without importing PySR")
    args = ap.parse_args()

    cv = Path(args.cv_dir).expanduser().resolve()
    fold_dir = cv / f"fold{int(args.fold)}"
    name = str(args.transform)
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"no such fold directory: {fold_dir}")

    manifest = json.loads((cv / "folds.json").read_text())
    offsets = manifest["to_ref_offset_dex"]
    fold_entry = next(f for f in manifest["folds"] if int(f["fold"]) == int(args.fold))
    test_halos = [int(h) for h in fold_entry["test_halos"]]

    train_csv = fold_dir / "transforms" / f"median_{name}.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"{train_csv} not found; run build_cv_folds first")
    tr = _read(train_csv, ("transformed_var", "log10f"))
    X = tr["transformed_var"].reshape(-1, 1)
    y = tr["log10f"]

    out = fold_dir / "pysr" / name
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "cv_dir": str(cv), "fold": int(args.fold), "transform": name,
        "n_train_points": int(X.size),
        "niterations": int(args.niterations), "populations": int(args.populations),
        "maxsize": int(args.maxsize),
        "binary_operators": list(args.binary_operators),
        "unary_operators": list(args.unary_operators),
        "nested_constraints": nested_constraints(list(args.unary_operators)),
        "seed": args.seed, "deterministic": bool(args.deterministic),
        "test_halos": test_halos,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if args.dry_run:
        print(f"[fold {args.fold}/{name}] dry run: config -> {out/'config.json'}, "
              f"{X.size} training points, test halos {test_halos}")
        return 0

    from pysr import PySRRegressor  # imported late: pulls in Julia

    model = PySRRegressor(
        niterations=int(args.niterations),
        populations=int(args.populations),
        maxsize=int(args.maxsize),
        binary_operators=list(args.binary_operators),
        unary_operators=list(args.unary_operators),
        nested_constraints=nested_constraints(list(args.unary_operators)),
        progress=False,
        output_directory=str(out),
        # PySR warns that a seed without serial evaluation does not make a search
        # reproducible, so the seed is only passed when determinism is actually asked for.
        **({"deterministic": True, "parallelism": "serial", "random_state": int(args.seed or 0)}
           if args.deterministic else {}),
        **({"procs": int(args.procs)} if int(args.procs) > 0 else {}),
        **({"timeout_in_seconds": int(args.timeout_s)} if args.timeout_s else {}),
    )
    model.fit(X, y, variable_names=["u"])

    # The whole Pareto front is the product of a run, not just the winner: the point of
    # running many searches is to see which families of expressions recur across folds
    # and transforms. equations.csv keeps every complexity level PySR retained.
    eq = model.equations_
    eq.to_csv(out / "equations.csv", index=False)
    print(f"[fold {args.fold}/{name}] front: {len(eq)} expressions, "
          f"complexity {int(eq['complexity'].min())}-{int(eq['complexity'].max())}")

    rows = []
    if args.score:
        med = _read(fold_dir / "train_median.csv", ("eps_hat", "median"))
        for h in test_halos:
            t = _read(fold_dir / "test" / f"test_halo{h:02d}.csv", ("eps_hat", "log10f"))
            u = apply_transform(name, t["eps_hat"])
            ok = np.isfinite(u)
            y_true = t["log10f"][ok] + float(offsets[str(h)])   # raw -> reference frame
            y_pred = model.predict(u[ok].reshape(-1, 1))
            resid = y_pred - y_true
            base = med["median"][ok] - y_true
            rows.append({
                "fold": int(args.fold), "transform": name, "halo": h,
                "n_points": int(np.count_nonzero(ok)),
                "mse_model": float(np.mean(resid**2)),
                "rmse_model": float(np.sqrt(np.mean(resid**2))),
                "max_abs_resid_model": float(np.max(np.abs(resid))),
                "mse_vs_train_median": float(np.mean(base**2)),
            })
            print(f"  halo {h:02d}: model MSE {rows[-1]['mse_model']:.5f}  "
                  f"median baseline {rows[-1]['mse_vs_train_median']:.5f}")

    best = model.get_best()
    result = {
        **cfg,
        "best_equation": str(best["equation"]),
        "best_complexity": int(best["complexity"]),
        "best_train_loss": float(best["loss"]),
        "n_front": int(len(eq)),
        "scored": bool(args.score),
        "scores": rows,
        **({"mse_model_mean": float(np.mean([r["mse_model"] for r in rows])),
            "mse_vs_train_median_mean": float(np.mean([r["mse_vs_train_median"] for r in rows]))}
           if rows else {}),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[fold {args.fold}/{name}] best (complexity {result['best_complexity']}): "
          f"{result['best_equation']}")
    if rows:
        print(f"[fold {args.fold}/{name}] mean model MSE {result['mse_model_mean']:.5f} "
              f"vs median baseline {result['mse_vs_train_median_mean']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
