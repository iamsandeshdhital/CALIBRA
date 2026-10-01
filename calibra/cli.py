"""Command-line entry point for CALIBRA.

Subcommands
-----------
``demo``      Fit and evaluate on the built-in high-tech line simulator
``step``      Print one online decision point, including the evidence card
``experiments``  Run the experiment suite (see experiments/run_experiments.py)
``coverage``  Marginal vs conditional coverage, and the Mondrian cell audit

Examples::

    python -m calibra demo --assets 34 --seed 7
    python -m calibra demo --matrix bearing --json
    python -m calibra step --seed 7 --index 1200
    python -m calibra coverage --seed 7
    python -m calibra experiments --seeds 7 11 13
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .pipeline import CalibraConfig, CalibraRunner, format_report
from .sim.hightech_line import HighTechLine, LineConfig


def _runner(args) -> CalibraRunner:
    runs = HighTechLine(LineConfig(seed=args.seed, n_assets=args.assets)).generate()
    cfg = CalibraConfig(
        seed=args.seed,
        cost_matrix=args.matrix,
        conformal_mode=args.conformal,
        conformal_alpha=args.alpha,
        response_rate=args.response_rate,
    )
    return CalibraRunner(cfg).fit(runs)


def cmd_demo(args) -> int:
    runner = _runner(args)
    rep = runner.evaluate("test", response_rate=args.response_rate)
    if args.json:
        print(json.dumps(rep, indent=2, default=float))
    else:
        print(format_report(rep))
    return 0


def cmd_step(args) -> int:
    runner = _runner(args)
    X = runner._data["Xte"]
    i = min(max(args.index, 0), len(X) - 1)
    out = runner.step(X[i])
    print(json.dumps(out, indent=2, default=float))
    return 0


def cmd_coverage(args) -> int:
    """Marginal vs conditional coverage, plus which Mondrian cells are real."""
    from .data.features import group_healthy
    from .metrics.score import coverage, interval_width

    runner = _runner(args)
    X, y, truth = runner._data["Xte"], runner._data["yte"], runner._data["rte"]
    mu, _, _, sig, lo, hi = runner._predict_distributions(X, group_healthy(truth))
    nom = 1.0 - args.alpha

    print(f"nominal coverage        : {nom:.2f}")
    print(f"marginal coverage       : {coverage(y, lo, hi):.4f}")
    print(f"mean interval width     : {interval_width(lo, hi):.2f}")
    print()
    print("conditional coverage by ground-truth regime:")
    for g in ("healthy", "degraded", "critical"):
        m = truth == g
        if m.sum():
            c = coverage(y[m], lo[m], hi[m])
            print(f"  {g:<10s} {c:.4f}   (n={int(m.sum())})")

    conf = runner.conformal_
    audit = conf.audit_groups(group_healthy(runner._data["rva"]))
    print()
    print(f"Mondrian cell audit (min size required: {audit['min_group_size_required']}):")
    for g, row in audit["groups"].items():
        flag = "OK" if row["certifiable"] else "FELL BACK TO GLOBAL"
        print(f"  {g:<12s} n={row['n']:>5d}  radius={row['radius']:>7.3f}  {flag}")
    if not audit["all_certifiable"]:
        print(
            "  -> split only as finely as you can certify; a cell that falls\n"
            "     back silently is worse than no conditioning at all."
        )
    return 0


def cmd_experiments(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.run_experiments import EXPERIMENTS

    out = {}
    for k in args.only or list(EXPERIMENTS):
        fn = EXPERIMENTS[k]
        r = fn(seeds=tuple(args.seeds)) if k == "E7" else fn(seed=args.seeds[0])
        out[k] = r
        print(f"[{k}] {r['study']}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(out, indent=2, default=lambda o: np.asarray(o).tolist())
        )
        print(f"wrote {args.json_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="calibra", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--assets", type=int, default=34)
        sp.add_argument("--matrix", choices=("fab", "bearing"), default="fab")
        sp.add_argument("--conformal", choices=("global", "mondrian"), default="mondrian")
        sp.add_argument("--alpha", type=float, default=0.10)
        sp.add_argument("--response-rate", type=float, default=0.85)
        return sp

    d = common(sub.add_parser("demo", help="fit and evaluate end to end"))
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_demo)

    s = common(sub.add_parser("step", help="one online decision point"))
    s.add_argument("--index", type=int, default=0)
    s.set_defaults(func=cmd_step)

    c = common(sub.add_parser("coverage", help="marginal vs conditional coverage"))
    c.set_defaults(func=cmd_coverage)

    e = sub.add_parser("experiments", help="run the experiment suite")
    e.add_argument("--only", nargs="*", default=None)
    e.add_argument("--seeds", type=int, nargs="*", default=[7])
    e.add_argument("--json-out", type=Path, default=None)
    e.set_defaults(func=cmd_experiments)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())