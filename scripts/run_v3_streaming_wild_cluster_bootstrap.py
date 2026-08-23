#!/usr/bin/env python3
"""Memory-safe 9,999-draw WCR11 bootstrap for frozen v3 H1/H2 inference.

The implementation follows the cluster-score algebra used by wildboottest for
WCR11, but processes Rademacher draws in checkpointed batches. Fixed effects
are removed once by alternating projections (Frisch--Waugh--Lovell).
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "robustness_results/v3/inference/inference_config.json"
OUT = ROOT / "robustness_results/v3/inference/wild_cluster_streaming"
PROGRESS = OUT / "progress.json"
DRAWS = OUT / "bootstrap_t_draws.npz"
REPS = 9_999
BATCH = 250
SEED = 20260807

Y = "delta_yes_price"
TOTAL = "signed_log_total_net_flow"
LARGE = "p99_signed_log_large_net_flow"
ORDINARY = "p99_signed_log_ordinary_net_flow"
CONTROLS = [
    "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume",
    "log_minutes_to_kickoff",
]
NUMERIC = [Y, TOTAL, LARGE, ORDINARY, *CONTROLS]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def absorb(values, groups, tolerance=1e-9, max_iterations=20_000):
    residual = np.asarray(values, dtype=np.float64).copy()
    codes = [pd.factorize(group, sort=False)[0] for group in groups]
    for iteration in range(1, max_iterations + 1):
        maximum_change = 0.0
        for code in codes:
            counts = np.bincount(code).astype(np.float64)
            for column in range(residual.shape[1]):
                means = np.bincount(code, weights=residual[:, column]) / counts
                residual[:, column] -= means[code]
                maximum_change = max(maximum_change, float(np.max(np.abs(means))))
        if iteration == 1 or iteration % 100 == 0:
            print(f"  absorption iteration {iteration:,}; max group mean {maximum_change:.3e}", flush=True)
        if maximum_change < tolerance:
            return residual, iteration, maximum_change
    raise RuntimeError("Fixed-effect absorption failed to converge")


def prepare_model(name, residualized, positions, cluster_codes, cluster_count):
    columns = positions[name]
    y = residualized[:, 0]
    x = residualized[:, columns]
    n, k = x.shape
    xxg = np.zeros((cluster_count, k, k), dtype=np.float64)
    xyg = np.zeros((cluster_count, k), dtype=np.float64)
    for group in range(cluster_count):
        mask = cluster_codes == group
        xg = x[mask]
        yg = y[mask]
        xxg[group] = xg.T @ xg
        xyg[group] = xg.T @ yg
    xx = xxg.sum(axis=0)
    xy = xyg.sum(axis=0)
    xx_inv = np.linalg.inv(xx)
    beta = xx_inv @ xy
    unrestricted_scores = xyg - np.einsum("gij,j->gi", xxg, beta)
    meat = unrestricted_scores.T @ unrestricted_scores
    ssc = ((n - 1) / (n - k)) * (cluster_count / (cluster_count - 1))
    vcov = ssc * xx_inv @ meat @ xx_inv
    return {
        "name": name, "x": x, "y": y, "n": n, "k": k, "xxg": xxg,
        "xyg": xyg, "xx_inv": xx_inv, "beta": beta, "vcov": vcov,
        "ssc": ssc,
    }


def prepare_test(model, label, restriction):
    r = np.asarray(restriction, dtype=np.float64)
    inv = model["xx_inv"]
    beta = model["beta"]
    denominator = float(r @ inv @ r)
    beta_restricted = beta - (inv @ r) * (float(r @ beta) / denominator)
    scores = model["xyg"] - np.einsum("gij,j->gi", model["xxg"], beta_restricted)
    scores = scores.T  # k x G
    cg = r @ inv @ scores  # G
    # H[g,h] = R'(X'X)^-1 Xg'Xg (X'X)^-1 score_h
    left = np.einsum("i,gij->gj", r @ inv, model["xxg"]) @ inv
    h = left @ scores
    observed_se = float(np.sqrt(r @ model["vcov"] @ r))
    observed_estimate = float(r @ beta)
    return {
        "label": label, "model": model["name"], "restriction": r,
        "cg": cg, "h": h, "ssc": model["ssc"],
        "observed_estimate": observed_estimate, "observed_se": observed_se,
        "observed_t": observed_estimate / observed_se,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    source = ROOT / config["input"]
    if sha256(source) != config["input_sha256"]:
        raise SystemExit("Frozen v3 input checksum mismatch")
    print("Loading frozen v3 input", flush=True)
    data = pd.read_csv(source, usecols=[*NUMERIC, "market_id", "event_id", "calendar_hour_utc"])
    if len(data) != config["expected_rows"] or data[NUMERIC].isna().any().any():
        raise RuntimeError("Input row-count or missing-value QC failed")
    event_codes, event_labels = pd.factorize(data.event_id, sort=True)
    if len(event_labels) != 104:
        raise RuntimeError("Expected 104 match-event clusters")

    print("Absorbing market and calendar-hour fixed effects", flush=True)
    residualized, iterations, final_change = absorb(
        data[NUMERIC].to_numpy(), [data.market_id.to_numpy(), data.calendar_hour_utc.to_numpy()]
    )
    positions = {
        "H1_TOTAL_CTRL": [NUMERIC.index(TOTAL), *[NUMERIC.index(c) for c in CONTROLS]],
        "H2_P99_SPLIT": [NUMERIC.index(LARGE), NUMERIC.index(ORDINARY),
                          *[NUMERIC.index(c) for c in CONTROLS]],
    }
    models = {
        name: prepare_model(name, residualized, positions, event_codes, len(event_labels))
        for name in positions
    }
    tests = [
        prepare_test(models["H1_TOTAL_CTRL"], "H1 total flow = 0", [1, 0, 0, 0, 0]),
        prepare_test(models["H2_P99_SPLIT"], "H2 P99 large flow = 0", [1, 0, 0, 0, 0, 0]),
        prepare_test(models["H2_P99_SPLIT"], "H2 ordinary flow = 0", [0, 1, 0, 0, 0, 0]),
        prepare_test(models["H2_P99_SPLIT"], "H2 large minus ordinary = 0", [1, -1, 0, 0, 0, 0]),
    ]

    rng = np.random.default_rng(SEED)
    weights = rng.choice(np.array([-1.0, 1.0]), size=(len(event_labels), REPS), replace=True)
    draw_arrays = {test["label"]: np.full(REPS, np.nan) for test in tests}
    completed = 0
    if PROGRESS.exists() and DRAWS.exists():
        progress = json.loads(PROGRESS.read_text())
        if (progress.get("input_sha256") == config["input_sha256"] and
                progress.get("reps") == REPS and progress.get("seed") == SEED):
            saved = np.load(DRAWS)
            for index, test in enumerate(tests):
                draw_arrays[test["label"]] = saved[f"test_{index}"].copy()
            completed = int(progress.get("completed_draws", 0))
            print(f"Resuming from draw {completed:,}", flush=True)

    for start in range(completed, REPS, BATCH):
        stop = min(start + BATCH, REPS)
        v = weights[:, start:stop]
        for test in tests:
            numerator = test["cg"] @ v
            z = test["cg"][:, None] * v - test["h"] @ v
            variance = test["ssc"] * np.square(z).sum(axis=0)
            draw_arrays[test["label"]][start:stop] = numerator / np.sqrt(variance)
        np.savez_compressed(DRAWS, **{
            f"test_{index}": draw_arrays[test["label"]]
            for index, test in enumerate(tests)
        })
        atomic_json(PROGRESS, {
            "status": "running" if stop < REPS else "draws_complete",
            "updated_at": now(), "input_sha256": config["input_sha256"],
            "reps": REPS, "seed": SEED, "batch_size": BATCH,
            "completed_draws": stop,
        })
        print(f"Bootstrap draws {stop:,}/{REPS:,}", flush=True)

    rows = []
    for test in tests:
        draws = draw_arrays[test["label"]]
        if not np.isfinite(draws).all():
            raise RuntimeError(f"Non-finite bootstrap draws: {test['label']}")
        exceed = int(np.sum(np.abs(draws) > abs(test["observed_t"])))
        rows.append({
            "test": test["label"], "model": test["model"],
            "observed_estimate": test["observed_estimate"],
            "observed_cluster_se": test["observed_se"],
            "observed_t": test["observed_t"], "bootstrap_p_value": exceed / REPS,
            "recommended_report": f"p<{1 / REPS:.4f}" if exceed == 0 else f"p={exceed / REPS:.4f}",
            "exceedances": exceed, "reps": REPS, "cluster": "event_id",
            "clusters": len(event_labels), "weights_type": "rademacher",
            "bootstrap_type": "WCR11", "impose_null": True, "seed": SEED,
        })
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "inference_bootstrap_9999.csv", index=False)
    metadata = {
        "status": "complete_qc_passed", "generated_at": now(),
        "method": "WCR11 cluster-score bootstrap, memory-safe batched implementation",
        "input": str(source.relative_to(ROOT)), "input_sha256": config["input_sha256"],
        "rows": len(data), "markets": int(data.market_id.nunique()),
        "clusters": len(event_labels), "calendar_hours": int(data.calendar_hour_utc.nunique()),
        "fixed_effect_absorption_iterations": iterations,
        "fixed_effect_absorption_final_max_change": final_change,
        "reps": REPS, "batch_size": BATCH, "seed": SEED,
        "weights_type": "rademacher", "impose_null": True,
        "bootstrap_type": "11", "k_adj": True, "G_adj": True,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "results": rows,
        "draws_sha256": sha256(DRAWS),
    }
    atomic_json(OUT / "summary.json", metadata)
    atomic_json(PROGRESS, {**json.loads(PROGRESS.read_text()), "status": "complete_qc_passed",
                           "completed_at": now()})
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
