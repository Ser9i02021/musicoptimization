import contextlib
import inspect
import os
import pickle
import time
import traceback
from pathlib import Path

import pandas as pd

from cost_matrix_construction import build_cost_matrix
from optimization import optimize


# ============================================================
# Additional paired r/s sensitivity experiments
# ============================================================
#
# This extends the completed Moderate L=62 sensitivity experiment to:
#   - Slow, SMF=0, L=43
#   - Fast, SMF=2, L=62
#
# For each baseline instance, the exact saved licks_list is reused. Therefore
# each r/s comparison is paired within the same sampled candidate set.
#
# Baseline setting (r=1, s=3) is reused and NOT re-solved.
# Four new settings are solved for each of 100 runs in each profile:
#   (2,3), (3,3), (1,4), (1,5)
# Total new MILP solves = 2 * 100 * 4 = 800.
# ============================================================

CONFIGURATIONS = [
    {"profile": "Slow", "smf": 0, "L": 43},
    {"profile": "Fast", "smf": 2, "L": 62},
]

B = 12
NUM_RUNS = 100

BASELINE_R = 1
BASELINE_S = 3

SENSITIVITY_SETTINGS = [
    (1, 3),
    (2, 3),
    (3, 3),
    (1, 4),
    (1, 5),
]

MAX_TOTAL_TIME = 300
MAX_CUT_ROUNDS = 1000
VERBOSE_OPTIMIZE = False
QUIET_INNER_FUNCTIONS = True
RETRY_FAILED = False

# Prefer the definitive directory used in the later statistical workflow.
# Fall back to the earlier name if needed.
BASELINE_RUNS_CANDIDATES = [
    Path("experiment_final") / "runs",
    Path("experiment_results") / "runs",
]

OUTPUT_DIR = Path("rs_sensitivity_results")
RUNS_DIR = OUTPUT_DIR / "runs"

# Preserve the original Moderate-only CSV produced earlier.
MODERATE_RESULTS_CSV = OUTPUT_DIR / "rs_sensitivity_all_runs.csv"

# New outputs from this extension.
ADDITIONAL_RESULTS_CSV = OUTPUT_DIR / "rs_sensitivity_slow_fast.csv"
COMBINED_RESULTS_CSV = OUTPUT_DIR / "rs_sensitivity_all_profiles.csv"


# ============================================================
# Helpers
# ============================================================

def resolve_baseline_runs_dir():
    for candidate in BASELINE_RUNS_CANDIDATES:
        if candidate.is_dir():
            return candidate

    attempted = "\n".join(f"  - {p}" for p in BASELINE_RUNS_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the baseline runs directory. Tried:\n"
        f"{attempted}\n"
        "Place this script in the project root or edit "
        "BASELINE_RUNS_CANDIDATES."
    )


def baseline_filename(baseline_runs_dir, profile, L, run_number):
    return (
        baseline_runs_dir
        / f"{profile.lower()}_L{L}_run_{run_number:03d}.pkl"
    )


def sensitivity_filename(profile, L, run_number, r, s):
    return (
        RUNS_DIR
        / f"{profile.lower()}_L{L}_run_{run_number:03d}_r{r}_s{s}.pkl"
    )


def save_record(record, filename):
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(record, f)
    os.replace(tmp, filename)


def load_record(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def validate_optimizer_signature():
    params = inspect.signature(optimize).parameters
    missing = [name for name in ("r", "s") if name not in params]
    if missing:
        raise RuntimeError(
            "The imported optimization.optimize() does not expose the "
            f"parameter(s) {missing}. Use the parameterized optimization.py "
            "from the completed Moderate sensitivity experiment."
        )


def load_baseline_run(
    baseline_runs_dir,
    profile,
    smf,
    L,
    run_number,
):
    filename = baseline_filename(
        baseline_runs_dir,
        profile,
        L,
        run_number,
    )

    if not filename.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {filename}")

    record = load_record(filename)

    if record.get("profile") != profile:
        raise ValueError(
            f"{filename} has profile={record.get('profile')!r}, "
            f"expected {profile!r}."
        )

    if int(record.get("SMF", -1)) != smf:
        raise ValueError(
            f"{filename} has SMF={record.get('SMF')!r}, expected {smf}."
        )

    if int(record.get("L", -1)) != L:
        raise ValueError(
            f"{filename} has L={record.get('L')!r}, expected {L}."
        )

    if int(record.get("run", -1)) != run_number:
        raise ValueError(
            f"{filename} has run={record.get('run')!r}, "
            f"expected {run_number}."
        )

    licks_list = record.get("licks_list")
    if not licks_list:
        raise ValueError(
            f"{filename} does not contain licks_list, so the exact paired "
            "instance cannot be reconstructed."
        )

    if len(licks_list) != L:
        raise ValueError(
            f"{filename} contains {len(licks_list)} candidate licks, expected {L}."
        )

    return record, licks_list


def baseline_as_sensitivity_record(
    baseline_record,
    profile,
    smf,
    L,
    run_number,
):
    return {
        "profile": profile,
        "SMF": smf,
        "L": L,
        "b": B,
        "run": run_number,
        "seed": baseline_record.get("seed"),
        "r": BASELINE_R,
        "s": BASELINE_S,
        "status": baseline_record.get("status"),
        "obj_val": baseline_record.get("obj_val"),
        "subt_count": baseline_record.get("subt_count"),
        "time_taken": baseline_record.get("time_taken"),
        "wall_time": baseline_record.get("wall_time"),
        "error_type": baseline_record.get("error_type"),
        "error_message": baseline_record.get("error_message"),
        "source": "baseline_checkpoint",
    }


def run_setting(
    profile,
    smf,
    L,
    licks_list,
    p,
    baseline_record,
    run_number,
    r,
    s,
):
    wall_start = time.perf_counter()

    record = {
        "profile": profile,
        "SMF": smf,
        "L": L,
        "b": B,
        "run": run_number,
        "seed": baseline_record.get("seed"),
        "r": r,
        "s": s,
        "status": None,
        "obj_val": None,
        "subt_count": None,
        "time_taken": None,
        "wall_time": None,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "source": "sensitivity_run",
    }

    try:
        def execute():
            return optimize(
                licks_list,
                p,
                B,
                max_total_time=MAX_TOTAL_TIME,
                max_cut_rounds=MAX_CUT_ROUNDS,
                verbose=VERBOSE_OPTIMIZE,
                r=r,
                s=s,
            )

        if QUIET_INNER_FUNCTIONS and not VERBOSE_OPTIMIZE:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    result = execute()
        else:
            result = execute()

        (
            graph_path_vertices_ordered,
            file_paths_for_the_ordered_licks_in_the_solution,
            obj_val,
            subt_count,
            time_taken,
        ) = result

        if obj_val is None or subt_count is None or time_taken is None:
            raise RuntimeError(
                "One or more required statistics were not returned by optimize()."
            )

        record.update(
            {
                "status": "OK",
                "obj_val": float(obj_val),
                "subt_count": int(subt_count),
                "time_taken": float(time_taken),
                "wall_time": time.perf_counter() - wall_start,
                "graph_path_vertices_ordered": graph_path_vertices_ordered,
                "file_paths_for_the_ordered_licks_in_the_solution":
                    file_paths_for_the_ordered_licks_in_the_solution,
            }
        )

    except TimeoutError as exc:
        record.update(
            {
                "status": "TIMEOUT",
                "wall_time": time.perf_counter() - wall_start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    except Exception as exc:
        record.update(
            {
                "status": "ERROR",
                "wall_time": time.perf_counter() - wall_start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    return record


def row_for_csv(record):
    return {
        "Profile": record.get("profile"),
        "SMF": record.get("SMF"),
        "L": record.get("L"),
        "b": record.get("b"),
        "Run": record.get("run"),
        "Seed": record.get("seed"),
        "r": record.get("r"),
        "s": record.get("s"),
        "Status": record.get("status"),
        "Quality": record.get("obj_val"),
        "Time_s": record.get("time_taken"),
        "Wall_time_s": record.get("wall_time"),
        "Subtours": record.get("subt_count"),
        "Error_type": record.get("error_type"),
        "Error_message": record.get("error_message"),
        "Source": record.get("source"),
    }


def dataframe_from_records(records):
    df = pd.DataFrame(row_for_csv(record) for record in records)
    if df.empty:
        return df
    return df.sort_values(
        ["Profile", "L", "Run", "r", "s"],
        kind="stable",
    ).reset_index(drop=True)


def save_additional_csv(records):
    df = dataframe_from_records(records)
    df.to_csv(ADDITIONAL_RESULTS_CSV, index=False)


def save_combined_csv(additional_df):
    frames = []

    if MODERATE_RESULTS_CSV.exists():
        frames.append(pd.read_csv(MODERATE_RESULTS_CSV))
    else:
        print(
            f"WARNING: {MODERATE_RESULTS_CSV} was not found. "
            "The combined file will contain only Slow/Fast for now."
        )

    frames.append(additional_df)
    combined = pd.concat(frames, ignore_index=True, sort=False)

    key = ["Profile", "L", "Run", "r", "s"]
    if all(column in combined.columns for column in key):
        combined = combined.drop_duplicates(subset=key, keep="last")
        combined = combined.sort_values(key, kind="stable").reset_index(drop=True)

    combined.to_csv(COMBINED_RESULTS_CSV, index=False)


def preflight(baseline_runs_dir):
    validate_optimizer_signature()

    print("Preflight validation of baseline checkpoints...")

    for config in CONFIGURATIONS:
        profile = config["profile"]
        smf = config["smf"]
        L = config["L"]

        status_counts = {}

        for run_number in range(1, NUM_RUNS + 1):
            baseline_record, _ = load_baseline_run(
                baseline_runs_dir,
                profile,
                smf,
                L,
                run_number,
            )
            status = baseline_record.get("status")
            status_counts[status] = status_counts.get(status, 0) + 1

        print(
            f"  {profile} L={L}: 100 checkpoints validated; "
            f"baseline statuses={status_counts}"
        )

    print("Preflight passed.\n")


# ============================================================
# Main experiment
# ============================================================

def main():
    baseline_runs_dir = resolve_baseline_runs_dir()

    OUTPUT_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("ADDITIONAL PAIRED r/s SENSITIVITY EXPERIMENTS")
    print("=" * 78)
    print(f"Baseline runs: {baseline_runs_dir}")
    print(f"Configurations: {CONFIGURATIONS}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Settings: {SENSITIVITY_SETTINGS}")
    print(
        "New MILP solves: "
        f"{len(CONFIGURATIONS) * NUM_RUNS * (len(SENSITIVITY_SETTINGS) - 1)}"
    )
    print(f"Time limit per solve: {MAX_TOTAL_TIME}s")
    print()

    preflight(baseline_runs_dir)

    all_records = []

    for config_index, config in enumerate(CONFIGURATIONS, start=1):
        profile = config["profile"]
        smf = config["smf"]
        L = config["L"]

        print("=" * 78)
        print(
            f"CONFIGURATION {config_index}/{len(CONFIGURATIONS)}: "
            f"Profile={profile}, SMF={smf}, L={L}, b={B}"
        )
        print("=" * 78)

        for run_number in range(1, NUM_RUNS + 1):
            baseline_record, licks_list = load_baseline_run(
                baseline_runs_dir,
                profile,
                smf,
                L,
                run_number,
            )

            # Rebuild transition costs once for this exact candidate set.
            if QUIET_INNER_FUNCTIONS:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull):
                        p = build_cost_matrix(licks_list)
            else:
                p = build_cost_matrix(licks_list)

            print(
                f"[{profile} L={L} run {run_number:03d}/{NUM_RUNS}] "
                f"seed={baseline_record.get('seed')}"
            )

            for r, s in SENSITIVITY_SETTINGS:
                if (r, s) == (BASELINE_R, BASELINE_S):
                    record = baseline_as_sensitivity_record(
                        baseline_record,
                        profile,
                        smf,
                        L,
                        run_number,
                    )
                    all_records.append(record)
                    print(f"  r={r}, s={s}: baseline -> {record['status']}")
                    continue

                filename = sensitivity_filename(
                    profile,
                    L,
                    run_number,
                    r,
                    s,
                )

                reuse = False
                record = None

                if filename.exists():
                    try:
                        old = load_record(filename)
                        same_setting = (
                            old.get("profile") == profile
                            and int(old.get("SMF", -1)) == smf
                            and int(old.get("L", -1)) == L
                            and int(old.get("run", -1)) == run_number
                            and int(old.get("r", -1)) == r
                            and int(old.get("s", -1)) == s
                            and old.get("seed") == baseline_record.get("seed")
                        )

                        if same_setting:
                            if old.get("status") == "OK" or not RETRY_FAILED:
                                reuse = True
                                record = old
                    except Exception:
                        reuse = False

                if reuse:
                    print(f"  r={r}, s={s}: checkpoint -> {record['status']}")
                else:
                    print(f"  r={r}, s={s}: solving ... ", end="", flush=True)

                    record = run_setting(
                        profile=profile,
                        smf=smf,
                        L=L,
                        licks_list=licks_list,
                        p=p,
                        baseline_record=baseline_record,
                        run_number=run_number,
                        r=r,
                        s=s,
                    )

                    save_record(record, filename)

                    if record["status"] == "OK":
                        print(
                            f"OK time={record['time_taken']:.2f}s, "
                            f"quality={record['obj_val']:.2f}, "
                            f"subtours={record['subt_count']}"
                        )
                    elif record["status"] == "TIMEOUT":
                        print(f"TIMEOUT after ~{record['wall_time']:.2f}s")
                    else:
                        print(
                            f"ERROR: {record['error_type']}: "
                            f"{record['error_message']}"
                        )

                all_records.append(record)

            # Continuously update the additional-results CSV for safe resumption.
            save_additional_csv(all_records)

        print()

    additional_df = dataframe_from_records(all_records)
    save_additional_csv(all_records)
    save_combined_csv(additional_df)

    print("=" * 78)
    print("Additional sensitivity experiments complete.")
    print(f"Slow/Fast results: {ADDITIONAL_RESULTS_CSV}")
    print(f"All-profile results: {COMBINED_RESULTS_CSV}")
    print(f"Checkpoints: {RUNS_DIR}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
