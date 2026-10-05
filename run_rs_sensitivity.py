import contextlib
import os
import pickle
import time
import traceback
from pathlib import Path

import pandas as pd

from cost_matrix_construction import build_cost_matrix
from optimization import optimize


# ============================================================
# Paired r/s sensitivity experiment
# ============================================================

PROFILE = "Moderate"
SMF = 1
L = 62
B = 12
NUM_RUNS = 100

BASELINE_R = 1
BASELINE_S = 3

# One-factor-at-a-time sensitivity design.
# (1, 3) is the baseline already available from the main experiment.
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

# IMPORTANT:
# Change this path so it points to the "runs" folder of the completed
# 700-run baseline experiment.
BASELINE_RUNS_DIR = Path("experiment_final/runs")

OUTPUT_DIR = Path("rs_sensitivity_results")
RUNS_DIR = OUTPUT_DIR / "runs"

# If False, existing sensitivity checkpoints are reused.
RETRY_FAILED = False


# ============================================================
# File helpers
# ============================================================

def baseline_filename(run_number):
    return BASELINE_RUNS_DIR / f"{PROFILE.lower()}_L{L}_run_{run_number:03d}.pkl"


def sensitivity_filename(run_number, r, s):
    return (
        RUNS_DIR
        / f"{PROFILE.lower()}_L{L}_run_{run_number:03d}_r{r}_s{s}.pkl"
    )


def save_record(record, filename):
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(record, f)
    os.replace(tmp, filename)


def load_record(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


# ============================================================
# Baseline handling
# ============================================================

def load_baseline_run(run_number):
    """
    Load the exact candidate set used in the main experiment.

    Reusing licks_list is what makes the sensitivity experiment paired:
    the same candidate set and transition-cost matrix are evaluated under
    different r and s values.
    """
    filename = baseline_filename(run_number)

    if not filename.exists():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: {filename}\n"
            "Set BASELINE_RUNS_DIR to the runs folder from the completed "
            "baseline experiment."
        )

    record = load_record(filename)

    if record.get("profile") != PROFILE:
        raise ValueError(
            f"{filename} has profile={record.get('profile')!r}, expected {PROFILE!r}."
        )

    if int(record.get("SMF", -1)) != SMF:
        raise ValueError(
            f"{filename} has SMF={record.get('SMF')!r}, expected {SMF}."
        )

    if int(record.get("L", -1)) != L:
        raise ValueError(
            f"{filename} has L={record.get('L')!r}, expected {L}."
        )

    licks_list = record.get("licks_list")
    if not licks_list:
        raise ValueError(
            f"{filename} does not contain licks_list, so the paired instance "
            "cannot be reconstructed."
        )

    return record, licks_list


def baseline_as_sensitivity_record(baseline_record, run_number):
    """
    Represent the already-computed r=1, s=3 result in the same schema as the
    new sensitivity results. No optimization is rerun.
    """
    return {
        "profile": PROFILE,
        "SMF": SMF,
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


# ============================================================
# One alternative parameter setting
# ============================================================

def run_setting(licks_list, p, baseline_record, run_number, r, s):
    wall_start = time.perf_counter()

    record = {
        "profile": PROFILE,
        "SMF": SMF,
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


# ============================================================
# CSV helper
# ============================================================

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


def save_current_csv(records):
    df = pd.DataFrame(row_for_csv(record) for record in records)
    df = df.sort_values(["Run", "r", "s"]).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "rs_sensitivity_all_runs.csv", index=False)


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    all_records = []

    total_new_solves = NUM_RUNS * (len(SENSITIVITY_SETTINGS) - 1)

    print("=" * 78)
    print("PAIRED r/s SENSITIVITY EXPERIMENT")
    print("=" * 78)
    print(f"Profile={PROFILE}, SMF={SMF}, L={L}, b={B}")
    print(f"Runs={NUM_RUNS}")
    print(f"Settings={SENSITIVITY_SETTINGS}")
    print(
        f"New MILP solves={total_new_solves} "
        f"(baseline r={BASELINE_R}, s={BASELINE_S} is reused)"
    )
    print(f"Time limit per solve={MAX_TOTAL_TIME}s")
    print()

    for run_number in range(1, NUM_RUNS + 1):
        baseline_record, licks_list = load_baseline_run(run_number)

        # Rebuild once per candidate set. The same p is used for every r/s pair.
        if QUIET_INNER_FUNCTIONS:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    p = build_cost_matrix(licks_list)
        else:
            p = build_cost_matrix(licks_list)

        print(
            f"[Run {run_number:03d}/{NUM_RUNS}] "
            f"seed={baseline_record.get('seed')}"
        )

        for r, s in SENSITIVITY_SETTINGS:

            # The baseline result already exists. Reuse it exactly.
            if (r, s) == (BASELINE_R, BASELINE_S):
                record = baseline_as_sensitivity_record(
                    baseline_record,
                    run_number,
                )
                all_records.append(record)

                print(
                    f"  r={r}, s={s}: baseline -> "
                    f"{record['status']}"
                )
                continue

            filename = sensitivity_filename(run_number, r, s)

            reuse = False
            record = None

            if filename.exists():
                try:
                    old = load_record(filename)

                    same_setting = (
                        int(old.get("run", -1)) == run_number
                        and int(old.get("r", -1)) == r
                        and int(old.get("s", -1)) == s
                        and int(old.get("L", -1)) == L
                        and int(old.get("SMF", -1)) == SMF
                        and old.get("seed") == baseline_record.get("seed")
                    )

                    if same_setting:
                        if old.get("status") == "OK" or not RETRY_FAILED:
                            reuse = True
                            record = old
                except Exception:
                    reuse = False

            if reuse:
                print(
                    f"  r={r}, s={s}: checkpoint -> "
                    f"{record['status']}"
                )
            else:
                print(
                    f"  r={r}, s={s}: solving ... ",
                    end="",
                    flush=True,
                )

                record = run_setting(
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
                        f"OK  time={record['time_taken']:.2f}s, "
                        f"quality={record['obj_val']:.2f}, "
                        f"subtours={record['subt_count']}"
                    )
                elif record["status"] == "TIMEOUT":
                    print(
                        f"TIMEOUT after ~{record['wall_time']:.2f}s"
                    )
                else:
                    print(
                        f"ERROR: {record['error_type']}: "
                        f"{record['error_message']}"
                    )

            all_records.append(record)

        # Keep a continuously updated flat file in case the experiment is stopped.
        save_current_csv(all_records)

    print()
    print("=" * 78)
    print("Sensitivity experiment complete.")
    print(f"Results: {OUTPUT_DIR / 'rs_sensitivity_all_runs.csv'}")
    print(f"Checkpoints: {RUNS_DIR}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
