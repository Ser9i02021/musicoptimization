import contextlib
import math
import os
import pickle
import random
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from label_chosen_licks import select_N_lick_samples
from cost_matrix_construction import build_cost_matrix
from optimization import optimize
from post_processing import post_process


# ============================================================
# Experimental design
# ============================================================

NUM_RUNS = 100
OPTIMIZE_PARAMETER = 12

# Maximum cumulative time used by optimize() for one sampled instance.
MAX_TOTAL_TIME = 300
MAX_CUT_ROUNDS = 1000

# For the full experiment, keep this False to avoid large console output.
VERBOSE_OPTIMIZE = False

# Suppress diagnostic prints still present inside sample selection / cost matrix
# construction / optimization. The driver itself still prints one compact line
# per run.
QUIET_INNER_FUNCTIONS = True

# Resume policy:
#   False -> keep previously recorded TIMEOUT/ERROR runs and continue past them.
#   True  -> rerun previously recorded TIMEOUT/ERROR runs using the SAME seed.
#           This is useful later if you raise MAX_TOTAL_TIME.
RETRY_FAILED = False

# SMF mapping:
# 0 = Slow
# 1 = Moderate
# 2 = Fast

EXPERIMENTS = [
    ("Slow",     0, [32, 43]),
    ("Moderate", 1, [32, 62, 160]),
    ("Fast",     2, [32, 62]),
]
'''
EXPERIMENTS = [
    ("Moderate", 1, [160]),
]
'''
BASE_SEED = 20260910

OUTPUT_DIR = Path("experiment_final")
RUNS_DIR = OUTPUT_DIR / "runs"

# ============================================================
# Optional MusicXML export
# ============================================================
#
# The benchmark itself should NOT export all solutions.
# Put only the specific successful runs you want to inspect here.
#
# Tuple format: (Profile, L, run_number)
#
# Example:
# EXPORT_XML_RUNS = {
#     ("Fast", 62, 1),
#     ("Fast", 62, 10),
# }
#
# Leave this empty to export no MusicXML files.
EXPORT_XML_RUNS = set()

XML_OUTPUT_DIR = Path("solutions") / "experiment_exports"


# ============================================================
# Reproducibility / file helpers
# ============================================================

def make_seed(smf, L, run_number):
    """
    Deterministic seed that depends only on the configuration and run number.

    Unlike an experiment-index-based seed, this remains the same when you run
    only one configuration for testing and later include it in the full grid.
    """
    return BASE_SEED + smf * 10_000_000 + L * 10_000 + run_number


def run_filename(profile, L, run_number):
    return RUNS_DIR / f"{profile.lower()}_L{L}_run_{run_number:03d}.pkl"


def save_record(record, filename):
    """Atomically save one run record so interruptions do not corrupt it."""
    tmp = filename.with_suffix(filename.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(record, f)
    os.replace(tmp, filename)


def load_record(filename):
    with open(filename, "rb") as f:
        record = pickle.load(f)

    # Backward compatibility with successful files created by the older driver.
    if "status" not in record:
        record["status"] = "OK"

    return record


def export_musicxml_if_requested(record):
    """
    Optional stages (5) and (6) of the original pipeline:

        (5) Postprocessing
        (6) Export as MusicXML

    A successful checkpoint already stores the ordered MusicXML source paths,
    so the selected solution can be exported without rerunning optimization.

    MusicXML export is intentionally not included in time_taken.
    """
    if record.get("status") != "OK":
        return None

    profile = record.get("profile")
    L = int(record.get("L"))
    run_number = int(record.get("run"))

    if (profile, L, run_number) not in EXPORT_XML_RUNS:
        return None

    ordered_paths = record.get(
        "file_paths_for_the_ordered_licks_in_the_solution"
    )

    if not ordered_paths:
        raise RuntimeError(
            f"Cannot export MusicXML for {profile}, L={L}, run={run_number}: "
            "the checkpoint does not contain the ordered MusicXML file paths."
        )

    XML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        XML_OUTPUT_DIR
        / f"{profile.lower()}_L{L}_run_{run_number:03d}.xml"
    )

    xml_result = post_process(
        ordered_paths,
        str(output_file),
    )

    exported_path = (
        str(xml_result) if xml_result is not None else str(output_file)
    )

    record["musicxml_output"] = exported_path
    record["musicxml_export_error"] = None

    print(f"  MusicXML exported: {exported_path}")

    return exported_path


def record_matches_request(record, profile, smf, L, run_number, seed):
    """
    Prevent stale test files from being mistaken for the requested full-run
    observation. In particular, the seed must match.
    """
    return (
        record.get("profile") == profile
        and int(record.get("SMF", -1)) == smf
        and int(record.get("L", -1)) == L
        and int(record.get("run", -1)) == run_number
        and int(record.get("seed", -1)) == seed
    )


# ============================================================
# One simulation
# ============================================================

def run_single_simulation(profile, smf, L, run_number):
    seed = make_seed(smf, L, run_number)

    random.seed(seed)
    np.random.seed(seed)

    record = {
        "profile": profile,
        "SMF": smf,
        "L": L,
        "run": run_number,
        "seed": seed,
        "status": None,
        "stage": "initialization",
    }

    licks_list = None
    wall_start = time.perf_counter()

    try:
        def execute_pipeline():
            nonlocal licks_list

            record["stage"] = "sample_selection"
            licks_list = select_N_lick_samples(L, smf)

            record["stage"] = "cost_matrix"
            p = build_cost_matrix(licks_list)

            record["stage"] = "optimization"
            return optimize(
                licks_list,
                p,
                OPTIMIZE_PARAMETER,
                max_total_time=MAX_TOTAL_TIME,
                max_cut_rounds=MAX_CUT_ROUNDS,
                verbose=VERBOSE_OPTIMIZE,
            )

        if QUIET_INNER_FUNCTIONS and not VERBOSE_OPTIMIZE:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    result = execute_pipeline()
        else:
            result = execute_pipeline()

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
                "stage": "completed",
                "licks_list": licks_list,
                "graph_path_vertices_ordered": graph_path_vertices_ordered,
                "file_paths_for_the_ordered_licks_in_the_solution":
                    file_paths_for_the_ordered_licks_in_the_solution,
                "obj_val": float(obj_val),
                "subt_count": int(subt_count),
                "time_taken": float(time_taken),
                "wall_time": time.perf_counter() - wall_start,
                "error_type": None,
                "error_message": None,
                "traceback": None,
            }
        )

    except TimeoutError as exc:
        record.update(
            {
                "status": "TIMEOUT",
                "licks_list": licks_list,
                "obj_val": None,
                "subt_count": None,
                "time_taken": None,
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
                "licks_list": licks_list,
                "obj_val": None,
                "subt_count": None,
                "time_taken": None,
                "wall_time": time.perf_counter() - wall_start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    return record


# ============================================================
# Summaries
# ============================================================

def summarize_experiment(profile, smf, L, records):
    successful = [r for r in records if r.get("status") == "OK"]
    timeouts = [r for r in records if r.get("status") == "TIMEOUT"]
    errors = [r for r in records if r.get("status") == "ERROR"]

    if successful:
        times = [r["time_taken"] for r in successful]
        subcycles = [r["subt_count"] for r in successful]
        qualities = [r["obj_val"] for r in successful]

        time_mean = statistics.mean(times)
        time_median = statistics.median(times)
        sub_mean = statistics.mean(subcycles)
        sub_median = statistics.median(subcycles)
        quality_mean = statistics.mean(qualities)
        quality_median = statistics.median(qualities)
    else:
        time_mean = math.nan
        time_median = math.nan
        sub_mean = math.nan
        sub_median = math.nan
        quality_mean = math.nan
        quality_median = math.nan

    return {
        "Perfil": profile,
        "SMF": smf,
        "L": L,
        "N successful": len(successful),
        "Timeouts": len(timeouts),
        "Errors": len(errors),
        "Tempo (s) - Média": time_mean,
        "Tempo (s) - Mediana": time_median,
        "Número de subciclos - Média": sub_mean,
        "Número de subciclos - Mediana": sub_median,
        "Qualidade - Média": quality_mean,
        "Qualidade - Mediana": quality_median,
    }


def status_row(record):
    return {
        "Perfil": record.get("profile"),
        "SMF": record.get("SMF"),
        "L": record.get("L"),
        "Run": record.get("run"),
        "Seed": record.get("seed"),
        "Status": record.get("status"),
        "Stage": record.get("stage"),
        "Tempo (s)": record.get("time_taken"),
        "Wall time (s)": record.get("wall_time"),
        "Número de subciclos": record.get("subt_count"),
        "Qualidade": record.get("obj_val"),
        "Error type": record.get("error_type"),
        "Error message": record.get("error_message"),
    }


def make_display_table(summary_df):
    display = summary_df[
        [
            "Perfil",
            "L",
            "Tempo (s) - Média",
            "Tempo (s) - Mediana",
            "Número de subciclos - Média",
            "Número de subciclos - Mediana",
            "Qualidade - Média",
            "Qualidade - Mediana",
        ]
    ].copy()

    display.columns = pd.MultiIndex.from_tuples(
        [
            ("Perfil", ""),
            ("L", ""),
            ("Tempo (s)", "Média"),
            ("Tempo (s)", "Mediana"),
            ("Número de subciclos", "Média"),
            ("Número de subciclos", "Mediana"),
            ("Qualidade", "Média"),
            ("Qualidade", "Mediana"),
        ]
    )

    profile_col = ("Perfil", "")
    display[profile_col] = display[profile_col].mask(
        display[profile_col].duplicated(), ""
    )

    return display


def format_decimal(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2f}".replace(".", ",")


def save_latex_table(summary_df, filename):
    """Write LaTeX directly; pandas Styler/Jinja2 is not required."""
    rows = []

    for _, row in summary_df.iterrows():
        rows.append(
            f'{row["Perfil"]} & {int(row["L"])} & '
            f'{format_decimal(row["Tempo (s) - Média"])} & '
            f'{format_decimal(row["Tempo (s) - Mediana"])} & '
            f'{format_decimal(row["Número de subciclos - Média"])} & '
            f'{format_decimal(row["Número de subciclos - Mediana"])} & '
            f'{format_decimal(row["Qualidade - Média"])} & '
            f'{format_decimal(row["Qualidade - Mediana"])} \\\\'
        )

    latex = "\n".join(
        [
            r"\begin{tabular}{llrrrrrr}",
            r"\hline",
            r"Perfil & $L$ & \multicolumn{2}{c}{Tempo (s)} & \multicolumn{2}{c}{Número de subciclos} & \multicolumn{2}{c}{Qualidade} \\",
            r" & & Média & Mediana & Média & Mediana & Média & Mediana \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
        ]
    )

    with open(OUTPUT_DIR / filename, "w", encoding="utf-8") as f:
        f.write(latex)


# ============================================================
# Main experiment loop
# ============================================================

def run_all_experiments():
    OUTPUT_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    all_records = []
    summary_rows = []

    total_experiments = sum(len(L_values) for _, _, L_values in EXPERIMENTS)
    experiment_number = 0

    for profile, smf, L_values in EXPERIMENTS:
        for L in L_values:
            experiment_number += 1

            print("\n" + "=" * 70)
            print(
                f"Experiment {experiment_number}/{total_experiments}: "
                f"Profile={profile}, SMF={smf}, L={L}"
            )
            print("=" * 70)

            experiment_records = []

            for run_number in range(1, NUM_RUNS + 1):
                seed = make_seed(smf, L, run_number)
                filename = run_filename(profile, L, run_number)

                reuse = False
                record = None

                if filename.exists():
                    try:
                        old_record = load_record(filename)

                        if record_matches_request(
                            old_record,
                            profile,
                            smf,
                            L,
                            run_number,
                            seed,
                        ):
                            if old_record.get("status") == "OK":
                                reuse = True
                                record = old_record
                            elif not RETRY_FAILED:
                                reuse = True
                                record = old_record
                        else:
                            print(
                                f"[{profile}, L={L}] Run {run_number:03d}: "
                                "existing file has a different seed/configuration; "
                                "rerunning."
                            )
                    except Exception as exc:
                        print(
                            f"[{profile}, L={L}] Run {run_number:03d}: "
                            f"could not read existing checkpoint ({exc}); rerunning."
                        )

                if not reuse:
                    print(
                        f"[{profile}, L={L}] Run {run_number:03d}/{NUM_RUNS} ... ",
                        end="",
                        flush=True,
                    )

                    record = run_single_simulation(
                        profile=profile,
                        smf=smf,
                        L=L,
                        run_number=run_number,
                    )

                    # Save every outcome immediately: OK, TIMEOUT, or ERROR.
                    save_record(record, filename)

                    if record["status"] == "OK":
                        print(
                            f"OK  time={record['time_taken']:.2f}s, "
                            f"subcycles={record['subt_count']}, "
                            f"quality={record['obj_val']:.2f}"
                        )
                    elif record["status"] == "TIMEOUT":
                        print(
                            f"TIMEOUT after ~{record['wall_time']:.2f}s "
                            f"(seed={record['seed']})"
                        )
                    else:
                        print(
                            f"ERROR at stage={record['stage']}: "
                            f"{record['error_type']}: {record['error_message']} "
                            f"(seed={record['seed']})"
                        )
                else:
                    print(
                        f"[{profile}, L={L}] Run {run_number:03d}/{NUM_RUNS} "
                        f"checkpoint -> {record['status']}"
                    )

                # ----------------------------------------------------
                # (5) Optional postprocessing
                # (6) Optional MusicXML export
                # ----------------------------------------------------
                try:
                    exported_path = export_musicxml_if_requested(record)

                    if exported_path is not None:
                        save_record(record, filename)

                except Exception as exc:
                    record["musicxml_export_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(
                        f"  WARNING: MusicXML export failed for "
                        f"{profile}, L={L}, run={run_number}: {exc}"
                    )
                    save_record(record, filename)

                experiment_records.append(record)
                all_records.append(record)

            summary = summarize_experiment(
                profile,
                smf,
                L,
                experiment_records,
            )
            summary_rows.append(summary)

            print(
                f"[{profile}, L={L}] "
                f"successful={summary['N successful']}/{NUM_RUNS}, "
                f"timeouts={summary['Timeouts']}, "
                f"errors={summary['Errors']}"
            )

            if summary["N successful"]:
                print(
                    f"  Tempo:     mean={summary['Tempo (s) - Média']:.2f}, "
                    f"median={summary['Tempo (s) - Mediana']:.2f}"
                )
                print(
                    f"  Subcycles: mean={summary['Número de subciclos - Média']:.2f}, "
                    f"median={summary['Número de subciclos - Mediana']:.2f}"
                )
                print(
                    f"  Quality:   mean={summary['Qualidade - Média']:.2f}, "
                    f"median={summary['Qualidade - Mediana']:.2f}"
                )

            if summary["N successful"] != NUM_RUNS:
                print(
                    "  WARNING: this configuration is incomplete. "
                    "Its statistics are provisional and use only successful runs."
                )

            # Checkpoint a human-readable status CSV after every configuration.
            pd.DataFrame([status_row(r) for r in all_records]).to_csv(
                OUTPUT_DIR / "run_status.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # --------------------------------------------------------
    # Save all run-level results and configuration summaries
    # --------------------------------------------------------
    status_df = pd.DataFrame([status_row(r) for r in all_records])
    status_df.to_csv(
        OUTPUT_DIR / "run_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    successful_df = status_df[status_df["Status"] == "OK"].copy()
    successful_df.to_csv(
        OUTPUT_DIR / "all_successful_runs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUTPUT_DIR / "summary_with_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    complete = all(
        row["N successful"] == NUM_RUNS
        and row["Timeouts"] == 0
        and row["Errors"] == 0
        for row in summary_rows
    )

    statistical_columns = [
        "Perfil",
        "L",
        "Tempo (s) - Média",
        "Tempo (s) - Mediana",
        "Número de subciclos - Média",
        "Número de subciclos - Mediana",
        "Qualidade - Média",
        "Qualidade - Mediana",
    ]
    table_df = summary_df[statistical_columns].copy()

    if complete:
        csv_name = "final_table.csv"
        txt_name = "final_table.txt"
        tex_name = "final_table.tex"
        title = f"FINAL TABLE - {NUM_RUNS} SUCCESSFUL RUNS PER CONFIGURATION"
    else:
        csv_name = "provisional_table.csv"
        txt_name = "provisional_table.txt"
        tex_name = "provisional_table.tex"
        title = "PROVISIONAL TABLE - AT LEAST ONE CONFIGURATION IS INCOMPLETE"

    table_df.to_csv(
        OUTPUT_DIR / csv_name,
        index=False,
        encoding="utf-8-sig",
    )

    display_table = make_display_table(summary_df)
    table_text = display_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )

    print("\n\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(table_text)

    with open(OUTPUT_DIR / txt_name, "w", encoding="utf-8") as f:
        f.write(table_text)

    # Formatting output is deliberately non-fatal.
    try:
        save_latex_table(summary_df, tex_name)
    except Exception as exc:
        print("\nWARNING: LaTeX table could not be generated.")
        print(f"Reason: {exc}")
        print("All numerical/checkpoint results remain saved.")

    print("\nFiles created/updated:")
    print(f"  {OUTPUT_DIR / 'run_status.csv'}")
    print(f"  {OUTPUT_DIR / 'all_successful_runs.csv'}")
    print(f"  {OUTPUT_DIR / 'summary_with_status.csv'}")
    print(f"  {OUTPUT_DIR / csv_name}")
    print(f"  {OUTPUT_DIR / txt_name}")
    print(f"  {OUTPUT_DIR / tex_name}")
    print(f"  Per-run checkpoints: {RUNS_DIR}/")

    if not complete:
        print(
            "\nDo not use the provisional table as the final paper table. "
            "Resolve the TIMEOUT/ERROR records first; their exact seeds and "
            "sampled lick lists have been preserved."
        )


if __name__ == "__main__":
    run_all_experiments()
