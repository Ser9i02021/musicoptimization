import pickle
from collections import Counter
from pathlib import Path

PAUSE_CODES = {"C2", "C3", "C4", "C5", "C6", "C7"}
B = 12
MAX_REPETITIONS = 1
MAX_PAUSES = 3

CHECKPOINTS = [
    Path("experiment_final_smoke_new/runs/slow_L32_run_001.pkl"),
    Path("experiment_final_smoke_new/runs/slow_L32_run_002.pkl"),
]


def flags(lick):
    classes = set(lick[2])
    return {
        "turnaround": "C8" in classes,
        "repetition": "C1" in classes,
        "pause": bool(classes & PAUSE_CODES),
        "classes": sorted(classes),
        "duration": lick[3],
        "path": lick[-1],
    }


def max_nonturnaround_selectable(nonturnaround_flags, rep_cap, pause_cap):
    """DP: maximum number of non-turnaround licks selectable under both caps."""
    if rep_cap < 0 or pause_cap < 0:
        return -1

    # dp[(repetitions_used, pauses_used)] = max number of selected licks
    dp = {(0, 0): 0}

    for f in nonturnaround_flags:
        rep = int(f["repetition"])
        pause = int(f["pause"])
        new_dp = dict(dp)

        for (r_used, p_used), count in dp.items():
            nr = r_used + rep
            npause = p_used + pause
            if nr <= rep_cap and npause <= pause_cap:
                key = (nr, npause)
                new_dp[key] = max(new_dp.get(key, -1), count + 1)

        dp = new_dp

    return max(dp.values(), default=-1)


def analyze_checkpoint(path):
    with path.open("rb") as f:
        record = pickle.load(f)

    licks = record["licks_list"]
    info = [flags(lick) for lick in licks]

    print("=" * 78)
    print(path)
    print("=" * 78)
    print("profile:", record.get("profile"))
    print("L:", record.get("L"))
    print("run:", record.get("run"))
    print("seed:", record.get("seed"))
    print("saved status:", record.get("status"))
    print("saved error:", record.get("error_message"))
    print()

    mask_counts = Counter(
        (
            int(f["turnaround"]),
            int(f["repetition"]),
            int(f["pause"]),
        )
        for f in info
    )

    n_turnaround = sum(f["turnaround"] for f in info)
    n_repetition = sum(f["repetition"] for f in info)
    n_pause = sum(f["pause"] for f in info)
    n_clean_nonturnaround = sum(
        (not f["turnaround"]) and (not f["repetition"]) and (not f["pause"])
        for f in info
    )

    print("Candidate composition")
    print("  total candidates:", len(info))
    print("  turnaround candidates:", n_turnaround)
    print("  repetition candidates:", n_repetition)
    print("  pause-related candidates:", n_pause)
    print("  clean non-turnaround/non-repetition/non-pause candidates:", n_clean_nonturnaround)
    print()

    print("Counts by (turnaround, repetition, pause):")
    for key in sorted(mask_counts):
        print(f"  {key}: {mask_counts[key]}")
    print()

    # With exactly one 2-bar turnaround and all other licks 1 bar,
    # b=12 requires 10 non-turnaround licks plus the turnaround = 11 licks.
    required_nonturnaround = B - 2
    print(
        f"For b={B}, exactly one 2-bar turnaround requires "
        f"{required_nonturnaround} non-turnaround licks."
    )

    turnarounds = [f for f in info if f["turnaround"]]
    nonturnarounds = [f for f in info if not f["turnaround"]]

    best = None
    for idx, t in enumerate(turnarounds, start=1):
        rep_cap = MAX_REPETITIONS - int(t["repetition"])
        pause_cap = MAX_PAUSES - int(t["pause"])
        max_non_t = max_nonturnaround_selectable(nonturnarounds, rep_cap, pause_cap)

        row = {
            "index": idx,
            "path": t["path"],
            "classes": t["classes"],
            "rep_cap": rep_cap,
            "pause_cap": pause_cap,
            "max_non_t": max_non_t,
        }
        if best is None or row["max_non_t"] > best["max_non_t"]:
            best = row

    if best is None:
        print("RESULT: INFEASIBLE -- no turnaround candidate exists.")
        return

    print()
    print("Best possible choice of turnaround under role-count caps:")
    print("  classes:", best["classes"])
    print("  file:", best["path"])
    print("  remaining repetition capacity:", best["rep_cap"])
    print("  remaining pause capacity:", best["pause_cap"])
    print("  maximum selectable non-turnaround licks:", best["max_non_t"])
    print("  required non-turnaround licks:", required_nonturnaround)
    print()

    if best["max_non_t"] >= required_nonturnaround:
        print("RESULT: role/duration constraints are combinatorially FEASIBLE.")
        print(
            "If CBC reports infeasible in round 1, inspect another model constraint, "
            "because the category counts alone admit a valid 12-bar selection."
        )
    else:
        print("RESULT: role/duration constraints are combinatorially INFEASIBLE.")
        print(
            f"Shortfall: at most {best['max_non_t']} eligible non-turnaround licks "
            f"can be selected, but {required_nonturnaround} are required."
        )


for checkpoint in CHECKPOINTS:
    analyze_checkpoint(checkpoint)
    print()
