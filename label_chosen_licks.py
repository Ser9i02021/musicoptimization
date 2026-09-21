import os
import random

from lxml import etree


# Function to extract pitch information
def get_note_info(note):
    pitch = note.find("pitch")
    if pitch is not None:
        step = pitch.find("step").text
        octave = pitch.find("octave").text
        alter = pitch.find("alter")

        if alter is not None:
            alter_value = int(alter.text)
            if alter_value == 1:
                step += "#"
            elif alter_value == -1:
                step += "♭"

        return f"{step}{octave}"

    duration = note.find("duration").text
    return "Rest w/ the duration (in beats) of " + duration


def lick_classification(lick_file_path: str):
    """Extract boundary-note information and classify one lick from C1 to C9."""
    tree = etree.parse(lick_file_path)
    root = tree.getroot()

    notes = root.findall(".//note")
    if not notes:
        raise ValueError(f"No notes found in XML file: {lick_file_path}")

    first_note = get_note_info(notes[0])
    last_note = get_note_info(notes[-1])

    lick_classes = []
    all_false = True

    # C1 (Repetition)
    if "repetition" in lick_file_path:
        lick_classes.append("C1")
        all_false = False

    # C2 (EwP <= 1)
    if last_note[0] == "R" and int(last_note[35:]) <= 1:
        lick_classes.append("C2")
        all_false = False

    # C3 (EwP > 1)
    if last_note[0] == "R" and int(last_note[35:]) > 1:
        lick_classes.append("C3")
        all_false = False

    # C4 (EwP > 2)
    if last_note[0] == "R" and int(last_note[35:]) > 2:
        lick_classes.append("C4")
        all_false = False

    # C5 (SwP <= 1)
    if first_note[0] == "R" and int(first_note[35:]) <= 1:
        lick_classes.append("C5")
        all_false = False

    # C6 (SwP > 1)
    if first_note[0] == "R" and int(first_note[35:]) > 1:
        lick_classes.append("C6")
        all_false = False

    # C7 (SwP > 2)
    if first_note[0] == "R" and int(first_note[35:]) > 2:
        lick_classes.append("C7")
        all_false = False

    # C8 (Turnaround)
    measure = root.find(".//measure")
    lick_label = (
        measure.find("lick-label").text
        if measure is not None and measure.find("lick-label") is not None
        else None
    )
    if lick_label == "turnaround":
        lick_classes.append("C8")
        all_false = False

    # C9 (Regular)
    if all_false:
        lick_classes.append("C9")

    # Turnarounds occupy two bars; every other lick occupies one bar.
    duration_in_bars = 2 if "C8" in lick_classes else 1

    return [
        first_note,
        last_note,
        lick_classes,
        duration_in_bars,
        lick_file_path,
    ]


# Selection of N actual candidate licks from the dataset.
#
# Every one of the n sampled files is an actual candidate lick.  The optimizer
# creates genuine dummy source/sink nodes internally.
#
# Sampling protocol
# -----------------
# A size-n subset is drawn uniformly without replacement from the eligible
# candidate pool.  If that subset cannot satisfy the model's fixed structural
# constraints for the 12-bar experiments, it is rejected and another uniform
# size-n subset is drawn.  Thus accepted samples are uniform conditional on
# structural feasibility; no category is manually forced into a position.
#
# Fixed structural conditions mirrored from optimization.py:
#   - total selected duration = 12 bars
#   - exactly one turnaround is selected
#   - at most one repetition lick is selected
#   - at most three pause-related licks are selected
#
# SMF convention:
#   0 -> slow
#   1 -> moderate
#   2 -> fast
#
# SMF applies only to speed-labelled FMS licks. Turnaround licks do not have
# speed-specific folders in the current dataset and therefore remain eligible
# for every SMF value.

_PAUSE_CODES = ("C2", "C3", "C4", "C5", "C6", "C7")
_CLASSIFICATION_CACHE = {}


def _classify_cached(path):
    """Classify a lick once and reuse the result across repeated experiments."""
    if path not in _CLASSIFICATION_CACHE:
        _CLASSIFICATION_CACHE[path] = lick_classification(path)
    return _CLASSIFICATION_CACHE[path]


def _is_structurally_feasible(
    classified_licks,
    total_bars=12,
    max_repetition=1,
    max_pause=3,
):
    """
    Exact feasibility screen for the role/duration constraints used by the MILP.

    This is a small 0/1 dynamic program over states
        (bars, turnaround_count, repetition_count, pause_count).
    It correctly handles category overlaps, e.g. a turnaround that is also
    pause-related.
    """
    # Start with the empty subset.
    states = {(0, 0, 0, 0)}

    for lick in classified_licks:
        classes = set(lick[2])
        duration = int(lick[3])
        is_turnaround = int("C8" in classes)
        is_repetition = int("C1" in classes)
        is_pause = int(any(code in classes for code in _PAUSE_CODES))

        next_states = set(states)

        for bars, turnarounds, repetitions, pauses in states:
            nb = bars + duration
            nt = turnarounds + is_turnaround
            nr = repetitions + is_repetition
            npause = pauses + is_pause

            if nb > total_bars:
                continue
            if nt > 1:
                continue
            if nr > max_repetition:
                continue
            if npause > max_pause:
                continue

            next_states.add((nb, nt, nr, npause))

        states = next_states

    return any(
        bars == total_bars and turnarounds == 1
        for bars, turnarounds, repetitions, pauses in states
    )


def select_N_lick_samples(n: int, SMF: int, max_sampling_attempts: int = 10000):
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")

    if SMF not in (0, 1, 2):
        raise ValueError("SMF must be 0 (slow), 1 (moderate), or 2 (fast).")

    if not isinstance(max_sampling_attempts, int) or max_sampling_attempts < 1:
        raise ValueError("max_sampling_attempts must be a positive integer.")

    speed = {0: "slow", 1: "moderate", 2: "fast"}[SMF]

    fms_root = r"licks_dataset_sampling/FMS"
    regular_dir = os.path.join(fms_root, "regular", speed)
    repetition_dir = os.path.join(fms_root, "repetition", speed)
    with_pause_dir = os.path.join(fms_root, "with_pause", speed)

    repetition_with_pause_dir = None
    if SMF == 2:
        repetition_with_pause_dir = os.path.join(
            fms_root, "repetition_with_pause", "fast"
        )
    elif SMF == 1:
        repetition_with_pause_dir = os.path.join(
            fms_root, "repetition_with_pause", "moderate"
        )

    turnaround_dir = r"licks_dataset_sampling/turnaround"
    turnaround_with_pause_dir = r"licks_dataset_sampling/turnaround_with_pause"

    def files_in(directory):
        if directory is None or not os.path.isdir(directory):
            return []
        return [
            os.path.join(directory, filename)
            for filename in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, filename))
        ]

    regular_files = files_in(regular_dir)

    repetition_files = files_in(repetition_dir)
    repetition_files += files_in(repetition_with_pause_dir)

    with_pause_files = files_in(with_pause_dir)

    turnaround_files = files_in(turnaround_dir)
    turnaround_files += files_in(turnaround_with_pause_dir)

    # One unique pool of ACTUAL candidate files.  No first/last files are
    # reserved and no role category is forced into the draw.
    candidate_pool = list(
        dict.fromkeys(
            regular_files
            + repetition_files
            + with_pause_files
            + turnaround_files
        )
    )

    if len(candidate_pool) < n:
        raise ValueError(
            f"Not enough distinct eligible lick files for n={n} and "
            f"SMF={SMF} ({speed}). Only {len(candidate_pool)} are available."
        )

    # Fail early if even the complete eligible pool cannot support the model's
    # structural constraints.  This distinguishes a dataset/configuration
    # problem from an unlucky random draw.
    full_pool_classified = [_classify_cached(path) for path in candidate_pool]
    if not _is_structurally_feasible(full_pool_classified):
        raise ValueError(
            f"The complete eligible pool for SMF={SMF} ({speed}) cannot satisfy "
            "the 12-bar role/duration constraints."
        )

    # Rejection sampling. random.sample() is uniform over size-n subsets, and
    # acceptance depends only on feasibility, so the returned subset is uniform
    # conditional on being structurally feasible.
    for _ in range(max_sampling_attempts):
        sampled_paths = random.sample(candidate_pool, n)
        classified = [_classify_cached(path) for path in sampled_paths]

        if _is_structurally_feasible(classified):
            return classified

    raise RuntimeError(
        f"Unable to draw a structurally feasible sample of n={n} candidates "
        f"for SMF={SMF} ({speed}) after {max_sampling_attempts} attempts. "
        "Increase max_sampling_attempts or inspect the pool composition."
    )
