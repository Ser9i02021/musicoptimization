````md
# MusicXML Lick Sequencing with Integer Programming

This project generates a MusicXML solo by selecting and ordering short MusicXML “licks” from a dataset aiming for the optimal solo generation according to previously defined criteria encounterd in the research paper from Cunha, Subramanian e Herremans (2018).

Pipeline:

1. **Sample** a subset of licks from a dataset folder tree
2. **Parse & classify** each lick (C1–C9) using its first/last note (or rest)
3. Build a **transition cost matrix** (rule-based heuristics)
4. Solve an **Integer Programming** model (PuLP) to pick and order licks subject to constraints
5. **Merge** the chosen MusicXML files into one final MusicXML output

---

## What you get

- ✅ A merged MusicXML file (the generated solo), written to `solutions/ordered_licks_optm_output.xml`
- ✅ A `.pkl` file storing the sampled licks, chosen ordering, objective value, subtour count, and runtime

---

## Requirements

- Python 3.x
- Packages:
  - `numpy`
  - `lxml`
  - `pulp`

Install:

```bash
pip install numpy lxml pulp
````

---

## Expected dataset structure

The code expects a dataset folder called `licks_dataset_sampling/` with (at minimum) the following structure:

```
licks_dataset_sampling/
  FMS/
    regular/
      fast/
      moderate/
      slow/
    repetition/
      fast/
      moderate/
      slow/
    repetition_with_pause/
      fast/
      moderate/
    with_pause/
      fast/
      moderate/
      slow/
  turnaround/
  turnaround_with_pause/
solutions/
```

Each leaf folder contains **MusicXML `.xml`** lick files.

> Note: the folder name used in the code is `FMS` (even though some variables are named `FSM_*`).

---

## How to run

Run the full pipeline:

```bash
python main.py
```

Outputs:

* `solutions/ordered_licks_optm_output.xml`
* `licks_list_1.pkl` (and additional `.pkl` files if you change the loop in `main.py`)

---

## Code overview

### `label_chosen_licks.py` — sampling + classification

**Sampling (`select_N_lick_samples`)**

* Randomly selects `n` licks from the dataset folders.
* Enforces:

  * the **first** and **last** sampled licks are **regular**
  * there is **at least one repetition** lick
  * there is **at least one turnaround** lick

**Classification (`lick_classification`)**
Parses a lick’s MusicXML and returns:

```
[first_note, last_note, classes, duration_in_bars, file_path]
```

Classes:

* **C1**: repetition (filename/path contains `"repetition"`)
* **C2**: ends with rest, duration ≤ 1 beat
* **C3**: ends with rest, duration > 1 beat
* **C4**: ends with rest, duration > 2 beats
* **C5**: starts with rest, duration ≤ 1 beat
* **C6**: starts with rest, duration > 1 beat
* **C7**: starts with rest, duration > 2 beats
* **C8**: turnaround (requires `<lick-label>turnaround</lick-label>` in the first measure)
* **C9**: regular (none of the above)

Duration in bars:

* Turnaround (**C8**) → **2 bars**
* Otherwise → **1 bar**

---

### `cost_matrix_construction.py` — transition scoring

Builds a matrix `p[i][j]` (size = number of sampled licks) based on heuristic rules **T1–T9**.

* Lower costs (especially **negative**) are “better” transitions because the optimization **minimizes** total cost.
* Diagonal transitions are blocked with cost **100**.
* If a transition matches none of the rules, it falls back to **100**.

---

### `optimization.py` — integer programming + subtour elimination

Solves for an ordered path using:

* `x[i,j] ∈ {0,1}`: whether transition `(i → j)` is used
* `y[i] ∈ {0,1}`: whether lick `i` is selected (interior nodes only)

Objective:

* Minimize `Σ p[i][j] * x[i,j]`

Key constraints (as implemented):

* Start node has exactly **one outgoing** arc
* End node has exactly **one incoming** arc
* Interior nodes obey flow constraints linked to `y[i]`
* Total duration (bars) constraint:

  * `Σ duration[i] * y[i] == b`
* Limits:

  * repetition licks ≤ `r` (currently `r = 1`)
  * pause-type licks ≤ `s` (currently `s = 3`)
* Start cannot directly connect to end (`x[start,end] = 0`)
* Turnaround placement constraint:

  * Start must connect to a **non-turnaround** lick
  * The lick right before the end must be a **turnaround** lick
* Iterative subtour elimination:

  * The model is solved repeatedly; if cycles (subtours) appear, new constraints are added to forbid them.

The function returns:

* ordered vertices,
* ordered MusicXML file paths,
* objective value,
* subtour count,
* solve time.

---

### `post_processing.py` — merge MusicXML files

Takes the ordered MusicXML file list and merges their measures into one `score-partwise` output.

* Measures are appended in order and renumbered sequentially.
* The MusicXML DOCTYPE is manually written (ElementTree does not preserve it).

---

### `main.py` — end-to-end runner

Typical run sequence:

1. sample licks
2. build cost matrix
3. optimize
4. save run data to `.pkl`
5. merge ordered MusicXML into `solutions/ordered_licks_optm_output.xml`

---

## Configuration knobs (practical)

You’ll most likely adjust these:

* **Sample size** (number of candidate licks):

  * in `main.py`, the `select_N_lick_samples(...)` call
* **Target bars** `b`:

  * in `main.py`, the `optimize(..., b)` call
* **Repetition / pause limits**:

  * in `optimization.py` (`r = 1`, `s = 3`)

---

## Known issues / important notes (current code)

1. **Signature mismatch in `select_N_lick_samples`**

* `label_chosen_licks.py` defines:

  ```py
  def select_N_lick_samples(n: int):
  ```
* but `main.py` currently calls:

  ```py
  select_N_lick_samples(14, 2)
  ```

If your repo is exactly as pasted, this will raise a `TypeError`.
Fix by either:

* updating the function to accept the second argument, or
* changing the call to `select_N_lick_samples(14)`.

2. **Pause-lick detection bug in optimization**
   In `optimization.py`:

```py
if ("C2" or "C3" or "C4" or "C5" or "C6" or "C7") in licks_list[i][2]:
```

In Python this collapses to `"C2" in ...`, so it only detects C2.
If you want “any of C2..C7”, use:

```py
if any(c in licks_list[i][2] for c in ["C2","C3","C4","C5","C6","C7"]):
```

3. **Verbose printing**
   `build_cost_matrix()` prints the entire matrix. For larger samples, this can be noisy.

---

## Reproducibility

Sampling is random. If you want repeatable outputs, set a seed at the top of your run (e.g., in `main.py` before sampling):

```py
import random
random.seed(0)
```

---

## Contact / authors

* Sergio Bonini - e-mail: mrsergiobonini@gmail.com
