# MusicXML Lick Sequencing with Integer Programming

This project generates 12-bar blues guitar solos by selecting and ordering short MusicXML “licks” with a mixed-integer linear programming (MILP) model.

The implementation is based on the optimization framework investigated by Cunha, Subramanian, and Herremans (2018). Candidate licks are represented as vertices in a directed graph, transitions between licks receive rule-based costs, and the optimization model searches for a minimum-cost sequence satisfying musical and structural constraints.

The main pipeline is:

1. **Sample** a feasible candidate set of licks from the dataset
2. **Parse and classify** each lick into categories C1–C9
3. Build a **transition cost matrix**
4. Build and solve a **MILP**
5. Iteratively eliminate disconnected **subtours**
6. Recover the ordered sequence of actual licks
7. Optionally **merge** the selected MusicXML files into a final MusicXML solo

---

## What you get

Depending on the runner being used, the project can produce:

- ✅ An ordered 12-bar sequence of MusicXML licks
- ✅ A merged MusicXML solo
- ✅ `.pkl` checkpoints containing the sampled candidate set and optimization results
- ✅ Objective value
- ✅ Number of subtours eliminated
- ✅ Optimization runtime
- ✅ Run status (`OK`, `TIMEOUT`, or `ERROR`)
- ✅ CSV and text/LaTeX summaries for simulation experiments

The standard MusicXML output is written under:

```text
solutions/
```

---

## Requirements

- Python 3.x
- Packages:
  - `numpy`
  - `lxml`
  - `pulp`
  - `pandas` for the experiment/simulation scripts

Install with:

```bash
pip install numpy lxml pulp pandas
```

The optimization experiments use CBC through PuLP.

---

## Expected dataset structure

The code expects a dataset folder called `licks_dataset_sampling/` with at least the following structure:

```text
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

Each leaf folder contains MusicXML `.xml` lick files.

The speed/profile convention used by the selector is:

```text
SMF = 0  -> Slow
SMF = 1  -> Moderate
SMF = 2  -> Fast
```

Turnaround folders are not speed-specific and are therefore available to all three profiles.

---

## Lick representation

Each MusicXML file is classified by `lick_classification()` and represented internally as:

```python
[first_note, last_note, classes, duration_in_bars, file_path]
```

The classes are:

- **C1**: repetition
- **C2**: ends with a rest of duration ≤ 1 beat
- **C3**: ends with a rest of duration > 1 beat
- **C4**: ends with a rest of duration > 2 beats
- **C5**: starts with a rest of duration ≤ 1 beat
- **C6**: starts with a rest of duration > 1 beat
- **C7**: starts with a rest of duration > 2 beats
- **C8**: turnaround
- **C9**: regular

A lick can belong to more than one category. For example, a turnaround can also be pause-related.

### Duration

The current model assumes:

```text
C8 turnaround -> 2 bars
all other licks -> 1 bar
```

Therefore, a 12-bar solution containing exactly one turnaround consists of:

```text
10 one-bar licks + 1 two-bar turnaround = 12 bars
```

or 11 actual licks in total.

---

## `label_chosen_licks.py` — sampling and classification

### `select_N_lick_samples(n, SMF)`

The selector constructs a candidate set of exactly `n` **actual musical licks**.

There are no longer any sampled licks reserved as artificial first or last nodes. Start and end conditions are handled inside the optimization model using separate dummy nodes.

The candidate pool contains the relevant Slow, Moderate, or Fast licks together with the turnaround folders.

Sampling is performed without replacement.

Because arbitrary random candidate sets can occasionally make the MILP structurally infeasible, the selector performs feasibility-aware rejection sampling:

1. draw a candidate set of size `n`;
2. test whether the set is capable of satisfying the fixed structural constraints;
3. accept it if feasible;
4. otherwise redraw another set of size `n`.

The feasibility test accounts for:

- total duration of 12 bars;
- exactly one turnaround;
- at most one repetition lick;
- at most three pause-related licks;
- overlaps among lick categories.

Thus the experiment samples uniformly from candidate sets **conditional on structural feasibility**.

This prevents candidate-generation artifacts from being confused with optimization failures.

### `lick_classification()`

The classifier parses the MusicXML file and determines:

- first note/rest;
- last note/rest;
- C1–C9 membership;
- duration in bars;
- original file path.

---

## `cost_matrix_construction.py` — transition scoring

The transition-cost matrix contains the cost of placing lick `j` immediately after lick `i`.

```python
p[i][j]
```

Costs are assigned according to the transition rules implemented in the project.

Important interpretation:

- lower cost is better;
- negative values represent favorable transitions under the scoring system;
- the optimization therefore **minimizes** total transition cost.

Diagonal/self-transitions are not used as valid musical transitions.

---

## `optimization.py` — MILP formulation

The optimization model searches for a single ordered sequence of actual licks.

### Dummy source and sink

Two additional vertices are created internally:

```text
source -> selected musical licks -> sink
```

These are **dummy boundary nodes**.

They:

- do not correspond to MusicXML files;
- have no musical duration;
- are not part of the sampled candidate set;
- are not returned in the final musical sequence;
- are not exported to MusicXML.

If `L` candidate licks are sampled, all `L` remain genuine candidate musical licks.

---

## Decision variables

For actual candidate lick `i`:

```text
y[i] = 1 if lick i is selected
```

For permitted directed arcs:

```text
x[i,j] = 1 if arc i -> j is selected
```

The graph also contains arcs involving the dummy source and sink for path construction.

---

## Objective

The model minimizes total transition cost:

```text
min Σ p[i][j] x[i,j]
```

Only transitions between actual musical licks contribute musical transition cost. Dummy boundary arcs do not represent lick-to-lick musical transitions.

Lower objective values indicate better solutions under the implemented cost system.

---

## Main structural constraints

### Single path

The dummy source has exactly one outgoing arc:

```text
source -> first actual lick
```

The dummy sink has exactly one incoming arc:

```text
last actual lick -> sink
```

Selected actual licks have matching incoming and outgoing flow.

---

### Fixed 12-bar duration

For experiments on 12-bar blues:

```text
b = 12
```

and the model enforces:

```text
Σ duration[i] * y[i] = 12
```

All actual selected licks are included in this duration calculation.

---

### Repetition limit

At most one selected lick can belong to C1:

```text
number of repetition licks <= 1
```

---

### Pause-related limit

At most three selected licks may belong to C2–C7:

```text
number of pause-related licks <= 3
```

Pause detection uses:

```python
any(
    code in licks_list[i][2]
    for code in ("C2", "C3", "C4", "C5", "C6", "C7")
)
```

---

### Exactly one turnaround

The model explicitly requires:

```text
number of selected C8 licks = 1
```

The unique turnaround must also be the final actual musical lick:

```text
turnaround -> dummy sink
```

The first actual lick cannot be a turnaround.

Consequently, with `b = 12`, a feasible final solo normally contains:

```text
10 one-bar non-turnaround licks
+
1 two-bar turnaround
=
12 bars
```

---

## Subtour elimination

Flow constraints alone can produce disconnected cycles in addition to the source-to-sink path.

The solver therefore uses iterative subtour elimination.

After each MILP solve:

1. extract the source-to-sink path;
2. detect disconnected cycles;
3. add a subtour-elimination constraint for each detected cycle;
4. solve the strengthened MILP again;
5. repeat until no subtours remain.

For a detected vertex set `S`, the added inequality is of the form:

```text
Σ x[i,j] <= |S| - 1
```

for arcs contained entirely in `S`.

The reported `subtours_count` is the cumulative number of individual subtours detected and eliminated across all solve rounds.

---

## Optimality and solver time limit

Experimental runs require a **proven optimum**.

PuLP/CBC can sometimes return a feasible incumbent when the solver reaches its time limit without proving optimality. Such a solution is not treated as optimal.

The experiments currently use a cumulative per-instance wall-clock limit of:

```python
MAX_TOTAL_TIME = 300
```

seconds.

If CBC reaches the time limit without proving optimality, the observation is recorded as:

```text
TIMEOUT
```

rather than as a program error.

A genuine infeasibility or unexpected implementation failure remains:

```text
ERROR
```

This distinction is important in the computational experiments because difficult instances are themselves part of the observed solver behavior.

---

## Returned solution

`optimize()` returns:

```python
(
    graph_path_vertices_ordered,
    file_paths_for_the_ordered_licks_in_the_solution,
    objective_value,
    subtours_count,
    time_taken,
)
```

Both:

```python
graph_path_vertices_ordered
```

and:

```python
file_paths_for_the_ordered_licks_in_the_solution
```

contain **actual musical licks only**.

The dummy source and sink are removed before the solution is returned.

The optimizer also performs consistency checks on successful solutions, including:

- total musical duration equals `b`;
- exactly one turnaround is selected;
- the turnaround is the last actual lick;
- returned vertices correspond to actual candidates;
- dummy nodes do not leak into MusicXML processing.

---

## `post_processing.py` — MusicXML merging

`post_process()` receives the ordered paths of the selected actual lick files.

It merges their MusicXML measures into one `score-partwise` document.

The measures are:

- appended in optimized order;
- renumbered sequentially.

Only actual lick files are passed to this stage. The dummy source and sink exist only inside the MILP formulation.

---

## Running a single generated solo

A standard end-to-end run follows the sequence:

```text
candidate generation
        ↓
classification
        ↓
cost matrix
        ↓
MILP optimization
        ↓
subtour elimination
        ↓
ordered actual lick files
        ↓
MusicXML post-processing
```

Depending on the version of `main.py`, run:

```bash
python main.py
```

---

## Computational experiments

The scaling experiments vary candidate-set size `L` across three profiles.

Current experimental grid:

```python
EXPERIMENTS = [
    ("Slow",     0, [32, 43]),
    ("Moderate", 1, [32, 62, 160]),
    ("Fast",     2, [32, 62]),
]
```

For the full study:

```python
NUM_RUNS = 100
```

giving:

```text
7 configurations x 100 runs = 700 instances
```

Each run uses a fresh feasible candidate set.

Recorded quantities include:

- wall-clock optimization time;
- number of subtours eliminated;
- objective value;
- completion status.

Typical statuses are:

```text
OK
TIMEOUT
ERROR
```

---

## Checkpoints and interrupted experiments

Each simulation is stored separately as a `.pkl` checkpoint.

This allows a long experiment to be interrupted and restarted without losing completed runs.

A resumed experiment can therefore produce messages such as:

```text
Run 001/100 checkpoint -> OK
Run 002/100 checkpoint -> OK
...
```

rather than solving those observations again.

The checkpoint normally preserves information such as:

```text
profile
SMF
L
run number
random seed
sampled licks
ordered solution
objective value
subtour count
runtime
status
error information
```

This is particularly useful for the largest configuration, `Moderate, L=160`, where occasional instances can be substantially more computationally demanding.

---

## Treatment of timeouts in experiments

Runs that fail to reach proven optimality within the fixed 300-second cumulative limit are not replaced by newly sampled instances.

They are preserved as `TIMEOUT` observations.

For example, a final experiment may report:

```text
successful = 98
timeouts = 2
errors = 0
```

Objective-value and subtour statistics should be computed from runs for which an optimum was proven.

The timeout frequency should also be reported because it is part of the computational behavior of the formulation.

---

## Reproducibility

The experiment runner uses deterministic seeds derived from:

```text
base seed
profile / SMF
candidate-set size L
run number
```

This allows a particular experiment instance to be reproduced without resampling a different candidate set.

Both Python's `random` module and NumPy are seeded before candidate generation.

For ad hoc runs, a simple fixed seed can also be used:

```python
import random
import numpy as np

random.seed(0)
np.random.seed(0)
```

---

## Practical configuration parameters

Frequently adjusted parameters include:

### Number of candidate licks

```python
L
```

Typical values used in the experiments are:

```text
Slow:      32, 43
Moderate:  32, 62, 160
Fast:      32, 62
```

### Target duration

```python
b = 12
```

### Repetition limit

```python
r = 1
```

### Pause-related limit

```python
s = 3
```

### Cumulative solver limit

```python
MAX_TOTAL_TIME = 300
```

seconds per sampled instance.

### Maximum subtour-cut rounds

A separate defensive limit is also maintained for the iterative subtour-elimination procedure.

---

## Important implementation notes

### Candidate-set feasibility

Randomly drawing `L` licks does not automatically imply that a 12-bar solution satisfying all role constraints exists.

For example, a small candidate set dominated by pause-related licks may be unable to provide:

```text
10 non-turnaround one-bar licks
```

while respecting:

```text
pause-related licks <= 3
```

For this reason, candidate generation rejects structurally infeasible candidate sets before optimization.

---

### Feasible incumbent is not necessarily optimal

When CBC reaches its time limit, it may have found a feasible integer solution without having proved that it is optimal.

The code therefore distinguishes:

```text
proven optimal solution
```

from:

```text
integer-feasible incumbent
```

and only the former receives status `OK`.

---

### Computational variability

MILP solution time can vary substantially even between instances with the same `L`.

In particular, large `Moderate, L=160` instances may require many successive MILP solves and subtour cuts, producing a heavy right tail in runtime.

For this reason, experiments report both means and medians and retain timeout information.

---

## Repository workflow

The intended workflow is:

```text
Data Input
    ↓
Lick Classification
    ↓
Feasible Candidate Sampling
    ↓
Transition Cost Matrix
    ↓
MILP Optimization
    ↓
Iterative Subtour Elimination
    ↓
Ordered Actual Lick Sequence
    ↓
MusicXML Post-processing / Export
```

---

## Contact / authors

- **Sergio Bonini** — mrsergiobonini@gmail.com
- **Sergio Da Silva** - professorsergiodasilva@gmail.com
