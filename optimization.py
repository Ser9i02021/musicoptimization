from itertools import combinations
from time import perf_counter

from pulp import (
    LpProblem,
    LpVariable,
    lpSum,
    LpMinimize,
    LpBinary,
    PULP_CBC_CMD,
    LpStatus,
    LpSolutionOptimal,
)


def power_set(iterable, min_size, max_size):
    """
    Returns the power set (all subsets) of the given iterable
    as a list of tuples.
    """
    s = list(iterable)
    subsets = []
    for r in range(min_size, max_size):
        for combo in combinations(s, r):
            subsets.append(list(combo))
    return subsets


def _selected_edges(A, x):
    """Return the arcs selected by the current MILP solution."""
    return [
        (i, j)
        for (i, j) in A
        if x[i, j].varValue is not None and x[i, j].varValue > 0.5
    ]


def _extract_main_path_and_subtours(selected_edges, source, sink):
    """
    Decompose the selected arcs into:
      1. the unique source-to-sink path;
      2. all remaining directed cycles (subtours).
    """
    successor = {}

    for i, j in selected_edges:
        if i in successor:
            raise RuntimeError(
                f"Invalid solution structure: vertex {i} has more than one "
                "selected outgoing arc."
            )
        successor[i] = j

    # Main path: source -> ... -> sink
    main_path = []
    current = source
    visited = set()

    while current != sink:
        if current in visited:
            raise RuntimeError(
                "A cycle was encountered while following the source-to-sink path."
            )

        visited.add(current)

        if current not in successor:
            raise RuntimeError(
                f"Invalid solution structure: no selected outgoing arc from "
                f"vertex {current} while constructing the main path."
            )

        nxt = successor[current]
        main_path.append((current, nxt))
        current = nxt

        if len(main_path) > len(selected_edges):
            raise RuntimeError(
                "Main-path extraction exceeded the number of selected arcs."
            )

    # Everything not in the main path must be a directed cycle
    remaining = set(selected_edges) - set(main_path)
    subtours = []

    while remaining:
        first_edge = next(iter(remaining))
        start_vertex = first_edge[0]

        subtour = []
        current = start_vertex
        visited_cycle = set()

        while True:
            if current in visited_cycle:
                if current != start_vertex:
                    raise RuntimeError(
                        "Malformed subtour encountered while extracting cycles."
                    )
                break

            visited_cycle.add(current)

            if current not in successor:
                raise RuntimeError(
                    f"Invalid subtour structure: vertex {current} has no "
                    "selected outgoing arc."
                )

            nxt = successor[current]
            edge = (current, nxt)

            if edge not in remaining:
                raise RuntimeError(
                    "Unexpected selected-edge structure while extracting a subtour."
                )

            subtour.append(edge)
            current = nxt

            if current == start_vertex:
                break

            if len(subtour) > len(remaining):
                raise RuntimeError(
                    "Subtour extraction exceeded the number of remaining arcs."
                )

        for edge in subtour:
            remaining.remove(edge)

        subtours.append(subtour)

    return main_path, subtours


def optimize(
    licks_list,
    p,
    b,
    max_total_time=300,
    max_cut_rounds=1000,
    verbose=False,
):
    """
    Solve the lick-sequencing MILP with iterative subtour elimination.

    Parameters
    ----------
    licks_list : list
        Classified candidate licks. Every entry is an actual musical lick;
        dummy source and sink boundary nodes are created internally.
    p : sequence
        Cost matrix.
    b : int or float
        Required total duration in bars.
    max_total_time : float, optional
        Maximum cumulative time, in seconds, allowed for all solve/cut
        iterations of this optimization instance. Default: 300 seconds.
    max_cut_rounds : int, optional
        Maximum number of MILP solve rounds. Default: 1000.
    verbose : bool, optional
        If True, print one compact progress line per solve round.

    Returns
    -------
    graph_path_vertices_ordered
    file_paths_for_the_ordered_licks_in_the_solution
    objective_value
    subtours_count
    time_taken

    Notes
    -----
    * time_taken is cumulative across ALL MILP solve rounds and subtour-cut
      generation, rather than only the final model.solve().
    * subtours_count is the total number of individual subtours detected
      across all rounds.
    """

    # ------------------------------------------------------------------
    # Nodes and arcs
    # ------------------------------------------------------------------
    # Every entry in licks_list is an ACTUAL candidate lick.  The source
    # and sink below are genuine dummy boundary nodes and therefore do not
    # correspond to MusicXML files, durations, or transition-cost entries.
    actual_nodes = list(range(len(licks_list)))

    if not actual_nodes:
        raise ValueError("licks_list must contain at least one candidate lick.")

    n = len(actual_nodes)
    source = n
    sink = n + 1

    # Only three kinds of arcs are required:
    #   dummy source -> actual lick
    #   actual lick  -> different actual lick
    #   actual lick  -> dummy sink
    # There are deliberately no arcs into source, out of sink, or directly
    # from source to sink.
    actual_arcs = [
        (i, j)
        for i in actual_nodes
        for j in actual_nodes
        if i != j
    ]
    source_arcs = [(source, j) for j in actual_nodes]
    sink_arcs = [(i, sink) for i in actual_nodes]
    A = source_arcs + actual_arcs + sink_arcs

    # Repetition licks
    R = [
        i
        for i in actual_nodes
        if "C1" in licks_list[i][2]
    ]

    # Turnaround licks
    T = [
        i
        for i in actual_nodes
        if "C8" in licks_list[i][2]
    ]

    # Licks with pause. Correctly tests C2, C3, ..., C7 individually.
    pause_codes = ("C2", "C3", "C4", "C5", "C6", "C7")
    P = [
        i
        for i in actual_nodes
        if any(code in licks_list[i][2] for code in pause_codes)
    ]

    # Duration of each ACTUAL lick, in bars.
    c = [lick[3] for lick in licks_list]

    # Constraints (5) and (6)
    r = 1
    s = 3

    # Model
    model = LpProblem("Integer_Programming_Model", LpMinimize)

    x = {
        (i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary)
        for (i, j) in A
    }

    # y exists only for actual candidate licks, never for dummy nodes.
    y = {
        i: LpVariable(f"y_{i}", cat=LpBinary)
        for i in actual_nodes
    }

    # Objective function (1): only transitions between actual licks have a
    # musician-derived transition cost.  Dummy-boundary arcs have zero cost
    # and therefore do not enter the objective.
    model += lpSum(
        p[i][j] * x[i, j]
        for (i, j) in actual_arcs
    )

    # Constraints (2) and (3): one source-to-sink path through selected
    # actual licks.  Dummy source has exactly one outgoing arc and dummy sink
    # exactly one incoming arc.
    model += lpSum(x[source, j] for j in actual_nodes) == 1
    model += lpSum(x[i, sink] for i in actual_nodes) == 1

    for i in actual_nodes:
        model += (
            lpSum(x[i, j] for j in actual_nodes if j != i)
            + x[i, sink]
            == y[i]
        )

    for j in actual_nodes:
        model += (
            x[source, j]
            + lpSum(x[i, j] for i in actual_nodes if i != j)
            == y[j]
        )

    # Constraint (4): total duration counts every selected ACTUAL lick.
    model += lpSum(c[i] * y[i] for i in actual_nodes) == b

    # Constraint (5)
    model += lpSum(y[i] for i in R) <= r

    # Constraint (6)
    model += lpSum(y[i] for i in P) <= s

    # Constraint (7): exactly one turnaround lick is selected, and that
    # unique turnaround is the final actual lick immediately before the
    # dummy sink.  The first actual lick is therefore required to be a
    # non-turnaround lick as well.
    if not T:
        raise ValueError("The candidate set contains no turnaround lick.")

    actual_nodes_minus_T = [i for i in actual_nodes if i not in T]

    # Exactly one turnaround is used anywhere in the 12-bar solution.
    model += lpSum(y[i] for i in T) == 1

    # The path must start with a non-turnaround and end with the unique
    # selected turnaround immediately before the dummy sink.
    model += lpSum(x[source, j] for j in actual_nodes_minus_T) == 1
    model += lpSum(x[i, sink] for i in T) == 1

    # Constraint (8): prohibit choosing both directions between the same two
    # actual licks.  Dummy-boundary arcs cannot form such two-node cycles.
    for i in actual_nodes:
        for j in actual_nodes:
            if i < j:
                model += x[i, j] + x[j, i] <= 1

    # Iterative solution + subtour elimination
    subtours_count = 0
    solve_round = 0

    # Defensive safeguard against adding the same vertex-set cut twice.
    added_subtour_sets = set()

    total_start_time = perf_counter()

    while True:
        solve_round += 1

        if solve_round > max_cut_rounds:
            elapsed = perf_counter() - total_start_time
            raise RuntimeError(
                f"Maximum number of subtour-elimination rounds "
                f"({max_cut_rounds}) reached after {elapsed:.2f} seconds."
            )

        elapsed = perf_counter() - total_start_time
        remaining_time = max_total_time - elapsed

        if remaining_time <= 0:
            raise TimeoutError(
                f"Optimization exceeded the {max_total_time}-second "
                "cumulative time limit."
            )

        # CBC receives only the time still available for this instance.
        solver = PULP_CBC_CMD(
            msg=verbose,
            timeLimit=remaining_time,
            gapRel=0.0,
            threads=1,
            timeMode="elapsed",
        )

        model.solve(solver)

        elapsed = perf_counter() - total_start_time
        status = LpStatus[model.status]

        if verbose:
            print(
                f"Round {solve_round}: "
                f"status={status}, elapsed={elapsed:.2f}s"
            )

        # Do not inspect variable values as if they were an exact optimum
        # unless CBC actually reports an optimal solution.  Recent PuLP
        # versions also expose model.sol_status, which distinguishes a proven
        # optimum from a merely feasible incumbent returned at a time limit.
        solution_status = getattr(model, "sol_status", None)

        proven_optimal = (
            status == "Optimal"
            and (solution_status is None or solution_status == LpSolutionOptimal)
        )

        if not proven_optimal:

            # CBC can report broad status "Optimal" through PuLP while
            # solution_status=2 means only an integer-feasible incumbent
            # was obtained before the time limit.
            if (
                solution_status == 2
                or status == "Not Solved"
                or elapsed >= max_total_time - 1e-6
            ):
                raise TimeoutError(
                    f"Optimization stopped before proven optimality "
                    f"after {elapsed:.2f}s: "
                    f"status='{status}', "
                    f"solution_status={solution_status}, "
                    f"round={solve_round}."
                )

            raise RuntimeError(
                f"Solver terminated without a proven optimum: "
                f"status='{status}', "
                f"solution_status={solution_status}, "
                f"round={solve_round}."
            )

        selected_edges = _selected_edges(A, x)

        if not selected_edges:
            raise RuntimeError(
                "The solver returned an optimal status but no selected arcs."
            )

        main_path, subtours = _extract_main_path_and_subtours(
            selected_edges,
            source,
            sink,
        )

        # No subtours: final connected solution
        if not subtours:
            break

        # Count INDIVIDUAL subcycles, rather than only cut rounds.
        subtours_count += len(subtours)

        if verbose:
            print(
                f"  detected {len(subtours)} subtour(s); "
                f"cumulative total={subtours_count}"
            )

        # Strong vertex-set subtour-elimination cuts:
        #   sum_{i,j in S, i != j} x_ij <= |S| - 1
        # This eliminates every directed cycle entirely contained in S,
        # not only the exact edge ordering found in the current solution.
        cuts_added_this_round = 0

        for subtour in subtours:
            S = frozenset(
                vertex
                for edge in subtour
                for vertex in edge
            )

            if len(S) < 2:
                raise RuntimeError(
                    "A detected subtour has fewer than two vertices."
                )

            if S in added_subtour_sets:
                raise RuntimeError(
                    "A previously eliminated subtour vertex set reappeared; "
                    "the algorithm is not making progress."
                )

            added_subtour_sets.add(S)

            model += lpSum(
                x[i, j]
                for i in S
                for j in S
                if i != j
            ) <= len(S) - 1

            cuts_added_this_round += 1

        if cuts_added_this_round == 0:
            raise RuntimeError(
                "Subtours were detected, but no new subtour-elimination "
                "constraints were added."
            )

        # Enforce the cumulative timeout between rounds too.
        if perf_counter() - total_start_time >= max_total_time:
            raise TimeoutError(
                f"Optimization exceeded the {max_total_time}-second "
                "cumulative time limit."
            )

    # Final statistics and ordered solution
    time_taken = perf_counter() - total_start_time

    if time_taken > max_total_time:
        raise TimeoutError(
            f"Optimization exceeded the {max_total_time}-second "
            "cumulative time limit."
        )

    graph_path_ordered = main_path

    # main_path includes the genuine dummy source and sink internally.  The
    # public return values contain only actual lick vertices/files so that
    # downstream post-processing never tries to interpret a dummy node as a
    # MusicXML lick.
    full_path_vertices = [source]
    full_path_vertices.extend(j for _, j in graph_path_ordered)

    graph_path_vertices_ordered = [
        vertex
        for vertex in full_path_vertices
        if vertex not in (source, sink)
    ]

    file_paths_for_the_ordered_licks_in_the_solution = [
        licks_list[vertex][-1]
        for vertex in graph_path_vertices_ordered
    ]

    # Defensive validation of the public solution representation.  Dummy
    # source/sink nodes exist only inside the MILP and must never escape into
    # post-processing or MusicXML export.
    if any(vertex not in actual_nodes for vertex in graph_path_vertices_ordered):
        raise RuntimeError(
            "Internal error: a dummy/non-lick vertex leaked into the returned path."
        )

    if len(graph_path_vertices_ordered) != len(
        file_paths_for_the_ordered_licks_in_the_solution
    ):
        raise RuntimeError(
            "Internal error: returned lick vertices and file paths have different lengths."
        )

    if any(not path for path in file_paths_for_the_ordered_licks_in_the_solution):
        raise RuntimeError(
            "Internal error: an empty lick file path was returned for post-processing."
        )

    # Validate the musical invariants represented by the returned actual licks.
    returned_duration = sum(c[vertex] for vertex in graph_path_vertices_ordered)
    if abs(returned_duration - b) > 1e-9:
        raise RuntimeError(
            f"Internal error: returned lick path has duration {returned_duration}, "
            f"expected {b}."
        )

    returned_turnarounds = [
        vertex for vertex in graph_path_vertices_ordered if vertex in T
    ]
    if len(returned_turnarounds) != 1:
        raise RuntimeError(
            "Internal error: returned lick path does not contain exactly one turnaround."
        )

    if not graph_path_vertices_ordered or graph_path_vertices_ordered[-1] not in T:
        raise RuntimeError(
            "Internal error: the final returned actual lick is not the turnaround."
        )

    objective_value = model.objective.value()

    if verbose:
        print(
            f"Final solution found after {solve_round} solve round(s) "
            f"in {time_taken:.2f} seconds."
        )
        print(f"Subtours detected: {subtours_count}")
        print(f"Objective function value: {objective_value}")

    return (
        graph_path_vertices_ordered,
        file_paths_for_the_ordered_licks_in_the_solution,
        objective_value,
        subtours_count,
        time_taken,
    )
