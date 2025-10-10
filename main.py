from label_chosen_licks import select_N_lick_samples
from cost_matrix_construction import build_cost_matrix
from optimization import optimize
from post_processing import post_process
import os
import pickle

def main():
    # (1) Data Input
    # (2) Selection of Subset of Licks
    for i in range(1, 2):
        filename = f"licks_list_{i}.pkl"
        
        # Select 30 random licks plus initial and final dummy licks
        licks_list = select_N_lick_samples(14, 2) # Select 30 random licks plus inital and final dummy licks
                                                # If the user has specified the desired tempo of the
                                                # solo as 100 BPM (1: moderate), then the subset will consist of
                                                # random licks taken from the original database with this same tempo.

        # (3) Calculation of Cost Matrix
        p = build_cost_matrix(licks_list)

        # (4) Optimization
        graph_path_vertices_ordered, file_paths_for_the_ordered_licks_in_the_solution, obj_val, subt_count, t_t_ = optimize(licks_list, p, 11)

        with open(filename, 'wb') as f:
            pickle.dump({
                'licks_list': licks_list,
                'graph_path_vertices_ordered': graph_path_vertices_ordered,
                'file_paths_for_the_ordered_licks_in_the_solution': file_paths_for_the_ordered_licks_in_the_solution,
                'obj_val': obj_val,
                'subt_count': subt_count,
                'time_taken': t_t_
            }, f)


        # (5) Postprocessing
        # (6) Export as MusicXML
        output_file = "solutions/ordered_licks_optm_output.xml"
        XML_optm_licks_file = post_process(file_paths_for_the_ordered_licks_in_the_solution, output_file)

        print(f"Merged MusicXML written to {XML_optm_licks_file}")


def load_chosen_licks_and_obj_val_and_subt_count_from_a_file(file_number: int) -> list:
    """
    Load the chosen licks from the a pickle file.
    """

    filename = f"licks_list_{file_number}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    
    return data['licks_list'], data['graph_path_vertices_ordered'], data['file_paths_for_the_ordered_licks_in_the_solution'], data['obj_val'], data['subt_count'], data['time_taken']
    
if __name__ == "__main__":
    main()
    
    for i in range(1, 2):
        l, vo, fpo, o, s, t = load_chosen_licks_and_obj_val_and_subt_count_from_a_file(i)
        print(vo, fpo, o, s, t)

licks_chosen_from_full_dataset = [l for l, _, _, _, _, _ in [load_chosen_licks_and_obj_val_and_subt_count_from_a_file(i) for i in range(1, 2)]]
print(f"Chosen licks from full dataset: {licks_chosen_from_full_dataset}")

vertices_ordered = [vo for _, vo, _, _, _, _ in [load_chosen_licks_and_obj_val_and_subt_count_from_a_file(i) for i in range(1, 2)]]
print(f"Vertices ordered: {vertices_ordered}")

licks_used_in_the_solution = [fpo for _, _, fpo, _, _, _ in [load_chosen_licks_and_obj_val_and_subt_count_from_a_file(i) for i in range(1, 2)]]
print(f"Licks used in the solution: {licks_used_in_the_solution}")

average_obj_val = sum(o for _, _, _, o, _, _ in [load_chosen_licks_and_obj_val_and_subt_count_from_a_file(i) for i in range(1, 2)]) / 1
print(f"Average objective value: {average_obj_val}")

average_subt_count = sum(s for _, _, _, _, s, _ in [load_chosen_licks_and_obj_val_and_subt_count_from_a_file(i) for i in range(1, 2)]) / 1
print(f"Average number of subtours: {average_subt_count:.2f}")

average_time_taken = sum(t for _, _, _, _, _, t in [load_chosen_licks_and_obj_val_and_subt_count_from_a_file(i) for i in range(1, 2)]) / 1
print(f"Average time taken: {average_time_taken:.2f} seconds")
