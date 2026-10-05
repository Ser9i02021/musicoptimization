import numpy as np
from lxml import etree
from label_chosen_licks import select_N_lick_samples 


def build_cost_matrix(licks_list: list):
    #licks_list = select_N_lick_samples(num_licks)
    '''
    for i in range(len(licks_list)):
        print(licks_list[i])
    '''
    cost_matrix = np.full((len(licks_list), len(licks_list)), 0)

    for i in range(len(licks_list)):
        for j in range(len(licks_list)):
            if i != j:
                all_false = True
                # T1
                if "C1" in licks_list[i][2] and ("C1" not in licks_list[j][2]):
                    cost_matrix[i][j] += -50
                    #print("cost_matrix[{}][{}]: T1".format(i, j))
                    all_false = False
                # T2
                if not (("C2" in licks_list[i][2]) or ("C3" in licks_list[i][2]) or ("C4" in licks_list[i][2])) and ("C2" in licks_list[j][2]):
                    cost_matrix[i][j] += -15
                    #print("cost_matrix[{}][{}]: T2".format(i, j))
                    all_false = False
                # T3
                if ("C2" in licks_list[i][2]) and (("C5" in licks_list[j][2]) or ("C6" in licks_list[j][2]) or ("C7" in licks_list[j][2])):
                    cost_matrix[i][j] += -15
                    #print("cost_matrix[{}][{}]: T3".format(i, j))
                    all_false = False
                # T4
                if licks_list[i][1][0] != "R" and licks_list[j][1][0] != "R": # Certify that none of the licks are a rest
                    if (((ord(licks_list[i][1][0]) - 2) % 7) == ((ord(licks_list[i][1][0]) - 3) % 7)) and  licks_list[i][1][1] == licks_list[j][1][1]:
                        cost_matrix[i][j] += -15
                        #print("cost_matrix[{}][{}]: T4".format(i, j))
                        all_false = False
                # T5
                if licks_list[i][1] == licks_list[j][0]:
                    cost_matrix[i][j] += -15
                    #print("cost_matrix[{}][{}]: T5".format(i, j))
                    all_false = False
                # T6
                if "C3" in licks_list[i][2] and "C6" in licks_list[j][2]:
                    cost_matrix[i][j] += 25
                    #print("cost_matrix[{}][{}]: T6".format(i, j))
                    all_false = False
                # T7
                if "C4" in licks_list[i][2] and ("C4" not in licks_list[j][2]):
                    cost_matrix[i][j] += 25
                    #print("cost_matrix[{}][{}]: T7".format(i, j))
                    all_false = False
                # T8
                if "C7" in licks_list[j][2]:
                    cost_matrix[i][j] += 25
                    #print("cost_matrix[{}][{}]: T8".format(i, j))
                    all_false = False
                # T9 (May be removed later. It is being used to identify if the transition does not fit in any of the previous categories) 
                if all_false:
                    cost_matrix[i][j] = 100
                    #print("cost_matrix[{}][{}]: T9".format(i, j))
            else:
                cost_matrix[i][j] = 100
    '''
    for i in range(len(licks_list)):
        for j in range(len(licks_list)):
            print(cost_matrix[i][j], end=" ")
        print()
    '''
    return cost_matrix
