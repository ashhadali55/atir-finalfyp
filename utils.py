import numpy as np




def softmax_grouped(values):
    if len(values) != 6:
        raise ValueError("The input list must contain exactly six values.")
    
    # Separate the values into two groups
    group1 = np.array(values[:2])
    group2 = np.array(values[2:])
    
    # Compute the softmax for each group
    softmax_group1 = np.exp(group1) / np.sum(np.exp(group1))
    softmax_group2 = np.exp(group2) / np.sum(np.exp(group2))
    
    # Combine the results into a single list
    result = list(softmax_group1) + list(softmax_group2)
    
    return result