import numpy as np
from random import random
from interface import *

if __name__ == "__main__":
    X_train = []
    y_train = []
    index_list = [0,0,0]
    total_sample_num = 1
    for sample_num in sample_num_list:
        total_sample_num *= sample_num + 1
    for i in tqdm(range(total_sample_num)):
        if generate_sample_rate != -1:
            assert generate_sample_rate > 0 and generate_sample_rate < 1, "Wrong sampling rate when generating!"
            if random() > generate_sample_rate:
                index_list = iterate(index_list)
                continue
        input_list = [min_list[i] + index_list[i] * gap_list[i] for i in range(input_num)]
        processed_input_list = preprocess(input_list)
        output = numericalMethod(input_list)
        X_train.append(processed_input_list)
        y_train.append(output)
        index_list = iterate(index_list)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)