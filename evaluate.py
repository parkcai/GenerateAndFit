from interface import *
from random import random
import torch

def calculateMeanErrorOf(model, evaluate_range):
    X_test = []
    y_true = []
    for i in range(evaluate_sample_num):
        input_list = [ vrange[0] + (vrange[1] - vrange[0]) * random() for vrange in evaluate_range]
        processed_input_list = preprocess(input_list)
        output = numericalMethod(input_list)
        X_test.append(processed_input_list)
        y_true.append(output)
    X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
    y_pred = model({'is_training': False, 'input_tensor': X_test})
    y_pred = y_pred.detach().cpu().numpy().flatten()
    return np.mean(np.abs(y_pred - y_true))

def evaluate(model, evaluate_range):
    n = 1
    res = calculateMeanErrorOf(model, evaluate_range)
    while True:
        add = calculateMeanErrorOf(model, evaluate_range)
        new_res = (res * n + add * 1) / (n + 1)
        if abs(new_res - res) / new_res < convergence_criterion:
            return new_res
        else:
            res = new_res
            n += 1
            
            