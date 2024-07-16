from copy import deepcopy
import numpy as np
from tqdm import tqdm

'''
----------------------------------需要更改的部分----------------------------------
'''
# 自变量个数
input_num = 3

# 各自变量变化范围
input_range = [
    (0,1),
    (3,8),
    (2,3),
]

# 各自变量采样数
sample_num_list = [
    200,
    100,
    200,
]

# 生成训练集时是否部分采样及采样率
# 值取-1，表示不部分采样，全部生成
# 值取一小数，即为部分采样的采样率
generate_sample_rate = 0.5

# 评估模型时的单次采样数
evaluate_sample_num = 10000

# 评估模型时的收敛判据
convergence_criterion = 1e-5


# 待拟合的input_num个自变量的单值函数
def numericalMethod(input_list):
    assert len(input_list) == input_num, "Wrong input variable number!"
    x = input_list[0]
    y = input_list[1]
    z = input_list[2]
    return x ** 2 + x * y * z + max(y,z) - 3 * max(0.2, x ** 2) + np.exp(2*x + 3*z)

# 预处理方法
# 模型学习的是从preprocess(input_list)到output的映射
processed_input_num = 4
def preprocess(input_list):
    input_list = deepcopy(input_list)
    x = input_list[0]
    z = input_list[2]
    input_list.append(np.exp(2 * x + 3 * z))
    return input_list

'''
-------------------------------------【END】-------------------------------------
'''


input_range = [ (float(vrange[0]), float(vrange[1])) for vrange in input_range]
min_list = [vrange[0] for vrange in input_range]
gap_list = [(input_range[i][1] - input_range[i][0]) / sample_num_list[i] for i in range(input_num)]

def iterate(index_list):
    if index_list == sample_num_list: return "END"
    res_list = deepcopy(index_list)
    for i in range(len(res_list)):
        if res_list[i] < sample_num_list[i]:
            res_list[i] += 1
            return res_list
        else:
            res_list[i] = 0 

def delete_lines_after(file_path, line_number):  
    # 创建一个列表来存储要保留的行  
    lines_to_keep = []  
  
    # 尝试打开文件并读取内容  
    try:  
        with open(file_path, 'r', encoding='utf-8') as file:  
            # 读取文件的每一行，直到达到指定的行号  
            for i, line in enumerate(file, 1):  
                if i <= line_number:  
                    lines_to_keep.append(line)  
  
        # 如果需要覆盖原文件，可以使用'w'模式打开文件  
        # 否则，可以写入一个新文件  
        with open(file_path, 'w', encoding='utf-8') as file:  
            # 将要保留的行写回到文件  
            for line in lines_to_keep:  
                file.write(line)  
  
    except FileNotFoundError:  
        print(f"File {file_path} not found, will automatically create a new one!")  
    except Exception as e:  
        print(f"发生错误: {e}")  

if __name__ == "__main__":
    # the following codes are merely used for testing
    assert len(input_range) == input_num, "Invalid input range!"
    assert len(sample_num_list) == input_num, "Invalid sample num list!"
    sample_num_list = [3,3,3]
    total_sample_num = 1
    for sample_num in sample_num_list:
        total_sample_num *= sample_num + 1
    index_list = [0,0,0]
    for i in tqdm(range(total_sample_num)):
        print(index_list)
        index_list = iterate(index_list)