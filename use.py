from interface import *
from model import *
from evaluate import *
import torch

'''
为使用模型，你需要手动拷贝所需的pkl文件，和诸多py文件放在同一个父文件夹下，并将其重命名为"mymodel.pkl"
'''

if __name__ == "__main__":
    model = FittingModel().to("cuda")
    model.load_state_dict(torch.load("mymodel.pkl"))
    
    input_list = [0.9, 4, 4]
    true_output = numericalMethod(input_list)
    processed_input_list = preprocess(input_list)
    processed_input_list = torch.tensor(processed_input_list, dtype=torch.float32).cuda()
    input_dict = {"is_training":False, "input_tensor": processed_input_list}
    pred_output = model(input_dict)
    pred_output = pred_output.detach().cpu().numpy().flatten()[0]
    print("Predicted output: %.4f"%pred_output)
    print("True output: %.4f"%true_output)
    
    
    # use_range = [
    #     (-0.5,1.5),
    #     (2.5,8.5),
    #     (1.5,3.5),
    # ]
    # print("MAE in use range: %.4f"%evaluate(model, use_range))
