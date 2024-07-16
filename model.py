from torch import nn
from interface import *
neural_no = 600
class FittingModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Linear(processed_input_num,neural_no),
            nn.ReLU(),
            nn.Linear(neural_no,neural_no),
            nn.ReLU(),
            nn.Linear(neural_no,neural_no),
            nn.ReLU(),
            nn.Linear(neural_no,neural_no),
            nn.ReLU(),
            nn.Linear(neural_no,neural_no),
            nn.ReLU(),
            nn.Linear(neural_no,neural_no),
            nn.ReLU(),
            nn.Linear(neural_no,1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        self.train(mode = input_dict["is_training"])
        input_tensor = input_dict["input_tensor"]
        return self._tower(input_tensor)