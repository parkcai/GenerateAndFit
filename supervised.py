from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from dataset import FittingDataset

from interface import *
from evaluate import *
from model import *

if __name__ == "__main__":
    logdir = "log"
    model = FittingModel().to("cuda")
    last_epoch = int(input("where to continue: "))
    end_epoch = int(input("where to end: "))
    
    batch_size = 256
    
    if last_epoch != -1:
        model.load_state_dict(torch.load(logdir+'/%d.pkl' % last_epoch))
        delete_lines_after("log/diary.txt", 2*last_epoch)
    else:
        last_epoch = 0
        delete_lines_after("log/diary.txt", 0)
    

    for e in range(last_epoch,114514000):
        lr = 5e-8
        print('Epoch', e)
        torch.save(model.state_dict(), logdir + '/%d.pkl' % e)
        print('Run validation:')
        mae = evaluate(model, input_range)
        print("Mean Absolute Error is %.4f"%(mae))
        with open("log/diary.txt","a",encoding="utf-8") as file:
            file.write("Epoch: %d MAE:%.4f lr: "%(e, mae) + '{:.2E}'.format(lr) + "\n")
            file.write("\n")
            
        if e == end_epoch:
            break
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        
      
        trainDataset = FittingDataset()
        loader = DataLoader(dataset = trainDataset, batch_size = batch_size, shuffle = True)
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'input_tensor': d[0].cuda()}
            outputs = model(input_dict)
            loss = F.mse_loss(outputs.squeeze(), d[1].cuda())
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batch_size + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        trainDataset = None
        loader = None
    
    