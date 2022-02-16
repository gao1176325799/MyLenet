from save_load_network import load_all, FILE
from alex_lib import torch
from config import DEVICE
from data_loaded import valid_loader
import matplotlib.pyplot as plt
loaded_model=load_all(FILE)
#loaded_model.eval()
print(loaded_model)

with torch.no_grad():
    for name, param in loaded_model.named_parameters():
        if name=='conv1.weight':
            out_f,in_f,k_H,k_W=param.shape
            print(out_f,in_f,k_H,k_W)
            total_num_weight=out_f*in_f*k_H*k_W
            print(param[0][0][0][0].detach().to('cpu').numpy())
            param[0][0][0][0]=0.1
            print(param[0][0][0][0].detach().to('cpu').numpy())
            for i in range(out_f):
                for k in range(in_f):
                    for j in range(k_H):
                        for l in range(k_W):
                            param[i][k][j][l]=0.1





test0=1
test1=0
if test0:
    with torch.no_grad():
        n_correct=0
        n_samples=0
        for i,(images, labels) in enumerate (valid_loader):
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            outputs=loaded_model(images)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
        acc=100*n_correct/n_samples#->9860/10000
        print(f'Accuracy of the network:{acc}%')

#### EXTRACT and MODIFY weight




def weight_ex_ex(model):
    return model
if test1:
    with torch.no_grad():
        n_correct=0
        n_samples=0
        for images,labels in valid_loader:
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            new_model=weight_ex_ex(loaded_model)
            outputs=loaded_model(images)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
        acc=100*n_correct/n_samples
        print(f'Accuracy of the network:{acc}%, correct:{n_correct}, total:{n_samples}')
