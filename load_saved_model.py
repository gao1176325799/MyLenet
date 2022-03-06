from Operation_LSM import *
from Utility_LSM import *
from save_load_network import load_all, FILE
from NN_lib import torch,np
from config import DEVICE
from data_loaded import valid_loader
import matplotlib.pyplot as plt
loaded_model=load_all(FILE)
loaded_model.eval()
print(loaded_model)


run_without_change(loaded_model)
para_m(loaded_model.named_parameters(),loaded_model)



test0=0
if test0:
    with torch.no_grad():
        for name, param in loaded_model.named_parameters():
            if name=='conv1.weight':
                plot(name,param)
                sum1=0
                sum2=0
                out_f,in_f,k_H,k_W=param.shape
                print(out_f,in_f,k_H,k_W)
                total_num_weight=out_f*in_f*k_H*k_W
                
                print(param[0][0][0][0].detach().to('cpu').numpy())
                p=param.detach().to('cpu').numpy()
                p=np.reshape(p,(1,total_num_weight))
                for i in range(total_num_weight):
                    sum1+=p[0][i]
                avg1=sum1/total_num_weight
                print('before modify the avg is:',avg1)
                for i in range(out_f):
                    for k in range(in_f):
                        for j in range(k_H):
                            for l in range(k_W):
                                param[i][k][j][l]=(param[i][k][j][l]+2.5)/3.5
                p=param.detach().to('cpu').numpy()
                p=np.reshape(p,(1,total_num_weight))
                for i in range(total_num_weight):
                    sum2+=p[0][i]
                avg2=sum2/total_num_weight
                plot(name,param)
                print('after modify avg is:',avg2)
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