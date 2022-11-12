from numpy import save
from Operation_LSM import *
from alex_model import *
from Utility_LSM import *
from save_load_network import save_load_method
from NN_lib import torch,np
import glo
from config import DEVICE,FILE_state,FILE_model
from data_loaded import valid_loader
import matplotlib.pyplot as plt
import xlwt
# from config import set_value,get_val
#model,_,_=run_test_model()
model,_,_=run_Lenet5()
#load trained model
#loaded_model=save_load_method(save_or_load="load",method="all",FILE_path=FILE_model)
loaded_model=save_load_method(save_or_load="load",method="state",model=model,FILE_path=FILE_state)

print(loaded_model)
# print('param start')
#para_m(loaded_model.named_parameters(),loaded_model)

glo._init()

#_____________________________
print('with myconv2d start')
loaded_model.my_conv1.bias.data=loaded_model.conv1.bias.data
loaded_model.my_conv2.bias.data=loaded_model.conv2.bias.data
loaded_model.my_conv3.bias.data=loaded_model.conv3.bias.data
loaded_model.my_conv1.weight.data=loaded_model.conv1.weight.data
loaded_model.my_conv2.weight.data=loaded_model.conv2.weight.data
loaded_model.my_conv3.weight.data=loaded_model.conv3.weight.data

run_with_My_conv2d(loaded_model,1000)

add_val=glo.get_value(0)
sub_val=glo.get_value(1)
mul_val=glo.get_value(2)
print(add_val,sub_val,mul_val)
print('original start')
run_without_change(loaded_model,1000)
#__________________________


#run_without_change_dig_gpu(loaded_model)


#para_m(loaded_model.named_parameters(),loaded_model)



# test0=0
# if test0:
#     with torch.no_grad():
#         for name, param in loaded_model.named_parameters(): 
#             if name=='conv1.weight':
#                 plot(name,param)
#                 sum1=0
#                 sum2=0
#                 out_f,in_f,k_H,k_W=param.shape
#                 print(out_f,in_f,k_H,k_W)
#                 total_num_weight=out_f*in_f*k_H*k_W
                
#                 print(param[0][0][0][0].detach().to('cpu').numpy())
#                 p=param.detach().to('cpu').numpy()
#                 p=np.reshape(p,(1,total_num_weight))
#                 for i in range(total_num_weight):
#                     sum1+=p[0][i]
#                 avg1=sum1/total_num_weight
#                 print('before modify the avg is:',avg1)
#                 for i in range(out_f):
#                     for k in range(in_f):
#                         for j in range(k_H):
#                             for l in range(k_W):
#                                 param[i][k][j][l]=(param[i][k][j][l]+2.5)/3.5
#                 p=param.detach().to('cpu').numpy()
#                 p=np.reshape(p,(1,total_num_weight))
#                 for i in range(total_num_weight):
#                     sum2+=p[0][i]
#                 avg2=sum2/total_num_weight
#                 plot(name,param)
#                 print('after modify avg is:',avg2)
#         n_correct=0
#         n_samples=0
#         for i,(images, labels) in enumerate (valid_loader):
#             images=images.to(DEVICE)
#             labels=labels.to(DEVICE)
#             outputs=loaded_model(images)
#             _,predicted=torch.max(outputs,1)
#             n_samples+=labels.size(0)
#             n_correct+=(predicted==labels).sum().item()
#         acc=100*n_correct/n_samples#->9860/10000
#         print(f'Accuracy of the network:{acc}%')    