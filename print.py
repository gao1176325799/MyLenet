from numpy import save
from Operation_LSM import *
from alex_model import *
from Utility_LSM import *
from save_load_network import save_load_method
from NN_lib import torch,np
import glo
import os
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
# for name, param in model.named_parameters():
#     print('------------------------------')
#     print(name)
#     print(param)
#     plot(name, param)
#     print('------------------------------')
def comb_detect(nparry):
    arr_flat=nparry.flatten() 
    arr_flat_sort=np.sort(arr_flat,axis=None)
    ary_len=len(arr_flat_sort)
    rounding_size=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
    print(ary_len,'elements being detect')
    pos=[]
    neg=[]
    for k in arr_flat_sort:
        if k <0:
            neg.append(k)
        else:
            pos.append(k)
    neg=-np.sort(-np.array(neg))
    pos=np.array(pos)
    print('There are ',len(neg),' negetive numbers, and ',len(pos),' positive numbers.')
    combs=[]    
    for i in rounding_size:
        sig=0
        temp=0
        idp=0
        for id in range(len(neg)):
                while neg[id]+pos[idp]<(-i) and idp<len(pos)-1:
                        idp+=1
                diff=neg[id]+pos[idp]
                if idp>=len(pos)-1:
                        break
                if abs(diff)<i:
                        idp+=1
                        temp+=1
                        #print('NP===','neg:',neg[id],', pos:',pos[idp],', diff is: ',diff,' current rounding size is ',i)
        combs.append(temp)
        temp=0
    combs.append('#')
    for i in rounding_size:
        temp=0
        id=0
        while id <len(neg)-1:
                if abs(neg[id]-neg[id+1])<i:
                        diff=abs(neg[id]-neg[id+1])
                        temp+=1
                        id+=2
                else: id+=1
        id=0
        while id <len(pos)-1:
                if abs(pos[id]-pos[id+1])<i:
                        diff=abs(pos[id]-pos[id+1])
                        temp+=1
                        id+=2
                else: id+=1
        combs.append(temp)
    combs.append('*')

    for i in rounding_size:
        temp=0
        id=0
        while id <ary_len-1:
            if abs(arr_flat_sort[id]-arr_flat_sort[id+1])<i:
                diff=abs(arr_flat_sort[id]-arr_flat_sort[id+1])
                #print('PP---','first:',arr_flat_sort[id],' second:',arr_flat_sort[id+1],' diff:',diff,' current rounding size is ',i)
                temp+=1
                id+=2
            else:id+=1
        combs.append(temp)

    print(combs)
    os.system("pause")

def potential_comb_analyzer(name,param):
    number=0
    if name.find("fc")==-1 and name.find("weight")!=-1:
        print("we are working on",name)
        arr=param.cpu().detach().numpy()
        print('input is',np.shape(arr),'\n')
        comb_detect(arr)

    print("Next\n\n")

test=2
if test==1:
        for name, param in model.named_parameters():
                print('------------------------------')
                print(name)
                #print(param)
                #plot(name, param)
                binwidth=0.05
                param=param.cpu().detach().numpy().ravel()
                plt.hist (param,bins=80,rwidth=0.5)
                plt.title('Weight distribution histogram '+'for '+name)
                plt.show()
                print('------------------------------')
                #break
elif test==2:
        for name, param in model.named_parameters():
                print('------------------------------')
                print(name)
                potential_comb_analyzer(name,param)
else:
        print('nothing to print')

