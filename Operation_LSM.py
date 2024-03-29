# TODO a function that can find the size after each layer. So we can sort and move them
from NN_lib import torch,np
from config import DEVICE
from data_loaded import valid_loader
from Utility_LSM import *

def run_without_change(loaded_model,samples):
    factor=int(samples/10)
    with torch.no_grad():
        n_correct=0
        n_samples=0
        for i,(images, labels) in enumerate (valid_loader):
            #print(images.shape)
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            #print(images.shape)
            outputs=loaded_model(images,0,1)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
            if n_samples==samples:
                break
        acc=100*n_correct/n_samples#->9860/10000
        print(f'Accuracy of the network using nn.conv2d:{acc}%')
def run_without_change_dig_gpu(loaded_model):
    
    with torch.no_grad():
        n_correct=0
        n_samples=0
        for i,(images, labels) in enumerate (valid_loader):
            print('before go to device',images.shape)
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            print('after push to device',images.shape)
            outputs=loaded_model(images,0,1)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
        acc=100*n_correct/n_samples#->9860/10000
        print(f'Accuracy of the network before modify:{acc}%')

def run_with_My_conv2d(loaded_model,samples):
    factor=int(samples/10)
    with torch.no_grad():
        n_correct=0
        n_samples=0
        for i,(images, labels) in enumerate (valid_loader):
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            outputs=loaded_model(images,1,1)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
            if n_samples%factor==0:
                print('progress:',n_samples/samples*100,'%')
            if n_samples==samples:
                break
        acc=100*n_correct/n_samples#->9860/10000
        print(f'Accuracy of the network using my conv2d:{acc}%')
#! use less 
def para_m_pos(parameters,loaded_model):
    #* using[0,1]quantization
    test=[0,0,0,0]
    with torch.no_grad():
        for name, param in parameters:
            if (name=='conv1.weight') or (name=='conv2.weight') or (name=='conv3.weight'):#later it will be all the weight
                print(f'Before modify')
                plot(name,param)
                sum1=0
                sum2=0
                #get shape
                out_f,in_f,k_H,k_W=param.shape
                #print(out_f,in_f,k_H,k_W)
                total_num_weight=out_f*in_f*k_H*k_W
            
                #print(param[0][0][0][0].detach().to('cpu').numpy())
                p=param.detach().to('cpu').numpy()
                p=np.reshape(p,(1,total_num_weight))
                for i in range(total_num_weight):
                    sum1+=p[0][i]
                avg1=sum1/total_num_weight
                min_=find_min(p)
                print('before modify the avg is:',avg1)
                print('before modify the min is:',min_)
                for i in range(out_f):
                    for k in range(in_f):
                        for j in range(k_H):
                            for l in range(k_W):
                                param[i][k][j][l]=pos_quantize(1,min_,param[i][k][j][l],4)
                                #param[i][k][j][l]=(param[i][k][j][l]+2.5)/3.5
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
            outputs=loaded_model(images,0)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
        acc=100*n_correct/n_samples#->9860/10000
        print(f'Accuracy of the network:{acc}%')  

def para_m(parameters,loaded_model):
    #*using [-1,1] quantization
    test=[0,0,0,0]
    with torch.no_grad():
        for name, param in parameters:
            if (name=='conv1.weight') or (name=='conv2.weight') or (name=='conv3.weight'):#later it will be all the weight
                print(f'Before modify')
                #plot(name,param)
                sum1=0
                sum2=0
                #get shape
                out_f,in_f,k_H,k_W=param.shape
                #print(out_f,in_f,k_H,k_W)
                total_num_weight=out_f*in_f*k_H*k_W
            
                #print(param[0][0][0][0].detach().to('cpu').numpy())
                p=param.detach().to('cpu').numpy()
                p=np.reshape(p,(1,total_num_weight))
                for i in range(total_num_weight):
                    sum1+=p[0][i]
                avg1=sum1/total_num_weight
                min_=find_min(p)
                #print('before modify the avg is:',avg1)
                #print('before modify the min is:',min_)
                for i in range(out_f):
                    for k in range(in_f):
                        for j in range(k_H):
                            for l in range(k_W):
                                param[i][k][j][l]=quantize(1,-1,param[i][k][j][l],4)
                                #param[i][k][j][l]=(param[i][k][j][l]+2.5)/3.5
                p=param.detach().to('cpu').numpy()
                p=np.reshape(p,(1,total_num_weight))
                for i in range(total_num_weight):
                    sum2+=p[0][i]
                avg2=sum2/total_num_weight
                #plot(name,param)
                #print('after modify avg is:',avg2)
            else:
                print(name)
        n_correct=0
        n_samples=0
        for i,(images, labels) in enumerate (valid_loader):
            
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            outputs=loaded_model(images,0,0)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
            if n_samples%20==0:
                print('progress:',n_samples,'%')
            if n_samples==100:
                break
        acc=100*n_correct/n_samples#->9860/10000
        print(f'Accuracy of the network:{acc}% with total {n_samples} samples')   
