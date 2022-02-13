from torch.utils.data import DataLoader
from torchvision import datasets, transforms,utils
import matplotlib.pyplot as plt
import numpy as np
from config import *
transforms=transforms.Compose([transforms.Resize((32,32)),
    transforms.ToTensor()])
train_dataset = datasets.MNIST(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.MNIST(root='mnist_data', 
                               train=False, 
                               transform=transforms)

# define the data loaders
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=BATCH_SIZE, 
                          shuffle=False)

##data stucture analysis
test0=0
if test0:
    def imshow(img):
        npimg=img.numpy()
        print(npimg)
        plt.imshow(np.transpose(npimg,(1,2,0)))
        print(np.transpose(npimg,(1,2,0)).shape)#-->h,w(122,242,3)一行八个数 有四行
        #after transforms.Resize(32,32) size become (138,274,3)
        plt.show()

    dataiter=iter(train_loader)
    images,labels=dataiter.next()
    imshow(utils.make_grid(images))
    print(type(valid_loader))
    print(len(train_loader.dataset))#-->60000
    print(len(valid_loader.dataset))#-->10000
test1=0
if test1:
    index=0
    max=0
    for i,(images,labels) in enumerate(valid_loader):
        if i<5:
            print('###images:',images.detach().numpy(),'###images:',images.detach().numpy().shape)#batch size 个副图的数据
            x=images.detach().numpy()
            for k,n in enumerate(x):
                if n.any()>max:
                    max=n
            print(f'in{i} group the max is{max}')
            print('@@@labels:',labels)#32个tensor
            print('###################################')
        index+=1
    print('Total samples in valid loader:',index)
        