#from pickletools import optimize
from NN_lib import *
from NN_lib import _pair
from config import DEVICE,LEARNING_RATE
from myconv2d import MYconv2d

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        #TODO: make test on our Conv2d, thus we make copy of all the conv provide 2 method
        #to able to choose which method is gonna using.
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.my_conv1=MYconv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.pool=nn.AvgPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.my_conv2=MYconv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.pool2=nn.AvgPool2d(2)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1)
        self.my_conv3=MYconv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1)
        self.fc1=nn.Linear(in_features=120,out_features=84)
        self.fc2=nn.Linear(in_features=84,out_features=10)
        self.index=0
        self.counter_trigger=0
    def forward(self,x,TRIGGERed,count):
        #if triggered run my conv2d else run nn.conv2d
        #the nn.conv2d is to train the model and save the model
        #the my conv2d is to test the method after we got a trained model
        if TRIGGERed:
            if self.counter_trigger==0:
                print('triggered')
            self.counter_trigger+=1
            s1=time.time()#for time measurement 
            x=self.pool(F.sigmoid(self.my_conv1(x)))
            e1=time.time()
            s2=time.time()
            x=self.pool2(F.sigmoid(self.my_conv2(x)))
            e2=time.time()
            print('time for conv1+pool is:',e1-s1)
            print('time for conv2+pool is:',e2-s2)
            x=F.sigmoid(self.my_conv3(x))
        else:
            if count:
                self.index+=1
            x=self.pool(F.sigmoid(self.conv1(x)))
            x=self.pool2(F.sigmoid(self.conv2(x)))
            x=F.sigmoid(self.conv3(x))
        #x=x.view(-1,120)
        x = torch.flatten(x, 1)
        x=F.sigmoid(self.fc1(x))
        x=self.fc2(x)
        
        return x
        
def run_Lenet5():
    model=Lenet5().to(DEVICE)# move model to GPU
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return model,criterion,optimizer

# class test_model(nn.Module):
    def __init__(self):
        super(test_model,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=4,kernel_size=17)#32-17+1=16
        self.myconv1=MYconv2d(in_channels=1,out_channels=4,kernel_size=17)
        self.pool=nn.MaxPool2d(kernel_size=2)#16/2=8
        self.conv2=nn.Conv2d(4,20,8)#8-8+1=1
        self.myconv2=MYconv2d(4,20,8)
        self.fc1=nn.Linear(20,10)
        self.index=0
        self.counter_trigger=0
    def forward(self,x,TRIGGERed,count):
        if TRIGGERed:
            if self.counter_trigger==0:
                print('triggered')
            self.counter_trigger=1
            x=self.myconv1(x)
            #print('myconv1',x)
            x=self.pool(F.sigmoid(x))
            x=F.sigmoid(self.myconv2(x))
        else:
            x=self.conv1(x)
            if TRIGGERed:
              print('conv1',x)
            x=self.pool(F.sigmoid(x))
            x=F.sigmoid(self.conv2(x))
        #x=x.view(-1,10)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        return x


# def run_test_model():
#     model=test_model().to(DEVICE)
#     criterion=nn.CrossEntropyLoss()
#     optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
#     return model,criterion,optimizer
