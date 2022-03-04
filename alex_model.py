#from pickletools import optimize
from alex_lib import *
from config import DEVICE,LEARNING_RATE
def test_func():
    print('hello')
    pass
class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.pool=nn.AvgPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.pool2=nn.AvgPool2d(2)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1)
        self.fc1=nn.Linear(in_features=120,out_features=84)
        self.fc2=nn.Linear(in_features=84,out_features=10)
    def forward(self,x):
        x=self.pool(F.sigmoid(self.conv1(x)))
        x=self.pool2(F.sigmoid(self.conv2(x)))
        x=F.sigmoid(self.conv3(x))
        #x=x.view(-1,120)
        x = torch.flatten(x, 1)
        x=F.sigmoid(self.fc1(x))
        x=self.fc2(x)
        test_func()
        return x
        
def run_Lenet5():
    model=Lenet5().to(DEVICE)# move model to GPU
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return model,criterion,optimizer
