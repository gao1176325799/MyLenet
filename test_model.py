from alex_lib import *
from config import DEVICE,LEARNING_RATE
class test_model(nn.Module):
    def __init__(self):
        super(test_model,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=17)#32-17+1=16
        self.pool=nn.MaxPool2d(kernel_size=2)#16/2=8
        self.conv2=nn.Conv2d(16,80,8)#8-8+1=1
        self.fc1=nn.Linear(80,10)
    def forward(self,x):
        x=self.pool(F.sigmoid(self.conv1(x)))
        print(x.weight.shape)
        x=F.sigmoid(self.conv2(x))
        #x=F.sigmoid(self.conv2(x))
        #x=x.view(-1,10)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        return x
def run_test_model():
    model=test_model().to(DEVICE)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return model,criterion,optimizer

