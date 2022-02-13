from alex_lib import *
from config import DEVICE,LEARNING_RATE
class test_model(nn.Module):
    def __init__(self):
        super(test_model,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=10,kernel_size=32)
        #self.pool=nn.MaxPool2d(kernel_size=3)
        #self.conv2=nn.Conv2d(10,50,8)
        self.fc1=nn.Linear(50,10)
    def forward(self,x):
        #x=self.pool(F.sigmoid(self.conv1(x)))
        x=F.sigmoid(self.conv1(x))
        #x=F.sigmoid(self.conv2(x))
        x=x.view(-1,10)
        #x=self.fc1(x)
        return x
def run_test_model():
    model=test_model().to(DEVICE)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return model,criterion,optimizer

