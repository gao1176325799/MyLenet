#from pickletools import optimize
from NN_lib import *

from config import DEVICE,LEARNING_RATE
from torch.nn.modules.utils import _single, _pair, _triple
def test_func(a):
    a+=1
    if a%100==0:
        print('hello@',a)
    return a
#TODO: program my conv2d to replace the Conv2d in nn.model
#*https://github.com/pskugit/custom-conv2d/blob/master/models/customconv.py
#*https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
#* transposed, output_padding been deleted in init
#* device, dtype been deleted in super
class My_Conv2d(nn.modules.conv._ConvNd):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1,
                 padding=0, dilation=1, 
                 groups: int=1, bias: bool= True, padding_mode: str='zeros', device=None, 
                 dtype=None) -> None:
        kernel_size=_pair(kernel_size)
        stride=_pair(stride)
        padding=_pair(padding)
        dilation=_pair(dilation)
        
        super(My_Conv2d,self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
              False, _pair(0),groups, bias, padding_mode)
    def conv2d_forward(self, input, weight):
        return myconv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

def myconv2d(input, weight, bias=None, stride=(1,1), padding=(0,0), 
                dilation=(1,1), groups=1):
    """
    Function to process an input with a standard convolution
    """
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()
    if bias is None:
        out_unf = inp_unf.transpose(1, 2).matmul(w_).transpose(1, 2)
    else:
        out_unf = (inp_unf.transpose(1, 2).matmul(w_) + bias).transpose(1, 2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)
    return out.float()









class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        #TODO: make test on our Conv2d, thus we make copy of all the conv provide 2 method
        #to able to choose which method is gonna using.
        self.conv1=nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.my_conv1=My_Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1)
        self.pool=nn.AvgPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.my_conv2=My_Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1)
        self.pool2=nn.AvgPool2d(2)
        self.conv3=nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1)
        self.my_conv3=My_Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1)
        self.fc1=nn.Linear(in_features=120,out_features=84)
        self.fc2=nn.Linear(in_features=84,out_features=10)
        self.index=0
        self.counter_trigger=0
    def forward(self,x,TRIGGERed):
        #if triggered run my conv2d else run nn.conv2d
        #the nn.conv2d is to train the model and save the model
        #the my conv2d is to test the method after we got a trained model
        if TRIGGERed:
            if self.counter_trigger==0:
                print('triggered')
            self.counter_trigger+=1
            x=self.pool(F.sigmoid(self.my_conv1(x)))
            x=self.pool2(F.sigmoid(self.my_conv2(x)))
            x=F.sigmoid(self.my_conv3(x))
        else:
            x=self.pool(F.sigmoid(self.conv1(x)))
            x=self.pool2(F.sigmoid(self.conv2(x)))
            x=F.sigmoid(self.conv3(x))
        #x=x.view(-1,120)
        x = torch.flatten(x, 1)
        x=F.sigmoid(self.fc1(x))
        x=self.fc2(x)
        index=self.index
        self.index=test_func(index)
        
        return x
        
def run_Lenet5():
    model=Lenet5().to(DEVICE)# move model to GPU
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return model,criterion,optimizer
