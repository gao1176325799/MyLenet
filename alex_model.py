#from pickletools import optimize
from NN_lib import *
from NN_lib import _pair
from config import DEVICE,LEARNING_RATE
def ocupied(comb_list,d,num):
  if comb_list[d]:
    for i in comb_list[d]:
      if i ==num:
        return True
    return False
  return False
def extract(comb,d):
  a= comb[d][0]
  del comb[d][0]
  b=comb [d][0]
  del comb[d][0]
  return a,b
def sort_w(weight,unit):
  d,c,k,j=weight.shape
  #create list comb to handle all the combinations. 
  comb=[]
  for i in range(d):
    x=[]
    comb.append(x)
  print('comb and its shape:\n',comb,np.shape(comb))
  temp_weight=weight.reshape(d,c*k*j)
  print('shape',np.shape(temp_weight))
  for dd in range(d):
    for ckj in range(c*k*j):
      cc=ckj//(k*j)
      kk=ckj%(k*j)//k
      jj=ckj%j
      temp_a=weight[dd,cc,kk,jj]
      if ckj!=c*k*j-1:
        for ckj_ in range(ckj+1,c*k*j):
          ccc=ckj_//(k*j)
          kkk=ckj_%(k*j)//k
          jjj=ckj_%k
          temp_b=weight[dd,ccc,kkk,jjj]
          absv=abs(temp_a+temp_b)
          if abs(temp_a+temp_b)<=unit:
            if (not ocupied(comb,dd,ckj)) and (not ocupied(comb,dd,ckj_)):
              comb[dd].append(ckj)
              comb[dd].append(ckj_)
              break 
  return comb

def vicsum_v2(inseq_after_pad,weight):
  n,c,h,w,k,j=inseq_after_pad.shape
  d,c,k,j=weight.shape
  out=torch.zeros(n,d,h,w)
  comb=sort_w(weight,0.11)
  for nn in range(n):
    for dd in range(d):
      for hh in range(h):
        for ww in range(w):
          FLAG=1#initialize internal flag
          # 0->no more comb thus no need to enquire new comb
          # 1->looking for ckj meets the comb
          temp_inseq_list=inseq_after_pad[nn,:,hh,ww].numpy().tolist()
          temp_inseq_list=list(chain.from_iterable(temp_inseq_list))#from 3d->2d
          temp_inseq_list=list(chain.from_iterable(temp_inseq_list))#from 2d->1d
          temp_weight_list=weight[dd].numpy().tolist()
          temp_weight_list=list(chain.from_iterable(temp_weight_list))#from 3d->2d
          temp_weight_list=list(chain.from_iterable(temp_weight_list))#from 2d->1d
          templist=[]
          while FLAG:
            if comb[dd]:#if comb exist
              pos1,pos2=extract(comb,dd)
              len1=len(temp_inseq_list)
              len2=len(temp_weight_list)
              out[nn,dd,hh,ww]+=(temp_inseq_list[pos1]-temp_inseq_list[pos2])*temp_weight_list[pos1]
              templist.append(pos1)
              templist.append(pos2)
            else:#if no more comb avaliable
              FLAG=0
          counter=0 
          templist.sort()
          for ele in templist:
            ele=ele-counter
            del temp_inseq_list[ele]
            del temp_weight_list[ele]
            counter+=1
          while len(temp_inseq_list)>0:#if stil have input and weight
            
            out[nn,dd,hh,ww]+=temp_inseq_list[0]*temp_weight_list[0]
            del temp_inseq_list[0]
            del temp_weight_list[0]
          #check if both list are empty
          assert len(temp_inseq_list)==0,'the inseq list is not empty'
          assert len(temp_weight_list)==0,'the weight list is not empty'  
  return out

def myconv2dv2(x,weight,bias,stride,pad):
  
  n,c_in,h_in,w_in=x.shape
  d,c_w,k,j=weight.shape
  x_pad=torch.zeros(n,c_in,h_in+2*pad[0],w_in+2*pad[0])
  if pad[0]>0:
    x_pad[:,:,pad[0]:-pad[0],pad[0]:-pad[0]]=x
  else:
    x_pad=x
  #double unfold-->window sliding based on kernel size
  x_pad=x_pad.unfold(2,k,stride[0])
  x_pad=x_pad.unfold(3,j,stride[0])
  print(x_pad.shape)
  n,c_in,h_in,w_in,k,j=x_pad.shape
  #now we replace einsum with vicsum
  #out=torch.einsum('nchwkj,dckj->ndhw',x_pad,weight)
  out=vicsum_v2(x_pad,weight)
  print('out',out.shape)
  out=out+bias.view(1,-1,1,1)
  print(out.shape)
  return out
from torch.nn.common_types import _size_2_t
from typing import Union
class MYconv2d(nn.modules.conv._ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(MYconv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
    def _conv_forward(self, input, weight, bias):
        print(self.weight.shape)
        print(self.bias.shape)
        return myconv2dv2(input,weight,bias,self.stride,self.padding)
    def forward(self, input):
        return self._conv_forward(input,self.weight,self.bias)
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
            x=self.pool(F.sigmoid(self.my_conv1(x)))
            x=self.pool2(F.sigmoid(self.my_conv2(x)))
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

class test_model(nn.Module):
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
            x=self.pool(F.sigmoid(self.myconv1(x)))
            x=F.sigmoid(self.myconv2(x))
        else:
            x=self.pool(F.sigmoid(self.conv1(x)))
            x=F.sigmoid(self.conv2(x))
        #x=x.view(-1,10)
        x=torch.flatten(x,1)
        x=self.fc1(x)
        return x
def run_test_model():
    model=test_model().to(DEVICE)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    return model,criterion,optimizer