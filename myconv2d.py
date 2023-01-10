from NN_lib import *
from NN_lib import _pair
import glo
from torch.nn.common_types import _size_2_t
from typing import Union
#from config import set_value,get_val

def ocupied(comb_list,d,num):
    #find out if the num exist in comb[d]
  if comb_list[d]:
    for i in comb_list[d]:
      if i ==num:
        return True
    return False
  return False
def extract(comb,d):
    #extract first two number as a and b, and delete them in comb list
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
  temp_weight=weight.reshape(d,c*k*j)
  for dd in range(d):
    for ckj in range(c*k*j):
      temp_a=temp_weight[dd,ckj]
      if ckj!=c*k*j-1:
        for ckj_ in range(ckj+1,c*k*j):
          temp_b=temp_weight[dd,ckj_]
          absv=abs(temp_a+temp_b)
          if abs(temp_a+temp_b)<=unit:
            if (not ocupied(comb,dd,ckj)) and (not ocupied(comb,dd,ckj_)):
              comb[dd].append(ckj)
              comb[dd].append(ckj_)
              break 
  return comb
def sort_v2(weight,unit):#weight is a tensor
    d,c,k,j=weight.shape
    comb=[]
    old_diff=0
    for dd in range(d):
        temp_weight=weight[dd]#temp_weight tensor,torch.float32
        temp_weight=temp_weight.reshape(c*k*j)
        sorted_,indices=torch.sort(temp_weight,stable=True)#both sorted_ and indices are tensor, torch.float32
        #next we are going to make sorted and indices as tuple
        #before that its need to convert to numpy array
        sorted_=sorted_.numpy()
        indices=indices.numpy()#now they change to array
        zipped=list(zip(sorted_,indices))
        print(type(zipped))
        temp=0
        for i in range(len(sorted_)):
            if sorted_[i]>=0:
                temp=i
                break
        neg_s=zipped[:temp]
        pos_s=zipped[temp:]
        #next step resort the negs part as descending
        neg_s=sorted(neg_s,key=lambda x:x[0],reverse=True)
        #so far we got a descending neg seq and a ascending pos seq
        n_ptr,p_ptr,p_ptr_bound=0,0,0
        cb1=[]
        cb2=[]
        while n_ptr<len(neg_s) and p_ptr<len(pos_s):#stop when n_ptr reach the maximum
            n_val,n_loc=neg_s[n_ptr]
            p_val,p_loc=pos_s[p_ptr]
            while (n_val+p_val)<=unit and p_ptr<len(pos_s):#stop when p_val is too big or p_ptr reach the maximum
                p_val,p_loc=pos_s[p_ptr]
                diff = abs(n_val+p_val)
                if diff<=unit:
                    if n_loc<p_loc:
                        cb1.append(n_loc)
                        cb2.append(p_loc)
                    else:
                        cb1.append(p_loc)
                        cb2.append(n_loc)
                    if diff>old_diff:
                        old_diff=diff
                        #print("{:4f}".format(n_val),"{:4f}".format(p_val),"{:4f}".format(diff),"--loc-->",n_loc,p_loc)
                    p_ptr_bound=p_ptr+1#set the new boundary
                    break #stop the current while loop, go find the next combination
                p_ptr+=1
            n_ptr+=1
            p_ptr=p_ptr_bound
        #at this point the location for combinations are stored at n_comb and p_comb
        cb_comb=list(zip(cb1,cb2))
        cb_comb=sorted(cb_comb,key=lambda x:x[0],reverse=False) 
        #sort the combinations along first element
        cb_comb=[item for sublist in cb_comb for item in sublist]
        comb.append(cb_comb)
    return comb



# def vicsum_v2(inseq_after_pad,weight):
#   n,c,h,w,k,j=inseq_after_pad.shape
#   d,c,k,j=weight.shape
#   out=torch.zeros(n,d,h,w)
#   start=time.time()
#   comb=sort_w(weight,0.0001)
#   end=time.time()
#   print('time elapse for sort:',end-start)
#   for nn in range(n):
#     for dd in range(d):
#       for hh in range(h):
#         for ww in range(w):
#           FLAG=1#initialize internal flag
#           # 0->no more comb thus no need to enquire new comb
#           # 1->looking for ckj meets the comb
#           temp_inseq_list=inseq_after_pad[nn,:,hh,ww].numpy().tolist()
#           temp_inseq_list=list(chain.from_iterable(temp_inseq_list))#from 3d->2d
#           temp_inseq_list=list(chain.from_iterable(temp_inseq_list))#from 2d->1d
#           temp_weight_list=weight[dd].numpy().tolist()
#           temp_weight_list=list(chain.from_iterable(temp_weight_list))#from 3d->2d
#           temp_weight_list=list(chain.from_iterable(temp_weight_list))#from 2d->1d
#           templist=[]
#           while FLAG:
#             if comb[dd]:#if comb exist
#               pos1,pos2=extract(comb,dd)
#               out[nn,dd,hh,ww]+=(temp_inseq_list[pos1]-temp_inseq_list[pos2])*temp_weight_list[pos1]
#               templist.append(pos1)
#               templist.append(pos2)
#             else:#if no more comb avaliable
#               FLAG=0
#           counter=0 
#           templist.sort()
#           for ele in templist:
#             ele=ele-counter
#             del temp_inseq_list[ele]
#             del temp_weight_list[ele]
#             counter+=1
#           while len(temp_inseq_list)>0:#if stil have input and weight
            
#             out[nn,dd,hh,ww]+=temp_inseq_list[0]*temp_weight_list[0]
#             del temp_inseq_list[0]
#             del temp_weight_list[0]
#           #check if both list are empty
#           assert len(temp_inseq_list)==0,'the inseq list is not empty'
#           assert len(temp_weight_list)==0,'the weight list is not empty'  
#   return out

def vicsum_v3(inseq_after_pad,weight):#a faster version
  n,c,h,w,k,j=inseq_after_pad.shape
  d,_,_,_=weight.shape
  out=torch.zeros(n,d,h,w)
  # start=time.time()
  comb=sort_v2(weight,0.5)
  count=0
  
  # end=time.time()
  # print('time elapse for sort:',end-start)
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
          for i in range (0,len(comb[dd]),2):
            pos1=comb[dd][i]
            pos2=comb[dd][i+1]
            out[nn,dd,hh,ww]+=(temp_inseq_list[pos1]-temp_inseq_list[pos2])*temp_weight_list[pos1]
            glo.set_value(0,glo.get_value(0)+1)
            glo.set_value(1,glo.get_value(1)+1)
            glo.set_value(2,glo.get_value(2)+1)
            templist.append(pos1)
            templist.append(pos2)
          counter=0 
          templist.sort()
          for ele in templist:
            ele=ele-counter
            del temp_inseq_list[ele]
            del temp_weight_list[ele]
            counter+=1
          ptr=0
          while ptr<len(temp_inseq_list):
            out[nn,dd,hh,ww]+=temp_inseq_list[ptr]*temp_weight_list[ptr]
            ptr+=1
            glo.set_value(0,glo.get_value(0)+1)
            glo.set_value(2,glo.get_value(2)+1)  
  return out

def myconv2dv2(x,weight,bias,stride,pad):
  #extract params in tensor
  n,c_in,h_in,w_in=x.shape
  d,c_w,k,j=weight.shape
  x_pad=torch.zeros(n,c_in,h_in+2*pad[0],w_in+2*pad[0])
  if pad[0]>0:
    x_pad[:,:,pad[0]:-pad[0],pad[0]:-pad[0]]=x
  else:
    x_pad=x
  #double unfold-->window sliding based on kernel size
  x_pad=x_pad.unfold(2,k,stride[0])
  x_pad=x_pad.unfold(3,j,stride[0])#xpad type=torch.float32 device CPU
  n,c_in,h_in,w_in,k,j=x_pad.shape
  out=vicsum_v3(x_pad,weight)#weight same as xpad with different size
  out=out+bias.view(1,-1,1,1)
  return out

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
        #print(self.weight.shape)
        #print(self.bias.shape)
        return myconv2dv2(input,weight,bias,self.stride,self.padding)
    def forward(self, input):
        return self._conv_forward(input,self.weight,self.bias)









