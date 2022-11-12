def _init():
    global _global_dict
    _global_dict=[]
    x=[]
    _global_dict.append(x)
    _global_dict.append(x)
    _global_dict.append(x)
    _global_dict[0]=0#add
    _global_dict[1]=0#sub
    _global_dict[2]=0#mul
def set_value(key,value):
    """0->add, 1->sub, 2->mul"""
    _global_dict[key] = value
 
 
def get_value(key,defValue=None):
    """0->add, 1->sub, 2->mul"""
    try:
        return _global_dict[key]
    except KeyError:
            return defValue
#some test
# _init()
# set_value(0,get_value(0)+1)
# set_value(0,get_value(0)+1)
# set_value(1,1)
# set_value(2,2)
# a=0
# b=0
# c=0
# a=get_value(0)
# b=get_value(1)
# c=get_value(2)
# print(a,b,c)
#result is 2 1 2