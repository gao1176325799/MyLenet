import matplotlib.pyplot as plt


def plot(name, param):
    dims = len(param[1].shape)
    y = param.detach().cpu().numpy().reshape(-1)
    x = range(len(y))

    y_min = min(y)
    x_min = 0
    for i, v in enumerate(y):
        if v == y_min:
            x_min = i
            break

    y_max = max(y)
    x_max = 0
    for i, v in enumerate(y):
        if v == y_max:
            x_max = i
            break

    plt.xlabel('x')
    plt.ylabel('y')

    l1 = plt.scatter(x, y, marker='.')
    l2 = plt.scatter(x_min, y_min, marker='x', color='red')
    l3 = plt.scatter(x_max, y_max, marker='x', color='red')

    plt.legend(handles=[l1, l2, l3], labels=[name, 'min: ' + str(y_min), 'max: ' + str(y_max)], loc='best')

    

    plt.show()

def quantize(max_r,min_r,input,bit_q):
    unit=1/pow(2,bit_q)
    output=0
    if input >max_r:
        output=max_r
    elif input<min_r:
        output=min_r
    else:
        output=int(input/unit)*unit
    return output

def pos_quantize(max_r,min_r,input,bit_q):
    unit=1/pow(2,bit_q)
    output=0
    if input>max_r:
        output=max_r
    else:
        input+=abs(min_r)
        input/=(max_r-min_r)
        output=int(input/unit)*unit
    return output

def find_min(input):
    temp=0
    for i in input.flat:
        if i <temp:
            temp=i
    return temp
