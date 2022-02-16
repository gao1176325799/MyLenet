from save_load_network import load_all, FILE
from alex_lib import torch
from config import DEVICE
from data_loaded import valid_loader

loaded_model=load_all(FILE)
loaded_model.eval()

with torch.no_grad():
    n_correct=0
    n_samples=0
    for i,(images, labels) in enumerate (valid_loader):
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)
        outputs=loaded_model(images)
        _,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()
    acc=100*n_correct/n_samples#->9860/10000
    print(f'Accuracy of the network:{acc}%')

#### EXTRACT and MODIFY weight
def weight_ex_ex(model):
    
    print(type(model))

    return model

with torch.no_grad():
    n_correct=0
    n_samples=0
    for images,labels in valid_loader:
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)
        new_model=weight_ex_ex(loaded_model)
        outputs=loaded_model(images)
        _,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()
    acc=100*n_correct/n_samples
    print(f'Accuracy of the network:{acc}%, correct:{n_correct}, total:{n_samples}')
