from NN_lib import torch


#0-> not save 
#1-> save all
#2-> save only state dict

#1
def save_all(model,file):
    torch.save(model, file)
def load_all(file):
    loaded_model=torch.load(file)
    loaded_model.eval()
    return loaded_model
#2
def save_state(model,file):
    torch.save(model.state_dict(),file)
def load_state(model,file):
    loaded_model=model
    loaded_model.load_state_dict(torch.load(file))
    loaded_model.eval()
    return loaded_model



def save_load_method(save_or_load,method,FILE_path,model=""):
    if save_or_load=="save":
        if method=="all":
            save_all(model,FILE_path)
        elif method=="state":
            save_state(model,FILE_path)
    else:
        if method=="all":
            loaded_model=load_all(FILE_path)
        elif method=="state":
            loaded_model=load_state(model,FILE_path)
        return loaded_model
    
