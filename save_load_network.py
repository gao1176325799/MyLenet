from train import*

FILE="model.pth"
method=0
#0-> not save 
#1-> save all
#2-> save only state dict
def save_all(model,file):
    torch.save(model, file)
def load_all(model,file):
    loaded_model=torch.load(FILE)
    loaded_model.eval()
    return loaded_model
