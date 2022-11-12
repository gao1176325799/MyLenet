# parameters
from NN_lib import torch
RANDOM_SEED     = 42
LEARNING_RATE   = 0.001
BATCH_SIZE      = 2
N_EPOCHS        = 10

IMG_SIZE        = 32
N_CLASSES       = 10
DEVICE          ='cpu'
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FILE_model      ="model.pth"
FILE_state      ="model_state.pth"
f_comb=         open("comb.txt","w")
