from alex_model import*
from test_model import*
from config import BATCH_SIZE, N_EPOCHS
from data_loaded import train_loader,valid_loader
from save_load_network import FILE,save_all
n_total_steps=len(train_loader)# used to support to calculate the step accuracy

#model,criterion,optimizer=run_test_model()
model,criterion,optimizer=run_Lenet5()
test0=0
for epoch in range(N_EPOCHS):
   for i, (images, labels) in enumerate(train_loader):
       #this for gives a batch at a time 
       #if batch is 20 and image size is 32,32, then the shape of images is 
       #20,1,32,32
       images=images.to(DEVICE)#move the data to the model location
       labels=labels.to(DEVICE)
       #forward pass
       outputs=model(images,0)
       if test0:
           print('In test 0')
           print('images size:',images.detach().to('cpu').numpy().shape)
           print('labels size:',labels.detach().to('cpu').numpy().shape)
           print('outputs size:',outputs.detach().to('cpu').numpy().shape)
           test0=0
       loss=criterion(outputs,labels)
       #backward and optimize
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
      
       if(i+1)%1000==0:
           print(f'Epoch[{epoch+1}/{N_EPOCHS}],Step[{i+1}/{n_total_steps}], Loss:{loss.item():.4f}')
print('Finished Training')
##we want to save the train.
with torch.no_grad():
    n_correct=0
    n_samples=0
    for i,(images, labels) in enumerate (valid_loader):
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)
        outputs=model(images,0)
        _,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()
    acc=100*n_correct/n_samples
    print(f'Accuracy of the network:{acc}%')
save_all(model,FILE)