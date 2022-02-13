from alex_model import*
from test_model import*
from config import BATCH_SIZE, N_EPOCHS
from data_loaded import train_loader,valid_loader
n_total_steps=len(train_loader)# used to support to calculate the step accuracy

model,criterion,optimizer=run_test_model()

for epoch in range(N_EPOCHS):
   for i, (images, labels) in enumerate(train_loader):
       images=images.to(DEVICE)
       labels=labels.to(DEVICE)
       #forward pass
       outputs=model(images)
       loss=criterion(outputs,labels)
       #backward and optimize
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if(i+1)%100==0:
           print(f'Epoch[{epoch+1}/{N_EPOCHS}],Step[{i+1}/{n_total_steps}], Loss:{loss.item():.4f}')

    with torch.no_grad():
        n_correct=0
        n_samples=0
        for i,(images, labels) in enumerate (valid_loader):
            images=images.to(DEVICE)
            labels=labels.to(DEVICE)
            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            n_samples+=labels.size(0)
            n_correct+=(predicted==labels).sum().item()
            acc=100*n_correct/n_samples
        print(f'Accuracy of the network:{acc}%')
print('Finished Training')

