#Traninng dataset is 60000 images or samples
#batch size is 32. Iteration=60000xepochs/32
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
       #!Critical update on Mar 5th, the trigger been added in to model(Alex_model.py)
       outputs=model(images,0)
       loss=criterion(outputs,labels)
       #backward and optimize
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if(i+1)%100==0:
           print(f'Epoch[{epoch+1}/{N_EPOCHS}],Step[{i+1}/{n_total_steps}], Loss:{loss.item():.4f}')
print('Finished Training')

test0=0

with torch.no_grad():#trainloader has 60000, testloader has 10000
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in range(10)]
    n_class_sample=[0 for i in range(10)]
    #magic=0
    label_counter=0
    counter=0
    #for images, labels in valid_loader:
    for i,(images, labels) in enumerate (valid_loader):
        images=images.to(DEVICE)
        labels=labels.to(DEVICE)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()
        if i>2 and i<5 and test0==1:
            print('#############################')
            print(predicted.detach().to('cpu').numpy())
            print(labels.detach().to('cpu').numpy())
            print(len(labels.detach().to('cpu').numpy()))
            print(type(predicted))
            print(type(labels))
        #magic+=1
        label_counter+=len(labels.detach().to('cpu').numpy())
        # if magic==313:
        #     size=16
        # else:
        #     size=BATCH_SIZE
        
        for i in range(BATCH_SIZE):#batchsize is 32 but the limit for i is 0-15,what happened?
            counter+=1
            label=labels[i]
            pred=predicted[i]
            if(label==pred):
                n_class_correct[label]+=1
            n_class_sample[label]+=1
        print(label_counter,counter)
    acc=100*n_correct/n_samples
    print(f'Accuracy of the network:{acc}%')
    for i in range(10):
        acc=100*n_class_correct[i]/n_class_sample[i]
        print(f'Accuracy of classes{i}: {acc} %')



#magic is because in for i in range(size) it appears the last one batch is not 32, but 16 which doesn't make sense
#because total step for one epoch is 1875 and for each step the batch size is 32,
#32*1875=60000, but now the data show that is 32*1875-16.
#><# later I find that the train loader has 60,000 images, but the test or valid loader only have 10,000 imgaes.