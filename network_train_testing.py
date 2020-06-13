import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from result_save_and_visualize import RetSAndV
from torchvision import transforms,utils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt


# ==============================  step 1/5:Initialization Setting ==============================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batchsize = 8
numworkers = 8
numepoch = 10
dataroot = 'data10'  # datasets path.
txt_name = dataroot+':'+str(numepoch)  # 保存training结果的txt文件名->根据不同训练任务进行更改->有助于识别
print('device:',device)
# Source image display.
# def show_img(data):
#     for i,imgs in enumerate([data[j][0] for j in range(8)]):
#         plt.imshow(imgs)
#         plt.title('Source image')
#         plt.axis('off')
#         plt.show()
#
#
# # show_img(dataset)


# ==============================  step 2/5:Data ==============================

data_transform = {
    'train':transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 将图片转换为Tensor的同时归一化到[0,1]
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  # 把像素值都归一化到[-1,1],(0-0.5)/0.5=-1,(1-0.5)/0.5=1
    ]),
    'test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    ])
}

datasets = {x:ImageFolder(os.path.join(dataroot,x),transform=data_transform[x]) for x in ['train','test']}

dataloader = {x:DataLoader(datasets[x],batch_size=batchsize,shuffle=True,num_workers=numworkers) for
              x in ['train','test']}

class_names = datasets['train'].classes
nums_classes = len(class_names)
data_size = {x:len(datasets[x]) for x in ['train','test']}


def show_batch(img_batch,title=None):
    """
    image show for Tensor.
    :param img_batch:a batch size of Tensor image
    :param title:
    :return:
    """
    img_batch = img_batch.numpy().transpose((1,2,0))  # Tensor to numpy(w,h,c)
    mean = [.5,.5,.5]
    std = [.5,.5,.5]
    img_batch = std*img_batch + mean
    img_batch = np.clip(img_batch,0,1)
    plt.imshow(img_batch)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# inputs,labels = next(iter(dataloader['train']))
#
#
# outputs = utils.make_grid(inputs)
# show_batch(outputs,title=[class_names[i] for i in labels])


# ==============================  step 3/5:Model ==============================

model_fc = torchvision.models.resnet18(pretrained=False)
num_ftrs = model_fc.fc.in_features
model_fc.fc = nn.Linear(num_ftrs, nums_classes)
model_fc = model_fc.to(device)
loss_CEL = nn.CrossEntropyLoss()

optimizer_fc = optim.SGD(model_fc.parameters(),lr=0.001,momentum=0.9)
# decay_lr_sheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
print('Starting Training!\nThere are {} categories in this model!'.format(nums_classes))


# ==============================  step 4/5:Training and Testing ==============================
def train_model(model,optimizer,loss_func,num_epochs=numepoch):
    # record start time.
    since = time.time()
    dict_loss_acc = {'train_loss':[],'test_loss':[],'train_acc':[],'test_acc':[]}

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('*'*20)

        # each epoch has a training and testing phase
        for phase in ['train','test']:

            running_loss = 0.0
            running_corrects = 0.0
            if phase == 'train':
                model.train()  # Set model to train model
            else:
                model.eval()  # Set model to evaluate model

            for inputs,labels in dataloader[phase]:
                # pass data to cuda(if cuda is available)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Reset grad as zero.
                optimizer.zero_grad()

                # train:generate gradient in the train phase to update parameters.
                # test:there is no need to generate gradient.
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    # caculate loss.This loss value here is average loss value of batchsize.
                    loss = loss_func(outputs,labels)
                    # _ is maximum in each rows,preds is index of it.
                    _,preds = torch.max(outputs,1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # The inputs.size(0) is the batchsize.
                # batchsize statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # epoch statistics
            epoch_loss = running_loss / data_size[phase]
            epoch_acc = float(running_corrects) / data_size[phase]

            # each epoch print train and test's loss and accuracy
            print('{} loss: {:.3f} acc: {:.3f}'.format(phase,epoch_loss,epoch_acc))
            if phase == 'train':
                dict_loss_acc['train_loss'].append(epoch_loss)
                dict_loss_acc['train_acc'].append(epoch_acc)
            else:
                dict_loss_acc['test_loss'].append(epoch_loss)
                dict_loss_acc['test_acc'].append(epoch_acc)
            # save the best model weights
            if phase is 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s.'.format(time_elapsed // 60,time_elapsed % 60))
    print('The best accuracy is {:.3f}.'.format(best_acc))

    return round(best_acc,2),dict_loss_acc


# ==============================  step 5/5:Ploting result ==============================
bestacc,diction_loss_acc = train_model(model_fc,optimizer_fc,loss_CEL)
retsv = RetSAndV(epoch=numepoch,bestacc=bestacc,name_txt=txt_name+'.txt',dict_loss_acc=diction_loss_acc)
retsv.save_ret()
# 这里使用**kw关键字参数的目的是->可以自由变化图片上的参数显示
retsv.visualize_ret(epoch=numepoch)
