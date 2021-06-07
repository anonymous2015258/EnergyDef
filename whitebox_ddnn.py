import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import csv
import torch.utils.data as torchdata2
import numpy as np
from torch.autograd import Variable
from torch import optim
import time
from torch import autograd
import sys
import torch.nn.functional as F
import torch.utils.data as Data
from model import base
from util import Partition
#from net_whitebox import DDNN
#import models_skip as models
from training_model import FlatResNet32




'''def partition(img2):
    #img2=torch.as_tensor(img2)
    img=img2[0]
    C, W, H = torch.as_tensor(img).shape


    parts = []
    part_W, part_H = W // 2, H // 3
    for r in range(3):
        for c in range(2):
            parts.append(img[:, c * part_W:(c + 1) * part_W, r * part_H:(r + 1) * part_H])
    #print(parts)
    img2=torch.zeros((1,6,3,16,10), device='cuda')
    img2[0]=torch.stack(parts)
    return img2'''

def partition(t, n_rows=3, n_cols=2):
    # img2=torch.as_tensor(img2)
    BS, C, W, H = t.shape
    parts = []
    part_W, part_H = W // n_cols, H // n_rows
    for r in range(n_rows):
        for c in range(n_cols):
            parts.append(t[:, :, c * part_W:(c + 1) * part_W, r * part_H:(r + 1) * part_H])
    # print(parts)
    # img2 = torch.zeros((1, 6, 3, 16, 10), device='cuda')
    return torch.stack(parts).permute([1, 0, 2, 3, 4])



transform_test = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                        ])



transform_train = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                        ])



#trainset = torchdata.CIFAR10(root='data/', train=True, download=True, transform=transform_train)
testset = torchdata.CIFAR10(root='data/', train=False, download=True, transform=transform_test)


#train_loader = torchdata2.DataLoader(trainset, batch_size=len(trainset))
#test_loader = torchdata2.DataLoader(testset, batch_size=1,shuffle=False,num_workers=4)
#train_dataset_array = next(iter(train_loader))[0].numpy()
test_loader = torchdata2.DataLoader(testset, batch_size=len(testset),shuffle=False,num_workers=4)
test_dataset_array = next(iter(test_loader))[0].numpy()
#print(test_dataset_array.shape)
'''test_dataset_array = next(iter(test_loader))
test_dataset_array=[t.numpy() for t in test_dataset_array]
print(type(test_dataset_array))

print(np.asarray(test_dataset_array).shape)'''

'''test_dataset_array = testset.data

print(test_dataset_array.shape)'''
#print(test_dataset_array[0][0])
# time.sleep(10)
'''training_arr = np.reshape(test_dataset_array, (10000, 3072))[:5000]

# test_dataset_array=np.reshape(training_arr, (5000,3,32,32))

# print(test_dataset_array[0][0])

time.sleep(5)
test_arr = np.reshape(test_dataset_array, (10000, 3072))[5000:]

lst = []
# print(np.reshape(test_dataset_array, (10000,3072)))
print(training_arr[0])'''
lst=[]
with open('skip_cifar_train_avg.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[1]))

#target_arr = np.asarray(lst)[0:8000]
target_arr2 = np.asarray(lst)


def test_model():
    # create model
    #model = net2.DDNN(in_channels, 10, num_devices)

    # checkpoint = torch.load('model_best.pth.tar')
    #model=model.load_state_dict(torch.load('model.pth'))
    model=torch.load('model2.pth')
    #model=DDNN
    #model.load_state_dict(torch.load('model2.pth')['state_dict'])
    return model






def tanh_rescale(x, x_min=-1.7, x_max=2.05):
    return (torch.tanh(x) + 1) * 0.5 * (x_max - x_min) + x_min

def reduce_sum(x, keepdim=True):
    # silly PyTorch, when will you get proper reducing sums/means?
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x


def l2_dist(x, y, keepdim=True):
    d = (x - y) ** 2
    return reduce_sum(d, keepdim=keepdim)

def loss_op(output, target, dist, scale_const):

    #loss1 =  target-output
    loss1 = torch.clamp(target - output, min=0.)
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss

def loss_op2(output, dist, scale_const):

    #loss1 =  target-output
    val=torch.tensor(0.0)
    for o in output:
        o1=o.tolist()
        if(o1<0.5):
            val=o
            print('val ',val)
            break
    target=torch.tensor(0.5)
    loss1=target - val
    #loss1 = torch.clamp(target - output, min=0.)
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss

'''ayer_config = [18, 18, 18]
net = FlatResNet32(base.BasicBlock, layer_config, num_classes=1)

model = net
model.load_state_dict(torch.load('ddnn_train_new2.pth'))
model.eval().cuda()'''

#x, y = Variable(torch.Tensor(test_dataset_array[8000:0])), Variable(torch.Tensor(target_arr2))

ddnn_model=test_model().eval().cuda()

def orig(p):
    py_list=p.tolist()
    p_list=py_list[0]
    #print(np.amax(np.asarray(p_list)))
    #print(np.amin(np.asarray(p_list)))
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    img = [[[0.0 for i1 in range(32)] for j1 in range(32)] for k1 in range(3)]


    for i in range(3):
        for j in range(32):
            for k in range(32):
                img[i][j][k]=(((p_list[i][j][k])*std[i])+ mean[i])
                #img[i][j][k] = int((p_list[i][j][k] * std[i] + mean[i]) * 255)
    py_list[0]=img
    #print(np.amax(np.asarray(img)))
    #print(np.amin(np.asarray(img)))

    return torch.FloatTensor(py_list)


def normalize(p):
    py_list=p.tolist()
    p_list=py_list[0]
    #print(np.amax(np.asarray(p_list)))
    #print(np.amin(np.asarray(p_list)))
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    img = [[[0.0 for i1 in range(32)] for j1 in range(32)] for k1 in range(3)]


    for i in range(3):
        for j in range(32):
            for k in range(32):
                img[i][j][k]=(((p_list[i][j][k])- mean[i])/std[i])
                #img[i][j][k] = int((p_list[i][j][k] * std[i] + mean[i]) * 255)
    py_list[0]=img
    #print(np.amax(np.asarray(img)))
    #print(np.amin(np.asarray(img)))

    return torch.FloatTensor(py_list)





def get_blocks(masks):
    skips = [mask.data.le(0.5).float().mean() for mask in masks]
    sum = 0
    #print(skips)
    for s in skips:
        # print(s)
        value = s.tolist()
        if (value == 1):
            sum += 1
    return (54-sum)
x, y = Variable(torch.Tensor(test_dataset_array)), Variable(torch.Tensor(target_arr2))
torch_dataset = Data.TensorDataset(x, y)
BATCH_SIZE = 1
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4, )

def get_exit(etas):

    for i in range(len(etas)):
        lst=etas[i].tolist()
        if(lst<0.5):
            return i+1
    return len(etas)+1

for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
    #print("epoch ", epoch)

    input_var = Variable(batch_x,requires_grad=True).cuda()
    input_var.retain_grad()
    b_y = Variable(batch_y,volatile=True).cuda()
    #input_var_skip=normalize(input_var)
    #print(input_var.size())
    predictions,etas=ddnn_model(partition(input_var))

    pred=predictions[0]
    pred2 = predictions[1][0]
    pred,indices=torch.sort(pred)

    '''lst=np.zeros((20,32,32),dtype=np.float64)
    pred=pred[0]
    for ind in range(10):
        #print(pred)
        #print(pred[ind])
        pred[ind].backward(retain_graph=True)
        saliency, _ = torch.max(input_var.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst[ind]=saliency
    for ind in range(10):
        pred2[ind].backward(retain_graph=True)
        saliency, _ = torch.max(input_var.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst[ind+10] = saliency'''
    #print(predictions)
    #time.sleep(100)

    lst = np.zeros((5, 32, 32), dtype=np.float64)
    #pred = pred[0]
    for ind in range(5):
        # print(pred)
        # print(pred[ind])
        etas[ind].backward(retain_graph=True)
        saliency, _ = torch.max(input_var.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst[ind] = saliency

    orig = get_exit(etas)
    #print(etas)

    #time.sleep(50)

    '''etas[0].backward()
    saliency, _ = torch.max(input_var.grad.data.abs(), dim=1)
    saliency = saliency.reshape(32, 32).data.cpu().numpy()'''
    np.save('sal_map/orig_sal_ddnn_etas'+str(orig) + '_'+str(step) + '.npy', lst)

    #sm1=get_blocks(masks)
    #temp = get_blocks(masks)
    #print("before attack sm ", sm1," ",step)
    index1000=0
    index2000 = 0
    index3000 = 0
    index4000=0

    modifier = torch.rand(input_var.size(), device="cuda").float()
    new_input_adv = torch.zeros(input_var.size()).float()
    modifier_var = autograd.Variable(modifier, requires_grad=True)
    optimizer = optim.Adam([modifier_var], lr=0.0005)
    target = torch.tensor(6 * [0.5], device="cuda")
    # target_var = autograd.Variable(target, requires_grad=False)
    min_loss = float("inf")
    # adv_img_min = np.zeros((1, 32, 32, 3))
    # min_output = torch.tensor(0.0)
    best=0
    for ind in range(5000):
        # print(ind)
        x = torch.tensor(10000)
        scale_const_var = autograd.Variable(x, requires_grad=False).cuda()
        # print(modifier_var)
        input_adv = tanh_rescale(modifier_var + input_var, -1.7, 2.05)

        predictions,etas = ddnn_model(partition(input_adv))
        # print(output)
        dist = l2_dist(input_adv, input_var, keepdim=False)
        # print("loss ", output)
        #loss = loss_op(torch.tensor(etas).cuda(), target, dist, scale_const_var)
        loss = loss_op2(etas, dist, scale_const_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        #output_np = output.data.cpu().numpy()

        #predictions, etas = ddnn_model(partition(input_adv))
        #print(etas)
        #time.sleep(2)

        # sm2 = get_blocks(masks)
        # print("min ", ind, " ", sm2)
        cnt=get_exit(etas)
        print(cnt," ",best)
        if (cnt > best):
            # temp=sm2
            best=cnt
            min_loss = loss
            # adv_img_min = input_adv_np
            new_input_adv = input_adv

            # min_output = output

            '''f = open("block.csv", "a")
            writer = csv.writer(f)
            # print(energy)
            blocks = []
            blocks.append(index2)
            writer.writerow(blocks)
            f.close()'''

    '''np.save(
        'np/' + str(step)  + '_data.npy', new_input_adv.data.cpu().numpy())'''
    # input_var_skip = normalize(new_input_adv)
    adv_clone = new_input_adv.clone().detach()

    adv_clone = Variable(adv_clone, requires_grad=True).cuda()
    adv_clone.retain_grad()

    predictions,etas = ddnn_model(partition(adv_clone))
    pred = predictions[0]
    pred2=predictions[1][0]
    pred, indices = torch.sort(pred)

    '''lst2 = np.zeros((20, 32, 32), dtype=np.float64)
    pred=pred[0]
    for ind in range(10):
        pred[ind].backward(retain_graph=True)
        saliency, _ = torch.max(adv_clone.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst2[ind] = saliency
    for ind in range(10):
        pred2[ind].backward(retain_graph=True)
        saliency, _ = torch.max(adv_clone.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst2[ind+10] = saliency'''
    lst2 = np.zeros((5, 32, 32), dtype=np.float64)
    #pred = pred[0]
    for ind in range(5):
        etas[ind].backward(retain_graph=True)
        saliency, _ = torch.max(adv_clone.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst2[ind] = saliency

    '''etas[0].backward()
    saliency, _ = torch.max(adv_clone.grad.data.abs(), dim=1)
    best_saliency = saliency.reshape(32, 32).data.cpu().numpy()'''
    np.save('sal_map/best_saliency_ddnn_etas'+str(best) + '_'+str(step) + '.npy', lst2)