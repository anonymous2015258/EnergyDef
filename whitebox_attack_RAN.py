import os
import math
import time
import shutil
import models
import time
from dataloader import get_dataloaders
from args import args
from adaptive_inference import dynamic_evaluate
from op_counter import measure_model
#from tx2_predict import PowerLogger,getNodes
import csv
import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
from torch import autograd
#torch.multiprocessing.set_start_method('spawn')
#os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from torch import optim
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')




model = getattr(models, 'RANet')(args)
model = torch.nn.DataParallel(model.to(device))
criterion = nn.CrossEntropyLoss().to(device)
train_loader, val_loader, test_loader = get_dataloaders(args)
'''trainset = torchdata.CIFAR10(root='data/', train=True, download=True, transform=transform_train)
testset = torchdata.CIFAR10(root='data/', train=False, download=True, transform=transform_test)
train_loader = torchdata2.DataLoader(trainset, batch_size=len(trainset))
test_loader = torchdata2.DataLoader(testset, batch_size=len(testset))
train_dataset_array = next(iter(train_loader))[0].numpy()
test_dataset_array = next(iter(test_loader))[0].numpy()

lst=[]
with open('cifar_training_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[2]))

target_arr = np.asarray(lst)'''
# target_arr=np.reshape(target_arr,(5000,1))
#target_arr2 = np.asarray(lst)[5000:]
# target_arr2=np.reshape(target_arr2,(5000,1))
# print(target_arr)

lst=[]

def get_block_count(agent,inputs):
    probs, _ = agent(inputs)
    # print(inputs)
    policy = probs.clone()
    #print(policy)
    policy[policy < 0.5] = 0.0
    policy[policy >= 0.5] = 1.0
    sm=0
    off_blocks=[]
    lst = policy.tolist()[0]
    for ind in range(len(lst)):
        # print("l ",l)
        if (lst[ind] == 1):
            sm += 1
        else:
            off_blocks.append(ind)
    f = open("off_blocks.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    writer.writerow(off_blocks)
    f.close()

    return policy,sm

'''with open('Cifar_Energy.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0

    for row in csv_reader:
        lst.append(float(row[2]))'''

#target_arr2 = np.asarray(lst)
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


def loss_op2(output, dist, scale_const):

    #sorted,indices=torch.sort(output,dim=1,descending=True)
    loss1 = output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss

    #return loss1


def loss_op3(output, dist, scale_const):

    #sorted,indices=torch.sort(output,dim=1,descending=True)
    target=torch.tensor(2.30)
    loss1 = torch.clamp(target - output, min=0.)
    #loss1 = output
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss
def loss_op(output, target, dist, scale_const):

    loss1 = torch.clamp(target - output, min=0.)
    loss1 = torch.sum(scale_const * loss1)
    loss2 = dist.sum()
    loss = loss1 + loss2
    return loss



#x, y = Variable(torch.Tensor(test_dataset_array)), Variable(torch.Tensor(target_arr2))

#state_dict = torch.load('model_best.pth.tar')['state_dict']

state_dict = torch.load('model_best2.pth.tar')
model.load_state_dict(state_dict)
model.eval().to(device)
#agent.eval().to(device)

#torch_dataset = Data.TensorDataset(x, y)
#BATCH_SIZE = 1
#loader = Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=4, )

for step, (batch_x, batch_y) in enumerate(test_loader):  # for each training step
    #print("epoch ", epoch)

    input_var = Variable(batch_x,requires_grad=True).to(device)
    #input_var = Variable(input, requires_grad=True).cuda()
    input_var.retain_grad()
    b_y = Variable(batch_y,volatile=True).to(device)
    #probs, _ = agent(input_var)

    output,ents = model(input_var)

    print(output)

    #time.sleep(200)
    orig=len(output)
    lst2 = np.zeros((orig, 32, 32), dtype=np.float64)
    ind = 0
    orig_idx=[]
    for g in ents:
        # print(g)

        # time.sleep(10)
        #g[0].backward(retain_graph=True)
        g.backward(retain_graph=True)
        #orig_idx.append(output[ind][1].tolist())
        saliency, _ = torch.max(input_var.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst2[ind] = saliency
        #pr.append(g.tolist()[1])
        ind += 1
    #print(pr)
    np.save('sal_map2/orig_saliency_RAN' + '_' + str(step) + '.npy', lst2)

    lst2 = np.zeros((orig, 10, 32, 32), dtype=np.float64)
    ind = 0
    orig_idx = []
    for g in output:
        # print(g)

        # time.sleep(10)
        # g[0].backward(retain_graph=True)
        #print(g)
        if(ind==0):
            orig_idx.append(g[0].tolist())
        for idx in range(10):
            #print(g[idx])
            g[0][idx].backward(retain_graph=True)

            saliency, _ = torch.max(input_var.grad.data.abs(), dim=1)
            saliency = saliency.reshape(32, 32).data.cpu().numpy()
            lst2[ind][idx] = saliency
        # pr.append(g.tolist()[1])
        ind += 1
    # print(pr)
    np.save('sal_map3/orig_saliency_RAN_acc' + '_' + str(step) +'_' +str(len(output))+  '.npy', lst2)

    #print(output)
    #time.sleep(20)
    #print("preds")
    #print(preds)
    #time.sleep(2)
    '''f = open("orig_preds_uni.csv", "a")
    writer = csv.writer(f)
    softm = torch.softmax(preds,dim = 1)
    # print(energy)
    preds = preds.data.cpu().numpy()
    # blocks.append(sm2)
    writer.writerow(preds)
    f.close()'''
    #policy = probs.clone()
    modifier = torch.zeros(input_var.size(), device=device).float()
    new_input_adv = torch.zeros(input_var.size()).float()
    #new_policy = torch.zeros(policy.size()).float()
    modifier_var = autograd.Variable(modifier, requires_grad=True)
    optimizer = optim.Adam([modifier_var], lr=0.0005)
    # target = torch.tensor(0.0)
    # target_var = autograd.Variable(target, requires_grad=False)
    min_loss = float("inf")
    adv_img_min = np.zeros((1, 32, 32, 3))
    #min_output = torch.zeros(policy.size()).float()
    #target = torch.tensor([0.5] * 54).to(device)
    final_sm=0
    for ind in range(1000):
        # print(ind)
        x = torch.tensor(100000)
        scale_const_var = autograd.Variable(x, requires_grad=False).to(device)
        input_adv = tanh_rescale(modifier_var + input_var, -1.7, 2.05)
        output,ents = model(input_adv)
        #probs, _ = agent(input_adv)
        # print(inputs)
        #policy = probs.clone()
        # print(output)

        dist = l2_dist(input_adv, input_var, keepdim=False)
        #loss = loss_op(policy,target, dist, scale_const_var)
        loss = loss_op3(ents[len(ents)-1][0], dist, scale_const_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss)
        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        #output_np = policy.data.cpu().numpy()
        # print(input_adv.data)
        input_adv_np = input_adv.data.cpu().numpy()
        # print(input_adv_np)
        max_output,m_ents = model(input_adv)

        #print("max_output")
        #print(len(max_output))
        if (len(m_ents) >final_sm):
        #if (loss < min_loss):
            min_loss = loss
            adv_img_min = input_adv_np
            new_input_adv = input_adv

            final_sm = len(m_ents)
            print("final_sm")
            print(final_sm)

            #time.sleep(1)

            #print(preds)
            #if(len(max_output)>final_sm):
            #    final_sm=len(max_output)

            #min_output = policy
            #print("min ", ind, " ", sm2)

            '''f = open("block2.csv", "a")
            writer = csv.writer(f)
            # print(energy)
            blocks = []
            blocks.append(sm2)
            writer.writerow(blocks)
            f.close()'''
            # time.sleep(2)
    # print("min ", min_output)
    #preds = rnet.forward_single(new_input_adv, new_policy.data.squeeze(0))
    '''policy, sm2 = get_block_count(agent, new_input_adv)
    f = open("block_changed_whitebox_uni.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    blocks = []
    blocks.append(sm)
    blocks.append(sm2)
    writer.writerow(blocks)
    f.close()
    preds = rnet.forward_single(new_input_adv, new_policy.data.squeeze(0))
    softm=torch.softmax(preds,dim = 1)
    f = open("preds_uni.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    preds = preds.data.cpu().numpy()
    #blocks.append(sm2)
    writer.writerow(preds)
    f.close()'''


    adv_clone = new_input_adv.clone().detach()
    adv_clone = Variable(adv_clone, requires_grad=True)
    adv_clone.retain_grad()

    output,ents = model(adv_clone)

    lst2 = np.zeros((len(output), 32, 32), dtype=np.float64)
    ind = 0
    best_idx=[]
    for g in ents:
        # print(g)

        # time.sleep(10)
        #g[0].backward(retain_graph=True)
        g.backward(retain_graph=True)
        #best_idx.append(output[ind][1].tolist())
        saliency, _ = torch.max(adv_clone.grad.data.abs(), dim=1)
        saliency = saliency.reshape(32, 32).data.cpu().numpy()
        lst2[ind] = saliency
        # pr.append(g.tolist()[1])
        ind += 1
    # print(pr)
    np.save('sal_map2/best_saliency_RAN' + '_' + str(step) + '.npy', lst2)

    lst2 = np.zeros((len(output), 10, 32, 32), dtype=np.float64)
    ind = 0
    #orig_idx = []
    #print(output)
    for g in output:
        # print(g)

        # time.sleep(10)
        # g[0].backward(retain_graph=True)
        if (ind == 0):
            best_idx.append(g[0].tolist())
        for idx in range(10):
            g[0][idx].backward(retain_graph=True)

            saliency, _ = torch.max(adv_clone.grad.data.abs(), dim=1)
            saliency = saliency.reshape(32, 32).data.cpu().numpy()
            lst2[ind][idx] = saliency
        # pr.append(g.tolist()[1])
        ind += 1
    # print(pr)
    np.save('sal_map3/best_saliency_RAN_acc' + '_' + str(step)+'_' +str(len(output))+ '.npy', lst2)
    f = open("exit_details_RANet2.csv", "a")
    writer = csv.writer(f)
    # print(energy)
    exits = []
    exits.append(orig)
    exits.append(final_sm)
    writer.writerow(exits)
    writer.writerow(orig_idx)
    writer.writerow(best_idx)
    # writer.writerow(lst_temp2)
    f.close()
    print(output)
    '''np.save(
        'np/' + str(step) +  '_'  + str(orig) + '_' + str(final_sm) + '_data.npy',
        adv_img_min)'''