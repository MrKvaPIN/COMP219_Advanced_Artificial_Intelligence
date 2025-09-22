############################################################################
### Written by Gaojie Jin and updated by Xiaowei Huang, 2021
###
### For a 2-nd year undergraduate student competition on 
### the robustness of deep neural networks, where a student 
### needs to develop 
### 1. an attack algorithm, and 
### 2. an adversarial training algorithm
###
### The score is based on both algorithms. 
############################################################################


import numpy as np
import pandas as pd
from torch.optim import LBFGS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

# input id
id_ = #your id here, e.g., 12345678

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')


args = parser.parse_args(args=[]) 

# judge cuda is available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")

torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

############################################################################
################    don't change the below code    #####################
############################################################################
train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

##############################################################################
#############    end of "don't change the below code"   ######################
##############################################################################

#generate adversarial data, you can define your adversarial method
def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)

    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################
    epsilon = 0.10994
    # 攻击迭代次数
    number = 15
    # 动量因子
    mu = 0.7
    # 扰动储存
    delta = torch.zeros_like(X).to(device)
    grad_accum = torch.zeros_like(X).to(device)
    # 定义扰动步长和迭代次数
    scale_steps = [0.2, 0.5, 0.4, 0.05]
    scale_number = [number // len(scale_steps) for _ in scale_steps]

    with torch.enable_grad():
        for scale, num_iters in zip(scale_steps, scale_number):  # 分阶段扰动更新
            for _ in range(num_iters):
                # 生成当前的对抗样本
                X_temp = X_adv + delta
                X_temp.requires_grad = True

                # 模型输出和目标类别置信度
                output = model(X_temp)
                softmax_output = F.softmax(output, dim=1)  # (batch_size, num_classes)
                batch_indices = torch.arange(softmax_output.size(0)).to(device)  # (batch_size,)
                target_confidence = softmax_output[batch_indices, y].mean().item()  # 平均置信度


                # 计算目标置信度的负值损失（最小化置信度）
                confidence_loss = target_confidence

                # 类别最大化项（梯度上升部分）
                class_loss = -output[range(output.size(0)), y].mean()  # 最大化目标类别分数

                # 添加动量更新项
                total_loss = class_loss + confidence_loss
                # 清空模型梯度并计算损失梯度
                model.zero_grad()
                total_loss.backward()

                # 更新动量梯度
                grad_accum = mu * grad_accum + X_temp.grad / (X_temp.grad.norm() + 1e-8)

                # 更新扰动 delta，使用动量和当前步长
                delta = torch.clamp(delta + scale * grad_accum.sign(), -epsilon, epsilon)
    X_adv = X_adv + delta
    return X_adv


#predict function
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        # 确保 data 启用梯度追踪
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28).requires_grad_()  # 启用梯度追踪

        # 动态生成对抗样本
        adv_data = adv_attack(model, data, target, device)

        # 数据增强
        data_aug = data + torch.normal(0, 0.1, data.shape).to(device)
        adv_data_aug = adv_data + torch.normal(0, 0.1, adv_data.shape).to(device)

        # 清梯度
        optimizer.zero_grad()

        # 计算原始数据和对抗样本的损失
        output = model(data_aug)
        loss_data = F.nll_loss(output, target)
        adv_output = model(adv_data_aug)
        loss_adv = F.nll_loss(adv_output, target)

        # 动态调整普通样本和对抗样本的损失权重
        weight = epoch / args.epochs
        loss = (1 - weight) * loss_data + weight * loss_adv

        # 添加KL散度正则化
        kl_div = F.kl_div(F.log_softmax(adv_output, dim=1), F.softmax(output, dim=1), reduction='batchmean')
        kl_weight = min(epoch / args.epochs, 0.2)  # KL 散度权重最大 0.2
        # 初始化 total_loss 确保其始终有值
        total_loss = loss + kl_weight * kl_div

        # 添加梯度惩罚正则化
        grad_norm = torch.autograd.grad(outputs=total_loss, inputs=data, create_graph=True)[0].norm()
        total_loss += 0.01 * grad_norm

        # 反向传播+更新（梯度裁剪）
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0),28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0),28*28)
            adv_data = adv_attack(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

#main function, train the dataset and print train loss, test loss for each epoch
def train_model():
    model = Net().to(device)
    
    ################################################################################################
    ## Note: below is the place you need to edit to implement your own training algorithm
    ##       You can also edit the functions such as train(...). 
    ################################################################################################
    
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)
    # 定义余弦学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        #training
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        #get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)
        
        #print trnloss and testloss
        print('Epoch '+str(epoch)+': '+str(int(time.time()-start_time))+'s', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))
        
    adv_tstloss, adv_tstacc = eval_adv_test(model, device, test_loader)
    print('Your estimated attack ability, by applying your attack method on your own trained model, is: {:.4f}'.format(1/adv_tstacc))
    print('Your estimated defence ability, by evaluating your own defence model over your attack, is: {:.4f}'.format(adv_tstacc))
    ################################################################################################
    ## end of training method
    ################################################################################################
    
    #save the model
    torch.save(model.state_dict(), str(id_)+'.pt')
    return model

#compute perturbation distance
def p_distance(model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0),28*28)
        data_ = copy.deepcopy(data.data)
        adv_data = adv_attack(model, data, target, device=device)
        p.append(torch.norm(data_-adv_data, float('inf')))
    print('epsilon p: ',max(p))

    
################################################################################################
## Note: below is for testing/debugging purpose, please comment them out in the submission file
################################################################################################
    
#Comment out the following command when you do not want to re-train the model
#In that case, it will load a pre-trained model you saved in train_model()
model = train_model()

#Call adv_attack() method on a pre-trained model'
#the robustness of the model is evaluated against the infinite-norm distance measure
#important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!
p_distance(model, train_loader, device)
