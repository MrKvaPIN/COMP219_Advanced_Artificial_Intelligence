import numpy as np
import pandas as pd
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
################################################################################################
## Note: below is the place you need to add your own attack algorithm
################################################################################################
def random_attack(model, X, y, device):
    X_adv = Variable(X.data)

    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################

    epsilon = 0.10994
    # 攻击迭代次数
    number = 15
    # 动量因子
    mu = 0.3
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

                # 计算损失，结合类别最大化和动量梯度更新
                # 类别最大化项（梯度上升部分）
                class_loss = -output[range(output.size(0)), y].mean()  # 最大化目标类别分数
                # 添加动量更新项
                total_loss = class_loss
                # 清空模型梯度并计算损失梯度
                model.zero_grad()
                total_loss.backward()

                # 更新动量梯度
                grad_accum = mu * grad_accum + X_temp.grad / (X_temp.grad.norm() + 1e-8)

                # 更新扰动 delta，使用动量和当前步长
                delta = torch.clamp(delta + scale * grad_accum.sign(), -epsilon, epsilon)
    X_adv = X_adv + delta

    return X_adv

# def adv_attack
################################################################################################

## end of attack method
################################################################################################

def eval_adv_test(model, device, test_loader, adv_attack_method):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0),28*28)
            adv_data = adv_attack_method(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def evaluate_all_models(model_file, attack_method, test_loader, device):

    model = Net().to(device)
    model.load_state_dict(torch.load(model_file))

    adv_attack = attack_method
    ls, acc = eval_adv_test(model, device, test_loader, adv_attack)
     
    del model
    return 1/acc, acc

def main():
    
    ################################################################################################
    ## Note: below is the place you need to load your own attack algorithm and defence model
    ################################################################################################

    # attack algorithms name, add your attack function name at the end of the list
    attack_method = [random_attack]

    # defense model name, add your attack function name at the end of the list
    model_file = ["201850501.pt","201690423.pt","2016904230001.pt","yindu.pt","sebV3.pt"]

    ################################################################################################
    ## end of load 
    ################################################################################################

    # number of attack algorithms
    num_all_attack = len(attack_method)
    # number of defense model
    num_all_model = len(model_file)

    # define the evaluation matrix number
    evaluation_matrix = np.zeros((num_all_attack, num_all_model))

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
    #device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")

    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    for i in range(num_all_attack):
        for j in range(num_all_model):
            attack_score, defence_score = evaluate_all_models(model_file[j], attack_method[i], test_loader, device)
            evaluation_matrix[i,j] = attack_score

    print("evaluation_matrix: ", evaluation_matrix)
    # Higher is better
    print("attack_score_mean: ", evaluation_matrix.mean(axis=1))
    # Higher is better
    print("defence_score_mean: ", 1/evaluation_matrix.mean(axis=0))

main()
