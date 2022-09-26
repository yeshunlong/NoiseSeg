import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

import utils
from dataset import DataSet, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from model import Net

# nohup python3 -u train.py >output.log 2>&1 &

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
args = parser.parse_args()

cudnn.benchmark = False
cudnn.deterministic = True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_iterations = 6000
T = 8

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def create_model(ema=False):
    net = Net(1, 2, 16, has_dropout=True).to(device)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def get_current_consistency_weight(epoch):
    return 0.1 * utils.sigmod_rampup(epoch, 40.0)

def update_emas(net, ema_net, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_net.parameters(), net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def train(noise_net, student_net, teacher_net, dataloader):
    optimizer_noise_net = torch.optim.SGD(noise_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    optimizer_student_net = torch.optim.SGD(student_net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    noise_net.train()
    student_net.train()
    teacher_net.train()
    lr_ = 0.01
    iteration_num = 0
    max_epoch = max_iterations // len(dataloader) + 1
    for _ in tqdm(range(max_epoch), ncols=70):
        iteration_num = 0
        for _, sample in enumerate(dataloader):
            image, label = sample['image'], sample['label']
            image, label = image.to(device), label.to(device)
            unlabeled_image = image[2:]
            noise = noise_net.gen_noise(unlabeled_image)
            student_net_output = student_net(image)
            with torch.no_grad():
                teacher_net_output = teacher_net(unlabeled_image + noise)
                
            loss_seg = F.cross_entropy(student_net_output[:2], label[:2])
            output_soft = F.softmax(student_net_output, dim=1)
            loss_seg_dice = utils.dice_loss(output_soft[:2, 1, :, :, :], label[:2] == 1)
            loss_supervised = 0.5 * (loss_seg + loss_seg_dice)
            
            unlabeled_image = unlabeled_image.repeat(2, 1, 1, 1, 1)
            preds = torch.zeros([2 * 8, 2, 112, 112, 80]).to(device)
            for i in range(4):
                input = unlabeled_image + noise_net.gen_noise(unlabeled_image) + torch.clamp(torch.randn_like(unlabeled_image) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * 2 * i:2 * 2 * (i + 1)] = teacher_net.forward(input)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape([8, 2, 2, 112, 112, 80])
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)

            consistency_weight = get_current_consistency_weight(iteration_num // 150)
            consistency_loss = utils.softmax_mse_loss(student_net_output[2:], teacher_net_output)
            H = (0.75 + 0.25 * utils.sigmod_rampup(iteration_num, max_iterations)) * np.log(2)
            mask = (uncertainty < H).float()
            consistency_loss = torch.sum(mask * consistency_loss) / (2 * torch.sum(mask) + 1e-16)
            loss_consistency = consistency_weight * consistency_loss
            
            loss_noise_net = F.mse_loss(uncertainty, noise)
            loss_student_net = loss_supervised + loss_consistency
            
            optimizer_noise_net.zero_grad()
            loss_noise_net.backward()
            optimizer_noise_net.step()
            
            optimizer_student_net.zero_grad()
            loss_student_net.backward()
            optimizer_student_net.step()
            
            update_emas(student_net, teacher_net, 0.99, iteration_num)
            iteration_num += 1
            print('Iteration: {}, loss_student_net: {:.4f}, loss_supervised: {:.4f}, loss_consistency: {:.8f}, loss_noise_net: {:.4f}'.format(iteration_num, loss_student_net, loss_supervised, loss_consistency, loss_noise_net))
            if iteration_num % 2500 == 0:
                lr_ = 0.01 * 0.1 ** (iteration_num // 2500)
                for param_group in optimizer_student_net.param_groups:
                    param_group['lr'] = lr_
            if iteration_num >= max_iterations:
                break
        if iteration_num >= max_iterations:
            break
    noise_net.save_model('./noise.pth')
    student_net.save_model('./model.pth')
    
if __name__ == '__main__':
    noise_net = create_model()
    strudent_net = create_model()
    teacher_net = create_model(ema=True)
    dataset = DataSet(transform=transforms.Compose([RandomRotFlip(), RandomCrop((112, 112, 80)), ToTensor()]))
    batch_sampler = TwoStreamBatchSampler(list(range(16)), list(range(16, 80)), 4, 2)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    train(noise_net, strudent_net, teacher_net, dataloader)