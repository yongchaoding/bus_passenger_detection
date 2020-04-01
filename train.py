import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init

from data import *
from data.bus_passenger_data import bus_passenger

from models.ssd import build_ssd
from models.functions.multibox_loss import MultiBoxLoss
def data_load(batch_size = 4, num_workers = 1):
    train_data = bus_passenger('train')
    train_data = data.DataLoader(train_data, batch_size, num_workers = num_workers, collate_fn=detection_collate)
    return train_data

use_cuda = False
resume = False
cfg = bus_passenger_cfg

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
## 数据集读入操作
## TODO: 需要进一步学习数据增强部分
train_data = data_load()

## 网络定义操作
ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
net = ssd_net
net.train()
#print(net)
if resume == False:
    print('Loading base network...')
    vgg_weights = torch.load('weight/vgg16_reducedfc.pth')
    ssd_net.vgg.load_state_dict(vgg_weights)
    print('Initializing other weights...')
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

## 优化求解器
optimizer = optim.SGD(net.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)

## 损失计算函数
criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, \
                        True, 3, 0.5, False, use_cuda)

## 开始训练
batch_iterator = iter(train_data)
for i in range(0, 100):
    try:
        images, targets = next(batch_iterator)
    except StopIteration:
        batch_iterator = iter(train_data)
        images, targets = next(batch_iterator)
    if use_cuda == False:
        images = images
        targets = [Variable(ann, volatile=True) for ann in targets]
    #print(images.size())
    out = net(images)
    #print(out[0].size(), out[1].size(), out[2].size())
    ##  优化进行时
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    print("Loss:", loss.item())
