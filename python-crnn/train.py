from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
from lib.models.crnn import CRNN
from lib.models.crnn import BidirectionalLSTM
from lib import dataset, utils
from Chinese_alphabet import alphabet

# 配置参数
parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()

# 英文字典
# alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

# 创建输出文件夹
if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

# 设置随机种子
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# 训练变量
image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)  # 图片尺寸
text = torch.IntTensor(opt.batchSize * 10)  # 假设每个句子长为5
length = torch.IntTensor(opt.batchSize)

# 输出类别数，即字符个数+空白符
nclass = len(alphabet) + 1
# 输入Channel
nc = 1

# 修改为指定字典集，使用英文字典时忽略大小写
converter = utils.strLabelConverter(alphabet, ignore_case=True)
# CTCLoss
criterion = torch.nn.CTCLoss()

# 创建crnn模型
crnn = CRNN(opt.imgH, nc, nclass, opt.nh)

if opt.cuda:
    crnn.cuda()
    image = image.cuda()
    criterion = criterion.cuda()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val(val_set, max_iter=100, flag=False):
    print('Start val')

    data_loader = torch.utils.data.DataLoader(
        val_set, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    loss_avg = utils.averager()
    if not flag:
        max_iter = min(max_iter, len(data_loader))
    else:
        max_iter = max(max_iter, len(data_loader))

    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        with torch.no_grad():
            crnn.eval()
            preds = crnn(image)
            crnn.train()

        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length)
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1

    if not flag:
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    if flag:
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    else:
        print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train():
    # 模型权重初始化
    crnn.apply(weights_init)

    # 如果指定预训练模型则加载
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        crnn.load_state_dict(torch.load(opt.pretrained))

    # loss averager
    loss_avg = utils.averager()

    # setup optimizer 从头训练
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                               betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters())
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    # # 微调
    # crnn.rnn = torch.nn.Sequential(
    #         BidirectionalLSTM(512, opt.nh, opt.nh),
    #         BidirectionalLSTM(opt.nh, opt.nh, nclass)).cuda()
    # optimizer = optim.Adam(crnn.rnn.parameters(), lr=opt.lr,
    #                        betas=(opt.beta1, 0.999))
    
    # 学习率衰减
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                               gamma=0.65)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=[2, 4, 6, 8, 10],
    #                                            gamma=0.65)

    # 检查设备
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # 加载数据集
    train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
    assert train_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

    test_dataset = dataset.lmdbDataset(
        root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))

    # 训练1个batch
    def train_batch():
        data = train_iter.next()
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = criterion(preds, text, preds_size, length)
        crnn.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    for epoch in range(opt.nepoch):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            crnn.train()
            cost = train_batch()
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % opt.valInterval == 0:
                val(test_dataset)

            # do checkpointing
            if i % opt.saveInterval == 0:
                torch.save(
                    crnn.state_dict(), '{0}/netCRNN.pth'.format(opt.expr_dir))

        scheduler.step()

    # test
    val(test_dataset, flag=True)


if __name__ == '__main__':
    train()
