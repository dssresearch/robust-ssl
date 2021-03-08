#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse, math, time, json, os
import random
import higher
from collections import OrderedDict
from tensorboardX import SummaryWriter
from datetime import datetime

from lib.wrn_weight import weight_WRN

from lib import transform
from config import config

# seed = 123
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="PL", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=5000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="cifar10", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument("--ood_ratio", default=0.5, type=float, help="ood ratio for unlabled set")
parser.add_argument("--n_label", default='class6', type=str, help="number of labeled set")
parser.add_argument("--meta_lr", default=0.01, type=float, help="learning rate for w")
parser.add_argument("--meta_val_batch", default=50, type=int, help="batch for meta val")
parser.add_argument("--w_clip", default=1e-5, type=float, help="batch for meta val")
parser.add_argument("--ood_weight", default=0.05, type=float, help="")
parser.add_argument("--reset", default=0.6, type=float, help="")


args = parser.parse_args()
print(args)
if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

condition = {}
exp_name = ""

print("dataset : {}".format(args.dataset))
condition["dataset"] = args.dataset
exp_name += str(args.dataset) + "_"

dataset_cfg = config[args.dataset]
transform_fn = transform.transform(*dataset_cfg["transform"]) # transform function (flip, crop, noise)

l_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "l_train")
if args.ood_ratio == 0:
    u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train")
else:
    u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_ood_{}".format(args.ood_ratio))
val_dataset = dataset_cfg["dataset"](args.root, args.n_label, "val")
test_dataset = dataset_cfg["dataset"](args.root, args.n_label, "test")

print("labeled data : {}, unlabeled data : {}, training data : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
condition["number_of_data"] = {
    "labeled":len(l_train_dataset), "unlabeled":len(u_train_dataset),
    "validation":len(val_dataset), "test":len(test_dataset)
}

class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

shared_cfg = config["shared"]
if args.alg != "supervised":
    # batch size = 0.5 x batch size
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
    )
else:
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"])
    )
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

u_loader = DataLoader(
    u_train_dataset, shared_cfg["batch_size"]//2, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"] * shared_cfg["batch_size"]//2)
)
val_loader_in_train = DataLoader(val_dataset, args.meta_val_batch, drop_last=True,
                                 sampler=RandomSampler(len(val_dataset), shared_cfg["iteration"] * args.meta_val_batch))

val_loader = DataLoader(val_dataset, 64, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 64, shuffle=False, drop_last=False)

print("maximum iteration : {}".format(min(len(l_loader), len(u_loader))))

alg_cfg = config[args.alg]
print("parameters : ", alg_cfg)
condition["h_parameters"] = alg_cfg

if args.em > 0:
    print("entropy minimization : {}".format(args.em))
    exp_name += "em_"
condition["entropy_maximization"] = args.em



# model = LeNet2().cuda()
model = weight_WRN(2, 6, transform_fn).cuda()
optimizer = optim.Adam(model.parameters(), lr=alg_cfg["lr"])
# model.load_state_dict(torch.load(os.path.join(args.output, "CIFAR10_4000_sup_model.pth")))

# trainable_paramters = sum([p.data.nelement() for p in model.parameters()])
# print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT":  # virtual adversarial training
    from lib.algs.vat import VAT_B_R

    ssl_obj = VAT_B_R(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL":  # pseudo label
    from lib.algs.pseudo_label import PL3

    ssl_obj = PL3(alg_cfg["threashold"])
elif args.alg == "MT":  # mean teacher
    from lib.algs.mean_teacher import MT_weight
    t_model = weight_WRN(2, 6, transform_fn).cuda()
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT_weight(t_model, alg_cfg["ema_factor"])
elif args.alg == "PI":  # PI Model
    from lib.algs.pimodel import PiModel_weight

    ssl_obj = PiModel_weight()
elif args.alg == "ICT":  # interpolation consistency training
    from lib.algs.ict import ICT

    t_model = weight_WRN(2, 6).cuda()
    # t_model = wrn.WRN(2, dataset_cfg["num_classes"], transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
elif args.alg == "MM":  # MixMatch
    from lib.algs.mixmatch import MixMatch

    ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
elif args.alg == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(args.alg))
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
print(TIMESTAMP)
path = os.path.join('TensorBoard', "Cluster_WRN",
                    "{}_{}_labeled_{}_ood_{}_lr_{}".format(args.dataset, args.alg, args.n_label, args.ood_ratio, args.meta_lr),
                    TIMESTAMP)
writer = SummaryWriter(path)
writer.add_text('Text', str(args), 0)

if args.ood_ratio == 0.25:
    ood_idx = [1,2,9,4,3,8,0]
    in_idx = [6,5,7]
elif args.ood_ratio == 0.5:
    ood_idx = [2, 4, 3, 9, 1, 6]
    in_idx = [5, 0, 7, 8]
else:
    ood_idx = [2, 7, 1]
    in_idx = [0, 3, 4, 5, 6, 8, 9]

aaaa = np.ones([20, 1]) / 20.0
# aaaa = np.ones([10, 1])
# aaaa[ood_idx] = args.ood_weight

emd_label = np.load('output/emd/cifar10/{}_{}_kmean_{}_new_label_{}_ood_{}.npy'.format(args.dataset, 'vgg19', 10, args.n_label, args.ood_ratio))

weight = torch.tensor(aaaa, dtype=torch.float32, device="cuda", requires_grad=True)
opt_w = optim.Adam([weight], lr=args.meta_lr)
# opt_w = optim.SGD([weight], lr=args.meta_lr, momentum=0.9)
# opt_w = optim.SGD([weight], lr=args.meta_lr)

u_label = u_train_dataset.dataset['labels']

iteration = 0
maximum_val_acc = 0
s = time.time()


for l_data, u_data, v_data in zip(l_loader, u_loader, val_loader_in_train):
    model.train()
    iteration += 1
    l_input, target, _ = l_data
    l_input, target = l_input.to(device).float(), target.to(device).long()

    u_input, dummy_target, idx = u_data
    u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
    val_input, val_target, _ = v_data
    val_input, val_target = val_input.to(device).float(), val_target.to(device).long()
    coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration / shared_cfg["warmup"], 1)) ** 2)

    w_l = torch.ones(len(l_input), device='cuda')
    w_v = torch.ones(len(val_input), device='cuda')

    w_u = weight[emd_label[idx]].squeeze()

    # w_u = weight[emd_label[idx[0]]]
    # for i in range(1, len(u_input)):
    #     w_u = torch.cat((w_u, weight[emd_label[idx[i]]]))
    with higher.innerloop_ctx(model, optimizer) as (meta_model, diffopt):
        for parameter in meta_model.parameters():
            parameter.requires_grad = False
        for parameter in meta_model.output.parameters():
            parameter.requires_grad = True
        meta_model.update_batch_stats(False)
        outputs_hat = meta_model(u_input, w_u)
        ssl_loss_meta = ssl_obj(u_input, outputs_hat.detach(), meta_model, w_u)
        cls_loss_meta = F.cross_entropy(meta_model(l_input, w_l), target, reduction="none", ignore_index=-1).mean()
        loss_meta = cls_loss_meta + ssl_loss_meta * coef
        meta_model.zero_grad()
        diffopt.step(loss_meta)
        val_loss_meta = F.cross_entropy(meta_model(val_input, w_v), val_target, reduction="none", ignore_index=-1).mean()
        opt_w.zero_grad()
        val_loss_meta.backward()
        opt_w.step()
    weight.data = torch.clamp(weight, min=0.0000000001, max=1)
    w_up = weight[emd_label[idx[0]]]
    for i in range(1, len(u_input)):
        w_up = torch.cat((w_up, weight[emd_label[idx[i]]]))
    w = w_up.clone().detach()
    #### training
    model.update_batch_stats(False)
    outputs_u = model(u_input, w)
    ssl_loss = ssl_obj(u_input, outputs_u.detach(), model, w)
    cls_loss = F.cross_entropy(model(l_input, w_l), target, reduction="none", ignore_index=-1).mean()
    loss = cls_loss + ssl_loss * coef
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ood_mask = (dummy_target == -1).float()

    writer.add_scalars('data/loss_group', {'loss': loss,
                                           'CE loss': cls_loss,
                                           'ssl_loss': ssl_loss}, iteration)
    w_ood = weight[ood_idx].data.cpu().numpy().mean()
    w_in = weight[in_idx].data.cpu().numpy().mean()
    writer.add_scalars('data/weight_group', {'w ood': w_ood,
                                             'w in': w_in,
                                             'truncated': 0}, iteration)
    # writer.add_scalars('data/Meta_loss_group', {'loss_meta': loss_meta,
    #                                             'CE loss_meta': cls_loss_meta,
    #                                             'ssl_loss_meta': ssl_loss_meta,
    #                                             'val_loss_meta': val_loss_meta}, iteration)
    if args.alg == "MT" or args.alg == "ICT":
        # parameter update with exponential moving average
        ssl_obj.moving_average(model.parameters())
    # display
    if iteration == 1 or (iteration % 100) == 0:
        wasted_time = time.time() - s
        rest = (shared_cfg["iteration"] - iteration) / 100 * wasted_time / 60
        print("iteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, w_ood : {:.2f}, w_in : {:.2f}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
            iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), w_ood, w_in, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]))
        s = time.time()
    if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
        with torch.no_grad():
            model.eval()
            # model.update_batch_stats(False)
            print()
            print("### validation ###")
            sum_acc = 0.
            s = time.time()
            for j, data in enumerate(val_loader):
                input, target, _ = data
                input, target = input.to(device).float(), target.to(device).long()
                w_l = torch.ones(len(input), device='cuda')
                # model.update_batch_stats(False)
                output = model(input, w_l)
                pred_label = output.max(1)[1]
                sum_acc += (pred_label == target).float().sum()
            acc = sum_acc / float(len(val_dataset))
            print("varidation accuracy : {}".format(acc))
            writer.add_scalar('data/val_acc', acc, iteration)
            # test
            if maximum_val_acc < acc:
                print("### test ###")
                maximum_val_acc = acc
                sum_acc = 0.
                s = time.time()
                for j, data in enumerate(test_loader):
                    input, target, _ = data
                    input, target = input.to(device).float(), target.to(device).long()
                    # model.update_batch_stats(False)
                    w_l = torch.ones(len(input), device='cuda')
                    output = model(input, w_l)
                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                test_acc = sum_acc / float(len(test_dataset))
                print("test accuracy : {}".format(test_acc))
                writer.add_scalar('data/test_acc', test_acc, iteration)
                torch.save(model.state_dict(), path + 'best_model.pth')
        model.train()
        s = time.time()
        if acc < args.reset:
            model.reset_param()
            # model.load_state_dict(torch.load(os.path.join(args.output, "CIFAR10_4000_sup_model.pth")))
            opt_w.param_groups[0]["lr"] = args.meta_lr
            # torch.nn.init.ones_(weight)
            print("### reset the model ###")
        else:
            opt_w.param_groups[0]["lr"] = 0.01
    # lr decay
    if iteration == shared_cfg["lr_decay_iter"]:
        optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]
t_acc = np.around(test_acc.cpu().data.numpy(), 4)*100
print("test acc : {}".format(test_acc))
condition["test_acc"] = test_acc.item()
writer.close()
