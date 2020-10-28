#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse, math, time, json, os
from collections import OrderedDict
from tensorboardX import SummaryWriter
from datetime import datetime

from lib2.meta_model import FFNN
from lib2 import transform
from config import config
from plot import plot_w
import pretrainedmodels
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from collections import Counter

seed = 123
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="VAT", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=3000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="mnist_idx_mix", type=str, help="dataset name : [svhn, cifar10, mnist]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="baseline", type=str, help="output dir")
parser.add_argument("--ood_ratio", default=0.5, type=float, help="ood ratio for unlabled set")
parser.add_argument("--n_label", default='boundary', type=str, help="number of labeled set")
parser.add_argument("--w_norm", default=0, type=int, help="normal initial ")
parser.add_argument("--less", default=0, type=int, help="remove ood in unlabeled set")
parser.add_argument("--meta_lr", default=0.01, type=float, help="learning rate for w")
parser.add_argument("--meta_val_batch", default=256, type=int, help="batch for meta val")
parser.add_argument("--reset", default=1, type=int, help="reset model when unstable training")

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
transform_fn = transform.transform(*dataset_cfg["transform"])  # transform function (flip, crop, noise)

l_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "l_train")
# u_train_dataset = dataset_cfg["dataset"](args.root,  args.n_label, "u_train")
if args.less == 1:
    u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_fashion_less_{}".format(args.ood_ratio))
else:
    u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_ood_{}".format(args.ood_ratio))
val_dataset = dataset_cfg["dataset"](args.root, args.n_label, "val")
test_dataset = dataset_cfg["dataset"](args.root, args.n_label, "test")

# if args.mix == 1:
#     u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_mix_ood_{}".format(args.ood_ratio))

print("labeled data : {}, unlabeled data : {}, training data : {}, OOD rario : {}".format(
    len(l_train_dataset), len(u_train_dataset), len(l_train_dataset) + len(u_train_dataset), args.ood_ratio))
print("validation data : {}, test data : {}".format(len(val_dataset), len(test_dataset)))
condition["number_of_data"] = {"OOD ratio": args.ood_ratio,
                               "labeled": len(l_train_dataset), "unlabeled": len(u_train_dataset),
                               "validation": len(val_dataset), "test": len(test_dataset)
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
        l_train_dataset, 64, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"][args.dataset] * 64)
    )
else:
    l_loader = DataLoader(
        l_train_dataset, shared_cfg["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), shared_cfg["iteration"][args.dataset] * shared_cfg["batch_size"])
    )
print("algorithm : {}".format(args.alg))
condition["algorithm"] = args.alg
exp_name += str(args.alg) + "_"

u_loader = DataLoader(
    u_train_dataset, 256, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), shared_cfg["iteration"][args.dataset] * 256)
)

val_loader_in_train = DataLoader(val_dataset, args.meta_val_batch, drop_last=True,
                                 sampler=RandomSampler(len(val_dataset), shared_cfg["iteration"][args.dataset] * args.meta_val_batch))

val_loader = DataLoader(val_dataset, 256, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, 256, shuffle=False, drop_last=False)
un_loader = DataLoader(u_train_dataset, 32, shuffle=False, drop_last=False)


model_name = 'vgg19' # could be fbresnet152 or inceptionresnetv2 or vgg19
n_clusters = 30
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet').cuda()
model.eval()
input_size = model.input_size
input_range = model.input_range
input_space = model.input_space
print(input_size)
emd = np.zeros([len(u_train_dataset), 4096])
for j, data in enumerate(un_loader):
    input, target, idx = data
    input, target = input.to(device).float(), target.to(device).long()
    feature = model.features(input)
    feature.squeeze()
    emd[idx] = feature.data.cpu().numpy()
np.save('output/emd/mix_{}_emd_{}_{}.npy'.format(model_name, args.ood_ratio, n_clusters), emd)

emd = np.load('output/emd/mix_{}_emd_{}_{}.npy'.format(model_name, args.ood_ratio, n_clusters))
kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=300).fit(emd)
k_label = kmeans.labels_
np.save('output/emd/mix_{}_kmean_{}_{}.npy'.format(model_name, n_clusters, args.ood_ratio), k_label)

# u_label = u_train_dataset.dataset['labels']
# k_label = np.load('output/emd/mix_{}_kmean_{}_{}.npy'.format(model_name, n_clusters, args.ood_ratio))
# idx_ood = np.argwhere(u_label < 0)
# idx_in = np.argwhere(u_label > -1)
# u_label[idx_ood] = 0
# u_label[idx_in] = 1
# k_label_ood = k_label[idx_ood]
# k_label_in = k_label[idx_in]
# # myList = [1,1,2,3,4,5,3,2,3,4,2,1,2,3]
# # print (Counter(myList))
# aaa = k_label_ood.squeeze().tolist()
# bbb = k_label_in.squeeze().tolist()
# print("OOD: ", Counter(aaa))
# print("In: ", Counter(bbb))
# ood_label = [0, 6, 17, 7, 15, 12, 14, 18, 1, 8, 11, 3, 10]
# in_label = [5, 2, 19, 13, 4, 9, 16]
# pred_k = np.zeros_like(k_label)
# for i in range(len(pred_k)):
#     if k_label[i] in in_label:
#         pred_k[i] = 1
#
# acc = accuracy_score(u_label, pred_k)
# print(acc)
print("xujiang zhao")