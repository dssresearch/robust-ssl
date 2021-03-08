#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "9"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse, math, time, json, os
import random
from lib import transform, meta_module_wrn
from config import config
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
print("now =", now)
# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--alg", "-a", default="VAT", type=str, help="ssl algorithm : [supervised, PI, MT, VAT, PL, ICT]")
parser.add_argument("--em", default=0, type=float, help="coefficient of entropy minimization. If you try VAT + EM, set 0.06")
parser.add_argument("--validation", default=5000, type=int, help="validate at this interval (default 25000)")
parser.add_argument("--dataset", "-d", default="cifar10", type=str, help="dataset name : [svhn, cifar10]")
parser.add_argument("--root", "-r", default="data", type=str, help="dataset dir")
parser.add_argument("--output", "-o", default="./exp_res", type=str, help="output dir")
parser.add_argument("--ood_ratio", default=0.5, type=float, help="ood ratio for unlabled set")
parser.add_argument("--n_label", default='class6', type=str, help="number of labeled set")
parser.add_argument("--meta_val_batch", default=50, type=int, help="batch for meta val")
parser.add_argument("--L1_trade_off", default=0e-07, type=float, help="L1_trade_off param")
parser.add_argument("--w_clip", default=1e-5, type=float, help="")
parser.add_argument("--meta_lr", default=0.01, type=float, help="learning rate for w")
parser.add_argument("--cluster", default=1, type=int, help="cluster or not")
parser.add_argument("--w_initial", default=0.05, type=float, help="weight initial")


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
    u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_in")
else:
    # u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_new_svhn_{}".format(args.ood_ratio))
    u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_ood_{}".format(args.ood_ratio))
    # u_train_dataset = dataset_cfg["dataset"](args.root, args.n_label, "u_train_ood_google_{}".format(args.ood_ratio))

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

model = meta_module_wrn.WideResNet(28, dataset_cfg["num_classes"], 2, transform_fn).to(device)
optimizer = optim.Adam(model.params(), lr=alg_cfg["lr"])

trainable_paramters = sum([p.data.nelement() for p in model.params()])
print("trainable parameters : {}".format(trainable_paramters))

if args.alg == "VAT": # virtual adversarial training
    from lib.algs.vat import VAT_base
    ssl_obj = VAT_base(alg_cfg["eps"][args.dataset], alg_cfg["xi"], 1)
elif args.alg == "PL": # pseudo label
    from lib.algs.pseudo_label import PL2
    ssl_obj = PL2(alg_cfg["threashold"])
elif args.alg == "MT": # mean teacher
    from lib.algs.mean_teacher import MT2
    t_model = meta_module_wrn.WideResNet(28, dataset_cfg["num_classes"], 2, transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT2(t_model, alg_cfg["ema_factor"])
elif args.alg == "PI": # PI Model
    from lib.algs.pimodel import PiModel2
    ssl_obj = PiModel2()
elif args.alg == "ICT": # interpolation consistency training
    from lib.algs.ict import ICT
    t_model = meta_module_wrn.WideResNet(28, dataset_cfg["num_classes"], 2, transform_fn).to(device)
    t_model.load_state_dict(model.state_dict())
    ssl_obj = ICT(alg_cfg["alpha"], t_model, alg_cfg["ema_factor"])
elif args.alg == "MM": # MixMatch
    from lib.algs.mixmatch import MixMatch
    ssl_obj = MixMatch(alg_cfg["T"], alg_cfg["K"], alg_cfg["alpha"])
elif args.alg == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(args.alg))

if args.ood_ratio == 0.25:
    ood_idx = [1,2,9,4,3,8,0]
    in_idx = [6,5,7]
elif args.ood_ratio == 0.5:
    ood_idx = [2, 4, 3, 9, 1, 6]
    in_idx = [5, 0, 7, 8]
else:
    ood_idx = [2, 7, 1]
    in_idx = [0, 3, 4, 5, 6, 8, 9]

emd_label = np.load('output/emd/cifar10/{}_{}_kmean_{}_new_label_{}_ood_{}.npy'.format(args.dataset, 'vgg19', 10, args.n_label, args.ood_ratio))
u_label = u_train_dataset.dataset['labels']

if args.cluster:
    weight = np.ones([20, 1])
else:
    weight = np.ones([len(u_label), 1])
weight = torch.tensor(weight* args.w_initial, dtype=torch.float32, device="cuda", requires_grad=True)
opt_w = optim.Adam([weight], lr=args.meta_lr)

iteration = 0
maximum_val_acc = 0
s = time.time()
start_time = time.time()
for l_data, u_data, v_data in zip(l_loader, u_loader, val_loader_in_train):
    iteration += 1
    l_input, target, _ = l_data
    l_input, target = l_input.to(device).float(), target.to(device).long()
    u_input, dummy_target, idx = u_data
    u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
    val_input, val_target, _ = v_data
    val_input, val_target = val_input.to(device).float(), val_target.to(device).long()
    coef = alg_cfg["consis_coef"] * math.exp(-5 * (1 - min(iteration / shared_cfg["warmup"], 1)) ** 2)
    if args.cluster:
        w_u = weight[emd_label[idx]]
    else:
        w_u = weight[idx].squeeze()
    meta_model = meta_module_wrn.WideResNet(28, dataset_cfg["num_classes"], 2, transform_fn).to(device)
    meta_model.load_state_dict(model.state_dict())
    outputs_hat = meta_model(u_input)
    ssl_loss_meta = ssl_obj(u_input, outputs_hat.detach(), meta_model)
    cls_loss_meta = F.cross_entropy(meta_model(l_input), target, reduction="none", ignore_index=-1).mean()
    loss_meta = cls_loss_meta + (ssl_loss_meta*w_u).mean() * coef
    meta_model.zero_grad()
    grads = torch.autograd.grad(loss_meta, meta_model.fc.params(), create_graph=True)
    meta_model.update_last_params(lr_inner=alg_cfg["lr"], source_params=grads)
    del grads
    val_loss_meta = F.cross_entropy(meta_model(val_input), val_target, reduction="none", ignore_index=-1).mean()
    val_loss_meta += args.L1_trade_off * torch.sum(torch.abs(weight))
    opt_w.zero_grad()
    val_loss_meta.backward()
    torch.nn.utils.clip_grad_norm_([weight], args.w_clip)
    opt_w.step()
    weight.data = torch.clamp(weight, min=0.0000000001, max=1)

    if args.cluster:
        w_up = weight[emd_label[idx]]
    else:
        w_up = weight[idx].squeeze()
    w = w_up.clone().detach()

    outputs_u = model(u_input)
    ssl_loss = ssl_obj(u_input, outputs_u.detach(), model) * coef
    # supervised loss
    outputs = model(l_input)
    cls_loss = F.cross_entropy(outputs, target, reduction="none", ignore_index=-1).mean()
    ssl_loss = (ssl_loss*w).mean()
    loss = cls_loss + ssl_loss * coef

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.alg == "MT" or args.alg == "ICT":
        # parameter update with exponential moving average
        ssl_obj.moving_average(model.parameters())
    # display
    if iteration == 1 or (iteration % 100) == 0:
        wasted_time = time.time() - s
        rest = (shared_cfg["iteration"] - iteration)/100 * wasted_time / 60
        print("iteration [{}/{}] cls loss : {:.6e}, SSL loss : {:.6e}, coef : {:.5e}, time : {:.3f} iter/sec, rest : {:.3f} min, lr : {}".format(
            iteration, shared_cfg["iteration"], cls_loss.item(), ssl_loss.item(), coef, 100 / wasted_time, rest, optimizer.param_groups[0]["lr"]))
        s = time.time()

    # validation
    if (iteration % args.validation) == 0 or iteration == shared_cfg["iteration"]:
        with torch.no_grad():
            model.eval()
            print("### validation ###")
            sum_acc = 0.
            s = time.time()
            for j, data in enumerate(val_loader):
                input, target, _ = data
                input, target = input.to(device).float(), target.to(device).long()

                output = model(input)

                pred_label = output.max(1)[1]
                sum_acc += (pred_label == target).float().sum()
                if ((j+1) % 10) == 0:
                    d_p_s = 10/(time.time()-s)
                    print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(j+1, len(val_loader), d_p_s, (len(val_loader) - j-1)/d_p_s))
                    s = time.time()
            acc = sum_acc/float(len(val_dataset))
            print("varidation accuracy : {}".format(acc))
            # test
            if maximum_val_acc < acc:
                print("### test ###")
                maximum_val_acc = acc
                sum_acc = 0.
                s = time.time()
                for j, data in enumerate(test_loader):
                    input, target, _ = data
                    input, target = input.to(device).float(), target.to(device).long()
                    output = model(input)
                    pred_label = output.max(1)[1]
                    sum_acc += (pred_label == target).float().sum()
                    if ((j+1) % 10) == 0:
                        d_p_s = 100/(time.time()-s)
                        print("[{}/{}] time : {:.1f} data/sec, rest : {:.2f} sec".format(j+1, len(test_loader), d_p_s, (len(test_loader) - j-1)/d_p_s))
                        s = time.time()
                test_acc = sum_acc / float(len(test_dataset))
                print("test accuracy : {}".format(test_acc))
                # torch.save(model.state_dict(), os.path.join(args.output, "best_model.pth"))
        model.train()
        s = time.time()
    # lr decay
    if iteration == shared_cfg["lr_decay_iter"]:
        optimizer.param_groups[0]["lr"] *= shared_cfg["lr_decay_factor"]

print("test acc : {}".format(test_acc))
condition["test_acc"] = test_acc.item()
end_time = time.time()
print("running time: ", (start_time-end_time)/60.0, " min")
exp_name += str(int(time.time())) # unique ID
if not os.path.exists(args.output):
    os.mkdir(args.output)
with open(os.path.join(args.output, exp_name + ".json"), "w") as f:
    json.dump(condition, f)
