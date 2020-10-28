from lib2.datasets import svhn, cifar10, mnist, synthetic
import numpy as np

shared_config = {
    # "iteration" : 50000,
    "iteration" : {"cifar10":500000, "cifar10_idx":500000, "svhn":300000, "two_moons": 10000, "mnist_idx": 10000, "mnist_idx_mix": 10000},
    "warmup" : 200000,
    "lr_decay_iter" : 400000,
    "lr_decay_factor" : 0.2,
    "batch_size" : 100,
}
### dataset ###
svhn_config = {
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
}
cifar10_config = {
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,
}
cifar10_idx_config = {
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10_idx,
    "num_classes" : 10,
}
mnist_config = {
    "transform" : [False, False, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,

}

mnist_idx_config = {
    # "transform" : [True, True, True],
    "transform": [False, False, False],
    "dataset" : mnist.MNIST_idx,
    "num_classes" : 10,

}

mnist_idx_mix_config = {
    "transform" : [False, False, False],
    "dataset" : mnist.MNIST_idx_mix,
    "num_classes" : 10,
}

two_moons_config = {
    "transform" : [False, False, False],
    "dataset" : synthetic.Two_moons,
    "num_classes" : 1,
}
### algorithm ###
vat_config = {
    # virtual adversarial training
    "xi" : 1e-6,
    "eps" : {"cifar10":6, "cifar10_idx":6, "svhn":1, "mnist":1, "mnist_idx":3, "mnist_idx_mix":3, "two_moons":0.5},
    "consis_coef" : 0.3,
    "lr" : 3e-3
}
pl_config = {
    # pseudo label
    "threashold" : 0.95,
    "lr" : 3e-4,
    "consis_coef" : 1,
}
mt_config = {
    # mean teacher
    "ema_factor" : 0.95,
    "lr" : 4e-3,
    "consis_coef" : 8,
}
pi_config = {
    # Pi Model
    "lr" : 3e-3,
    "consis_coef" : 20.0,
}
ict_config = {
    # interpolation consistency training
    "ema_factor" : 0.999,
    "lr" : 4e-4,
    "consis_coef" : 100,
    "alpha" : 0.1,
}
mm_config = {
    # mixmatch
    "lr" : 3e-3,
    "consis_coef" : 100,
    "alpha" : 0.75,
    "T" : 0.5,
    "K" : 2,
}
supervised_config = {
    "lr" : 3e-3
}
### master ###
config = {
    "shared" : shared_config,
    "svhn" : svhn_config,
    "cifar10" : cifar10_config,
    "cifar10_idx": cifar10_idx_config,
    "mnist" : mnist_config,
    "mnist_idx": mnist_idx_config,
    "mnist_idx_mix": mnist_idx_mix_config,
    "two_moons": two_moons_config,
    "VAT" : vat_config,
    "PL" : pl_config,
    "MT" : mt_config,
    "PI" : pi_config,
    "ICT" : ict_config,
    "MM" : mm_config,
    "supervised" : supervised_config
}