import numpy as np
import argparse
from skimage.transform import resize
from sklearn.utils import shuffle


def mnist_ood_mixup():
    cifar_un = np.load("data/mnist/u_train.npy", allow_pickle=True)
    cifar_x = cifar_un.item()['images']
    cifar_y = cifar_un.item()['labels']

    ood_image = []
    ood_label = []
    for i in range(10):
        ci_images = cifar_x[(cifar_y == i)][:1000]
        for j in range(i+1, 10):
            cj_images = cifar_x[(cifar_y == j)][:1000]
            ood_i = (ci_images + cj_images) * 0.5
            c_labels = cifar_y[(cifar_y == j)][:1000]
            ood_image += [ood_i]
            ood_label += [np.zeros_like(c_labels) - 1]
    ood_image = np.concatenate(ood_image, 0)
    ood_label = np.concatenate(ood_label, 0)
    un_set = {"images": ood_image, "labels": ood_label}
    np.save('data/mnist/u_train_mix', un_set)

    return


def show_image(data):
    from matplotlib import pyplot as plt
    plt.imshow(data, interpolation='nearest')
    plt.show()
    plt.clf()

def mnist_ood_boundary(ratio):
    mnist_un = np.load("data/mnist/u_train.npy", allow_pickle=True)
    fashion_mnist_un = np.load("data/mnist/u_train_mix.npy", allow_pickle=True)

    mnist_y = mnist_un.item()['labels']
    mnist_x = mnist_un.item()['images']
    fashion_mnist_y = fashion_mnist_un.item()['labels']
    fashion_mnist_x = fashion_mnist_un.item()['images']
    print("mnist_x shape:", mnist_x.shape)
    fashion_mnist_y = np.zeros_like(fashion_mnist_y) - 1
    num_un = len(mnist_x)
    mnist_x, mnist_y = shuffle(mnist_x, mnist_y, random_state=0)
    fashion_mnist_x = shuffle(fashion_mnist_x)
    for i in range(10):
        show_image(fashion_mnist_x[i][0])
    if ratio == 0:
        ood_un = mnist_x
        ood_y = mnist_y
    elif ratio == 1:
        ood_un = fashion_mnist_x[:num_un]
        ood_y = fashion_mnist_y[:num_un]
    else:
        un_in = mnist_x[:int((1 - ratio) * num_un)]
        ood_in_y = mnist_y[:int((1 - ratio) * num_un)]
        un_ood = fashion_mnist_x[:int(ratio * num_un)]
        ood_un_y = fashion_mnist_y[:int(ratio * num_un)]
        print('un_in shape:', un_in.shape)
        print('un_ood shape:', un_ood.shape)
        ood_un = np.concatenate((un_in, un_ood), axis=0)
        ood_y = np.concatenate((ood_in_y, ood_un_y), axis=0)
    un = {"images": ood_un, "labels": ood_y}
    np.save('data/mnist/u_train_boundary_ood_{}'.format(ratio), un)
    return un

def mnist_ood_val():
    mnist_un = np.load("data/mnist_100/val.npy", allow_pickle=True)
    mnist_y = mnist_un.item()['labels']
    mnist_x = mnist_un.item()['images']
    mnist_x, mnist_y = shuffle(mnist_x, mnist_y, random_state=0)
    num_un = [25, 50, 100, 500, 1000, 2000, 5000]
    for n in num_un:
        n_in = mnist_x[:n]
        ood_in_y = mnist_y[:n]
        un = {"images": n_in, "labels": ood_in_y}
        np.save('data/mnist_100/val_{}'.format(n), un)
    return

def mnist_ood(ratio):
    mnist_un = np.load("data/mnist/u_train.npy", allow_pickle=True)
    fashion_mnist_un = np.load("data/fashion_mnist/u_train.npy", allow_pickle=True)

    mnist_y = mnist_un.item()['labels']
    mnist_x = mnist_un.item()['images']
    fashion_mnist_y = fashion_mnist_un.item()['labels']
    fashion_mnist_x = fashion_mnist_un.item()['images']
    print("mnist_x shape:", mnist_x.shape)
    fashion_mnist_y = np.zeros_like(fashion_mnist_y) - 1
    num_un = len(mnist_x)
    mnist_x, mnist_y = shuffle(mnist_x, mnist_y, random_state=0)
    fashion_mnist_x = shuffle(fashion_mnist_x)

    if ratio == 0:
        ood_un = mnist_x
        ood_y = mnist_y
    elif ratio == 1:
        ood_un = fashion_mnist_x[:num_un]
        ood_y = fashion_mnist_y[:num_un]
    else:
        un_in = mnist_x[:int((1 - ratio) * num_un)]
        ood_in_y = mnist_y[:int((1 - ratio) * num_un)]
        # un_ood = np.random.uniform(-1, 1, [int(num_un * ratio) - 1, 1, 28, 28])
        un_ood = fashion_mnist_x[:int(ratio * num_un)]
        ood_un_y = fashion_mnist_y[:int(ratio * num_un)]
        print('un_in shape:', un_in.shape)
        print('un_ood shape:', un_ood.shape)
        ood_un = np.concatenate((un_in, un_ood), axis=0)
        ood_y = np.concatenate((ood_in_y, ood_un_y), axis=0)
        un_less = {"images": un_in, "labels": ood_in_y}
        np.save('data/mnist/u_train_fashion_less_{}'.format(ratio), un_less)
    un = {"images": ood_un, "labels": ood_y}
    np.save('data/mnist/u_train_fashion_ood_{}'.format(ratio), un)
    return un


ratio = [0.0, 0.25, 0.5, 0.75]
for r in ratio:
    mnist_ood(r)
mnist_ood_mixup()
for r in ratio:
    mnist_ood_boundary(r)

