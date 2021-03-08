import numpy as np
import os
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt

class MNIST:
    def __init__(self, root, n_label, split="l_train"):
        self.dataset = np.load(os.path.join(root, "mnist_{}".format(n_label), split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        image = (image/255. - 0.5)/0.5
        return image, label

    def __len__(self):
        return len(self.dataset["images"])


class MNIST_idx:
    def __init__(self, root, n_label, split="l_train"):
        self.dataset = np.load(os.path.join(root, "mnist_{}".format(n_label), split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        image = (image/255. - 0.5)/0.5
        image = resize(image, (1, 32, 32))
        return image, label, idx

    def __len__(self):
        return len(self.dataset["images"])


class MNIST_idx_mix:
    def __init__(self, root, n_label, split="l_train"):
        self.dataset = np.load(os.path.join(root, "mnist_{}".format(n_label), split+".npy"), allow_pickle=True).item()

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        image = image/255.
        image = np.squeeze(image)
        image = resize(image, (224, 224))
        img = np.zeros([224, 224, 3])
        img[:, :, 0] = image
        img[:, :, 1] = image
        img[:, :, 2] = image
        image = np.transpose(img, (2, 0, 1))
        return image, label, idx

    def __len__(self):
        return len(self.dataset["images"])