import torch
import jax
import jax.random as jrng
import jax.numpy as jnp
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import v2 as transforms

def _to_tensor(x):
    if isinstance(x, jax.Array):
        return torch.from_numpy(np.array(x))
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list):
        return torch.Tensor(x)
    else:
        return x

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape((-1,))

class OneHot(torch.nn.Module):
    def __init__(self, num_classes, soft_onehot = False):
        super().__init__()
        self.soft_onehot = soft_onehot
        self.num_classes = num_classes

    def forward(self, label):
        if isinstance(label, int):
            label = torch.tensor(label)

        if self.soft_onehot:
            return 0.94 * F.one_hot(label, self.num_classes) + 0.03
        else:
            return F.one_hot(label, self.num_classes)

class EMNISTFlip(torch.nn.Module):
    def forward(self, img):
        # Fix EMNIST orientation: swap height and width axes
        return img.permute(0, 2, 1)

class JaxImageOrder(torch.nn.Module):
    def forward(self, img):
        # CHW --> HWC
        return torch.permute(img, (1, 2, 0))

def default_data_transforms(cfg):
    if cfg.name == "mnist":
        img_transforms = [transforms.ToImage(),
                          transforms.ToDtype(torch.float32, scale=True)]
        if cfg.normalize:
            img_transforms.append(transforms.Normalize((0.1307,), (0.3081,)))
        img_transforms.append(JaxImageOrder())
        if cfg.flatten:
            img_transforms.append(Flatten())

        return transforms.Compose(img_transforms), OneHot(10, soft_onehot=True)
    elif cfg.name == "fashionmnist":
        img_transforms = [transforms.ToImage(),
                          transforms.ToDtype(torch.float32, scale=True)]
        if cfg.normalize:
            img_transforms.append(transforms.Normalize((0.2860,), (0.3530,)))
        img_transforms.append(JaxImageOrder())
        if cfg.flatten:
            img_transforms.append(Flatten())

        return transforms.Compose(img_transforms), OneHot(10, soft_onehot=True)
    elif cfg.name == "cifar10":
        img_transforms = [transforms.ToImage(),
                          transforms.ToDtype(torch.float32, scale=True)]
        if cfg.normalize:
            img_transforms.append(transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            ))
        img_transforms.append(JaxImageOrder())
        if cfg.flatten:
            img_transforms.append(Flatten())

        return transforms.Compose(img_transforms), OneHot(10, soft_onehot=True)
    elif cfg.name == "emnist":
        img_transforms = [transforms.ToImage(),
                          transforms.ToDtype(torch.float32, scale=True),
                          EMNISTFlip()] # fix orientation
        if cfg.normalize:
            img_transforms.append(transforms.Normalize((0.1736,), (0.3248,)))
        img_transforms.append(JaxImageOrder())
        if cfg.flatten:
            img_transforms.append(Flatten())
        return transforms.Compose(img_transforms), OneHot(47, soft_onehot=True)
    else:
        return None, None

def load_dataset(cfg, rng_key, transforms = (None, None)):
    if cfg.name == "mnist":
        return {
            "train": datasets.MNIST(
                "~/torchvision",
                train=True,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
            "test": datasets.MNIST(
                "~/torchvision",
                train=False,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
        }
    elif cfg.name == "emnist":
        return {
            "train": datasets.EMNIST(
                "~/torchvision",
                split=cfg.split,
                train=True,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
            "test": datasets.EMNIST(
                "~/torchvision",
                split=cfg.split,
                train=False,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
        }
    elif cfg.name == "fashionmnist":
        return {
            "train": datasets.FashionMNIST(
                "~/torchvision",
                train=True,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
            "test": datasets.FashionMNIST(
                "~/torchvision",
                train=False,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
        }
    elif cfg.name == "cifar10":
        return {
            "train": datasets.CIFAR10(
                "~/torchvision",
                train=True,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
            "test": datasets.CIFAR10(
                "~/torchvision",
                train=False,
                download=True,
                transform=transforms[0],
                target_transform=transforms[1],
            ),
        }
    elif cfg.name == "gaussian":
        xs_trn_key, xs_tst_key, w_key = jrng.split(rng_key, 3)
        xs_trn = jrng.normal(xs_trn_key, (cfg.train_size, cfg.input_dim)) * cfg.input_stddev
        xs_tst = jrng.normal(xs_tst_key, (cfg.test_size, cfg.input_dim)) * cfg.input_stddev
        w = jrng.normal(w_key, (cfg.input_dim, cfg.output_dim))
        ys_trn = xs_trn @ w
        ys_tst = xs_tst @ w
        if ys_trn.ndim == 1:
            ys_trn = ys_trn[:, None]
            ys_tst = ys_tst[:, None]
        # convert to torch
        xs_trn = _to_tensor(xs_trn)
        xs_tst = _to_tensor(xs_tst)
        ys_trn = _to_tensor(ys_trn)
        ys_tst = _to_tensor(ys_tst)

        return {
            "train": TensorDataset(xs_trn, ys_trn),
            "test": TensorDataset(xs_tst, ys_tst),
        }
    elif cfg.name == "gaussian-clamped":
        xs_trn_key, xs_tst_key, w_key = jrng.split(rng_key, 3)
        xs_trn = jrng.normal(xs_trn_key, (cfg.train_size, cfg.input_dim)) * cfg.input_stddev
        xs_tst = jrng.normal(xs_tst_key, (cfg.test_size, cfg.input_dim)) * cfg.input_stddev
        ys_trn = jnp.ones_like(xs_trn, shape=(cfg.train_size, cfg.output_dim))
        ys_tst = jnp.ones_like(xs_tst, shape=(cfg.test_size, cfg.output_dim))
        if ys_trn.ndim == 1:
            ys_trn = ys_trn[:, None]
            ys_tst = ys_tst[:, None]
        # convert to torch
        xs_trn = _to_tensor(xs_trn)
        xs_tst = _to_tensor(xs_tst)
        ys_trn = _to_tensor(ys_trn)
        ys_tst = _to_tensor(ys_tst)

        return {
            "train": TensorDataset(xs_trn, ys_trn),
            "test": TensorDataset(xs_tst, ys_tst),
        }
    else:
        raise RuntimeError(f"Unknown {cfg.name=}.")

def build_dataloader(data,
                     batch_size=1,
                     shuffle=True,
                     drop_remainder=False,
                     collate_fn=None,
                     nworkers=2):
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nworkers,
        persistent_workers=True,
        drop_last=drop_remainder,
        collate_fn=collate_fn,
        multiprocessing_context="spawn"
    )
