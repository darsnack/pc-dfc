import torch
import jax
import jax.random as jrng
import jax.numpy as jnp
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

def _to_tensor(x):
    if isinstance(x, jax.Array):
        return torch.from_numpy(np.array(x))
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list):
        return torch.Tensor(x)
    else:
        return x

class OneHot(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, label):
        if isinstance(label, int):
            label = torch.tensor(label)

        return F.one_hot(label, self.num_classes)

class JaxImageOrder(torch.nn.Module):
    def forward(self, img):
        # CHW --> HWC
        return torch.permute(img, (1, 2, 0))

def load_dataset(cfg, rng_key, transforms = None):
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
    elif cfg.name == "gaussian":
        xs_trn_key, xs_tst_key, w_key = jrng.split(rng_key, 3)
        xs_trn = jrng.normal(xs_trn_key, (cfg.train_size, cfg.input_dim)) * cfg.input_stddev
        xs_tst = jrng.normal(xs_tst_key, (cfg.test_size, cfg.input_dim)) * cfg.input_stddev
        w = jrng.normal(w_key, (cfg.input_dim,))
        ys_trn = xs_trn @ w
        ys_tst = xs_tst @ w
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
        raise RuntimeError(f"Unknown {dataset=}.")

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
