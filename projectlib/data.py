import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

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

def load_dataset(dataset, transforms):
    if dataset == "mnist":
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
