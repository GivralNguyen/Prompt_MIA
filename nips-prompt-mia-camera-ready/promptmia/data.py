"""Data: CIFAR-10, Dirichlet client split, and GPU-resident caches.

Every image goes through the standard pipeline exactly once (PIL Resize(256) ->
CenterCrop(224)) and is kept on the GPU as uint8. Normalisation is a lookup into a
3x256 table built with torchvision's own ToTensor+Normalize, so a cached batch is
bit-identical to what the per-sample DataLoader pipeline would produce, and no CPU
work happens during training.

The backbone is frozen, so the query q(x) of every image is also computed once.
"""
import os
import numpy as np
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

NORM_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Dataset')


def _n_workers():
    return max(1, len(os.sched_getaffinity(0)) - 1)


def load_cifar10(root=DATA_ROOT):
    pre = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])  # uint8, no normalisation yet
    train = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=pre)
    test = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=pre)
    return train, test, 10


class ImageCache:
    """A whole dataset on the GPU as uint8 (N, 3, 224, 224) plus labels."""

    def __init__(self, dataset, device, batch_size=500):
        loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=_n_workers())
        imgs, labels = zip(*[(x.to(device), torch.as_tensor(y).to(device)) for x, y in loader])
        self.images = torch.cat(imgs)
        self.labels = torch.cat(labels).long()
        levels = TF.to_tensor(np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, axis=2))  # (3, 256, 1)
        self.lut = TF.normalize(levels, *NORM_STATS).reshape(-1).to(device)                       # 3*256 values
        self.chan = (torch.arange(3, device=device) * 256).view(1, 3, 1, 1)

    def __len__(self):
        return self.images.size(0)

    def get(self, idx):
        return self.lut[self.images[idx].long() + self.chan], self.labels[idx]


@torch.no_grad()
def compute_queries(model, cache, batch_size=500):
    out = [model.query(cache.get(ids)[0]) for ids in torch.arange(len(cache), device=cache.images.device).split(batch_size)]
    return torch.cat(out)


def dirichlet_split(labels, n_clients, alpha, seed, min_size=32):
    """Label-skewed partition (the standard Dirichlet scheme); redraws until every client has min_size samples."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_classes = labels.max() + 1
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p)) for p in parts]
