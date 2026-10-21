"""Data: the four benchmarks, Dirichlet client split, and GPU-resident caches.

Every image goes through the standard pipeline exactly once (PIL Resize(256) ->
CenterCrop(224), as uint8). Normalisation is a lookup into a 3x256 table built with
torchvision's own ToTensor+Normalize, so a normalised batch is bit-identical to what
the per-sample DataLoader pipeline would produce, and no CPU work happens per step.

The backbone is frozen, so the query q(x) of every image is also computed once.
"""
import os
import numpy as np
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

NORM_STATS = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Dataset')
PRE = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])  # uint8, normalised later


def _n_workers():
    return max(1, len(os.sched_getaffinity(0)) - 1)


class Normalizer:
    """uint8 (B, 3, H, W) -> ToTensor+Normalize output, bit for bit, via a 3x256 lookup table."""

    def __init__(self, device):
        levels = TF.to_tensor(np.arange(256, dtype=np.uint8).reshape(256, 1, 1).repeat(3, axis=2))  # (3, 256, 1)
        self.lut = TF.normalize(levels, *NORM_STATS).reshape(-1).to(device)
        self.chan = (torch.arange(3, device=device) * 256).view(1, 3, 1, 1)

    def __call__(self, u8):
        return self.lut[u8.long() + self.chan]


class ImageCache:
    """A whole dataset on the GPU as uint8 (N, 3, 224, 224) plus labels."""

    def __init__(self, dataset, device, batch_size=500):
        loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=_n_workers())
        imgs, labels = zip(*[(x.to(device), torch.as_tensor(y).to(device)) for x, y in loader])
        self.images = torch.cat(imgs)
        self.labels = torch.cat(labels).long()
        self.norm = Normalizer(device)

    def __len__(self):
        return self.images.size(0)

    def get(self, idx):
        return self.norm(self.images[idx]), self.labels[idx]


@torch.no_grad()
def compute_queries(model, cache, batch_size=500):
    out = [model.query(cache.get(ids)[0]) for ids in torch.arange(len(cache), device=cache.images.device).split(batch_size)]
    return torch.cat(out)


@torch.no_grad()
def dataset_queries(query_fn, dataset, device, batch_size=500):
    """q(x) for every sample, streaming the images once (same batches and inputs as compute_queries)."""
    norm = Normalizer(device)
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=_n_workers(), pin_memory=True)
    return torch.cat([query_fn(norm(x.to(device, non_blocking=True))) for x, _ in loader])


# ---------------------------------------------------------------- datasets
class _Images(Data.Dataset):
    """(PIL-loadable item, label) pairs -> (uint8 3x224x224, label)."""

    def __init__(self, get_pil, labels):
        self.get_pil, self.labels = get_pil, np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return PRE(self.get_pil(i).convert('RGB')), int(self.labels[i])


def load_cifar10(root=DATA_ROOT):
    train = torchvision.datasets.CIFAR10(root, train=True, download=False, transform=PRE)
    test = torchvision.datasets.CIFAR10(root, train=False, download=False, transform=PRE)
    return train, test, 10


def _torchvision_split(cls, root, train):
    ds = cls(root, train=train, download=False)
    return _Images(lambda i: ds[i][0], ds.targets)


def _tinyimagenet(root, train):
    base = os.path.join(root, 'tiny-imagenet-200')
    classes = sorted(os.listdir(os.path.join(base, 'train')))
    cidx = {c: i for i, c in enumerate(classes)}
    if train:
        files = [(os.path.join(base, 'train', c, 'images', f), cidx[c])
                 for c in classes for f in sorted(os.listdir(os.path.join(base, 'train', c, 'images')))]
    else:
        lines = open(os.path.join(base, 'val', 'val_annotations.txt')).read().split('\n')
        files = [(os.path.join(base, 'val', 'images', l.split('\t')[0]), cidx[l.split('\t')[1]]) for l in lines if l]
    return _Images(lambda i: Image.open(files[i][0]), [y for _, y in files])


def _balanced_subset(labels, n_classes, size, seed=0):
    """The original four_dataset_reader's selection: the first size//C samples of every class, shuffled with seed 0."""
    np.random.seed(seed)
    idx = np.concatenate([np.where(np.asarray(labels) == c)[0][:size // n_classes] for c in range(n_classes)])
    np.random.shuffle(idx)
    return idx


def _fourdataset(root, train):
    """MNIST-M, Fashion-MNIST, CINIC-10, MMAFEDB with labels shifted to 0-9, 10-19, 20-29, 30-36 (App. A.6)."""
    size = 30000 if train else 2500
    split = 'train' if train else 'test'
    sources = []
    mx, my = torch.load(os.path.join(root, 'MNISTM', 'processed', f'mnist_m_{split}.pt'))
    sources.append((lambda i, d=mx: Image.fromarray(d[i].numpy(), mode='RGB'), my.numpy(), 10))
    fm = torchvision.datasets.FashionMNIST(os.path.join(root, 'FashionMNIST'), train=train, download=False)
    sources.append((lambda i, d=fm: d[i][0], fm.targets.numpy(), 10))
    for name, n_cls in (('cinic10-py', 10), ('MMAFEDB', 7)):
        f = torchvision.datasets.ImageFolder(os.path.join(root, name, split))
        sources.append((lambda i, d=f: Image.open(d.samples[i][0]), np.array(f.targets), n_cls))
    getters, labels, source = [], [], []
    shift = 0
    for s, (get, lab, n_cls) in enumerate(sources):
        n = size if size % n_cls == 0 else int(np.ceil(size / n_cls)) * n_cls  # MMAFEDB rounding, as in the original
        for j in _balanced_subset(lab, n_cls, n):
            getters.append((get, int(j)))
            labels.append(int(lab[j]) + shift)
            source.append(s)
        shift += n_cls
    return _Images(lambda i: getters[i][0](getters[i][1]), labels), np.array(source)


def load_dataset(name, root=DATA_ROOT):
    """Returns dict(train, test, n_classes, train_labels, train_source, test_source); *_source is None
    except for fourdataset, where it gives each sample's source dataset (clients are split per source)."""
    out = dict(train_source=None, test_source=None)
    if name == 'cifar10':
        out.update(train=_torchvision_split(torchvision.datasets.CIFAR10, root, True),
                   test=_torchvision_split(torchvision.datasets.CIFAR10, root, False), n_classes=10)
    elif name == 'cifar100':
        out.update(train=_torchvision_split(torchvision.datasets.CIFAR100, root, True),
                   test=_torchvision_split(torchvision.datasets.CIFAR100, root, False), n_classes=100)
    elif name == 'tinyimagenet':
        out.update(train=_tinyimagenet(root, True), test=_tinyimagenet(root, False), n_classes=200)
    elif name == 'fourdataset':
        (tr, trs), (te, tes) = _fourdataset(root, True), _fourdataset(root, False)
        out.update(train=tr, test=te, n_classes=37, train_source=trs, test_source=tes)
    else:
        raise ValueError(name)
    out['train_labels'] = out['train'].labels
    return out


def dirichlet_split(labels, n_clients, alpha, seed, min_size=32):
    """Label-skewed partition (the standard Dirichlet scheme); redraws until every client has min_size samples."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p)) for p in parts]


def client_split(data, n_clients, alpha, seed):
    """Dirichlet split; for fourdataset, n_clients/4 clients per source dataset, as in the original setup."""
    if data['train_source'] is None:
        return dirichlet_split(data['train_labels'], n_clients, alpha, seed)
    parts = []
    for s in np.unique(data['train_source']):
        idx = np.flatnonzero(data['train_source'] == s)
        parts += [idx[p] for p in dirichlet_split(data['train_labels'][idx], n_clients // 4, alpha, seed + int(s))]
    return parts
