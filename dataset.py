import codecs
import string
from pathlib import Path
import warnings

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, download_and_extract_archive, \
                                       extract_archive, verify_str_arg

from redis_client import RedisClient


resources = [
    ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
]

root_dir = "data"
dataset_name = "mnist"
raw_dir = Path(root_dir, dataset_name, 'raw').resolve()
processed_dir = Path(root_dir, dataset_name, 'processed').resolve()
training_file = 'training.pt'
test_file = 'test.pt'

classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
           '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']


class MnistDataset(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, train=True, transform=None, target_transform=None):
        super().__init__()
        self.train = train  # training set or test set
        self.db_key = f"{dataset_name}_train" if train else f"{dataset_name}_test"
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        rdb = RedisClient()
        d = rdb.get(self.db_key)[index]
        img, target = d[0], int(d[1])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        rdb = RedisClient()
        return len(rdb.get(self.db_key))


def download():
    """Download the MNIST data if it doesn't exist in raw_folder already."""

    if raw_dir.joinpath('train-images-idx3-ubyte').exists() and \
       raw_dir.joinpath('train-labels-idx1-ubyte').exists() and \
       raw_dir.joinpath('t10k-images-idx3-ubyte').exists() and \
       raw_dir.joinpath('t10k-labels-idx1-ubyte').exists():
        return

    print('Downloading...')

    raw_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    processed_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    # download files
    for url, md5 in resources:
        filename = url.rpartition('/')[2]
        download_and_extract_archive(url, download_root=str(raw_dir), filename=filename, md5=md5)

    print('Done!')


def preprocess():
    """Load the MNIST raw data and make preprocessed data"""
    if processed_dir.joinpath(training_file).exists() and \
       processed_dir.joinpath(test_file).exists():
        return

    print('Processing...')

    training_set = (
        read_image_file(raw_dir.joinpath('train-images-idx3-ubyte')),
        read_label_file(raw_dir.joinpath('train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(raw_dir.joinpath('t10k-images-idx3-ubyte')),
        read_label_file(raw_dir.joinpath('t10k-labels-idx1-ubyte'))
    )

    with open(processed_dir.joinpath(training_file), 'wb') as f:
        torch.save(training_set, f)
    with open(processed_dir.joinpath(test_file), 'wb') as f:
        torch.save(test_set, f)

    print('Done!')


def store_to_redis():
    train_x, train_y = torch.load(processed_dir.joinpath(training_file))
    test_x, test_y = torch.load(processed_dir.joinpath(test_file))

    trainset = [(x.numpy(), y.numpy()) for x, y in zip(train_x, train_y)]
    testset = [(x.numpy(), y.numpy()) for x, y in zip(test_x, test_y)]

    db = RedisClient()

    db.set_data_list(f"{dataset_name}_train", trainset)
    db.set_data_list(f"{dataset_name}_test", testset)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path):
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        import gzip
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def read_sn3_pascalvincent_tensor(path, strict=True):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # typemap
    if not hasattr(read_sn3_pascalvincent_tensor, 'typemap'):
        read_sn3_pascalvincent_tensor.typemap = {
            8: (torch.uint8, np.uint8, np.uint8),
            9: (torch.int8, np.int8, np.int8),
            11: (torch.int16, np.dtype('>i2'), 'i2'),
            12: (torch.int32, np.dtype('>i4'), 'i4'),
            13: (torch.float32, np.dtype('>f4'), 'f4'),
            14: (torch.float64, np.dtype('>f8'), 'f8')}
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = read_sn3_pascalvincent_tensor.typemap[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path):
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x


if __name__ == "__main__":
    download()
    preprocess()
    store_to_redis()

