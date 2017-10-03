import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import torchvision

torch.manual_seed(0)

# load mnist dataset
def load_mnist_dataset(shuffle=True, batch_size = 64, num_workers=1, root_dir='./data/mnist'):
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = dset.MNIST(root=root_dir, train=True, transform=data_transform, download=False)
    test_set = dset.MNIST(root=root_dir, train=False, transform=data_transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    return train_loader, test_loader

# load fashion-mnist dataset
def load_fashion_mnist_dataset(shuffle=True, batch_size = 64, num_workers=1, root_dir='./data/fashion_mnist'):
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = FashionMNIST(root=root_dir, train=True, transform=data_transform, download=False)
    test_set = FashionMNIST(root=root_dir, train=False, transform=data_transform, download=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    return train_loader, test_loader

# load mnist dataset
def load_cifar10_dataset(shuffle=True, batch_size = 64, num_workers=1, root_dir='./data/cifar10'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=num_workers)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    return train_loader, test_loader


class FashionMNIST(dset.MNIST):
    """`Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]