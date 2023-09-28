# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold

from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import napari


def load_model(input_shape, L=1, K=4, hidden_channels=64, split_mode='channel', scale=True):
    """returns nf.MultiscaleFlow model. We are currently hacking this model to work with 3d images by casting z planes
    as channels. Seems to work ok. input_shape=(Z,Y,X)"""
    # Set up flows, distributions and merge operations
    channels = input_shape[0]
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                          split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                            input_shape[2] // 2 ** L)
        # q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes=num_classes)]
        q0 += [nf.distributions.DiagGaussian(latent_shape)]

    # Construct flow model with the multiscale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)

    return model


def train_flow(df, max_iter=100_000, input_shape=(10, 12, 12), batch_size=128, L=1, K=4, hidden_channels=64, split_mode='channel', scale=True,
               enable_cuda=True):
    # initialize cuda device
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

    # load data
    train_iter, train_loader, _ = load_data(df, batch_size=batch_size)

    # load model
    model = load_model(input_shape, L=L, K=K, hidden_channels=hidden_channels, split_mode=split_mode, scale=scale).to(device)

    # array for storing loss history
    loss_hist = np.array([])

    # initialize optimizer
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # loop over iterations and minimze the log likelihood
    for i in tqdm(range(max_iter)):
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)

        optimizer.zero_grad()
        x = x.to(device)
        x = x.type(torch.cuda.FloatTensor)
        loss = model.forward_kld(x)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())

    return loss_hist


def load_data(df, batch_size=128):
    probs = df.iloc[:, -1]
    data = pd.DataFrame(df[probs < 0.1].data, columns=['data'])
    train_data_list = data.data.iloc[:50_000].to_list()
    test_data_list = data.data.iloc[50_000:55_000].to_list()

    tmp = [np.expand_dims(d, axis=0) for d in train_data_list]
    train_data = np.concatenate(tmp, axis=0)
    train_data = train_data.astype('float32')
    tmp = [np.expand_dims(d, axis=0) for d in test_data_list]
    test_data = np.concatenate(tmp, axis=0)
    test_data = test_data.astype('float32')

    for i, voxel in enumerate(train_data):
        train_data[i] = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))

    for i, voxel in enumerate(test_data):
        test_data[i] = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))

    # note: seems like data needs to have even dimensions
    train_data_pad = np.zeros((len(train_data), 10, 12, 12))
    for i, voxel in enumerate(train_data):
        train_data_pad[i] = np.pad(voxel, ((1, 0), (1, 0), (1, 0)))
    train_data = train_data_pad

    test_data_pad = np.zeros((len(test_data), 10, 12, 12))
    for i, voxel in enumerate(test_data):
        test_data_pad[i] = np.pad(voxel, ((1, 0), (1, 0), (1, 0)))
    test_data = test_data_pad

    transform = tv.transforms.ToTensor()
    train_data = Dataset(train_data, transform=transform)
    test_data = Dataset(test_data, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    train_iter = iter(train_loader)

    return train_iter, train_loader, test_loader


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, transform=None):
        'Initialization'
        self.x = x

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.x[index]