# Import required packages
import torch
import torchvision as tv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from tqdm import tqdm


class CNN3d(nn.Module):
    """for voxels of size 10x12x12"""
    def __init__(self, channels_1=4, channels_2=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=channels_1, kernel_size=(3, 3, 3), padding='same')
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d(in_channels=4, out_channels=channels_2, kernel_size=(3, 3, 3), padding='same')
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=channels_2 * (10 // 4) * (12 // 4) * (12 // 4), out_features=512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.sigmoid(self.fc2(x))

        return x


class LabelledDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, x, y, transform=None):
        'Initialization'
        self.x = x
        self.y = y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.x[index], self.y[index]


def weighted_bce_loss(weights):
    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss


def train_classifier(df, channels_1=4, channels_2=4, learning_rate=1e-4, batch_size=8,
                n_epochs=100, enable_cuda=True):
    """df = dataframe with columns ['data', 'manual_classification']"""
    # initialize cuda device
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

    # remove spots that were not classified
    df = df[[label is not None for label in df.manual_classification]]

    # load data and labels into arrays.
    train_loader, test_loader = load_data(df, batch_size=batch_size)

    # get class weights
    weights = get_class_weights(df)

    # define loss function
    criterion = weighted_bce_loss(weights=torch.tensor(weights).to(device))

    # set up the model
    model = CNN3d(channels_1=channels_1, channels_2=channels_2).to(device)

    # set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # array for recording output
    loss_hist = np.zeros(n_epochs)

    # loop over epochs and train model
    for j in tqdm(range(n_epochs)):

        # keep a running total of the loss function across batches
        running_loss = 0.0
        for i, val in enumerate(train_loader):
            x, y = val
            x, y = x.to(device).type(torch.cuda.FloatTensor), y.to(device).type(torch.cuda.FloatTensor)

            # run the usual pytorch training commands
            # zero optimizer gradients
            optimizer.zero_grad()

            # evaluate the model. note: final output of model has shape (batch_size, 1), so we need to squeeze
            outputs = model(x).squeeze()

            # evaluate loss function
            loss = criterion(outputs, y)

            # backpropagate
            loss.backward()

            # step the optimizer
            optimizer.step()

            # add the loss from this batch to the running loss
            running_loss += loss.item()

        # average the loss over the batches and store the value for this epoch
        loss_hist[j] = running_loss / len(train_loader)

    return loss_hist


def load_data(df, batch_size=8):
    # TODO: update this to make it scale better for large datasets.
    data_list = df.data.to_list()
    data_list_expanded = [np.expand_dims(voxel, axis=0) for voxel in data_list]
    data = np.array(data_list_expanded)
    labels = np.array(df.manual_classification.to_list(), dtype=int)

    # pad the data for now
    data_pad = np.zeros((len(data), 1, 10, 12, 12))
    for i, voxel in enumerate(data):
        data_pad[i] = np.pad(voxel, ((0, 0), (1, 0), (1, 0), (1, 0)))
    data = data_pad

    # normalize the data
    data_norm = np.zeros((len(data), 1, 10, 12, 12))
    for i, voxel in enumerate(data):
        data_norm[i] = (voxel - np.min(voxel)) / (np.max(voxel) - np.min(voxel))
    data = data_norm

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

    # define the transform for data loading.
    train_transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.RandomRotation(180)])
    test_transform = tv.transforms.ToTensor()

    train_data = LabelledDataset(X_train, y_train, transform=train_transform)
    test_data = LabelledDataset(X_test, y_test, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def get_class_weights(df):
    labels = np.array(df.manual_classification.to_list(), dtype=int)
    # class weights --- helps a lot with the class imbalance problem!
    neg, pos = np.bincount(labels)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    return [weight_for_0, weight_for_1]