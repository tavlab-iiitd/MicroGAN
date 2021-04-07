import torch
import getopt, sys
from torch import autograd
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

Tensor = torch.cuda.FloatTensor


class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim, output_dim),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim, latent_dim2, output_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim2, output_dim),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


def generate_wgan(PATH, num_samples, pca, minmax):
    print(PATH)
    G = torch.load(PATH)["G"].cuda()
    print(G)
    G.eval()
    noise = Tensor(np.random.normal(0, 1, (num_samples, 100)))
    with torch.no_grad():
        samples = G(noise)
        samples = samples.cpu().numpy()
    print(samples.shape)
    inv_pca = pca.inverse_transform(samples)
    inv_scaled = minmax.inverse_transform(inv_pca)
    ret_data = pd.DataFrame(np.log2(inv_scaled))
    return ret_data


def generate_gmm(PATH, num_samples, pca):
    loaded_gmm = pickle.load(open(PATH, "rb"))
    data = loaded_gmm.sample(num_samples)
    data = pca.inverse_transform(data[0])
    ret_data = pd.DataFrame(data)
    print(ret_data.shape)
    print(ret_data.head())
    return ret_data


if __name__ == "__main__":
    model = None
    data_type = None
    num_samples = None
    argumentList = sys.argv[1:]
    options = "m:n:d:"
    try:
        arguments, values = getopt.getopt(argumentList, options)
        print(arguments)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-m"):
                model = currentValue
            elif currentArgument in ("-n"):
                num_samples = int(currentValue)
            elif currentArgument in ("-d"):
                data_type = currentValue
    except getopt.error as err:
        print(str(err))

    print(model, data_type, num_samples)

    if model == "wgan":
        if data_type == "healthy":
            path = "./saves/wgan_gp1.pth"
            pca = pickle.load(open("./saves/gp1_pca.pkl", "rb"))
            scaler = pickle.load(open("./saves/gp1_scaler.pkl", "rb"))
            generate_wgan(path, num_samples, pca, scaler)
        elif data_type == "tb":
            path = "./saves/wgan_gp2.pth"
            pca = pickle.load(open("./saves/gp2_pca.pkl", "rb"))
            scaler = pickle.load(open("./saves/gp2_scaler.pkl", "rb"))
            generate_wgan(path, num_samples, pca, scaler)
    elif model == "gmm":
        if data_type == "healthy":
            path = "./saves/GMM_log2_GP1.sav"
            pca = pickle.load(open("./saves/pca_gmm_gp1.pkl", "rb"))
            generate_gmm(path, num_samples, pca)
        if data_type == "tb":
            path = "./saves/GMM_log2_GP2.sav"
            pca = pickle.load(open("./saves/pca_gmm_gp2.pkl", "rb"))
            generate_gmm(path, num_samples, pca)
