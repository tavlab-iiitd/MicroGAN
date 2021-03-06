# -*- coding: utf-8 -*-
"""VAE PyTorch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1484hwapatM9DqxsXJA-bWSPn_KJX49HW
"""

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import os
import random
import glob

"""# Dataset"""

data = np.random.randn(573, 66)   # n_patients x n_genes # fake data for now


class CustomDataset(Dataset):
    def __init__(self, actual, shuffle=True, scale=False):
        if isinstance(actual, str):
            self.df = pd.read_csv(actual)
        elif isinstance(actual, np.ndarray):
            self.df = pd.DataFrame(actual)
        else:
            self.df = actual

        for col in ["Unnamed: 0", "tuberculosis", "hiv", "host_name"]:
            if col in self.df.columns:
                self.df.drop(col, axis=1, inplace=True)

        self.len = self.df.shape[0]
        self.df = self.df.values
        self.original_dim = self.df.shape[1]-1

        if scale:
            scaler = MinMaxScaler()
            self.df[:, :-1] = scaler.fit_transform(self.df[:, :-1])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.df[idx, :-1], self.df[idx, -1]


""" Model"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

""" Encoder"""


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        self.mean = nn.Sequential(
            nn.Linear(input_dim+1, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )

        self.var = nn.Sequential(
            nn.Linear(input_dim+1, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        mean = self.mean(out)
        std = self.var(out)

        return mean, std


"""## Decoder"""


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(latent_dim+1, latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),

            nn.Linear(latent_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer(x)


"""## Connect Encoder and Decoder"""


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        global device
        self.enc = Encoder(input_dim, latent_dim).to(device)
        self.dec = Decoder(input_dim, latent_dim).to(device)

    def forward(self, x, y):
        z_mean, z_var = self.enc(x, y)

        std = torch.exp(z_var/2)
        eps = torch.randn_like(std)
        sample = eps.mul(std).add(z_mean)
        sample = torch.cat([sample, y], dim=1)
        gen = self.dec(sample)

        return gen, z_mean, z_var


"""# Training Loop"""


def train(original_dim, latent_dim, learning_rate, epochs, dataloader, rawdataset):
    global device
    vae = VAE(original_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    losslist = []
    for epoch in range(epochs):
        running_loss = 0
        for _, (data, label) in enumerate(dataloader):
            data = data.float().to(device)
            label = label.float().to(device)
            label = torch.unsqueeze(label, dim=1)

            optimizer.zero_grad()

            sample, z_mean, z_var = vae(data, label)
            reconstruction_loss = nn.functional.binary_cross_entropy(sample, data)
            kl_loss = 0.5 * 0.5 * torch.sum(torch.exp(z_var) + z_mean**2 - 1.0 - z_var)

            loss = reconstruction_loss + kl_loss
            loss.backward()
            running_loss += loss.item()

            optimizer.step()

        losslist.append(running_loss/len(rawdataset))
        if(epoch % 500 == 499):
            print(f"[{epoch+1}/{epochs}] Done. Average loss = {sum(losslist)/len(losslist)}")
    return vae, losslist


def generate(decoder, numsamples, latent_dim):
    global device
    labels = np.array([0, 1, 2, 3, 4, 5]*numsamples)
    labels = torch.from_numpy(labels).float().to(device)
    labels = torch.unsqueeze(labels, dim=1)
    noise = np.random.randn(6*numsamples, latent_dim)
    noise = torch.from_numpy(noise).float().to(device)
    noise = torch.cat([noise, labels], dim=1).to(device)
    decoder.eval()
    with torch.no_grad():
        gen = decoder(noise).cpu().numpy()
        labels = labels.cpu().numpy()

    gen = pd.DataFrame(gen)
    labels = pd.DataFrame(labels, columns=["class"])
    gen = pd.concat([gen, labels], axis=1)

    return gen


def plot(plotpath, losslist):
    x = list(range(1, len(losslist)+1))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.plot(x, losslist)
    plt.savefig(os.path.join(plotpath, "LossVsEpoch.png"))
    plt.close()


def save(encoder, decoder, losslist, modelpath):
    encoder.cpu()
    decoder.cpu()
    torch.save({
      "Enc": encoder,
      "Dec": decoder,
      "state_dict": {
        "enc": encoder.state_dict(),
        "dec": decoder.state_dict()
      }
    }, os.path.join(modelpath, "models.pth"))
    f = open("losslist.txt", "w+")
    for i, val in enumerate(losslist):
        f.write(f"Epoch: {i+1}, Loss: {losslist}\n")
    f.close()
    f = open("Info.txt", "w+")
    f.write("Models saved in form of dictionary of the following structure\n")
    f.write("""
            'Enc':encoder,
            'Dec':decoder,
            'state_dict':{
              'enc':encoder state dict,
              'dec':decoder state dict
            } 
            \n""")
    f.close()


def main(root):
    origpath = os.path.join(root, "origdata", "data.csv")
    genpath = os.path.join(root, "gendata")
    modelpath = os.path.join(root, "models")
    plotpath = os.path.join(root, "figures")
    os.makedirs(genpath, exist_ok=True)
    os.makedirs(modelpath, exist_ok=True)
    os.makedirs(plotpath, exist_ok=True)

    # Hyperparameters
    latent_dim = 50
    batch_size = 50
    epochs = 1000
    learning_rate = 0.0005

    print(f"[INFO] Loading Dataset {origpath}...")
    rawdataset = CustomDataset(origpath, scale=False)
    dataloader = DataLoader(rawdataset, batch_size=batch_size, shuffle=True)
    original_dim = rawdataset.original_dim

    print("[INFO] Training Begin")
    vae, losslist = train(original_dim, latent_dim, learning_rate, epochs,
                          dataloader, rawdataset)
    encoder = vae.enc
    decoder = vae.dec
    print("[INFO] Generate Samples")
    gen_sample = generate(decoder, 100, latent_dim)
    gen_sample.to_csv(os.path.join(genpath, "gendata.csv"))
    print("[INFO] Plotting loss")
    plot(plotpath, losslist)
    print("[INFO] Saving models and list")
    save(encoder, decoder, losslist, modelpath)
    encoder.cpu()
    decoder.cpu()
    vae.cpu()
    del encoder
    del decoder
    del vae
    del rawdataset
    del dataloader
    del losslist

if __name__ == "__main__":
    import time
    roots = glob.glob("Chromosome_*")
    for root in roots:
        print(f"{'-'*5} {root} {'-'*5}")
        start = time.time()
        main(root)
        print(f"Time required = {time.time()-start}")
        print(f"-"*20)
        # break
