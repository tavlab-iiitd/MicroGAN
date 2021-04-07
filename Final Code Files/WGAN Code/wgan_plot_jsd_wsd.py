# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import glob
from torch import autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from scipy.stats import wasserstein_distance
import torch
from scipy.stats import entropy
from sklearn.preprocessing import PowerTransformer
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import random
import sys
from time import time

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
original = sys.stdout
Tensor = torch.cuda.FloatTensor

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, actual, shuffle=True, scale=False):
        if isinstance(actual, str):
            self.df = pd.read_csv(actual)
        elif isinstance(actual, np.ndarray):
            self.df = pd.DataFrame(actual)
        else:
            self.df = actual

        self.len = self.df.shape[0]
        self.df = self.df.values

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.df[idx, :]


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


def compute_gradient_penalty(D, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = Variable(
        Tensor(real_samples.shape[0], 1).fill_(1.0).to(device), requires_grad=False
    )
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    # grad_norm = torch.sqrt(torch.sum(gradients ** 2,dim=1)+1e-12) # Remove this later
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    # gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty


def prep(data):
    if "Unnamed: 0" in data.columns.values:
        data.drop(["Unnamed: 0"], axis=1, inplace=True)
    if "Unnamed: 0.1" in data.columns.values:
        data.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    if "Unnamed: 0.1.1" in data.columns.values:
        data.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)
    return data


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def calculate_wstd(org, gen):
    org = prep(org)
    gen = prep(gen)
    wsd = 0
    orig_data = org.values
    gen_data = gen.values
    # if orig_data.shape[1] != gen_data.shape[1]:
    #    pca = PCA(n_components=gen_data.shape[1])
    #    pca = pca.fit(orig_data)
    #    gen_data = pca.inverse_transform(gen_data)
    columns = orig_data.shape[1]
    w = 0
    print(orig_data.shape, gen_data.shape, " Test WSTD Function")
    for i in range(columns):
        orig_data_ = orig_data[:, i].reshape(-1, 1)
        gen_data_ = gen_data[:, i].reshape(-1, 1)
        orig_data_ = orig_data_.reshape(
            len(orig_data),
        )
        gen_data_ = gen_data_.reshape(
            len(gen_data),
        )
        # orig_data_ += 1
        # gen_data_ += 1
        w += wasserstein_distance(orig_data_, gen_data_)
        # break
    w /= columns
    wsd += w
    return wsd


def calculate_jsd(org, gen):
    org = prep(org)
    gen = prep(gen)
    KLD = 0
    orig_data = org.values
    gen_data = gen.values
    # if orig_data.shape[1] != gen_data.shape[1]:
    #    pca = PCA(n_components=gen_data.shape[1])
    #    pca = pca.fit(orig_data)
    #    gen_data = pca.inverse_transform(gen_data)
    columns = orig_data.shape[1]
    scaler = MinMaxScaler()
    softmax = torch.nn.Softmax(0)
    kld = 0
    print(orig_data.shape, gen_data.shape, " Test JSD Function")
    for i in range(columns):
        orig_data_ = scaler.fit_transform(orig_data[:, i].reshape(-1, 1))
        gen_data_ = scaler.fit_transform(gen_data[:, i].reshape(-1, 1))
        orig_data_ = orig_data_.reshape(
            len(orig_data),
        )
        gen_data_ = gen_data_.reshape(
            len(gen_data),
        )
        orig_data_ = softmax(torch.from_numpy(orig_data_)).numpy()
        gen_data_ = softmax(torch.from_numpy(gen_data_)).numpy()
        # orig_data_ += 1
        # gen_data_ += 1
        kld += js_divergence(orig_data_, gen_data_)
        # break
    kld /= columns
    KLD += kld
    return KLD


def generate(G, genpath, sample_size, noise_input_size=100):
    noise = Tensor(np.random.normal(0, 1, (sample_size, noise_input_size))).to(
        device
    )  # Tensor(np.random.randn(sample_size, noise_input_size))

    G = G.eval()
    with torch.no_grad():
        samples = G(noise)
        samples = samples.cpu().numpy()

    return samples


def main(root, device, doPca=False, doMinMax=False):

    # Hyperparameters
    n_train_steps = 4000
    Batch_size = 1
    noise_input_size = 100
    inflate_to_size = 600
    disc_internal_size = 200
    num_cells_generate = 100
    learn_rate = 1e-4
    n_critic = 10
    lambda_gp = 10
    datapath = root
    genpath = os.path.join(root.split("/")[-1][:-4], "gendata")
    losspath = os.path.join(root.split("/")[-1][:-4], "LossDetails")

    print("[INFO] Making Nesessary Directories")
    os.makedirs(genpath, exist_ok=True)
    os.makedirs(losspath, exist_ok=True)

    print(genpath)
    print(losspath)

    sample_sizes = [25, 35, 50, 65, 75, 80]
    for sample_size in sample_sizes:
        data = pd.read_csv(datapath)
        if "Unnamed: 0" in data.columns.values:
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        if "Unnamed: 0.1" in data.columns.values:
            data.drop(["Unnamed: 0.1"], axis=1, inplace=True)
        if "Unnamed: 0.1.1" in data.columns.values:
            data.drop(["Unnamed: 0.1.1"], axis=1, inplace=True)
        print(data.head(1))
        try:
            data.drop(["host_name", "tuberculosis", "hiv"], axis=1, inplace=True)
        except:
            pass
        orig_data = data
        data = data.sample(sample_size)
        columns = data.columns
        print(len(columns), "Num Columns")

        if doMinMax:
            print("In transformer")
            scaler = MinMaxScaler(feature_range=(-1, 1))
            print("In MinMax")
            data = pd.DataFrame(scaler.fit_transform(data), columns=columns)
        if doPca:
            print("inpca")
            pca = PCA(0.99)
            data = pd.DataFrame(pca.fit_transform(data))

        print(data.columns, "FEATURE Space")

        print("[INFO] Load Data")
        rawdataset = CustomDataset(data)
        dataloader = DataLoader(rawdataset, batch_size=Batch_size, shuffle=True)
        gex_size = rawdataset.df.shape[1]
        num_cells_train = rawdataset.df.shape[0]

        # Training Loop
        print("[INFO] Training Loop start2")
        G = Generator(noise_input_size, inflate_to_size, gex_size).to(device)
        D = Discriminator(gex_size, gex_size // 2, disc_internal_size, 1).to(device)
        optimizer_G = optim.Adam(G.parameters())
        optimizer_D = optim.Adam(D.parameters())
        G.train()
        D.train()

        batches_done = 0
        start = time()  # timer start
        sys.stdout = open(f"{losspath}/loss{sample_size}.txt", "w+")
        for epoch in range(n_train_steps):
            for i, data in enumerate(dataloader):
                # Configure input
                real_data = data.float().to(device)
                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()
                # Sample noise as generator input
                z = Variable(
                    Tensor(
                        np.random.normal(0, 1, (real_data.shape[0], noise_input_size))
                    ).to(device)
                )
                # Generate a batch of images
                fake_data = G(z)
                # Real images
                real_validity = D(real_data)
                # Fake images
                fake_validity = D(fake_data)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    D, real_data.data, fake_data.data
                )
                # Adversarial loss
                d_loss = (
                    -torch.mean(real_validity)
                    + torch.mean(fake_validity)
                    + lambda_gp * gradient_penalty
                )

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_data = G(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = D(fake_data)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    batches_done += n_critic
        gendatax = generate(G, genpath, len(orig_data), noise_input_size)
        if doPca:
            gendatax = pca.inverse_transform(gendatax)
            gendatax = pd.DataFrame(gendatax, columns=columns)
            gendatax.to_csv(f"{genpath}/WGAN_{sample_size}.csv", index=False)
            jsd = calculate_jsd(orig_data, gendatax)
            wst_distance = calculate_wstd(orig_data, gendatax)
            print(orig_data.shape, gendatax.shape)
            print("WSD JSD", epoch, sample_size, wst_distance, jsd)

        sys.stdout = original
        print(f"[INFO] Time required = {time()-start}")

        G = G.cpu()
        D = D.cpu()

        print("[INFO] Saving models")
        # save_path = os.path.join(genpath, "models.pth")

        # torch.save({"G": G, "D": D}, save_path)

        del G
        del D
        del dataloader
        del rawdataset


# %%
allfiles = sorted(glob.glob(os.path.join("../..", "saad18409", "Group*Data.csv")))
print(allfiles)
allfiles = [allfiles[0], allfiles[1]]
print(allfiles)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

for root in allfiles:
    print("-------- ", root, " --------")
    main(root, device, True, False)
    print("-" * (len(root) + 18))
