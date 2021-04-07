from torch import autograd
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
import random
import sys

original = sys.stdout

Tensor = torch.cuda.FloatTensor

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
        #self.df = self.df.values

        if scale:
            self.scaler = PCA(0.9)
            label = self.df[["class"]]
            self.df.drop("class", axis=1, inplace=True)
            self.df = pd.DataFrame(self.scaler.fit_transform(self.df))
            self.df = pd.concat([self.df, label], axis=1)
        
        self.df = self.df.values

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.df[idx, :-1], self.df[idx, -1]

    def getScaler(self):
        return self.scaler

# Generator Network
class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(Generator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(inplace=True),

            nn.Linear(latent_dim, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = self.layer(out)
        return out

# Discriminator Network


class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim, latent_dim2, output_dim):
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(inplace=True),

            nn.Linear(latent_dim, latent_dim2),
            nn.LeakyReLU(inplace=True),

            nn.Linear(latent_dim2, 1),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = self.layer(out)

        return out


def compute_gradient_penalty(D, real_samples, fake_samples, y):
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                                            * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, y)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)
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
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def generate(G, genpath, sample_size, noise_input_size, scaler=None):
    labels = [0, 1, 2, 3, 4, 5] * sample_size
    labels = np.array(labels).reshape(-1, 1)
    # random.shuffle(labels)
    labels = Tensor(labels)

    noise = Tensor(np.random.randn(sample_size*6, noise_input_size))

    G = G.eval()
    with torch.no_grad():
        samples = G(noise, labels)
        samples = samples.cpu().numpy()
        if scaler is not None:
           samples = scaler.inverse_transform(samples)
        labels = labels.cpu().numpy()
        samples = pd.DataFrame(samples)
        labels = pd.DataFrame(labels, columns=["label"])
        gendata = pd.concat([samples, labels], axis=1)

    gen_data_dict = {}
    for i in [0, 1, 2, 3, 4, 5]:
        gen_data_dict[i] = gendata[gendata["label"] == i]

    for i in gen_data_dict:
        gen_data_dict[i].to_csv(os.path.join(
            genpath, f"data{i}_{sample_size}_samples.csv"))


def main(root, device):
    datapath = os.path.join(root, "origdata", "data.csv")
    genpath = os.path.join(root, "gendata")
    losspath = os.path.join(root, "LossDetails")

    print("[INFO] Making Nesessary Directories")
    os.makedirs(genpath, exist_ok=True)
    os.makedirs(losspath, exist_ok=True)

    data = pd.read_csv(datapath)
    if "Unnamed: 0" in data.columns.values:
        data.drop(["Unnamed: 0"], axis=1, inplace=True)

    data.drop(["host_name", "tuberculosis", "hiv"], axis=1, inplace=True)

    print("[INFO] Load Data")
    scale = False
    if root.find("Full")!=-1:
       scale = True
    rawdataset = CustomDataset(data, scale=scale)
    dataloader = DataLoader(rawdataset, batch_size=32, shuffle=True)

    # Hyperparameters
    n_train_steps = 1000
    batch_size = 32
    noise_input_size = 100
    inflate_to_size = 600
    gex_size = rawdataset.df.shape[1]-1
    disc_internal_size = 200
    num_cells_train = rawdataset.df.shape[0]
    num_cells_generate = 100
    learn_rate = 1e-5
    n_critic = 2
    lambda_gp = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training Loop
    print("[INFO] Training Loop start")

    G = Generator(noise_input_size+1, inflate_to_size, gex_size).cuda(device)
    D = Discriminator(gex_size*2, gex_size, disc_internal_size, 1).cuda(device)
    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())
    G.train()
    D.train()

    batches_done = 0
    sys.stdout = open(f"{losspath}/loss.txt", "w+")
    for epoch in range(n_train_steps):
        for i, (data, label) in enumerate(dataloader):

            # Configure input
            real_data = data.cuda(device).float()
            label = label.cuda(device).float()
            # ---------------------
            #  Train Discriminator
            # ---------------------

            label = torch.unsqueeze(label, dim=1)
            optimizer_D.zero_grad()
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(
                0, 1, (real_data.shape[0], noise_input_size))))
            label_expanded = label + \
                Tensor(np.zeros((real_data.shape[0], gex_size)))
            # Generate a batch of images
            fake_data = G(z, label)
            # Real images
            real_validity = D(real_data, label_expanded)
            # Fake images
            fake_validity = D(fake_data, label_expanded)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                D, real_data.data, fake_data.data, label_expanded)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + \
                torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_data = G(z, label)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = D(fake_data, label_expanded)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_train_steps, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                # if batches_done % opt.sample_interval == 0:
                #     save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

                batches_done += n_critic

    sys.stdout = original
    # Generate Data
    scaler = None
    if root.find("Full")!=-1:
       scaler = rawdataset.getScaler()
    for i in [30, 50, 100]:
        generate(G, genpath, i, noise_input_size, scaler)

    G = G.cpu()
    D = D.cpu()

    print("[INFO] Saving models")
    save_path = os.path.join(root, "models.pth")

    torch.save({
        'G': G,
        'D': D
    }, save_path)

    del G
    del D
    del dataloader
    del rawdataset


if __name__ == "__main__":
    import glob
    allfiles = glob.glob("Chromosome_*")
    allfiles.sort()
    device = 1

    import argparse
    parser = argparse.ArgumentParser(description="GPU Device number")
    parser.add_argument("-d", "--device", type=int,
                        required=True, help="GPU number")
    args = parser.parse_args()
    #allfiles.remove("Chromosome_Full")
    allfiles = ["Chromosome_Nan", "Chromosome_X", "Chromosome_Y"]
    for file in allfiles:
        print(f"---- {file} ----")
        main(file, args.device)
        print("-"*(len(file)+10))
