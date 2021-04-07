import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, ep, plotpath):
    plt.plot(ep, x, label='Loss_G')
    plt.plot(ep, y, label='Loss_D')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{plotpath}/loss.png")
    plt.close()


def parse_txt(root):
    plotpath = os.path.join(root, "plots")
    file = open(os.path.join(root, "LossDetails", "loss.txt"), 'r')
    text = file.read().strip().split("\n")
    loss_G, loss_D = {}, {}
    for line in text:
        line = line.split(" ")
        n = line[1].find("/")
        epochnum = line[1][:n]

        if epochnum not in loss_G.keys():
            loss_G[epochnum] = 0

        lossval = float(line[-1][:-1])
        loss_G[epochnum] += lossval

        if epochnum not in loss_D.keys():
            loss_D[epochnum] = 0

        lossval = float(line[-4][:-1])
        loss_D[epochnum] += lossval

    x, y = [], []

    for it in zip(loss_G.items(), loss_D.items()):
        g = it[0][1]
        d = it[1][1]

        x.append(g)
        y.append(d)

    assert len(x) == len(y), "Something is wrong"
    plot(x, y, list(range(1, len(x)+1)), plotpath)


allfiles = glob.glob("Chromosome_*")
allfiles.sort()

for f in allfiles:
    print(f"---- {f} ----")
    parse_txt(f)
    print("-"*(len(f)+10))
