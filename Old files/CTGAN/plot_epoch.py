import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def parse_text(f):
    text = f.read().strip().split("\n")
    loss_G = []
    loss_D = []
    for line in text:
        line = line.split(", ")
        loss_G.append(float(line[1].split(" ")[-1]))
        loss_D.append(float(line[2].split(" ")[-1]))

    assert len(loss_G) == len(loss_D), "Something is not right"

    return loss_G, loss_D, list(range(1, len(loss_G)+1))


def plot(loss_G, loss_D, ep, plotpath, model_num):
    plt.plot(ep, loss_G, label='loss G')
    plt.plot(ep, loss_D, label='loss D')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{plotpath}/loss_{model_num}.png")
    plt.close()


def main(root):
    modelpath = os.path.join(root, 'LossDetails', '*')
    plotpath = os.path.join(root, 'plots')
    allfiles = sorted(glob.glob(modelpath))
    for file in allfiles:
        if file[-3:] == 'txt':
            model_num = file[-5]
        else:
            model_num = file[-1]
        f = open(file, "r")
        loss_G, loss_D, ep = parse_text(f)
        f.close()
        plot(loss_G, loss_D, ep, plotpath, model_num)


allfiles = glob.glob("Chromosome_*")
for file in allfiles:
    print(f"---- {file} ----")
    main(file)
    print("-"*(len(file)+10))
