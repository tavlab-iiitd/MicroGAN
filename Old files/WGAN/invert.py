import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import glob

def main(root):
    origpath = os.path.join(root, 'origdata', 'data.csv')
    genpath = os.path.join(root, 'gendata', '*')

    orig = pd.read_csv(origpath)
    genfiles = glob.glob(genpath)
    print(orig.columns)
    gen = [pd.read_csv(file) for file in genfiles]
    label = []
    print(gen[0].columns)
    if "label" in orig.columns:
        orig.drop("label", axis=1, inplace=True)
    orig.drop(["Unnamed: 0", "host_name", "tuberculosis", "hiv"], axis=1, inplace=True)
    orig.reset_index(drop=True, inplace=True)
    for i in range(len(gen)):
        gen[i].drop(["Unnamed: 0"], axis=1, inplace=True)
        try:
            label.append(gen[i][["label"]])
            gen[i].drop("label", axis=1, inplace=True)
        except:
            continue
    
    scaler = MinMaxScaler()
    scaler = scaler.fit(orig)

    for i in range(len(gen)):
        gen[i] = pd.DataFrame(scaler.inverse_transform(gen[i]))

    if len(label) > 0:
        for i in range(len(gen)):
            gen[i] = pd.concat([gen[i], label[i]], ignore_index=True, axis=1)

    for i in range(len(gen)):
        gen[i].to_csv(genfiles[i])

mfiles = glob.glob("Chromosome_Full")
for i in mfiles:
    print("-"*10)
    print(f"{i}")
    main(i)
    print("-"*10)