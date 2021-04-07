import glob
import pandas as pd
import numpy as np
import os


def main(root):
    orig = pd.read_csv(os.path.join(root, "origdata", "data.csv"))
    genfiles = glob.glob(os.path.join(root, "gendata", "sample*"))
    gendict = {}
    for i in genfiles:
        df = pd.read_csv(i)
        df.drop("Unnamed: 0", axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        if i.split("/")[-1][-5] not in gendict.keys():
            gendict[i.split("/")[-1][-5]] = df
        else:
            gendict[i.split("/")[-1][-5]] = pd.concat([gendict[i.split("/")[-1][-5]], df], axis=0, ignore_index=True)
    
    for i in gendict:
        size = orig[orig['class']==int(i)].shape[0]
        df = gendict[i].sample(size)
        df.to_csv(os.path.join(root, "gendata", f"equal_size_data{i}_{size}_samples.csv"))
    
allfiles = glob.glob("Chromosome_*")
for i in allfiles:
    print(i)
    main(i)
    print("-"*10)


