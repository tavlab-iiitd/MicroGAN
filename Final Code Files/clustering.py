import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gen_wgan = ["/home/ayushig/TB/WGAN/Group1Axes/minmax/gendata/gendata_unnorm_log25.csv","/home/ayushig/TB/WGAN/Group2Axes/minmax/gendata/gendata_unnorm_log25.csv"]
gen_ctgan = ["/home/ayushig/TB/CTGAN/1/gendataaxes/gen_sample_model550.csv","/home/ayushig/TB/CTGAN/2/gendataaxes/gen_sample_model550.csv"]
gen_gmm = ["/home/ayushig/TB/GMMGAN/Group1Axes/GMM1_550.csv","/home/ayushig/TB/GMMGAN/Group2Axes/GMM2_550.csv"]
org = ["/home/ayushig/saad18409/Group1Axes.csv", "/home/ayushig/saad18409/Group2Axes.csv"]
lbl = ["1", "2"]



for i in range(len(gen_wgan)):
    df = pd.read_csv(gen_wgan[i])
    if 'Unnamed: 0' in list(df.columns.values):
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'Unnamed: 0.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1'], axis=1, inplace=True)
    if 'Unnamed: 0.1.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1.1'], axis=1, inplace=True)

    corr = df.corr()

    # Generate a mask for the upper triangle
    #arr_corr = corr.as_matrix()
    # mask out the top triangle
    #arr_corr[np.triu_indices_from(arr_corr)] = np.nan

    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(24, 18))
    
    hm = sns.clustermap(corr, method="complete", cmap='RdBu', annot=False, 
               annot_kws={"size": 3}, vmin=-1, vmax=1, figsize=(28,20))
    
    ticks = np.arange(corr.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(corr.columns, rotation=360, fontsize=8)
    
    ax.set_title('correlation matrix')
    plt.savefig("./GP{}_WGAN_Cluster.png".format(lbl[i]), dpi=300)


for i in range(len(gen_wgan)):
    df = pd.read_csv(gen_ctgan[i])
    if 'Unnamed: 0' in list(df.columns.values):
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'Unnamed: 0.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1'], axis=1, inplace=True)
    if 'Unnamed: 0.1.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1.1'], axis=1, inplace=True)

    corr = df.corr()
    # Generate a mask for the upper triangle
    #arr_corr = corr.as_matrix()
    # mask out the top triangle
    #arr_corr[np.triu_indices_from(arr_corr)] = np.nan

    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(24, 18))
    
    hm = sns.clustermap(corr, method="complete", cmap='RdBu', annot=False, 
               annot_kws={"size": 3}, vmin=-1, vmax=1, figsize=(28,20))

    
    ticks = np.arange(corr.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(corr.columns, rotation=360, fontsize=8)
    
    ax.set_title('correlation matrix')
    
    plt.savefig("./GP{}_CTGAN_Cluster.png".format(lbl[i]), dpi=300)

for i in range(len(gen_gmm)):
    df = pd.read_csv(gen_gmm[i])
    if 'Unnamed: 0' in list(df.columns.values):
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'Unnamed: 0.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1'], axis=1, inplace=True)
    if 'Unnamed: 0.1.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1.1'], axis=1, inplace=True)

    corr = df.corr()

    # Generate a mask for the upper triangle
    # Generate a mask for the upper triangle
    #arr_corr = corr.as_matrix()
    # mask out the top triangle
    #arr_corr[np.triu_indices_from(arr_corr)] = np.nan

    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(24, 18))
    
    hm = sns.clustermap(corr, method="complete", cmap='RdBu', annot=False, 
               annot_kws={"size": 3}, vmin=-1, vmax=1, figsize=(28,20))


    ticks = np.arange(corr.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(corr.columns, rotation=360, fontsize=8)
    
    ax.set_title('correlation matrix')
    plt.savefig("./GP{}_GMM_Cluster.png".format(lbl[i]), dpi=300)

for i in range(len(gen_wgan)):
    df = pd.read_csv(org[i])
    if 'Unnamed: 0' in list(df.columns.values):
        df.drop(['Unnamed: 0'], axis=1, inplace=True)
    if 'Unnamed: 0.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1'], axis=1, inplace=True)
    if 'Unnamed: 0.1.1' in list(df.columns.values):
        df.drop(['Unnamed: 0.1.1'], axis=1, inplace=True)

    corr = df.corr()

    # Generate a mask for the upper triangle
    #arr_corr = corr.as_matrix()
    # mask out the top triangle
    #arr_corr[np.triu_indices_from(arr_corr)] = np.nan

    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(24, 18))
    
    hm = sns.clustermap(corr, method="complete", cmap='RdBu', annot=False, 
               annot_kws={"size": 3}, vmin=-1, vmax=1, figsize=(28,20))

    
    ticks = np.arange(corr.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(corr.columns, rotation=360, fontsize=8)
    
    ax.set_title('correlation matrix')
    plt.savefig("./GP{}_ORG_Cluster.png".format(lbl[i]), dpi=300)


