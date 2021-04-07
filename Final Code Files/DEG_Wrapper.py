import os
import glob
import subprocess


wgan_files1 = sorted(
    glob.glob(os.path.join("/home/ayushig/TB/WGAN/Group1Data/gendata/WGAN_*.csv"))
)
wgan_files2 = sorted(
    glob.glob(os.path.join("/home/ayushig/TB/WGAN/Group2Data/gendata/WGAN_*.csv"))
)


ctgan_files1 = sorted(
    glob.glob(
        os.path.join("/home/ayushig/TB", "CTGAN", "Group1Data", "gendata", "*.csv")
    )
)
ctgan_files2 = sorted(
    glob.glob(
        os.path.join("/home/ayushig/TB", "CTGAN", "Group2Data", "gendata", "*.csv")
    )
)


gmm_files1 = sorted(
    glob.glob(os.path.join("/home/ayushig/TB", "GMMGAN", "Group1Data", "*.csv"))
)
gmm_files2 = sorted(
    glob.glob(os.path.join("/home/ayushig/TB", "GMMGAN", "Group2Data", "*.csv"))
)

vae_files1 = sorted(
    glob.glob(os.path.join("/home/ayushig/TB", "VAE", "gen_final_org1", "*.csv"))
)
vae_files2 = sorted(
    glob.glob(os.path.join("/home/ayushig/TB", "VAE", "gen_final_org2", "*.csv"))
)

# wgan_files1 = ["/home/ayushig/saad18409/Group1Axes.csv"]
# wgan_files2 = ["/home/ayushig/saad18409/Group2Axes.csv"]
# ctgan_files1 = []
# ctgan_files2 = []
# gmm_files1 = []
# gmm_files2 = []

print(gmm_files1, gmm_files2)
print()
# print(ctgan_files1, ctgan_files2)
# print()
# print(gmm_files1, gmm_files2)
print("********** WGAN ***********")
for w_file in range(len(wgan_files1)):
    direct_output = subprocess.check_output(
        "Rscript DEG.R -f {} -o {} -s {}".format(
            wgan_files1[w_file], wgan_files2[w_file], 3
        ),
        shell=True,
    )
    print()
    direct_output = direct_output.decode("utf-8")
    print(direct_output)
    print()
    t = " ".join(direct_output.split())
    t = t.replace('"', "")
    t = t.split(" ")
    x = [i for i in t if not (i[0] == "[" and i[-1] == "]")]
    for i in range(len(x)):
        if "[" in x[i]:
            start = x[i].find("[")
            end = x[i].find("]")
            x[i] = x[i][:start] + x[i][end + 1 :]

    ### Handle Edge Cases
    for i in range(len(x)):
        if x[i] == "HBII.52.45":
            x[i] = "HBII-52-45"
        elif x[i] == "HOM.TES.103":
            x[i] = "HOM-TES-103"
        elif x[i] == "KU.MEL.3":
            x[i] = "KU-MEL-3"
        elif x[i] == "NY.REN.7":
            x[i] = "NY-REN-7"
        elif "." in x[i] and x[i][:3] != "HS.":
            x[i] = x[i].replace(".", "-", 1)

    ### Write DE Genes to a file
    file_to_write = "./WGAN_S_" + wgan_files1[w_file][-7:-4] + ".txt"
    with open(file_to_write, "w") as out_f:
        for item in x:
            out_f.write("%s\n" % item)

print("********** CTGAN ************")
for c_file in range(len(ctgan_files1)):
    direct_output = subprocess.check_output(
        "Rscript DEG.R -f {} -o {} -s {}".format(
            ctgan_files1[c_file], ctgan_files2[c_file], 2
        ),
        shell=True,
    )
    direct_output = direct_output.decode("utf-8")
    print(direct_output)
    t = " ".join(direct_output.split())
    t = t.replace('"', "")
    t = t.split(" ")
    x = [i for i in t if not (i[0] == "[" and i[-1] == "]")]
    for i in range(len(x)):
        if "[" in x[i]:
            start = x[i].find("[")
            end = x[i].find("]")
            x[i] = x[i][:start] + x[i][end + 1 :]

    ### Handle Edge Cases
    for i in range(len(x)):
        if x[i] == "HBII.52.45":
            x[i] = "HBII-52-45"
        elif x[i] == "HOM.TES.103":
            x[i] = "HOM-TES-103"
        elif x[i] == "KU.MEL.3":
            x[i] = "KU-MEL-3"
        elif x[i] == "NY.REN.7":
            x[i] = "NY-REN-7"
        elif "." in x[i] and x[i][:3] != "HS.":
            x[i] = x[i].replace(".", "-", 1)

    ### Write DE Genes to a file
    file_to_write = "CTGAN_S_" + ctgan_files1[c_file][-7:-4] + ".txt"
    with open(file_to_write, "w") as out_f:
        for item in x:
            out_f.write("%s\n" % item)


print("********** GMM ***********")
for g_file in range(len(gmm_files1)):
    direct_output = subprocess.check_output(
        "Rscript DEG.R -f {} -o {} -s {}".format(
            gmm_files1[g_file], gmm_files2[g_file], 2
        ),
        shell=True,
    )
    direct_output = direct_output.decode("utf-8")
    print(direct_output)
    t = " ".join(direct_output.split())
    t = t.replace('"', "")
    t = t.split(" ")
    x = [i for i in t if not (i[0] == "[" and i[-1] == "]")]
    for i in range(len(x)):
        if "[" in x[i]:
            start = x[i].find("[")
            end = x[i].find("]")
            x[i] = x[i][:start] + x[i][end + 1 :]

    ### Handle Edge Cases
    for i in range(len(x)):
        if x[i] == "HBII.52.45":
            x[i] = "HBII-52-45"
        elif x[i] == "HOM.TES.103":
            x[i] = "HOM-TES-103"
        elif x[i] == "KU.MEL.3":
            x[i] = "KU-MEL-3"
        elif x[i] == "NY.REN.7":
            x[i] = "NY-REN-7"
        elif "." in x[i] and x[i][:3] != "HS.":
            x[i] = x[i].replace(".", "-", 1)

    ### Write DE Genes to a file
    file_to_write = "GMM_S_" + gmm_files1[g_file][-7:-4] + ".txt"
    with open(file_to_write, "w") as out_f:
        for item in x:
            out_f.write("%s\n" % item)


print("********** VAE ***********")
for v_file in range(len(vae_files1)):
    direct_output = subprocess.check_output(
        "Rscript DEG.R -f {} -o {} -s {}".format(
            vae_files1[v_file], vae_files2[v_file], 2
        ),
        shell=True,
    )
    direct_output = direct_output.decode("utf-8")
    print(direct_output)
    t = " ".join(direct_output.split())
    t = t.replace('"', "")
    t = t.split(" ")
    x = [i for i in t if not (i[0] == "[" and i[-1] == "]")]
    for i in range(len(x)):
        if "[" in x[i]:
            start = x[i].find("[")
            end = x[i].find("]")
            x[i] = x[i][:start] + x[i][end + 1 :]

    ### Handle Edge Cases
    for i in range(len(x)):
        if x[i] == "HBII.52.45":
            x[i] = "HBII-52-45"
        elif x[i] == "HOM.TES.103":
            x[i] = "HOM-TES-103"
        elif x[i] == "KU.MEL.3":
            x[i] = "KU-MEL-3"
        elif x[i] == "NY.REN.7":
            x[i] = "NY-REN-7"
        elif "." in x[i] and x[i][:3] != "HS.":
            x[i] = x[i].replace(".", "-", 1)

    ### Write DE Genes to a file
    file_to_write = "VAE_S_" + vae_files1[v_file][-7:-4] + ".txt"
    with open(file_to_write, "w") as out_f:
        for item in x:
            out_f.write("%s\n" % item)
