import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.patches as mpl_patches

org = pd.read_csv("./CTGAN_550.tsv", sep="\t")

gen = pd.read_csv("./CTGAN_Axes_550.tsv", sep="\t")

extras_file = "CTGAN_Axes_vs_CTGAN_Extra"
absentees_file = "CTGAN_Axes_vs_CTGAN_Missing"
lgnd = ("CTGAN 550", "CTGAN Axes 550")
img_name = "CTGAN_Axes_vs_CTGAN"

org_lst = []
for i in org.iloc[:, 0]:
    org_lst.append(i)

gen_lst = []
for i in gen.iloc[:, 0]:
    gen_lst.append(i)


print(len(list(set(org_lst) & set(gen_lst))), len(org_lst), len(gen_lst))

extras = []
absentees = []

for i in gen_lst:
    if i not in org_lst:
        extras.append(i)

for i in org_lst:
    if i not in gen_lst:
        absentees.append(i)

overlaps = list(set(org_lst) & set(gen_lst))


go_map = {}

org_enrichment = {}
for i in range(len(org.iloc[:, 0])):
    if org.iloc[i, 0] in overlaps and "Unclassified" not in org.iloc[i, 0]:
        key_s = org.iloc[i, 0].find("(GO")
        key_e = org.iloc[i, 0].find(")")
        go = org.iloc[i, 0][key_s + 1 : key_e]
        rest = org.iloc[i, 0][: key_s - 1]
        go_map[go] = rest
        org_enrichment[rest] = org.iloc[i, 2]


enrichment = {}
for i in range(len(gen.iloc[:, 0])):
    if gen.iloc[i, 0] in overlaps and "Unclassified" not in gen.iloc[i, 0]:
        key_s = gen.iloc[i, 0].find("(GO")
        key_e = gen.iloc[i, 0].find(")")
        go = gen.iloc[i, 0][key_s + 1 : key_e]
        rest = gen.iloc[i, 0][: key_s - 1]
        enrichment[rest] = gen.iloc[i, 2]

org_enrichment = OrderedDict(sorted(org_enrichment.items()))
enrichment = OrderedDict(sorted(enrichment.items()))

X = np.arange(len(org_enrichment))
rcParams.update({"figure.autolayout": True})
fig = plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
ax.bar(X, org_enrichment.values(), width=0.2, color="r", align="center")
ax.bar(X - 0.2, enrichment.values(), width=0.2, color="g", align="center")
ax.legend(lgnd)
plt.xticks(X, org_enrichment.keys(), rotation=(65), fontsize=10, ha="right")
plt.title(
    "Overlaps from Overrepresentation Analysis on GO Biological Processes", fontsize=17
)
plt.xlabel("GO Biological Process")
plt.ylabel("Number of Genes")
ax.grid(linestyle="--")
# plt.tight_layout()
plt.savefig("./{}.png".format(img_name))


with open("./{}.txt".format(extras_file), "w") as f:
    for item in extras:
        f.write("%s\n" % item)

with open("./{}.txt".format(absentees_file), "w") as f:
    for item in absentees:
        f.write("%s\n" % item)
