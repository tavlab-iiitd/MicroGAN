import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px

df = pd.read_csv("./Overlaps.csv")
df["Type"].replace(
    {
        "Original Overlap": "Overlap with Original Data",
        "KEGG Overlap": "Overlap with KEGG Genes",
    },
    inplace=True,
)


fig = px.bar(
    df,
    x="Sample_size",
    y="Overlap",
    color="Type",
    barmode="group",
    facet_col="Model",
    category_orders={"Model": ["VAE", "CTGAN", "WGAN", "GMM"]},
    labels=dict(
        Sample_size="Training sample size",
        # JSD="Jensen-Shannon divergence",
        Type="Type of Overlap",
    ),
)
fig.update_xaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    mirror=True,
    matches=None,
    type="category",
)
fig.update_yaxes(
    # range=[0, 0.3],
    showline=True,
    linewidth=2,
    linecolor="black",
    mirror=True,
)
# fig.update_traces(mode="markers+lines")
# fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Group=", "")))
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Model=", "")))
# fig.show()
fig.write_image("./SampleWiseOverlaps.png", width=1300, height=600, scale=3)


"""
df = pd.read_csv("./Overlaps.csv")
# df["Type"].replace(
#     {
#         "Original Overlap": "Overlap with Original Data",
#         "KEGG Overlap": "Overlap with KEGG Genes",
#     },
#     inplace=True,
# )


fig = px.bar(
    df,
    x="Sample_size",
    y="Num_Genes",
    facet_col="Model",
    category_orders={"Model": ["VAE", "CTGAN", "WGAN", "GMM"]},
    labels=dict(
        Sample_size="Training sample size",
        # JSD="Jensen-Shannon divergence",
        Num_Genes="Number of DE Genes",
    ),
)
fig.update_xaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    mirror=True,
    matches=None,
    type="category",
)
fig.update_yaxes(
    # range=[0, 0.3],
    showline=True,
    linewidth=2,
    linecolor="black",
    mirror=True,
)
# fig.update_traces(mode="markers+lines")
# fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Group=", "")))
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Model=", "")))
# fig.show()
fig.write_image("./SampleWiseDEGenes.png", width=1300, height=600, scale=3)
"""