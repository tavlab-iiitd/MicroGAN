import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px

df = pd.read_csv("EpochWiseMetrices.csv")
df["Group"].replace({1: "Healthy", 2: "TB"}, inplace=True)
dfdl = df[df["Model"] != "GMM"]
dfst = df[df["Model"] == "GMM"]

fig = px.line(
    dfst,
    x="Value",
    y="WSD",
    facet_col="Model",
    facet_row="Group",
    category_orders={
        "Model": ["GMM"],
        "Group": ["Healthy", "TB"],
    },
    labels=dict(WSD="Wasserstein Distance", Value="Number of Components"),
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
    range=[0, 0.3], showline=True, linewidth=2, linecolor="black", mirror=True
)
fig.update_traces(mode="markers+lines")
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Group=", "")))
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Model=", "")))
fig.write_image("./GMMEpochwiseWSD.png", width=400, height=600, scale=6)


# import plotly.express as px

# df = px.data.tips()
# print(df)
# fig = px.histogram(
#     df,
#     x="total_bill",
#     y="tip",
#     color="sex",
#     facet_row="time",
#     facet_col="day",
#     category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]},
# )
# fig.show()
