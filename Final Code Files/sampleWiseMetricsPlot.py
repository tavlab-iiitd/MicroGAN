import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px

df = pd.read_csv("Distance_metrics_sample_wise.csv")
df["Group"].replace({1: "Healthy", 2: "TB"}, inplace=True)

fig = px.line(
    df,
    x="Value",
    y="JSD",
    color="Model",
    facet_row="Group",
    category_orders={
        "Group": ["Healthy", "TB"],
    },
    labels=dict(
        #WSD="Wasserstein Distance",
        JSD="Jensen-Shannon divergence",
        Value="Num Training samples",
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
fig.update_traces(mode="markers+lines")
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Group=", "")))
#fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Model=", "")))
fig.write_image("./SampleWiseDistJSD.png", width=400, height=400, scale=3)


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
