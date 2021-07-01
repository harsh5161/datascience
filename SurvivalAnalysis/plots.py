import matplotlib.pyplot as plt
import pandas as pd
from lifelines.plotting import *


def bar_metrics(resultsDict):
    df = pd.DataFrame.from_dict(resultsDict)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(20, 15))

    # MAE plot
    fig.add_subplot(2, 2, 1)
    df.loc["mae"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["mae"].sort_values().index], )
    plt.legend()
    plt.title("MAE Metric, lower is better")

    # RMSE plot
    fig.add_subplot(2, 2, 2)
    df.loc["rmse"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["rmse"].sort_values().index], )
    plt.legend()
    plt.title("RMSE Metric, lower is better")

    # MAPE plot
    fig.add_subplot(2, 2, 3)
    df.loc["mape"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["mape"].sort_values().index], )
    plt.legend()
    plt.title("MAPE Metric, lower is better")

    # R2 plot
    fig.add_subplot(2, 2, 4)
    df.loc["r2"].sort_values(ascending=False).plot(
        kind="bar",
        colormap="Paired",
        color=[
            color_dict.get(x, "#333333")
            for x in df.loc["r2"].sort_values(ascending=False).index
        ],
    )
    plt.legend()
    plt.title("R2 Metric, higher is better")
    plt.tight_layout()
    # plt.savefig("results/metrics.png")
    plt.show()


def plot_cindex_(resultsDict):
    df = pd.DataFrame(resultsDict, index=['_concordance_index'])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(20, 15))

    # C-Index plot
    df.loc["_concordance_index"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["_concordance_index"].sort_values().index], )
    plt.legend()
    plt.title("Concordance Index, higher is better")


def plot_simple_cindex(resultsDict):
    df = pd.DataFrame(resultsDict, index=["simple_cindex"])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(20, 15))

    # C-Index plot
    df.loc["simple_cindex"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["simple_cindex"].sort_values().index], )
    plt.legend()
    plt.title("Concordance Index, higher is better")


def plot_aic(resultsDict):
    df = pd.DataFrame(resultsDict, index=['aic'])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(20, 15))

    # C-Index plot
    df.loc["aic"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["aic"].sort_values().index], )
    plt.legend()
    plt.title("AIC score, lower is better")


# Only for Right/Uncensored and Left censoring for Univariate models
def plot_rmst(modelsDict, t_mean):
    df = pd.DataFrame(modelsDict, index=['models'])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    plt.figure(figsize=(24, 12))

    ax = plt.subplot(111)
    for m in df.loc['models'].values.tolist():
        rmst_plot(m, t=t_mean, ax=ax)
    plt.show()

    plt.legend()
    plt.title("RMST score, higher is better")


# Only for Right/Uncensored and Left censoring for Univariate models
def plot_cdf(modelsDict):
    df = pd.DataFrame(modelsDict, index=['models'])
    df.drop(['KaplanMeierFitter', 'GeneralizedGammaFitter'], axis=1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(24, 12))
    model_instance = df.loc['models'].values.tolist()

    fig.add_subplot(2, 2, 1)
    cdf_plot(model_instance[0])
    fig.add_subplot(2, 2, 2)
    cdf_plot(model_instance[1])
    fig.add_subplot(2, 2, 3)
    cdf_plot(model_instance[2])
    fig.add_subplot(2, 2, 4)
    cdf_plot(model_instance[3])

    plt.title("Cumulative Distribution Function")