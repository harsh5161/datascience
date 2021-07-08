import matplotlib.pyplot as plt
import pandas as pd
from lifelines.plotting import *
from random import randint


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


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
    fig = plt.figure(figsize=(8, 5))

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
    fig = plt.figure(figsize=(8, 5))

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
    fig = plt.figure(figsize=(10, 5))

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
    plt.figure(figsize=(20, 8))
    plt.title("RMST score, higher is better")
    ax = plt.subplot(111)
    for m in df.loc['models'].values.tolist():
        rmst_plot(m, t=t_mean, ax=ax)
    plt.show()


# Only for Interval censoring with UNivariate analysis
def plot_rmst_interval(modelsDict, lt_mean, ut_mean):
    df = pd.DataFrame(modelsDict, index=['models'])
    df = df.drop('KaplanMeierFitter', axis=1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    plt.figure(figsize=(20, 8))
    ax = plt.subplot(211)
    plt.title('RMST at lower bound of duration, higher is better')
    for m in df.loc['models'].values.tolist():
        rmst_plot(m, t=lt_mean, ax=ax)
    ax = plt.subplot(212)
    plt.title('RMST at higher bound of duration, higher is better')
    for m in df.loc['models'].values.tolist():
        rmst_plot(m, t=ut_mean, ax=ax)
    plt.show()


# Only for Right/Uncensored and Left censoring for Univariate models
def plot_cdf(modelsDict):
    df = pd.DataFrame(modelsDict, index=['models'])
    df = df.drop(['KaplanMeierFitter', 'GeneralizedGammaFitter', 'ExponentialFitter'], axis=1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(24, 12))
    model_instance = df.loc['models'].values.tolist()
    ax = fig.add_subplot(3, 1, 1)
    plt.title('Cumulative Dustribution Function')
    cdf_plot(model_instance[0], ax=ax)
    ax = fig.add_subplot(3, 1, 2)
    cdf_plot(model_instance[1], ax=ax)
    ax = fig.add_subplot(3, 1, 3)
    cdf_plot(model_instance[2], ax=ax)


def winnerPlots(winnerModelName, winnerModel, T, E):
    if winnerModelName in ['ExponentialFitter', 'GeneralizedGammaFitter', 'LogNormalFitter',
                           'LogLogisticFitter', 'WeibullFitter']:
        winnerModel.fit(T, E)
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(231)
        plt.title('Desc plot')
        winnerModel.plot()
        fig.add_subplot(232)
        plt.title('Cumulative hazard plot')
        winnerModel.plot_cumulative_hazard()
        fig.add_subplot(233)
        plt.title('Hazard plot')
        winnerModel.plot_hazard()
        fig.add_subplot(234)
        plt.title('Density plot')
        winnerModel.plot_density()
        fig.add_subplot(235)
        plt.title('Survival function plot')
        winnerModel.plot_survival_function()
        fig.add_subplot(236)
        plt.title('Cumulative density plot')
        winnerModel.plot_cumulative_density()
        plt.show()
    elif winnerModelName == 'KaplanMeierFitter':
        winnerModel.fit(T, E)
        fig = plt.figure(figsize=(12, 10))
        fig.add_subplot(221)
        plt.title('Desc plot')
        winnerModel.plot()
        fig.add_subplot(222)
        plt.title('Cumulative density')
        winnerModel.plot_cumulative_density()
        fig.add_subplot(223)
        plt.title('Survival Function')
        winnerModel.plot_survival_function()
        fig.add_subplot(224)
        plt.title('Log Logs plot')
        winnerModel.plot_loglogs()
        plt.show()
    elif winnerModelName == 'NelsonAalenFitter':
        winnerModel.fit(T, E)
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(111)
        plt.title('Cumulative Hazard')
        winnerModel.plot_cumulative_hazard()
        plt.show()


def winnerPlots_interval(winnerModelName, winnerModel, lower_bound, upper_bound, E):
    if winnerModelName in ['ExponentialFitter', 'GeneralizedGammaFitter', 'LogNormalFitter',
                           'LogLogisticFitter', 'WeibullFitter']:
        winnerModel.fit_interval_censoring(lower_bound, upper_bound, E)
        fig = plt.figure(figsize=(15, 10))
        fig.add_subplot(231)
        plt.title('Desc plot')
        winnerModel.plot()
        fig.add_subplot(232)
        plt.title('Cumulative hazard plot')
        winnerModel.plot_cumulative_hazard()
        fig.add_subplot(233)
        plt.title('Hazard plot')
        winnerModel.plot_hazard()
        fig.add_subplot(234)
        plt.title('Density plot')
        winnerModel.plot_density()
        fig.add_subplot(235)
        plt.title('Survival function plot')
        winnerModel.plot_survival_function()
        fig.add_subplot(236)
        plt.title('Cumulative density plot')
        winnerModel.plot_cumulative_density()
        plt.show()
    elif winnerModelName == 'KaplanMeierFitter':
        winnerModel.fit_interval_censoring(lower_bound, upper_bound, E)
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(221)
        plt.title('Desc plot')
        winnerModel.plot()
        fig.add_subplot(222)
        plt.title('Cumulative density')
        winnerModel.plot_cumulative_density()
        fig.add_subplot(223)
        plt.title('Survival Function')
        winnerModel.plot_survival_function()
        fig.add_subplot(224)
        plt.title('Log Logs plot')
        winnerModel.plot_loglogs()
        plt.show()


def winner_multi(predictionsDict, result):
    if result in ['LogLogisticAFTFitter', 'LogNormalAFTFitter', 'WeibullAFTFitter']:
        fig = plt.figure(figsize=(20, 10))
        cmap = get_cmap(50)
        fig.add_subplot(131)
        plt.title('Survival Function')
        plt.plot(predictionsDict[result][0])
        fig.add_subplot(132)
        plt.title('Cumulative Hazard')
        plt.plot(predictionsDict[result][1])
        fig.add_subplot(133)
        plt.title('Hazard')
        plt.plot(predictionsDict[result][2])
        plt.show()
    # elif result == 'GeneralizedGammaRegressionFitter':
    #     fig = plt.figure(figsize=(10, 10))
    #     cmap = get_cmap(50)
    #     fig.add_subplot(121)
    #     plt.title('Survival Function')
    #     plt.plot(predictionsDict[0])
    #     fig.add_subplot(122)
    #     plt.title('Cumulative Hazard')
    #     plt.plot(predictionsDict[1])
    #     plt.show()
    elif result == 'CoxPHFitter':
        fig = plt.figure(figsize=(15, 15))
        cmap = get_cmap(50)
        fig.add_subplot(221)
        plt.title('Survival Function')
        plt.plot(predictionsDict[result][0])
        fig.add_subplot(222)
        plt.title('Cumulative Hazard')
        plt.plot(predictionsDict[result][1])
        fig.add_subplot(223)
        plt.title('Partial Hazard')
        plt.plot(predictionsDict[result][2])
        fig.add_subplot(224)
        plt.title('Log Hazard')
        plt.plot(predictionsDict[result][3])
        plt.show()
