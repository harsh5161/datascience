import matplotlib.pyplot as plt
import pandas as pd
from lifelines.plotting import *
from random import randint
from sklearn.preprocessing import MinMaxScaler
from lifelines.utils import qth_survival_time
from pysurvival.models.non_parametric import KaplanMeierModel
from pysurvival.utils.display import display_non_parametric


def kmf_valid(df, target, dur):
    kmf = KaplanMeierModel()
    kmf.fit(df[dur['indicator']], df[target], alpha=0.95)
    display_non_parametric(kmf)


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
    # with joblib.parallel_backend('dask'):
    df = pd.DataFrame(resultsDict, index=['Concordance Index'])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(8, 5))

    # C-Index plot
    df.loc["Concordance Index"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["Concordance Index"].sort_values().index], )
    plt.legend()
    plt.title("Concordance Index, higher is better")
    plt.show()


def plot_simple_cindex(resultsDict):
    # with joblib.parallel_backend('dask'):
    df = pd.DataFrame(resultsDict, index=["Concordance Index"])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(8, 5))

    # C-Index plot
    df.loc["Concordance Index"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["Concordance Index"].sort_values().index], )
    plt.legend()
    plt.title("Concordance Index, higher is better")
    plt.show()


def plot_aic(resultsDict):
    # with joblib.parallel_backend('dask'):
    df = pd.DataFrame(resultsDict, index=['AIC'])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(10, 5))

    # C-Index plot
    df.loc["AIC"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["AIC"].sort_values().index], )
    plt.legend()
    plt.title("AIC score, lower is better")
    plt.show()


def plot_aic_(resultsDict, resultsDict_):
    # with joblib.parallel_backend('dask'):
    _df = pd.DataFrame(resultsDict, index=['AIC'])
    df_ = pd.DataFrame(resultsDict_, index=['AIC'])
    df = pd.concat([_df, df_], axis=1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pallette = plt.cm.get_cmap("tab20c", len(df.columns))
    colors = [pallette(x) for x in range(len(df.columns))]
    color_dict = dict(zip(df.columns, colors))
    fig = plt.figure(figsize=(10, 5))

    # C-Index plot
    df.loc["AIC"].sort_values().plot(
        kind="bar", colormap="Paired", color=[
            color_dict.get(
                x, "#333333") for x in df.loc["AIC"].sort_values().index], )
    plt.legend()
    plt.title("AIC score, lower is better")
    plt.show()


# Only for Right/Uncensored and Left censoring for Univariate models
def plot_rmst(modelsDict, t_mean):
    # with joblib.parallel_backend('dask'):
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


# Only for Interval censoring with Univariate analysis
def plot_rmst_interval(modelsDict, lt_mean, ut_mean):
    # with joblib.parallel_backend('dask'):
    df = pd.DataFrame(modelsDict, index=['models'])
    # df = df.drop('KaplanMeierFitter', axis=1)
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
    # with joblib.parallel_backend('dask'):
    df = pd.DataFrame(modelsDict, index=['models'])
    df = df.drop(['ExponentialFitter'], axis=1)
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
    plt.show()


def winnerPlots(winnerModelName, winnerModel):
    # with joblib.parallel_backend('dask'):
    if winnerModelName in ['ExponentialFitter', 'GeneralizedGammaFitter', 'LogNormalFitter',
                           'LogLogisticFitter', 'WeibullFitter']:
        # winnerModel.fit(T, E)
        fig = plt.figure(figsize=(21, 8))
        fig.add_subplot(131)
        plt.title('Desc plot')
        winnerModel.plot()
        fig.add_subplot(132)
        plt.title('Hazard plot')
        winnerModel.plot_hazard()
        fig.add_subplot(133)
        plt.title('Survival function plot')
        winnerModel.plot_survival_function()
        plt.show()
    # elif winnerModelName == 'KaplanMeierFitter':
    #     winnerModel.fit(T, E)
    #     fig = plt.figure(figsize=(12, 10))
    #     fig.add_subplot(221)
    #     plt.title('Desc plot')
    #     winnerModel.plot()
    #     fig.add_subplot(222)
    #     plt.title('Cumulative density')
    #     winnerModel.plot_cumulative_density()
    #     fig.add_subplot(223)
    #     plt.title('Survival Function')
    #     winnerModel.plot_survival_function()
    #     fig.add_subplot(224)
    #     plt.title('Log Logs plot')
    #     winnerModel.plot_loglogs()
    #     plt.show()
    elif winnerModelName == 'NelsonAalenFitter':
        # winnerModel.fit(T, E)
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(111)
        plt.title('Cumulative Hazard')
        winnerModel.plot_cumulative_hazard()
        plt.show()


def winnerPlots_interval(winnerModelName, winnerModel):
    # with joblib.parallel_backend('dask'):
    if winnerModelName in ['ExponentialFitter', 'GeneralizedGammaFitter', 'LogNormalFitter',
                           'LogLogisticFitter', 'WeibullFitter']:
        # winnerModel.fit_interval_censoring(lower_bound, upper_bound, E)
        fig = plt.figure(figsize=(21, 8))
        fig.add_subplot(131)
        plt.title('Desc plot')
        winnerModel.plot()
        fig.add_subplot(132)
        plt.title('Hazard plot')
        winnerModel.plot_hazard()
        fig.add_subplot(133)
        plt.title('Survival function plot')
        winnerModel.plot_survival_function()
        plt.show()
    elif winnerModelName == 'KaplanMeierFitter':
        # winnerModel.fit_interval_censoring(lower_bound, upper_bound, E)
        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(121)
        plt.title('Desc plot')
        winnerModel.plot()
        fig.add_subplot(122)
        plt.title('Survival Function')
        winnerModel.plot_survival_function()
        plt.show()


def winner_multi(predictionsDict, result):
    # with joblib.parallel_backend('dask'):
    if result in ['LogLogisticAFTFitter', 'LogNormalAFTFitter', 'WeibullAFTFitter']:
        fig = plt.figure(figsize=(21, 13))
        cmap = get_cmap(50)
        fig.add_subplot(211)
        plt.title('Survival Function')
        plt.plot(predictionsDict[result][0])
        fig.add_subplot(212)
        plt.title('Hazard')
        plt.plot(predictionsDict[result][1])
        plt.show()
    elif result == 'GeneralizedGammaRegressionFitter':
        fig = plt.figure(figsize=(10, 10))
        cmap = get_cmap(50)
        fig.add_subplot(111)
        plt.title('Survival Function')
        plt.plot(predictionsDict[result[0]])
        plt.show()
    elif result == 'CoxPHFitter':
        fig = plt.figure(figsize=(21, 13))
        cmap = get_cmap(50)
        fig.add_subplot(211)
        plt.title('Survival Function')
        plt.plot(predictionsDict[result][0])
        fig.add_subplot(212)
        plt.title('Cumulative Hazard')
        plt.plot(predictionsDict[result][1])
        plt.show()


def pysurv_winner(predictionsDict, result):
    # with joblib.parallel_backend('dask'):
    for i in [0, 1, 2]:
        predictionsDict[result][i] = pd.DataFrame(predictionsDict[result][i])

    fig = plt.figure(figsize=(25, 19.5))
    cmap = get_cmap(50)

    len_0 = predictionsDict[result][0].shape[1]
    len_2 = predictionsDict[result][2].shape[1]
    len_1 = predictionsDict[result][1].shape
    print(len_0, len_2, len_1)

    # Survival Function plots
    if len_0 >= 50:
        fig.add_subplot(311)
        try:
            plt.title('Survival Function')
            plt.plot(predictionsDict[result][0].iloc[:, int(0.02*len_0)])
            plt.plot(predictionsDict[result][0].iloc[:, int(0.25*len_0):int(0.27*len_0)])
            plt.plot(predictionsDict[result][0].iloc[:, int(0.75*len_0):int(0.77*len_0)])
            plt.plot(predictionsDict[result][0].iloc[:, int(0.97*len_0):])
        except:
            plt.title('Survival Function')
            plt.plot(predictionsDict[result][0].iloc[:, int(0.05 * len_0)])
            plt.plot(predictionsDict[result][0].iloc[:, int(0.25 * len_0):int(0.30 * len_0)])
            plt.plot(predictionsDict[result][0].iloc[:, int(0.75 * len_0):int(0.80 * len_0)])
            plt.plot(predictionsDict[result][0].iloc[:, int(0.95 * len_0):])

    elif len_0 < 50:
        fig.add_subplot(311)
        plt.title('Survival Function')
        plt.plot(predictionsDict[result][0].iloc[:, int(len_0)])

    # Hazard Function plots
    if len_2 >= 50:
        fig.add_subplot(312)
        try:
            plt.title('Hazard Function')
            plt.plot(predictionsDict[result][2].iloc[:, int(0.02 * len_2)])
            plt.plot(predictionsDict[result][2].iloc[:, int(0.25 * len_2):int(0.27 * len_2)])
            plt.plot(predictionsDict[result][2].iloc[:, int(0.75 * len_2):int(0.77 * len_2)])
            plt.plot(predictionsDict[result][2].iloc[:, int(0.97 * len_2):])
        except:
            plt.title('Hazard Function')
            plt.plot(predictionsDict[result][2].iloc[:, int(0.05 * len_2)])
            plt.plot(predictionsDict[result][2].iloc[:, int(0.25 * len_2):int(0.30 * len_2)])
            plt.plot(predictionsDict[result][2].iloc[:, int(0.75 * len_2):int(0.80 * len_2)])
            plt.plot(predictionsDict[result][2].iloc[:, int(0.95 * len_2):])
    elif len_2 < 50:
        fig.add_subplot(312)
        plt.title('Hazard Function')
        plt.plot(predictionsDict[result][2].iloc[:, int(0.02 * len_2)])

    if len_1[1] != 1:
        fig.add_subplot(313)
        plt.title('Risk Scores')
        plt.plot(predictionsDict[result][1])
        plt.show()
    else:
        pass


def qth_survival(percentile, winnerModel, df, target, dur, type_='multi'):
    tt = df[dur['indicator']]
    tt = pd.DataFrame(tt)
    if  percentile >= 1:
        percentile = MinMaxScaler(feature_size=(0, 1))
    if type_ != 'multi':
        return qth_survival_time(percentile, winnerModel)
    else:
        surv = winnerModel.predict_survival(df.drop([target, dur['indicator']], axis=1))
        surv = pd.DataFrame(surv)
        surv_t = pd.concat([surv, tt], axis=1)
        # surv_pot = surv_t.loc[(surv_t[dur['indicator']] > POT) & (surv_t[dur['indicator']] < upper), :]
        # print(surv_pot.head(2))
        surv_pot = surv_t.copy()
        surv_pot['surv'] = surv_t.drop([dur['indicator']], axis=1).sum(axis=1)
        surv_pot = surv_pot[[dur['indicator'], 'surv']]
        surv_pot.rename(columns={'surv':'Survival'}, inplace=True)
        surv_pot = surv_pot[surv_pot['Survival'] >= percentile]

        surv_pot.plot(x=dur['indicator'], y='Survival', title='Survival Probability', figsize=(12, 6),
                      kind='scatter')
        return surv_pot