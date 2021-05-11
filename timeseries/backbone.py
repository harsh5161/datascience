# Central Pipeline where Data will be passed around
from modules import *


def INIT():
    df, target = userInputs()

    if df is None or target is None:
        return None

    action_list = processIdentifier(df)

    if len(action_list) == 0:
        return None

    resampled_frames = dataResampler(df, action_list)
    # We split our dataset to be able to evaluate our models

    resultsDict = {}
    predictionsDict = {}

    print(
        f"We have {len(df_training)} days of training data and {len(df_test)} days of testing data ")

    for val in action_list:
        if val == 'Y':
            # call function passing df,target and other parameters specific to Y
            print("Performing Yearly Analysis")
            seasonalDecompose(df[target][:365])
            stationaryNormalityPlots(df[target], 30, 7)
            tsplot(df[target], lags=30)
            modellingInit(df, target, resultsDict, predictionsDict)
        if val == 'M':
            # call function passing df,target and other parameters specific to M
            print("Performing Monthly Analysis")
            seasonalDecompose(df[target][:300])
            stationaryNormalityPlots(df[target], 15, 7)
            tsplot(df[target], lags=15)
        if val == 'D':
            # call function passing df,target and other parameters specific to D
            print("Performing Daily Analysis")
            seasonalDecompose(df[target][:250])
            stationaryNormalityPlots(df[target], 7, 7)
            tsplot(df[target], lags=7)
        if val == 'RQ':
            # call function passing resampled_frames['Quarter'],target and other parameters specific to RQ
            print("Performing Resampled Quarterly Analysis")
            seasonalDecompose(resampled_frames['Quarter'][target])
            stationaryNormalityPlots(
                resampled_frames['Quarter'][target], 10, 7)
            tsplot(resampled_frames['Quarter'][target], lags=10)
        if val == 'RM':
            # call function passing resampled_frames['Monthly'],target and other parameters specific to RM
            print("Performing Resampled Monthly Analysis")
            seasonalDecompose(resampled_frames['Month'][target])
            stationaryNormalityPlots(resampled_frames['Month'][target], 10, 7)
            tsplot(resampled_frames['Month'][target], lags=10)
        if val == 'RW':
            # call function passing resampled_frames['Daily'],target and other parameters specific to RW
            print("Performing Resampled Weekly Analysis")
            seasonalDecompose(resampled_frames['Week'][target])
            stationaryNormalityPlots(resampled_frames['Week'][target], 10, 7)
            tsplot(resampled_frames['Week'][target], lags=10)
