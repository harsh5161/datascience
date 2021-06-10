# Central Pipeline where Data will be passed around
from modules import *
import matplotlib.pyplot as plt


def INIT():
    try:
        df, target = userInputs()
    except:
        return None

    if df is None or target is None:
        return None

    action_list = processIdentifier(df)

    if len(action_list) == 0:
        return None

    resampled_frames = dataResampler(df, action_list)
    print("Initial Length",len(df))
    # Need to update logic for yearly, monthly and daily because the same models are being called using the same data for these three. Need to make change in data by resampling
    for val in action_list:
        if val == 'Y':
            # call function passing df,target and other parameters specific to Y
            period = 350 #sometimes there are holes in the dataset, but we shouldn't perform yearly analysis if there isn't at least 2 * 350 days of data in it
            resultsDict = {}
            predictionsDict = {}
            print("Performing Yearly Analysis")
            seasonalDecompose(df[target][:720], period)
            stationaryNormalityPlots(df[target], 30, 7)
            tsplot(df[target], lags=30)
            winnerModel = modellingInit(df, target, resultsDict, predictionsDict,period)
            winnerModelTrainer(df,target,winnerModel)
        if val == 'M':
            # call function passing df,target and other parameters specific to M
            period = 30
            resultsDict = {}
            predictionsDict = {}
            print("Performing Monthly Analysis")
            seasonalDecompose(df[target][:300], period)
            stationaryNormalityPlots(df[target], 15, 7)
            tsplot(df[target], lags=15)
            winnerModel = modellingInit(df, target, resultsDict, predictionsDict,period)
            winnerModelTrainer(df,target,winnerModel)
        if val == 'D':
            # call function passing df,target and other parameters specific to D
            period = 24
            resultsDict = {}
            predictionsDict = {}
            print("Performing Daily Analysis")
            seasonalDecompose(df[target][:250], period)
            stationaryNormalityPlots(df[target], 7, 7)
            tsplot(df[target], lags=7)
            winnerModel = modellingInit(df, target, resultsDict, predictionsDict,period)
            winnerModelTrainer(df,target,winnerModel)
        if val == 'RQ':
            # call function passing resampled_frames['Quarter'],target and other parameters specific to RQ
            try:
                period = 4
                resultsDict = {}
                predictionsDict = {}
                print("Performing Resampled Quarterly Analysis")
                seasonalDecompose(resampled_frames['Quarter'][target], period)
                stationaryNormalityPlots(
                    resampled_frames['Quarter'][target], 10, 7)
                tsplot(resampled_frames['Quarter'][target], lags=10)
                winnerModel = modellingInit(resampled_frames['Quarter'],
                              target, resultsDict, predictionsDict,period)
                winnerModelTrainer(resampled_frames['Quarter'],target,winnerModel)
            except ValueError:
                print("Not enough Data to perform resampled analysis")
        if val == 'RM':
            # call function passing resampled_frames['Monthly'],target and other parameters specific to RM
            try:
                period = 12
                resultsDict = {}
                predictionsDict = {}
                print("Performing Resampled Monthly Analysis")
                seasonalDecompose(resampled_frames['Month'][target], 10)
                stationaryNormalityPlots(
                    resampled_frames['Month'][target], 10, 7)
                tsplot(resampled_frames['Month'][target], lags=10)
                winnerModel = modellingInit(resampled_frames['Month'],
                              target, resultsDict, predictionsDict,period)
                winnerModelTrainer(resampled_frames['Month'],target,winnerModel)
            except ValueError:
                print("Not enough Data to perform resampled analysis")
        if val == 'RW':
            # call function passing resampled_frames['Daily'],target and other parameters specific to RW
            try:
                period = 52
                resultsDict = {}
                predictionsDict = {}
                print("Performing Resampled Weekly Analysis")
                seasonalDecompose(resampled_frames['Week'][target], 50)
                stationaryNormalityPlots(
                    resampled_frames['Week'][target], 10, 7)
                tsplot(resampled_frames['Week'][target], lags=10)
                winnerModel = modellingInit(resampled_frames['Week'],
                              target, resultsDict, predictionsDict,period)
                winnerModelTrainer(resampled_frames['Week'],target,winnerModel)
            except ValueError:
                print("Not enough Data to perform resampled analysis")

    return True
