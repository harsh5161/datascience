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

    # Need to update logic for yearly, monthly and daily because the same models are being called using the same data for these three. Need to make change in data by resampling
    for val in action_list:
        if val == 'Y':
            # call function passing df,target and other parameters specific to Y
            resultsDict = {}
            predictionsDict = {}
            print("Performing Yearly Analysis")
            seasonalDecompose(df[target][:720], 350)
            stationaryNormalityPlots(df[target], 30, 7)
            tsplot(df[target], lags=30)
            modellingInit(df, target, resultsDict, predictionsDict)
        if val == 'M':
            # call function passing df,target and other parameters specific to M
            resultsDict = {}
            predictionsDict = {}
            print("Performing Monthly Analysis")
            seasonalDecompose(df[target][:300], 30)
            stationaryNormalityPlots(df[target], 15, 7)
            tsplot(df[target], lags=15)
            modellingInit(df, target, resultsDict, predictionsDict)
        if val == 'D':
            # call function passing df,target and other parameters specific to D
            resultsDict = {}
            predictionsDict = {}
            print("Performing Daily Analysis")
            seasonalDecompose(df[target][:250], 24)
            stationaryNormalityPlots(df[target], 7, 7)
            tsplot(df[target], lags=7)
            modellingInit(df, target, resultsDict, predictionsDict)
        if val == 'RQ':
            # call function passing resampled_frames['Quarter'],target and other parameters specific to RQ
            try:
                resultsDict = {}
                predictionsDict = {}
                print("Performing Resampled Quarterly Analysis")
                seasonalDecompose(resampled_frames['Quarter'][target], 4)
                stationaryNormalityPlots(
                    resampled_frames['Quarter'][target], 10, 7)
                tsplot(resampled_frames['Quarter'][target], lags=10)
                modellingInit(resampled_frames['Quarter'],
                              target, resultsDict, predictionsDict)
            except ValueError:
                print("Not enough Data to perform resampled analysis")
        if val == 'RM':
            # call function passing resampled_frames['Monthly'],target and other parameters specific to RM
            try:
                resultsDict = {}
                predictionsDict = {}
                print("Performing Resampled Monthly Analysis")
                seasonalDecompose(resampled_frames['Month'][target], 10)
                stationaryNormalityPlots(
                    resampled_frames['Month'][target], 10, 7)
                tsplot(resampled_frames['Month'][target], lags=10)
                modellingInit(resampled_frames['Month'],
                              target, resultsDict, predictionsDict)
            except ValueError:
                print("Not enough Data to perform resampled analysis")
        if val == 'RW':
            # call function passing resampled_frames['Daily'],target and other parameters specific to RW
            try:
                resultsDict = {}
                predictionsDict = {}
                print("Performing Resampled Weekly Analysis")
                seasonalDecompose(resampled_frames['Week'][target], 50)
                stationaryNormalityPlots(
                    resampled_frames['Week'][target], 10, 7)
                tsplot(resampled_frames['Week'][target], lags=10)
                modellingInit(resampled_frames['Week'],
                              target, resultsDict, predictionsDict)
            except ValueError:
                print("Not enough Data to perform resampled analysis")

    return True
