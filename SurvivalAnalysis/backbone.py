from Modules import *
from modules_multi import *


def INIT():
    try:
        df, target, dur, censoring_type = userInputs()
    except:
        return None

    if df is None or target is None or censoring_type is None or target is None:
        return None

    print('Initial Length of the data: ', len(df))
    resultsDict, predictionsDict, modelsDict = {}, {}, {}

    if censoring_type != 'Interval':
        lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
        normalityPlots(df[target])
        print('Performing Univariate Survival Analysis.')
        winnerModel = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
        if winnerModel is not None:
            winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type)
            return True
        else:
            return False
    else:
        lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
        normalityPlots(df[target])
        winnerModel = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)
        if winnerModel is not None:
            winnerModelTrainer_uni_interval(df, target, winnerModel, dur)
            return True
        else:
            return False


def INIT_multi():
    try:
        df, target, dur, censoring_type = userInputs()
    except:
        return None

    if df is None or target is None or censoring_type is None or target is None:
        return None
    resultsDict, predictionsDict, modelsDict = {}, {}, {}
    lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
    normalityPlots(df[target])
    print('Performing Multivariate Survival Analysis')
    try:
        winnerModel = modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
        if winnerModel is not None:
            winnerModelTrainer_multi(df, target, winnerModel, dur, censoring_type)
            return True
    except:
        # try:
        df[target] = df[target] / 100
        resultsDict, predictionsDict, modelsDict = {}, {}, {}
        print('Performing Multivariate Survival Analysis')

        winnerModel = modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict, censoring_type,
                                          dur)
        if winnerModel is not None:
            winnerModelTrainer_multi(df, target, winnerModel, dur, censoring_type)
            return True
        # except:
        #     print('Please check your data for the following problems:\n')
        #     print("""1. Is there high-collinearity in the dataset? Try using the variance inflation factor (VIF) to find redundant variables.
        #         2. Trying adding a small penalizer (or changing it, if already present).
        #          Example: `GeneralizedGammaRegressionFitter(penalizer=0.01).fit(...)`.\n""")
        #     print('Your current version of data supports only Univariate Analysis')
        # return True
