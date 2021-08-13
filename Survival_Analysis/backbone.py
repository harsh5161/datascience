from Modules import *
from modules_multi import *


def INIT():
    try:
        df, target, dur, censoring_type, rate = userInputs()
    except:
        return None

    # Slower results
    if not rate:
        print('Initial Length of the data: ', len(df), '\n')

        # Left censoring
        if censoring_type == 'LEFT':
            resultsDict, predictionsDict, modelsDict = {}, {}, {}
            normalityPlots(df[dur['indicator']])
            targetDistribution(df, target)
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)

            # Univariate Analysis
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
            if winnerModel is not None:
                winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type)

                # Multivariate Analysis
                print('\n---------------------------------------------------\n\nPerforming Multivariate'
                      ' Survival Analysis')
                resultsDict, predictionsDict, modelsDict = {}, {}, {}
                if df.shape[1] > 25:
                    df = featureSelection(df, target, dur, censoring_type)
                print(f'Number of feature variables are {df.shape[1] - 1}\n')
                try:
                    winnerModel_ = modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict,
                                                       censoring_type, dur)
                    if winnerModel_ is not None:
                        winnerModelTrainer_multi(df, target, winnerModel_, dur, censoring_type)
                        return True
                    else:
                        print('\nRUNNING AGAIN...\n')
                        df[target] = df[target] / 100
                        resultsDict, predictionsDict, modelsDict = {}, {}, {}
                        print('Performing Multivariate Survival Analysis')
                        try:
                            winnerModel_1 = modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict,
                                                                censoring_type, dur)
                            if winnerModel_1 is not None:
                                winnerModelTrainer_multi(df, target, winnerModel_1, dur, censoring_type)
                                return True
                            else:
                                print('Please take your observations form Univariate Analysis, data is not suited for'
                                      ' Multivariate analysis.')
                        except:
                            pass
                except:
                    pass
                return True
            else:
                return False

        # Interval censoring
        elif censoring_type == 'INTERVAL':
            resultsDict, predictionsDict, modelsDict = {}, {}, {}
            normalityPlots(df[dur['indicator']['lower']])
            normalityPlots(df[dur['indicator']['upper']])
            targetDistribution(df, target)
            lifetimes(None, df[target], censoring_type, df[dur['indicator']['lower']], df[dur['indicator']['upper']])

            # Univariate analysis
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)
            if winnerModel is not None:
                winnerModelTrainer_uni_interval(df, target, winnerModel, dur)

                # Multivariate analysis
                print('\n---------------------------------------------\n\nPerforming Multivariate Survival Analysis')
                resultsDict, predictionsDict, modelsDict = {}, {}, {}
                if df.shape[1] > 25:
                    df = featureSelection(df, target, dur, censoring_type)
                print(f'Number of feature variables are {df.shape[1] - 1}\n')
                try:
                    winnerModel_ = modellingInit_multi_interval(df, target, resultsDict, predictionsDict, dur)
                    if winnerModel_ is not None:
                        winnerModelTrainer_multi_interval(df, target, winnerModel_, dur)
                        return True
                    else:
                        print('\nRUNNING AGAIN...\n')
                        df[dur['indicator']['lower']] = df[dur['indicator']['lower']] / 100
                        df[dur['indicator']['upper']] = df[dur['indicator']['upper']] / 100
                        resultsDict, predictionsDict, modelsDict = {}, {}, {}
                        print('Performing Multivariate Survival Analysis')
                        try:
                            winnerModel_1 = modellingInit_multi_interval(df, target, resultsDict, predictionsDict, dur)
                            if winnerModel_1 is not None:
                                winnerModelTrainer_multi_interval(df, target, winnerModel_1, dur)
                                return True
                            else:
                                print('Your data is not suited for Multivariate analysis. Please take observations from'
                                      ' Univariate analysis.')
                        except:
                            pass
                except:
                    pass
                return True
            else:
                return False

        # Right/Uncensored data
        elif censoring_type in ['RIGHT', 'UNCENSORED']:
            resultsDict, predictionsDict, modelsDict = {}, {}, {}
            normalityPlots(df[dur['indicator']])
            targetDistribution(df, target)
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)

            # Univariate Analysis
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
            if winnerModel is not None:
                winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type)

                # Multivariate Analysis
                print('\n---------------------------------------------\n\nPerforming Multivariate Survival Analysis')

                if df.shape[1] > 25:
                    df = featureSelection(df, target, dur, censoring_type)
                print('\n')
                display(df.head())
                print(f'Number of feature variables are {df.shape[1] - 1}\n')
                try:
                    resultsDict, predictionsDict, modelsDict = {}, {}, {}
                    winnerModel_ = modelling_pysurvival(df, target, resultsDict, predictionsDict, dur,
                                                        modelsDict)
                    if winnerModel_ is not None:
                        winnerModeltrainer_pysurv(df, target, dur, winnerModel_)
                        return True
                except:
                    print('\n\n')
                    resultsDict, predictionsDict, modelsDict = {}, {}, {}
                    try:
                        winnerModel_1 = modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict,
                                                            censoring_type, dur)
                        if winnerModel_1 is not None:
                            winnerModelTrainer_multi(df, target, winnerModel_1, dur, censoring_type)
                            return True
                    except:
                        print('\nRUNNING AGAIN...\n')
                        df[target] = df[target] / 100
                        resultsDict, predictionsDict, modelsDict = {}, {}, {}
                        print('-------------------------------------------\n\n'
                              'Performing Multivariate Survival Analysis')
                        try:
                            winnerModel_2 = modellingInit_multi(df, target, resultsDict, predictionsDict,
                                                                modelsDict, censoring_type,
                                                                dur)
                            if winnerModel_2 is not None:
                                winnerModelTrainer_multi(df, target, winnerModel_2, dur, censoring_type)
                                return True
                        except:
                            print('Your data is not suited for Multivariate analysis.'
                                  ' Please take observations from Univariate analysis.')
                return True
            else:
                return False

    # Quick results
    else:
        print('Initial Length of the data: ', len(df), '\n')

        if censoring_type != 'INTERVAL':
            resultsDict, predictionsDict, modelsDict = {}, {}, {}
            normalityPlots(df[dur['indicator']])
            targetDistribution(df, target)
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
            normalityPlots(df[target])

            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
            if winnerModel is not None:
                winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type)
                return True
            else:
                return False

        # Interval censoring
        else:
            resultsDict, predictionsDict, modelsDict = {}, {}, {}
            normalityPlots(df[dur['indicator']['lower']])
            normalityPlots(df[dur['indicator']['upper']])
            targetDistribution(df, target)
            lifetimes(None, df[target], censoring_type, df[dur['indicator']['lower']], df[dur['indicator']['upper']])

            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)

            if winnerModel is not None:
                winnerModelTrainer_uni_interval(df, target, winnerModel, dur)
                return True
            else:
                return False
