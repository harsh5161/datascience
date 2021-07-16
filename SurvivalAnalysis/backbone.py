from Modules import *
from modules_multi import *


def INIT():
    try:
        df, target, dur, censoring_type, rate = userInputs()
    except:
        return None

    if df is None or target is None or censoring_type is None or target is None:
        return None

    if not rate:

        print('Initial Length of the data: ', len(df))
        resultsDict, predictionsDict, modelsDict = {}, {}, {}

        if censoring_type != 'INTERVAL':
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
            normalityPlots(df[target])
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
            if winnerModel is not None:
                winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type)
                print('\n---------------------------------------------------\n\nPerforming Multivariate'
                      ' Survival Analysis')
                resultsDict, predictionsDict, modelsDict = {}, {}, {}
                try:
                    try:
                        winnerModel = modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict,
                                                          censoring_type,
                                                          dur)
                    except:
                        pass
                    finally:
                        if winnerModel is not None:
                            winnerModelTrainer_multi(df, target, winnerModel, dur, censoring_type)
                            return True
                except:
                    try:
                        print('\nRUNNING AGAIN...\n')
                        df[target] = df[target] / 100
                        resultsDict, predictionsDict, modelsDict = {}, {}, {}
                        print('Performing Multivariate Survival Analysis')
                        try:
                            winnerModel = modellingInit_multi(df, target, resultsDict, predictionsDict, modelsDict,
                                                              censoring_type,
                                                              dur)
                        except:
                            pass
                        finally:
                            if winnerModel is not None:
                                winnerModelTrainer_multi(df, target, winnerModel, dur, censoring_type)
                                return True
                    except:
                        print(
                            'Please take your observations form Univariate Analysis, data is not suited for'
                            ' Multivariate analysis.')
                return True
            else:
                return False

        elif censoring_type in ['LEFT', 'UNCENSORED']:
            lifetimes(None, df[target], censoring_type, df[dur['indicator']['lower']], df[dur['indicator']['upper']])
            normalityPlots(df[target])
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)

            if winnerModel is not None:
                winnerModelTrainer_uni_interval(df, target, winnerModel, dur)
                print('\n---------------------------------------------\n\nPerforming Multivariate Survival Analysis')
                resultsDict, predictionsDict, modelsDict = {}, {}, {}
                try:
                    try:
                        winnerModel = modellingInit_multi_interval(df, target, resultsDict, predictionsDict, dur)
                    except:
                        pass
                    finally:
                        if winnerModel is not None:
                            winnerModelTrainer_multi_interval(df, target, winnerModel, dur)
                            return True
                except:
                    try:
                        print('\nRUNNING AGAIN...\n')
                        df[dur['indicator']['lower']] = df[dur['indicator']['lower']] / 100
                        df[dur['indicator']['upper']] = df[dur['indicator']['upper']] / 100
                        resultsDict, predictionsDict, modelsDict = {}, {}, {}
                        print('Performing Multivariate Survival Analysis')

                        try:
                            winnerModel = modellingInit_multi_interval(df, target, resultsDict, predictionsDict, dur)
                        except:
                            pass
                        finally:
                            if winnerModel is not None:
                                winnerModelTrainer_multi_interval(df, target, winnerModel, dur)
                                return True
                    except:
                        print(
                            'Your data is not suited for Multivariate analysis. Please take observations from'
                            ' Univariate analysis.')
                return True

            else:
                return False

        elif censoring_type == 'RIGHT':
            lifetimes(None, df[target], censoring_type, df[dur['indicator']['lower']], df[dur['indicator']['upper']])
            normalityPlots(df[target])
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)

            if winnerModel is not None:
                winnerModelTrainer_uni_interval(df, target, winnerModel, dur)
                print('\n---------------------------------------------\n\nPerforming Multivariate Survival Analysis')
                resultsDict, predictionsDict, modelsDict = {}, {}, {}
                try:
                    try:
                        winnerModel = modelling_pysurvival(df, target, resultsDict, predictionsDict, dur, modelsDict)
                    except:
                        pass
                    finally:
                        if winnerModel is not None:
                            winnerModeltrainer_pysurv(df, target, dur, winnerModel)
                            return True
                except:
                    try:
                        resultsDict, predictionsDict, modelsDict = {}, {}, {}
                        try:
                            winnerModel = modellingInit_multi_interval(df, target, resultsDict, predictionsDict, dur)
                        except:
                            pass
                        finally:
                            if winnerModel is not None:
                                winnerModelTrainer_multi_interval(df, target, winnerModel, dur)
                                return True
                    except:
                        try:
                            print('\nRUNNING AGAIN...\n')
                            df[dur['indicator']['lower']] = df[dur['indicator']['lower']] / 100
                            df[dur['indicator']['upper']] = df[dur['indicator']['upper']] / 100
                            resultsDict, predictionsDict, modelsDict = {}, {}, {}
                            print('Performing Multivariate Survival Analysis')

                            try:
                                winnerModel = modellingInit_multi_interval(df, target, resultsDict, predictionsDict,
                                                                           dur)
                            except:
                                pass
                            finally:
                                if winnerModel is not None:
                                    winnerModelTrainer_multi_interval(df, target, winnerModel, dur)
                                    return True
                        except:
                            print(
                                'Your data is not suited for Multivariate analysis. Please take observations from'
                                ' Univariate analysis.')
                return True
            else:
                return False

    else:
        print('Initial Length of the data: ', len(df))
        resultsDict, predictionsDict, modelsDict = {}, {}, {}

        if censoring_type != 'INTERVAL':
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
            normalityPlots(df[target])
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
            if winnerModel is not None:
                winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type)
                return True
        else:
            lifetimes(None, df[target], censoring_type, df[dur['indicator']['lower']], df[dur['indicator']['upper']])
            normalityPlots(df[target])
            print('Performing Univariate Survival Analysis.')
            winnerModel = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)

            if winnerModel is not None:
                winnerModelTrainer_uni_interval(df, target, winnerModel, dur)
                return True
