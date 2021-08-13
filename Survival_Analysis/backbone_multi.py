from modules_multi import *
from Modules import *
from lifetimes import lifetimes, mrl, lifetime_plot
from input import *
from plots import qth_survival, kmf_valid


def INIT_multi():
    try:
        df, target, dur, censoring_type, orig, cols_to_drop, dur_cols, encoder = userInputs()
        print(df.dtypes)
        # print(df.isnull().sum())
    except:
        return None

    if censoring_type in ['RIGHT', 'UNCENSORED']:
        try:
            if df.shape[1] > 25:
                df = featureSelection(df, target, dur, censoring_type)
                # print(df.isnull().sum())
        except: return None
        # print(df.isnull().sum())
        df[dur['indicator']] = df[dur['indicator']].fillna(1e-5)
        df = df.fillna(0)
        normalityPlots(df[dur['indicator']])
        targetDistribution(df, target)
        print('\nVisualizing the overall lifetime variation')
        lfp = input('Enter point of time around which you want to check lifetime variation (Optional/None): ')
        if lfp is not None:
            try:
                lfp = float(lfp)
                lifetime_plot(df[dur['indicator']], df[target], None, None, lfp, censoring_type)
            except:
                lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)

        display(df.head())
        print(f'Number of feature variables are {df.shape[1] - 1}\n')
        print('Performing Survival Analysis.')
        resultsDict, predictionsDict, modelsDict, resultsDict_, predictionsDict_, modelsDict_ = {}, {}, {}, {}, {}, {}
        winnerModel, instance_ = right_cens(df, target, resultsDict, predictionsDict, dur, modelsDict,
                                            resultsDict_, predictionsDict_, modelsDict_)

        if winnerModel is not None:
            winnerCens(df, target, dur, winnerModel, instance_)
            mrl_ = input('Do you want to calculate Mean Residual Lifetime? (Y/N): ').upper()
            if mrl_ == 'Y':
                pot = input('Specify point of time to calculate mean residual lifetime? (Optional): ')
                if winnerModel not in ['LogLogisticFitter', 'LogNormalFitter', 'ExponentialFitter', 'WeibullFitter']:
                    if pot != '':
                        pot = float(pot)
                        mrl_result, data = mrl(instance_, df, target, dur, pot)
                        print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
                    else:
                        pot = None
                        mrl_result, data = mrl(instance_, df, target, dur, pot)
                        print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
                else:
                    if pot != '':
                        pot = float(pot)
                        mrl_result, data = mrl(instance_, df, target, dur, pot, type_='uni')
                        print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
                    else:
                        pot = None
                        mrl_result, data = mrl(instance_, df, target, dur, pot, type_='uni')
                        print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
            else:
                pass
            qth_surv = float(input('Enter percentile(0-1) to calculate Qth survival time: '))
            if winnerModel in ['LogLogisticFitter', 'LogNormalFitter', 'ExponentialFitter', 'WeibullFitter']:
                data_surv = qth_survival(qth_surv, instance_, df, target, dur, type_='uni')
            else:
                data_surv = qth_survival(qth_surv, instance_, df, target, dur, type_='multi')
            if winnerModel in ['LogLogisticFitter', 'LogNormalFitter', 'ExponentialFitter', 'WeibullFitter']:
                print('\nDisplaying non-parametric Kaplan-Meier Plots for validation\n')
                kmf_valid(df, target, dur)

            ll = input('Enter "y" if you want to carry out prediction: ').upper()
            if ll == 'Y':
                print('\nTo carry out predictions, read the following instructions:\n')
                print('1. Ensure the column names match to the data provided before')
                print('2. Ensure there is no target event column\n')
                df_, dur_, censor_, target_ = lead_input(orig, censoring_type, dur, target, cols_to_drop, dur_cols,
                                                         encoder)
                winnerCens(df_, target_, dur_, winnerModel, instance_)
            return True
        else:
            print('Your data is not suited for Multivariate analysis.'
                  ' Please use quick results.')

    elif censoring_type == 'LEFT':
        resultsDict, predictionsDict, modelsDict = {}, {}, {}
        normalityPlots(df[dur['indicator']])
        targetDistribution(df, target)
        print('\nVisualizing the overall lifetime variation')
        lfp = float(input('Enter point of time around which you want to check lifetime variation (Optional/None): '))
        if lfp != 'None':
            lifetime_plot(df[dur['indicator']], df[target], None, None, lfp, censoring_type)
        elif lfp == 'None':
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)

        print('Performing Univariate Survival Analysis.')
        winnerModel, instance_ = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type, dur)
        if winnerModel is not None:
            winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type, instance_)
            mrl_ = input('Do you want to calculate Mean Residual Lifetime? (Y/N): ').upper()
            if mrl_ == 'Y':
                pot = input('Specify point of time to calculate mean residual lifetime? (Optional): ')
                if pot != '':
                    pot = float(pot)
                    mrl_result, data = mrl(instance_, df, target, dur, pot, type_='uni')
                    print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
                else:
                    pot = None
                    mrl_result, data = mrl(instance_, df, target, dur, pot, type_='uni')
                    print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
            else:
                pass
            qth_surv = float(input('Enter percentile(0-1) to calculate Qth survival time: '))
            data_surv = qth_survival(qth_surv, instance_, df, target, dur, type_='uni')
            print('\nDisplaying non-parametric Kaplan-Meier Plots for validation\n')
            kmf_valid(df, target, dur)
            ll = input('Enter "y" if you want to carry out prediction: ').upper()
            if ll == 'Y':
                print('\nTo carry out predictions, read the following instructions:\n')
                print('1. Ensure the column names match to the data provided before')
                print('2. Ensure there is no target event column\n')
                df_, dur_, censor_, target_ = lead_input(orig, censoring_type, dur, target, cols_to_drop, dur_cols,
                                                         encoder)
                winnerModelTrainer_uni(df_, target_, winnerModel, dur_, censor_, instance_)

            return True
        else:
            return False

    elif censoring_type == 'INTERVAL':
        resultsDict, predictionsDict, modelsDict = {}, {}, {}
        normalityPlots(df[dur['indicator']['lower']])
        normalityPlots(df[dur['indicator']['upper']])
        targetDistribution(df, target)
        print('\nVisualizing the overall lifetime variation')
        lfp = float(input('Enter point of time around which you want to check lifetime variation (Optional/None): '))
        if lfp != 'None':
            lifetime_plot(None, df[target], df[dur['indicator']['lower']], df[dur['indicator']['upper']],
                          lfp, censoring_type)
        elif lfp == 'None':
            lifetimes(None, df[target], censoring_type, df[dur['indicator']['lower']], df[dur['indicator']['upper']])

        print('Performing Univariate Survival Analysis.')
        winnerModel, instance_ = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)

        if winnerModel is not None:
            winnerModelTrainer_uni_interval(df, target, winnerModel, instance_, dur)
            mrl_ = input('Do you want to calculate Mean Residual Lifetime? (Y/N): ').upper()
            if mrl_ == 'Y':
                pot = input('Specify point of time to calculate mean residual lifetime? (Optional): ')
                if pot != '':
                    pot = float(pot)
                    mrl_result, data = mrl(instance_, df, target, dur, pot, type_='uni')
                    print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
                else:
                    pot = None
                    mrl_result, data = mrl(instance_, df, target, dur, pot, type_='uni')
                    print(f'Mean residual lifetime at t={pot} is: ', mrl_result)
            else:
                pass
            qth_surv = float(input('Enter percentile(0-1) to calculate Qth survival time: '))
            data_surv = qth_survival(qth_surv, instance_, df, target, dur, type_='uni')
            print('\nDisplaying non-parametric Kaplan-Meier Plots for validation\n')
            kmf_valid(df, target, dur)

            ll = input('Enter "y" if you want to carry out prediction: ').upper()
            if ll == 'Y':
                print('\nTo carry out predictions, read the following instructions:\n')
                print('1. Ensure the column names match to the data provided before')
                print('2. Ensure there is no target event column\n')
                df_, dur_, censor_, target_ = lead_input(orig, censoring_type, dur, target, cols_to_drop, dur_cols,
                                                         encoder)
                winnerModelTrainer_uni_interval(df_, target_, winnerModel, instance_, dur_)
            return True
        else:
            return False
