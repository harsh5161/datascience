from modules_multi import *
from Modules import *
from lifetimes import lifetimes, mrl
from input import *
from plots import kmf_valid
from uni_input import *


def INIT_multi():
    try:
        df, target, dur, censoring_type, orig, dur_cols, encoder = userInputs()
        print(df.dtypes)
        # print(df.isnull().sum())
    except:
        return None

    if censoring_type in ['RIGHT', 'UNCENSORED']:
        try:
            if df.shape[1] > 25:
                df = featureSelection(df, target, dur, censoring_type)
                # print(df.isnull().sum())
        except:
            return None
        # print(df.isnull().sum())
        df[dur['indicator']] = df[dur['indicator']].fillna(1e-5)
        df = df.fillna(0)
        normalityPlots(df[dur['indicator']])
        targetDistribution(df, target)

        display(df.head())
        print(f'Number of feature variables are {df.shape[1] - 1}\n')
        print('Performing Survival Analysis.')
        resultsDict, predictionsDict, modelsDict, resultsDict_, predictionsDict_, modelsDict_ = {}, {}, {}, {}, {}, {}
        winnerModel, instance_ = right_cens(df, target, resultsDict, predictionsDict, dur, modelsDict,
                                            resultsDict_, predictionsDict_, modelsDict_)

        if winnerModel is not None:

            winnerCens(df, target, dur, winnerModel, instance_,orig, mode='default', target_availability='None')
            scoring_multi(df, target, dur, winnerModel, instance_, orig)

            if winnerModel not in ['LogLogisticFitter', 'LogNormalFitter', 'ExponentialFitter', 'WeibullFitter']:
                data = mrl(instance_, df, target, dur, winnerModel, censoring_type, orig)
                data.to_csv('Mean_Residual_Lifetime.csv', index=False)
                print('Mean residual lifetime has been calculated!')
            else:
                data = mrl(instance_, df, target, dur, censoring_type, winnerModel, orig, type_='uni')
                data.to_csv('Mean_Residual_Lifetime.csv', index=False)
                print('Mean residual lifetime has been calculated!')

            print('\nVisualizing the overall lifetime variation')
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)

            if winnerModel in ['LogLogisticFitter', 'LogNormalFitter', 'ExponentialFitter', 'WeibullFitter']:
                print('\nDisplaying non-parametric Kaplan-Meier Plots for validation\n')
                kmf_valid(df, target, dur)

            ll = input('Enter "y" if you want to carry out prediction: ').upper()
            if ll == 'Y':
                print('\nTo carry out predictions, read the following instructions:\n')
                print('1. Ensure the column names match to the data provided before')
                tt_in = input("\nEnter 'y' if you the dataset has the target column: ").upper()
                if tt_in == 'Y':
                    df_, dur_, censor_, target_ = lead_input(orig, censoring_type, dur, target, dur_cols,
                                                             encoder)
                    df_ = df_[(df_.columns) & (df.columns)]
                    winnerCens(df_, target_, dur_, winnerModel, instance_,orig, mode='scoring', target_availability='tt_in')
                else:
                    df_, dur_, censor_, target_ = lead_input(orig, censoring_type, dur, target, dur_cols,
                                                             encoder)
                    df_ = df_[(df_.columns) & (df.columns)]
                    winnerCens(df_, target_, dur_, winnerModel, instance_,orig, mode='scoring', target_availability=None)
            return True
        else:
            print('Your data is not suited for Multivariate analysis.'
                  ' Please use quick results.')

    elif censoring_type == 'LEFT':
        resultsDict, predictionsDict, modelsDict = {}, {}, {}
        normalityPlots(df[dur['indicator']])
        targetDistribution(df, target)

        print('Performing Survival Analysis.')
        winnerModel, instance_ = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict, censoring_type,
                                                   dur)
        if winnerModel is not None:
            winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type, instance_,orig, mode='default',
                                   target_availability=None)

            scoring_prediction(df, target, winnerModel, dur, censoring_type, instance_, orig)

            data = mrl(instance_, df, target, dur, censoring_type, winnerModel, orig, type_='uni')
            data.to_csv('Mean_Residual_Lifetime.csv', index=False)
            print('Mean residual Lifetime calculated!')

            print('\nVisualizing the overall lifetime variation')
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
            # qth_surv = float(input('Enter percentile(0-1) to calculate Qth survival time: '))
            # data_surv = qth_survival(qth_surv, instance_, df, target, dur, type_='uni')
            # print(data_surv)
            print('\nDisplaying non-parametric Kaplan-Meier Plots for validation\n')
            kmf_valid(df, target, dur)
            ll = input('Enter "y" if you want to carry out prediction: ').upper()
            if ll == 'Y':
                print('\nTo carry out predictions, read the following instructions:\n')
                print('1. Ensure the column names match to the data provided before')
                tt_in = input("\nEnter 'y' if you the dataset has the target column: ").upper()
                if tt_in == 'Y':
                    df_, dur_, censor_, target_ = lead_input_target_(orig, censoring_type, dur, target, dur_cols)
                    winnerModelTrainer_uni(df_, target_, winnerModel, dur_, censor_, instance_,orig, mode='scoring',
                                           target_availability='tt_in')
                else:
                    df_, dur_, censor_, target_ = lead_input_(orig, censoring_type, dur, target, dur_cols)
                    winnerModelTrainer_uni(df_, target_, winnerModel, dur_, censor_, instance_,orig, mode='scoring',
                                           target_availability=None)

            return True
        else:
            return False

    elif censoring_type == 'INTERVAL':
        resultsDict, predictionsDict, modelsDict = {}, {}, {}
        normalityPlots(df[dur['indicator']['lower']])
        normalityPlots(df[dur['indicator']['upper']])
        targetDistribution(df, target)

        print('Performing Survival Analysis.')
        winnerModel, instance_ = modellingInit_uni_interval(df, target, resultsDict, predictionsDict, modelsDict, dur)

        if winnerModel is not None:
            winnerModelTrainer_uni_interval(df, target, winnerModel, instance_, dur,orig, mode='default',
                                            target_availability=None)

            scoring_pred_interval(df, target, winnerModel, instance_, dur, orig)

            data = mrl(instance_, df, target, dur, censoring_type, winnerModel,orig, type_='uni')
            data.to_csv('Mean_Residual_Lifetime.csv', index=False)
            print('Mean residual Lifetime calculated!')

            print('\nVisualizing the overall lifetime variation')
            lifetimes(None, df[target], censoring_type, df[dur['indicator']['lower']], df[dur['indicator']['upper']])
            # qth_surv = float(input('Enter percentile(0-1) to calculate Qth survival time: '))
            # data_surv = qth_survival(qth_surv, instance_, df, target, dur, type_='uni')
            # print(data_surv)

            ll = input('Enter "y" if you want to carry out prediction: ').upper()
            if ll == 'Y':
                print('\nTo carry out predictions, read the following instructions:\n')
                print('1. Ensure the column names match to the data provided before')

                tt_in = input("\nEnter 'y' if you the dataset has the target column: ").upper()
                if tt_in == 'Y':
                    df_, dur_, censor_, target_ = lead_input_target_(orig, censoring_type, dur, target, dur_cols)
                    winnerModelTrainer_uni_interval(df_, target_, winnerModel, instance_, dur_,orig, mode='scoring',
                                                    target_availability='tt_in')
                else:
                    df_, dur_, censor_, target_ = lead_input_(orig, censoring_type, dur, target, dur_cols)
                    winnerModelTrainer_uni_interval(df_, target_, winnerModel, instance_, dur_,orig, mode='scoring',
                                                    target_availability=None)
            return True
        else:
            return False
