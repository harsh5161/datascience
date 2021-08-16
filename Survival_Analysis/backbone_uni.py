from Modules import *
from lifetimes import lifetimes, mrl
from uni_input import *
from plots import kmf_valid


def INIT():
    try:
        df, target, dur, censoring_type, orig, dur_cols = userInputs_()
    except:
        return None

    print('Initial Length of the data: ', len(df), '\n')

    if censoring_type != 'INTERVAL':
        resultsDict, predictionsDict, modelsDict = {}, {}, {}
        normalityPlots(df[dur['indicator']])
        targetDistribution(df, target)

        print('Performing Univariate Survival Analysis.')
        winnerModel, instance_ = modellingInit_uni(df, target, resultsDict, predictionsDict, modelsDict,
                                                   censoring_type, dur)
        if winnerModel is not None:
            winnerModelTrainer_uni(df, target, winnerModel, dur, censoring_type, instance_,orig, mode='default',
                                   target_availability=None)

            scoring_prediction(df, target, winnerModel, dur, censoring_type, instance_, orig)

            data = mrl(instance_, df, target, dur, censoring_type, winnerModel,orig,  type_='uni')
            print(data.columns)
            data.to_csv('Mean_Residual_Lifetime.csv', index=False)
            print('Mean residual Lifetime calculated!')

            print('\nVisualizing the overall lifetime variation')
            lifetimes(df[dur['indicator']], df[target], censoring_type, None, None)
            # qth_surv = float(input('Enter percentile(0-1) to calculate Qth survival time: '))
            # data_surv = qth_survival(qth_surv, instance_, df, target, dur, type_='uni')
            # print(data_surv)
            print('\nDisplaying non-parametric Kaplan-Meier Plots for validation\n')
            kmf_valid(df, target, dur)

            # If a dataset is provided with no target columns
            ll = input('Enter "y" if you want to carry out prediction: ').upper()
            if ll == 'Y':
                print('\nTo carry out predictions, read the following instructions:\n')
                print('1. Ensure the column names match to the data provided before')
                tt_in = input("\nEnter 'y' if you the dataset has the target column: ").upper()
                if tt_in == 'Y':
                    df_, dur_, censor_, target_ = lead_input_target_(orig, censoring_type, dur, target, dur_cols)
                    winnerModelTrainer_uni(df_, target_, winnerModel, dur_, censor_, instance_, orig, mode='scoring',
                                           target_availability='tt_in')
                else:
                    df_, dur_, censor_, target_ = lead_input_(orig, censoring_type, dur, target, dur_cols)
                    winnerModelTrainer_uni(df_, target_, winnerModel, dur_, censor_, instance_, orig, mode='scoring',
                                           target_availability=None)

            return True
        else:
            return False

    # Interval censoring
    else:
        resultsDict, predictionsDict, modelsDict = {}, {}, {}
        normalityPlots(df[dur['indicator']['lower']])
        normalityPlots(df[dur['indicator']['upper']])
        targetDistribution(df, target)

        print('Performing Univariate Survival Analysis.')
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