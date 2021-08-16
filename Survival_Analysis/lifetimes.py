import matplotlib.pyplot as plt
from lifelines.plotting import *
import matplotlib as mpl
import numpy as np
import pandas as pd

seed = 42

np.random.seed(seed)
plt.style.use('bmh')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'


def lifetimes(time_events, actual, censor,
              lower_bound_t, upper_bound_t):
    # with joblib.parallel_backend('dask'):
    if censor != 'INTERVAL':
        if 30000 <= len(time_events) < 50000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.001 * (len(time_events)))],
                                actual[:int(0.001 * (len(time_events)))])

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.501 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.501 * (len(time_events)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.999 * (len(time_events))):],
                                actual[int(0.999 * (len(time_events))):])
            plt.show()
        elif 9000 <= len(time_events) < 30000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.002 * (len(time_events)))],
                                actual[:int(0.002 * (len(time_events)))])

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.502 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.502 * (len(time_events)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.998 * (len(time_events))):],
                                actual[int(0.998 * (len(time_events))):])
            plt.show()
        elif 50000 < len(time_events) <= 300000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.0002 * (len(time_events)))],
                                actual[:int(0.0002 * (len(time_events)))])

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.5002 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.5002 * (len(time_events)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.9998 * (len(time_events))):],
                                actual[int(0.9998 * (len(time_events))):])
            plt.show()
        elif 300000 < len(time_events) <= 600000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.0001 * (len(time_events)))],
                                actual[:int(0.0001 * (len(time_events)))])

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.5001 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.5001 * (len(time_events)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.9999 * (len(time_events))):],
                                actual[int(0.9999 * (len(time_events))):])
            plt.show()
        elif len(time_events) > 600000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.00005 * (len(time_events)))],
                                actual[:int(0.00005 * (len(time_events)))])

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.50005 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.50005 * (len(time_events)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.99995 * (len(time_events))):],
                                actual[int(0.99995 * (len(time_events))):])
            plt.show()
        elif len(time_events) < 9000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.0025 * (len(time_events)))],
                                actual[:int(0.0025 * (len(time_events)))])

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.5025 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.5025 * (len(time_events)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.9975 * (len(time_events))):],
                                actual[int(0.9975 * (len(time_events))):])
            plt.show()

    else:
        if 30000 <= len(lower_bound_t) < 50000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.001 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.001 * (len(upper_bound_t)))])
            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.501 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.501 * (len(upper_bound_t)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.99 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.99 * (len(upper_bound_t))):])
            plt.show()
        elif 9000 <= len(lower_bound_t) < 30000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.002 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.002 * (len(upper_bound_t)))])
            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.502 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.502 * (len(upper_bound_t)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.998 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.998 * (len(upper_bound_t))):])
            plt.show()
        elif 50000 < len(lower_bound_t) <= 300000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.0002 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.0002 * (len(upper_bound_t)))])
            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5002 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5002 * (len(upper_bound_t)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.9998 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.9998 * (len(upper_bound_t))):])
            plt.show()
        elif 300000 < len(lower_bound_t) <= 600000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.0001 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.0001 * (len(upper_bound_t)))])
            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5001 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5001 * (len(upper_bound_t)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.9999 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.9999 * (len(upper_bound_t))):])
            plt.show()
        elif len(lower_bound_t) > 600000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.00005 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.00005 * (len(upper_bound_t)))])
            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.50005 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.50005 * (len(upper_bound_t)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.99995 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.99995 * (len(upper_bound_t))):])
            plt.show()
        elif len(lower_bound_t) < 9000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.0025 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.0025 * (len(upper_bound_t)))])
            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5025 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5025 * (len(upper_bound_t)))])

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.9975 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.9975 * (len(upper_bound_t))):])
            plt.show()


def mrl(winnerModel, df, target, dur, censor, modelName,orig, type_='multi'):
    if type_ == 'multi':
        # Survival - Hazard
        tt = df[dur['indicator']]
        tt = pd.DataFrame(tt)

        # Survival
        surv = winnerModel.predict_survival(df.drop([target, dur['indicator']], axis=1))
        surv = pd.DataFrame(surv)
        surv = surv.replace([np.inf, -np.inf], 0)
        surv_t = pd.concat([surv, tt], axis=1)
        # surv_pot = surv_t.loc[(surv_t[dur['indicator']] > POT) & (surv_t[dur['indicator']] < upper), :]
        # print(surv_pot.head(2))
        surv_pot = surv_t.copy()
        surv_pot['surv'] = surv_t.drop([dur['indicator']], axis=1).sum(axis=1)
        # print(surv_pot.head(5))
        surv_pot['multi'] = surv_pot['surv'] * surv_pot[dur['indicator']]
        # print(surv_pot.head(5))
        # prob_surv = surv_pot['multi'].sum(axis=0)/len(surv_pot)

        # Hazard
        hazard = winnerModel.predict_hazard(df.drop([target, dur['indicator']], axis=1))
        hazard = pd.DataFrame(hazard)
        hazard = hazard.replace([np.inf, -np.inf], 0)
        # print(hazard.head())
        hz_t = pd.concat([hazard, tt], axis=1)
        # print(hz_t.head())
        # hz_pot = hz_t.loc[(hz_t[dur['indicator']] > POT) & (hz_t[dur['indicator']] < upper), :]
        # print(hz_pot.head(2))
        hz_pot = hz_t.copy()
        hz_pot['hazard'] = hz_t.drop([dur['indicator']], axis=1).sum(axis=1)
        hz_pot['multi1'] = hz_pot['hazard'] * hz_pot[dur['indicator']]
        # prob_hazard = hz_pot['multi'].sum(axis=0)/len(hz_pot)

        total = pd.concat([surv_pot[['multi', dur['indicator']]], hz_pot['multi1']], axis=1)
        total['MRL'] = (total['multi'] + total['multi1']) / len(total)
        total = total.loc[total['MRL'] > 0, :]
        total = total[[dur['indicator'], 'MRL']]
        length = len(total)
        total = pd.concat([total, orig], axis=1)
        total = total.iloc[:length, :]
        total = total.sort_values(dur['indicator'])
        total.plot(x=dur['indicator'], y='MRL', figsize=(15, 8), title='Mean Residual Lifetime Variation')
        return total

    elif type_ == 'uni':
        if censor != 'INTERVAL':
            # print(winnerModel.hazard_)
            # print('-------------------\n\n', winnerModel.survival_function_, '\n')
            haz = winnerModel.hazard_
            haz['prob'] = haz.index * haz.iloc[:, 0]
            # haz = haz.rename(columns={'Hazard Estimate':'Estimate'})

            surv = winnerModel.survival_function_
            surv['prob'] = surv.index * surv.iloc[:, 0]
            # surv = surv.rename(columns={'Survival Estimate': 'Estimate'})
            # print(haz, '--------------\n', surv, '------------\n')
            total = haz + surv

            total['MRL'] = total.iloc[:, 1]
            total = total.reset_index(drop=False)
            total.rename(columns={'index': 'Point of Time'}, inplace=True)
            total = total[['Point of Time', 'MRL']]
            length = len(total)
            total = pd.concat([total, orig], axis=1)
            total = total.iloc[:length, :]
            # print(total)
            if len(total) != 1:
                total.plot(x='Point of Time', y='MRL', figsize=(15, 8), title='Mean Residual Lifetime Variation')
            return total

        elif censor == 'INTERVAL':
            haz = winnerModel.hazard_
            haz['prob'] = haz.index * haz.iloc[:, 0]
            # haz = haz.rename(columns={'Hazard Estimate':'Estimate'})

            surv = winnerModel.survival_function_
            surv['prob'] = surv.index * surv.iloc[:, 0]
            # surv = surv.rename(columns={'Survival Estimate': 'Estimate'})
            # print(haz, '----\n', surv)
            total = haz + surv
            # print(total)
            total.reset_index(drop=False, inplace=True)
            # print('-----\n', total)
            # total.rename(columns={'index':'lower_bound'}, inplace=True)
            # print('-----\n', total)
            df = df.sort_values(by=dur['indicator']['lower'], ascending=True)
            total['upper_bound'] = df[dur['indicator']['upper']]
            total['Duration'] = total['upper_bound'] - total.iloc[:, 0]
            # print(total)
            total['MRL'] = total.iloc[:, 2]
            total = total[['Duration', 'MRL']]
            length = len(total)
            total = pd.concat([total, orig], axis=1)
            total = total.iloc[:length, :]
            # print(total)
            if len(total) != 1:
                total.plot(x='Duration', y='MRL', figsize=(15, 8), title='Mean Residual Lifetime Variation')
            return total
