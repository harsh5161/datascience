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


def lifetime_plot(time_events, actual, lower_bound_t,
                  upper_bound_t, pot, censor):
    if censor != 'INTERVAL':
        if 30000 <= len(time_events) < 50000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.001 * (len(time_events)))],
                                actual[:int(0.001 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[:int(0.001 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.501 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.501 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[int(0.50 * (len(time_events))):int(0.501 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.999 * (len(time_events))):],
                                actual[int(0.999 * (len(time_events))):])
            ax.set_xlim(0, max(time_events[int(0.999 * (len(time_events))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            plt.show()

        elif len(time_events) < 30000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.0024 * (len(time_events)))],
                                actual[:int(0.0024 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[:int(0.0024 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.5024 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.5024 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[int(0.50 * (len(time_events))):int(0.5024 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.9976 * (len(time_events))):],
                                actual[int(0.9976 * (len(time_events))):])
            ax.set_xlim(0, max(time_events[int(0.9976 * (len(time_events))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

        elif 50000 < len(time_events) <= 300000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.0002 * (len(time_events)))],
                                actual[:int(0.0002 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[:int(0.0002 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.5002 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.5002 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[int(0.50 * (len(time_events))):int(0.5002 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.9998 * (len(time_events))):],
                                actual[int(0.9998 * (len(time_events))):])
            ax.set_xlim(0, max(time_events[int(0.9998 * (len(time_events))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

        elif 300000 < len(time_events) <= 600000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.0001 * (len(time_events)))],
                                actual[:int(0.0001 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[:int(0.0001 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.5001 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.5001 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[int(0.50 * (len(time_events))):int(0.5001 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.9999 * (len(time_events))):],
                                actual[int(0.9999 * (len(time_events))):])
            ax.set_xlim(0, max(time_events[int(0.9999 * (len(time_events))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

        elif len(time_events) > 600000:
            fig = plt.figure(figsize=(21, 10))

            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_lifetimes(time_events[:int(0.00005 * (len(time_events)))],
                                actual[:int(0.00005 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[:int(0.00005 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_lifetimes(time_events[int(0.50 * (len(time_events))):int(0.50005 * (len(time_events)))],
                                actual[int(0.50 * (len(time_events))):int(0.50005 * (len(time_events)))])
            ax.set_xlim(0, max(time_events[int(0.50 * (len(time_events))):int(0.50005 * (len(time_events)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_lifetimes(time_events[int(0.99995 * (len(time_events))):],
                                actual[int(0.99995 * (len(time_events))):])
            ax.set_xlim(0, max(time_events[int(0.99995 * (len(time_events))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

    else:
        if 30000 <= len(lower_bound_t) < 50000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.001 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.001 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[:int(0.001 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.501 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.501 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.501 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.99 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.99 * (len(upper_bound_t))):])
            ax.set_xlim(0, max(upper_bound_t[int(0.99 * (len(upper_bound_t))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

        elif len(lower_bound_t) < 30000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.002 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.002 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[:int(0.002 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.502 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.502 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.502 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.998 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.998 * (len(upper_bound_t))):])
            ax.set_xlim(0, max(upper_bound_t[int(0.998 * (len(upper_bound_t))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

        elif 50000 < len(lower_bound_t) <= 300000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.0002 * (len(lower_bound_t)))],
                                                  upper_bound_t[int(0.998 * (len(upper_bound_t))):])
            ax.set_xlim(0, max(upper_bound_t[int(0.998 * (len(upper_bound_t))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5002 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5002 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5002 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.9998 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.9998 * (len(upper_bound_t))):])
            ax.set_xlim(0, max(upper_bound_t[int(0.9998 * (len(upper_bound_t))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

        elif 300000 < len(lower_bound_t) <= 600000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.0001 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.0001 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[:int(0.0001 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            fig.add_subplot(132)

            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5001 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5001 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.5001 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.9999 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.9999 * (len(upper_bound_t))):])
            ax.set_xlim(0, max(upper_bound_t[int(0.9999 * (len(upper_bound_t))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()

        elif len(lower_bound_t) > 600000:
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(131)
            plt.title('Lifetime variation from the beginning')
            ax = plot_interval_censored_lifetimes(lower_bound_t[:int(0.00005 * (len(lower_bound_t)))],
                                                  upper_bound_t[:int(0.00005 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[:int(0.00005 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(132)
            plt.title('Lifetime variation from the middle')
            ax = plot_interval_censored_lifetimes(
                lower_bound_t[int(0.50 * (len(lower_bound_t))):int(0.50005 * (len(lower_bound_t)))],
                upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.50005 * (len(upper_bound_t)))])
            ax.set_xlim(0, max(upper_bound_t[int(0.50 * (len(lower_bound_t))):int(0.50005 * (len(upper_bound_t)))]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')

            fig.add_subplot(133)
            plt.title('Lifetime variation from the end')
            ax = plot_interval_censored_lifetimes(lower_bound_t[int(0.99995 * (len(lower_bound_t))):],
                                                  upper_bound_t[int(0.99995 * (len(upper_bound_t))):])
            ax.set_xlim(0, max(upper_bound_t[int(0.99995 * (len(upper_bound_t))):]))
            ax.vlines(pot, 0, 70, lw=2, linestyles='--')
            plt.show()


def mrl(winnerModel, df, target, dur, POT=None, type_='multi'):
    if type_ == 'multi':
        if POT is not None:
            # Survival - Hazard
            tt = df[dur['indicator']]
            tt = pd.DataFrame(tt)
            upper = POT + 1

            # Survival
            surv = winnerModel.predict_survival(df.drop([target, dur['indicator']], axis=1))
            surv = pd.DataFrame(surv)
            surv = surv.replace([np.inf, -np.inf], 0)
            surv_t = pd.concat([surv, tt], axis=1)
            surv_pot = surv_t.loc[(surv_t[dur['indicator']] >= POT) & (surv_t[dur['indicator']] < upper), :]
            # print(surv_pot.head(2))
            surv_pot['surv'] = surv_pot.drop([dur['indicator']], axis=1).sum(axis=1)
            # print(surv_pot.head(5))
            surv_pot['multi'] = surv_pot['surv'] * surv_pot[dur['indicator']]
            # print(surv_pot.head(5))
            prob_surv = surv_pot['multi'].sum(axis=0) / len(surv_pot)

            # Hazard
            hazard = winnerModel.predict_hazard(df.drop([target, dur['indicator']], axis=1))

            hazard = pd.DataFrame(hazard)
            hazard = hazard.replace([np.inf, -np.inf], 0)
            # print(hazard.head())
            hz_t = pd.concat([hazard, tt], axis=1)
            # print(hz_t.head())
            hz_pot = hz_t.loc[(hz_t[dur['indicator']] >= POT) & (hz_t[dur['indicator']] < upper), :]
            # print(hz_pot.head(2))
            hz_pot['hazard'] = hz_pot.drop([dur['indicator']], axis=1).sum(axis=1)
            hz_pot['multi1'] = hz_pot['hazard'] * hz_pot[dur['indicator']]
            prob_hazard = hz_pot['multi1'].sum(axis=0) / len(hz_pot)

            total = pd.concat([surv_pot[['multi', dur['indicator']]], hz_pot['multi1']], axis=1)
            total['MRL'] = (total['multi'] + total['multi1']) / len(total)
            total = total.loc[total['MRL'] > 0, :]
            total = total[[dur['indicator'], 'MRL']]
            total = total.sort_values(dur['indicator'])
            if len(total) != 1:
                total.plot(x=dur['indicator'], y='MRL', figsize=(15, 8), title='Mean Residual Lifetime Variation')

            return abs(prob_surv + prob_hazard), total

        else:
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
            total = total.sort_values(dur['indicator'])
            if len(total) != 1:
                total.plot(x=dur['indicator'], y='MRL', figsize=(15, 8), title='Mean Residual Lifetime Variation')

            return total['MRL'].sum() / len(total), total

    elif type_ == 'uni':
        if POT is None:
            total = winnerModel.hazard_ + winnerModel.survival_function_
            total['MRL'] = total.index * total.iloc[:, 0]
            total = total.reset_index(drop=False)
            if len(total) != 1:
                total.plot(x='Point of Time', y='MRL', figsize=(15, 8), title='Mean Residual Lifetime Variation')
            return total['MRL'].sum() / len(total), total
        else:
            upper = POT + 10
            total = winnerModel.hazard_ + winnerModel.survival_function_
            total = total.loc[(total.index >= POT) & (total.index < upper)]
            print(total.head())
            total['MRL'] = total.index * total.iloc[:, 0]
            total = total.reset_index(drop=False)
            print(total.head())
            if len(total) != 1:
                total.plot(x='Point of Time', y='MRL', figsize=(15, 8), title='Mean Residual Lifetime Variation')
            return total['MRL'].sum() / len(total), total
