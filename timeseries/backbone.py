# Central Pipeline where Data will be passed around
from modules import *


def INIT():
    df, target = userInputs()

    if df is None or target is None:
        return None

    action_list = processIdentifier(df)

    if len(action_list) == 0:
        return None

    resampled_frames = dataResampler(df, action_list)

    for val in action_list:
        if val == 'Y':
            # call function passing df,target and other parameters specific to Y
            print("Performing Yearly Analysis")
            seasonalDecompose(df[target][:365])
        if val == 'M':
            # call function passing df,target and other parameters specific to M
            print("Performing Monthly Analysis")
            seasonalDecompose(df[target][:300])
        if val == 'D':
            # call function passing df,target and other parameters specific to D
            print("Performing Daily Analysis")
            seasonalDecompose(df[target][:250])
        if val == 'RQ':
            # call function passing resampled_frames['Quarter'],target and other parameters specific to RQ
            print("Performing Resampled Quarterly Analysis")
            seasonalDecompose(resampled_frames['Quarter'][target])
        if val == 'RM':
            # call function passing resampled_frames['Monthly'],target and other parameters specific to RM
            print("Performing Resampled Monthly Analysis")
            seasonalDecompose(resampled_frames['Month'][target])
        if val == 'RW':
            # call function passing resampled_frames['Daily'],target and other parameters specific to RW
            print("Performing Resampled Weekly Analysis")
            seasonalDecompose(resampled_frames['Week'][target])
