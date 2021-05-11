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
            print("function Y")
            seasonalDecompose(df[target][:365])
        if val == 'M':
            # call function passing df,target and other parameters specific to M
            print("function M")
            seasonalDecompose(df[target][:300])
        if val == 'D':
            # call function passing df,target and other parameters specific to D
            print("function D")
            seasonalDecompose(df[target][:250])
        if val == 'RQ':
            # call function passing resampled_frames['Quarter'],target and other parameters specific to RQ
            print("function RQ")
            seasonalDecompose(resampled_frames['Quarter'][target][:91])
        if val == 'RM':
            # call function passing resampled_frames['Monthly'],target and other parameters specific to RM
            print("function RM")
            seasonalDecompose(resampled_frames['Month'][target][:300])
        if val == 'RW':
            # call function passing resampled_frames['Daily'],target and other parameters specific to RW
            print("function RW")
            seasonalDecompose(resampled_frames['Week'][target][:52])
