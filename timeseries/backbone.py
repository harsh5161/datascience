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
        if val == 'M':
            # call function passing df,target and other parameters specific to M
            print("function M")
        if val == 'D':
            # call function passing df,target and other parameters specific to D
            print("function D")
        if val == 'RQ':
            # call function passing resampled_frames['Quarter'],target and other parameters specific to RQ
            print("function RQ")
        if val == 'RM':
            # call function passing resampled_frames['Monthly'],target and other parameters specific to RM
            print("function RM")
        if val == 'RW':
            # call function passing resampled_frames['Daily'],target and other parameters specific to RW
            print("function RW")
