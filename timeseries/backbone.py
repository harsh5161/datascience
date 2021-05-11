# Central Pipeline where Data will be passed around
from modules import *


def INIT():
    df = userInputs()

    if df is None:
        return None

    action_list = processIdentifier(df)
