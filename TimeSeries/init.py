from userInputs import importFile
from engineerings import numeric_engineering,date_engineering

def TimeSeriesInit(path,info):
    df,_ = importFile(path)

    # Numeric Engineering
    df = numeric_engineering(df)

    return None,None

