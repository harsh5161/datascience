import matplotlib.pyplot as plt

def basicPlot(props):
    # This will be replaced with PlotLy.JS
    df = props['df']
    df.index = df[props['info']['PrimaryDate']]
    info = props['info']
    plt.plot(df[info['Target']])
    return None