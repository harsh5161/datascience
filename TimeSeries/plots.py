import matplotlib.pyplot as plt

def basicPlot(props):
    # This will be replaced with PlotLy.JS
    df = props['df']
    info = props['info']
    plt.plot(df[info['Target']])