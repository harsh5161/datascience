import matplotlib.pyplot as plt

def basicPlot(props):
    df = props['df']
    info = props['info']
    plt.plot(df[info['Target']])