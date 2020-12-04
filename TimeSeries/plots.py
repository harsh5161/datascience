import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def basicPlot(props):
    # This will be replaced with PlotLy.JS
    print('\nPlotting Basic Target without any changes!')
    df = props['df']
    df.index = df[props['info']['PrimaryDate']]
    info = props['info']
    plt.plot(df[info['Target']])
    return None

def decompositionPlot(props):
    df = props['df']
    df.index = df[props['info']['PrimaryDate']]
    target = props['info']['Target']
    resultPlot = seasonal_decompose(df[target],model='multiplicative')
    resultPlot.plot()
    plt.show()
    
def fbprophet_plots(model,forecast):
    model.plot(forecast)
    plt.show()
    model.plot_components(forecast)
    plt.show()
    
def neural_prophet_plots(model,forecasts):
    model.plot_components(forecasts)
    plt.show()
    model.plot_parameters()
    plt.show()