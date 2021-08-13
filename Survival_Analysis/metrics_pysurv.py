def aic(model):
    return abs(model.aic)


pysurv_metric = {'AIC': aic}


def pysurv_eval(model):
    results = {'AIC': pysurv_metric['AIC'](model)}
    return results
