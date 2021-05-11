class Modelling:
    def __init__(self, train, test, target, resultsDict, predictionsDict):
        self.train = train
        self.test = test
        self.target = target
        self.resultsDict = resultsDict
        self.predictionsDict = predictionsDict

    def naiveModel(self):
        mean = test[target].mean()
        mean = np.array([mean for u in range(len(test))])
        resultsDict['Naive mean'] = evaluate(test[target], mean)
        predictionsDict['Naive mean'] = mean
