import numpy as np

class GaussianNaiveBayesFromScratch:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator