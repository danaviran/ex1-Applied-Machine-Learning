import numpy as np
import pandas as pd
from tqdm import tqdm


def sample_prophets(k, min_p, max_p):
    """
    Samples a set of k prophets
    :param k: number of prophets
    :param min_p: minimum probability
    :param max_p: maximum probability
    :return: list of prophets
    """
    prophets_err_rate = np.random.uniform(min_p, max_p, size=k)
    prophets = []
    for i in range(k):
        p = Prophet(prophets_err_rate[i])
        prophets.append(p)
    return prophets


class Prophet:

    def __init__(self, err_prob):
        """
        Initializes the Prophet model
        :param err_prob: the probability of the prophet to be wrong
        """
        ############### YOUR CODE GOES HERE ###############
        self.err_prob = err_prob

    def predict(self, y):
        """
        Predicts the label of the input point
        draws a random number between 0 and 1
        if the number is less than the probability, the prediction is correct (according to y)
        else the prediction is wrong
        NOTE: Realistically, the prophet should be a function from x to y (without getting y as an input)
        However, for the simplicity of our simulation, we will give the prophet y straight away
        :param y: the true label of the input point
        :return: a prediction for the label of the input point
        """
        # Generate random numbers in [0,1] for each point in array and use threshold
        random_threshes = np.random.rand(y.shape[0])
        correct_predictions = random_threshes > self.err_prob
        predictions = np.where(correct_predictions, y, 1 - y)
        return predictions
