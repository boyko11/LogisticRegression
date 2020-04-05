import numpy as np
from base.base_learner import BaseLearner


class LogisticRegressionLearner(BaseLearner):

    def __init__(self, theta_vector_size, learning_rate=0.001):
        self.theta = np.random.rand(1, theta_vector_size)
        self.learning_rate = learning_rate
        self.cost_history = []
        self.theta_history = []

    def predict(self, feature_data):

        return self.predict_for_theta(feature_data, self.theta)

    @staticmethod
    def predict_for_theta(feature_data, theta):

        z = np.dot(np.insert(feature_data, 0, 1, axis=1), np.transpose(theta)).flatten()
        return 1/(1 + np.exp(-z))

    def calculate_cost(self, predictions, labels):
        predictions[predictions == 1] -= 0.00001
        predictions[predictions == 0] += 0.00001

        predictions_logs = np.log(predictions)
        one_minus_prediction_logs = np.log(1 - predictions)
        one_errors = labels * predictions_logs
        one_error = np.sum(one_errors)
        zero_errors = (1 - labels) * one_minus_prediction_logs
        zero_error = np.sum(zero_errors)
        all_errors_sum = one_error + zero_error
        return -all_errors_sum/predictions.shape[0]

    def train(self, feature_data, labels):

        for i in range(4000):
            predictions = self.predict(feature_data)
            current_cost = self.calculate_cost(predictions, labels)
            # print('current cost: ', current_cost)
            self.cost_history.append(current_cost)
            self.theta_history.append(self.theta)
            self.update_theta_gradient_descent(predictions, feature_data, labels)

        min_cost_index = np.argmin(self.cost_history)
        self.theta = self.theta_history[min_cost_index]

        print('min_cost_index: ', min_cost_index)
        # print('min_cost_theta: ', self.theta)

        self.cost_history = self.cost_history[:min_cost_index + 1]

    def update_theta_gradient_descent(self, predictions, feature_data, labels):

        predictions_minus_labels = np.transpose(predictions - labels)

        predictions_minus_labels = predictions_minus_labels.reshape(predictions_minus_labels.shape[0], 1)

        gradient = np.mean(predictions_minus_labels * feature_data, axis=0)
        #add 1 for the bias
        gradient = np.concatenate(([1], gradient))

        self.theta = self.theta - self.learning_rate * gradient


