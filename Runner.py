from service.data_service import DataService
import numpy as np
from logistic_regression_learner import LogisticRegressionLearner
from service.report_service import ReportService


class Runner:

    def __init__(self, normalization_method='z'):
        self.logistic_regression_learner = None
        self.normalization_method = normalization_method
        self.report_service = ReportService()

    def run(self):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        feature_data = data[:, 2:]
        labels = data[:, 1]
        theta_vector_size = data.shape[1] - 1

        self.logistic_regression_learner = LogisticRegressionLearner(theta_vector_size, learning_rate=0.001)

        normalized_data = DataService.normalize(data, method='z')
        normalized_feature_data = normalized_data[:, 2:]

        self.train_with_gradient_descent(data, labels, normalized_feature_data, labels)

    def train_with_gradient_descent(self, data, labels_data, normalized_feature_data, normalized_labels):

        self.logistic_regression_learner.train(normalized_feature_data, normalized_labels)

        predictions = self.logistic_regression_learner.predict(normalized_feature_data)
        rounded_predictions = np.rint(predictions)

        self.report_service.report(predictions, rounded_predictions, labels_data, self.logistic_regression_learner)


if __name__ == "__main__":

    Runner(normalization_method='min-max').run()
