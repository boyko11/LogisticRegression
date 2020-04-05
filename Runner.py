from service.data_service import DataService
import numpy as np
from logistic_regression_learner import LogisticRegressionLearner
from service.plot_service import PlotService


class Runner:

    def __init__(self, normalization_method='z'):
        self.logistic_regression_learner = None
        self.normalization_method = normalization_method

    def run(self):

        data = DataService.load_csv("data/wdbc.data")
        # column 1 is the id, column 2 is the label, the rest are features
        feature_data = data[:, 2:]
        labels = data[:, 1]
        theta_vector_size = data.shape[1] - 1

        self.logistic_regression_learner = LogisticRegressionLearner(theta_vector_size, learning_rate=0.001)

        normalized_data = DataService.normalize(data, method='z')
        normalized_feature_data = normalized_data[:, 2:]

        self.train_with_gradient_descent(feature_data, labels, normalized_feature_data, labels)

    def train_with_gradient_descent(self, feature_data, labels_data, normalized_feature_data, normalized_labels):

        self.logistic_regression_learner.train(normalized_feature_data, normalized_labels)

        predictions = self.logistic_regression_learner.predict(normalized_feature_data)
        rounded_predictions = np.rint(predictions)

        accuracy = 1 - np.sum(np.abs(rounded_predictions - labels_data)) / labels_data.shape[0]

        print("Accuracy: ", accuracy)

        positive_labels_count = np.count_nonzero(labels_data)
        negative_labels_count = labels_data.shape[0] - positive_labels_count
        positive_predictions_count = np.count_nonzero(rounded_predictions)
        negative_predictions_count = labels_data.shape[0] - positive_predictions_count

        print("Positive Labels, Positive Predictions: ", positive_labels_count, positive_predictions_count)
        print("Negative Labels, Negative Predictions: ", negative_labels_count, negative_predictions_count)

        labels_for_class1_predictions = labels_data[rounded_predictions == 1]
        true_positives_class1 = np.count_nonzero(labels_for_class1_predictions)
        false_negatives_class0 = labels_for_class1_predictions.shape[0] - true_positives_class1

        labels_for_class0_predictions = labels_data[rounded_predictions == 0]
        false_negatives_class1 = np.count_nonzero(labels_for_class0_predictions)
        true_positives_class0 = labels_for_class0_predictions.shape[0] - false_negatives_class1

        print('Class 1, true_positives, false_positives: ', true_positives_class1,
              positive_predictions_count - true_positives_class1)
        precision_class1 = np.around(true_positives_class1/positive_predictions_count, 3)
        recall_class1 = np.around(true_positives_class1 / (true_positives_class1 + false_negatives_class1), 3)
        class1_f1_score = np.around(2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1), 3)

        print('Class 0, true_positives, false_positives: ', true_positives_class0,
              negative_predictions_count - true_positives_class0)
        precision_class0 = np.around(true_positives_class0/negative_predictions_count, 3)
        recall_class0 = np.around(true_positives_class0 / (true_positives_class0 + false_negatives_class0), 3)
        class0_f1_score = np.around(2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0), 3)

        print('precision class1: ', precision_class1)
        print('recall class1: ', recall_class1)
        print('f1 score class1: ', class1_f1_score)
        print('precision class0: ', precision_class0)
        print('recall class0: ', recall_class0)
        print('f1 score class0: ', class0_f1_score)

        PlotService.plot_line(
            x=range(1, len(self.logistic_regression_learner.cost_history) + 1),
            y=self.logistic_regression_learner.cost_history,
            x_label="Iteration",
            y_label="Training Cost",
            title="Training Learning Curve")

        cost = self.logistic_regression_learner.calculate_cost(predictions, normalized_labels)
        print("Final Normalized Cost: ", cost)


if __name__ == "__main__":

    Runner(normalization_method='min-max').run()