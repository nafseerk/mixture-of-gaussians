from data_loader import DataLoader
import numpy as np
import pandas as pd
import math


class MOG:

    def __init__(self, M):
        self.pi1 = None
        self.pi2 = None
        self.mu1 = None
        self.mu2 = None
        self.sigma = None
        self.y_map = {5.0: 1, 6.0: 0}
        self.M = M
        self.N1 = None
        self.N2 = None
        self.S1 = None
        self.S2 = None
        self.w = None
        self.w0 = None
        self.train_accuracy = None

    def learn(self, full_dataset, report_acc=False):

        self.N1 = 0
        self.N2 = 0
        self.pi1 = 0.0
        self.mu1 = np.zeros((self.M, 1), dtype=float)
        self.mu2 = np.zeros((self.M, 1), dtype=float)

        for train_set_attrs, train_set_labels in full_dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            for i, row in train_set_attrs.iterrows():

                xi = row.values.reshape((self.M, 1))
                yi = self.y_map[train_set_labels.iat[i, 0]]

                if yi == 1:
                    self.mu1 = np.add(self.mu1, yi * xi)
                    self.N1 += 1
                else:
                    self.mu2 = np.add(self.mu2, (1 - yi) * xi)
                    self.N2 += 1
                self.pi1 += yi

        N = self.N1 + self.N2
        self.pi1 = self.pi1 / N
        self.pi2 = 1 - self.pi1
        self.mu1 = self.mu1 / self.N1
        self.mu2 = self.mu2 / self.N2

        self.S1 = np.zeros((self.M, self.M), dtype=float)
        self.S2 = np.zeros((self.M, self.M), dtype=float)

        for train_set_attrs, train_set_labels in full_dataset:
            for i, row in train_set_attrs.iterrows():

                xi = row.values.reshape((self.M, 1))
                yi = self.y_map[train_set_labels.iat[i, 0]]
                if yi == 1:
                    self.S1 = np.add(self.S1, np.matmul(xi - self.mu1, np.transpose(xi - self.mu1)))
                else:
                    self.S2 = np.add(self.S2, np.matmul(xi - self.mu2, np.transpose(xi - self.mu2)))

        self.S1 = self.S1 / self.N1
        self.S2 = self.S2 / self.N2
        self.sigma = (self.N1 / N) * self.S1 + (self.N2 / N) * self.S2

        sigma_inv = np.linalg.inv(self.sigma)
        self.w = np.matmul(sigma_inv, self.mu1 - self.mu2)
        self.w0 = -0.5 * np.matmul(np.matmul(np.transpose(self.mu1), sigma_inv), self.mu1) + \
            0.5 * np.matmul(np.matmul(np.transpose(self.mu2), sigma_inv), self.mu2) + \
            math.log(self.pi1 / self.pi2)

        if report_acc:
            self.train_accuracy = self.k_fold_cross_validation(full_dataset)
            print('Training Accuracy = %.2f %%' % self.train_accuracy)

    def sigmoid(self, x):
        power = -1 * np.add(np.matmul(np.transpose(self.w), x), self.w0)
        return 1 / (1 + np.exp(power))

    def classify_point(self, x):
        prob_y_given_x = self.sigmoid(x)[0][0]
        y = 1 if prob_y_given_x >= 0.5 else 0
        for label, yi in self.y_map.items():
            if yi == y:
                return label

    def classify(self, test_attrs, true_labels=None, verbose=False):

        N = len(test_attrs)
        if not true_labels.empty:
            if len(test_attrs) != len(true_labels):
                raise ValueError('count mismatch in attributes and labels')

        correct = 0
        predicted_labels = []
        for i, row in test_attrs.iterrows():
            xi = row.values.reshape((self.M, 1))
            predicted_label = self.classify_point(xi)
            predicted_labels.append(predicted_label)
            if not true_labels.empty:
                true_label = true_labels.iat[i, 0]

                if verbose:
                    print('Predicted Label =', predicted_label, 'True Label =', true_label)

                if predicted_label == true_label:
                    correct += 1

        accuracy = None
        if true_labels is not None:
            accuracy = correct / N * 100

        predicted_labels = pd.DataFrame(np.array(predicted_labels))
        return predicted_labels, accuracy

    def k_fold_cross_validation(self, full_dataset, k=10):
        avg_accuracy = 0.0
        for i in range(k):
            test_attrs, test_labels = full_dataset.pop(0)
            accuracy = self.classify(test_attrs, true_labels=test_labels)[1]
            full_dataset.append((test_attrs, test_labels))
            avg_accuracy += accuracy
        return avg_accuracy / k


if __name__ == '__main__':
    full_dataset = DataLoader.load_full_dataset('./dataset')
    model = MOG(M=64)
    model.learn(full_dataset, report_acc=True)

    # Test the model with test data taken from training data
    train_dataset, test_attrs, test_labels = DataLoader.load_with_test_data(
        './dataset',
        split_ratio=0.1)
    model = MOG(M=64)
    model.learn(train_dataset, report_acc=True)
    predictions, acc = model.classify(test_attrs, true_labels=test_labels)

    print('Test Accuracy = %.2f %%' % acc)
    print('=====Predictions=====')
    print(predictions)


