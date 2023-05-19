import numpy as np
import copy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from utils.data_utils import mini_batches, index_bootstrap


class RBFNetwork(object):

    def __init__(self, k_size, d_out, optim, n_epochs, lr, alpha):
        self.k_size = k_size        # kernel size
        self.d_out = d_out          # output dimension
        self.optim = optim          # optimizer
        self.n_epochs = n_epochs    # number of epochs
        self.lr = lr                # learning rate
        self.alpha = alpha

        self.centers, self.std = None, None
        self.w = np.random.randn(self.k_size, self.d_out)
        self.b = np.random.randn(1, self.d_out)

    def get_params(self):
        """
        :return: model parameters (deep copied)
        """
        tmp_c, tmp_w, tmp_b, tmp_s = copy.deepcopy(self.centers), copy.deepcopy(self.w), \
                                     copy.deepcopy(self.b), copy.deepcopy(self.std)
        return {'centers': tmp_c, 'w': tmp_w, 'b': tmp_b, 'std': tmp_s}

    def set_params(self, m_tuple: tuple):
        """
        Since model parameters are deep copied in func get_params,
        :param m_tuple: tuple of model parameters
        :return:
        """
        tmp_w, tmp_b = m_tuple[0], m_tuple[1]
        self.w = tmp_w
        self.b = tmp_b

    @staticmethod
    def sort_centers(centers):
        """
        To sort the centers according to the distance from zero vector
        Please note that this fun has not consider the direction of the centers, should be change
        :param centers:
        :return: sorted centers & index
        """
        tmp_centers = copy.deepcopy(centers)
        distance = np.sum(tmp_centers ** 2, axis=1)
        sorted_index = np.argsort(distance)
        tmp_centers = tmp_centers[sorted_index, :]
        return tmp_centers, sorted_index

    @staticmethod
    def _dist(Mat1, Mat2):
        """
        rewrite euclidean distance function in Matlab: dist
        :param Mat1: matrix 1, M x N
        :param Mat2: matrix 2, N x R
        output: Mat3. M x R
        """
        Mat2 = Mat2.T
        return cdist(Mat1, Mat2)

    def find_center(self, X):
        k_means = KMeans(n_clusters=self.k_size).fit(X)
        centers = k_means.cluster_centers_
        centers, tmp_index = self.sort_centers(centers)
        AllDistances = self._dist(centers, centers.T)
        dMax = AllDistances.max(initial=-10)

        for i in range(self.k_size):
            AllDistances[i, i] = dMax + 1
        AllDistances = np.where(AllDistances != 0, AllDistances, 0.000001)
        std = self.alpha * np.min(AllDistances, axis=0)
        return centers, std

    def fit(self, X, y):
        assert len(X.shape) == 2 and len(y.shape) == 1

        n_samples = X.shape[0]
        self.centers, self.std = self.find_center(X)
        Distance = self._dist(self.centers, X.T)
        SpreadsMat = np.tile(self.std.reshape(-1, 1), (1, n_samples))
        A = np.exp(-(Distance / SpreadsMat) ** 2).T

        if self.optim == 'sgd':
            # SGD
            # for each sample, loss = (y' - y)**2 / 2 = (wx+b - y)**2/2
            # dw = (wx+b -y)*x, db = wx+b
            for _ in range(self.n_epochs):
                for i in range(n_samples):
                    F = A[i].T.dot(self.w) + self.b
                    error = F - y[i]
                    dw = A[i].reshape(-1, 1) * error.reshape(1, self.d_out)
                    db = error
                    # update
                    self.w = self.w - self.lr * dw
                    self.b = self.b - self.lr * db

        elif self.optim == 'max-gd':
            # max error gd
            # for each batch, loss = sum(y' - y)**2 / (2m) = sum(wx+b - y)**2/(2m), m is the batch size
            # dw = sum(wx+b - y)/m, db = sum(wx+b)/m
            for _ in self.n_epochs:
                F = A.dot(self.w) + self.b
                error = F - y
                sum_error = np.sum(error ** 2, axis=1)
                max_index = np.argmax(sum_error)
                dw = A[max_index].reshape(-1, 1) * error[max_index].reshape(1, self.d_out)
                db = error[max_index].reshape(1, self.d_out)
                # update
                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db

        elif self.optim == 'm-sgd':
            # mini-batch SGD
            # for each batch, loss = sum(y' - y)**2 / (2m) = sum(wx+b - y)**2/(2m), m is the batch size
            # dw = sum(wx+b - y)/m, db = sum(wx+b)/m
            batch_size = 12
            for _ in self.n_epochs:
                for (x_batch, y_batch, A_tmp) in mini_batches(X, y, A, batch_size):
                    real_bs = x_batch.shape[0]
                    F = A_tmp.dot(self.w) + self.b
                    error = F - y_batch
                    tmp_A = A_tmp.reshape(real_bs, -1, 1)
                    tmp_error = error.reshape(real_bs, -1, self.d_out)
                    out = tmp_A * tmp_error
                    dw = np.sum(out, axis=0) / real_bs
                    db = np.sum(error, axis=0).reshape(1, self.d_out) / real_bs
                    # update
                    self.w = self.w - self.lr * dw
                    self.b = self.b - self.lr * db

        elif self.optim == '1-sgd':
            # one batch SGD
            # for each batch, loss = sum(y' - y)**2 / (2m) = sum(wx+b - y)**2/(2m), m is the batch size
            # dw = sum(wx+b - y)/m, db = sum(wx+b)/m
            wanted_size = 32
            prob = wanted_size / n_samples
            for _ in self.n_epochs:
                batch_index = index_bootstrap(n_samples, prob)
                batch_x, batch_y = X[batch_index], y[batch_index]
                batch_size = batch_x.shape[0]
                A_tmp = A[batch_index]

                F = A_tmp.dot(self.w) + self.b
                error = F - batch_y
                tmp_A = A_tmp.reshape(batch_size, -1, 1)
                tmp_error = error.reshape(batch_size, -1, self.d_out)
                out = tmp_A * tmp_error
                dw = np.sum(out, axis=0) / batch_size
                db = np.sum(error, axis=0).reshape(1, self.d_out) / batch_size
                # update
                self.w = self.w - self.lr * dw
                self.b = self.b - self.lr * db

        else:
            print('Error in loss name')

    def predict(self, X):
        assert len(X.shape) == 2
        N = X.shape[0]
        TestDistance = self._dist(self.centers, X.T)
        TestSpreadMat = np.tile(self.std.reshape(-1, 1), (1, N))
        TestHiddenOut = np.exp(-(TestDistance / TestSpreadMat) ** 2).T
        y_pred = np.dot(TestHiddenOut, self.w) + self.b
        return y_pred
