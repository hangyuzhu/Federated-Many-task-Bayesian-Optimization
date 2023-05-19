from pyDOE import lhs
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score


def calibrate_x(x_r, x_lb, x_up):
    return np.clip(x_r, x_lb, x_up)


def _checker(method):
    def inner(ref, x, *args, **kwargs):
        if not type(x).__module__ == np.__name__:
            x = np.array(x)
            if len(x.shape) > 1:
                x = np.squeeze(x)
        # dimension check
        assert len(x) == ref.d
        try:
            if ref.normalized:
                # retrieve to the original value
                x = x * (ref.x_ub - ref.x_lb) + ref.x_lb
            if ref.transform is not None:
                # for multi-task problems
                x = ref.transform(x)
            # ensure all the decision variables are located within the predefined boundary
            x = calibrate_x(x, x_lb=ref.x_lb, x_up=ref.x_ub)
            return method(ref, x, *args, **kwargs)
        except Exception as inst:
            if inst.__str__() == "Error":
                print("input out of domain: " + str((ref.x_lb, ref.x_ub)))
                print("input must be vector: [1,2,...]")
            else:
                print(inst)
    return inner


class Transform:
    pass


class MultiTaskTransform(Transform):

    def __init__(self, rot_mat=1, shift_mat=0):
        self.rot_mat = rot_mat
        self.shift_mat = shift_mat

    def __call__(self, x: np.ndarray):
        assert len(x.shape) == 1
        return self._rotate(self._shift(x))

    def _shift(self, x):
        return x - self.shift_mat

    def _rotate(self, x):
        return np.dot(self.rot_mat, x)


class ObjectiveFunction:

    def __init__(self, x_lb, x_ub, d, normalized, transform):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.d = d
        try:
            self.b_dim = len(self.x_lb)
        except:
            self.b_dim = 1
        if self.b_dim > 1:
            assert self.b_dim == len(self.x_ub) == self.d
        self.normalized = normalized
        self.transform = transform

    @property
    def x_bound(self):
        if self.normalized:
            x_bound = [0, 1]
        else:
            if self.b_dim == 1:
                x_bound = [self.x_lb, self.x_ub]
            else:
                x_bound = np.hstack((np.expand_dims(self.x_lb, 1), np.expand_dims(self.x_ub, 1)))
        return x_bound

    def __repr__(self):
        return type(self).__name__

    def init_x_with_bound(self, size, iid, pb=None):
        """
        Initialize random decision variables within the (partitioned) boundary
        :param size: number of init samples
        :param iid: if using iid split method
        :param pb: partition boundary
        :return: initialized decision variables with shape size x d
        """
        if iid:
            # LHS sampling
            result = lhs(self.d, samples=size)
        else:
            result = np.random.uniform(pb[0], pb[1], size=(size, self.d))
        if self.normalized:
            return result
        else:
            # element wise multiplication
            return result * (self.x_ub - self.x_lb) + self.x_lb

    def random_uniform_x(self, size, iid, pb=None):
        if iid:
            out = np.random.uniform(low=0, high=1, size=(size, self.d))
        else:
            out = np.random.uniform(pb[0], pb[1], size=(size, self.d))
        if not self.normalized:
            out = out * (self.x_ub - self.x_lb) + self.x_lb
        return out

    @_checker
    def __call__(self, x, *args, **kwargs):
        raise NotImplementedError


class Griewank(ObjectiveFunction):

    def __init__(self, x_lb=-600, x_ub=600, d=10, normalized=False, transform=None):
        super(Griewank, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        F1 = 0
        F2 = 1
        d_x = len(x)
        for i in range(0, d_x):
            # z = x[i] - griewank[i]
            z = x[i]
            F1 = F1 + (z ** 2 / 4000)
            F2 = F2 * (np.cos(z / np.sqrt(i + 1)))
        return F1 - F2 + 1


class Rastrigin(ObjectiveFunction):

    def __init__(self, x_lb=-5, x_ub=5, d=10, normalized=False, transform=None):
        super(Rastrigin, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        F = 0
        d_x = len(x)
        for i in range(0, d_x):
            z = x[i]
            F = F + (z ** 2 - 10 * np.cos(2 * np.pi * z) + 10)
        return F


class Ackley(ObjectiveFunction):

    def __init__(self, x_lb=-32.768, x_ub=32.768, d=10, normalized=False, transform=None):
        super(Ackley, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        sum1 = 0
        sum2 = 0
        d_x = len(x)
        # M =1 no rotated
        M = 1
        for i in range(0, d_x):
            # z = M * (x[i] - ackley[i])
            z = x[i]
            sum1 = sum1 + z ** 2
            sum2 = sum2 + np.cos(2 * np.pi * z)
        out = -20 * np.exp(-0.2 * np.sqrt(sum1 / d_x)) - np.exp(sum2 / d_x) + 20 + np.e
        # F += f_bias[5]
        return out


class Schwefel(ObjectiveFunction):

    def __init__(self, x_lb=-500, x_ub=500, d=10, normalized=False, transform=None):
        super(Schwefel, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        out = 0
        d_x = len(x)
        for i in range(d_x):
            out += x[i] * np.sin(np.sqrt(abs(x[i])))
        return 418.9829 * d_x - out


class Sphere(ObjectiveFunction):

    def __init__(self, x_lb=-100, x_ub=100, d=10, normalized=False, transform=None):
        super(Sphere, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        return x.dot(x)


class Rosenbrock(ObjectiveFunction):
    def __init__(self, x_lb=-2.048, x_ub=2.048, d=10, normalized=False, transform=None):
        super(Rosenbrock, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        out = 0
        d_x = len(x)
        for i in range(0, d_x - 1):
            tmp = 100 * np.power(x[i + 1] ** 2 - x[i], 2) + np.power(x[i] - 1, 2)
            out += tmp
        # F += f_bias[2]
        return out


class Weierstrass(ObjectiveFunction):

    def __init__(self, x_lb=-0.5, x_ub=0.5, d=10, normalized=False, transform=None):
        super(Weierstrass, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        D = len(x)
        a = 0.5
        b = 3
        kmax = 21
        obj = 0
        for i in range(D):
            for k in range(kmax):
                obj += a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5))
        for k in range(kmax):
            obj -= D * a**k * np.cos(2 * np.pi * b**k * 0.5)
        return obj


class Ellipsoid(ObjectiveFunction):

    def __init__(self, x_lb=-5.12, x_ub=5.12, d=10, normalized=False, transform=None):
        super(Ellipsoid, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, *args, **kwargs):
        out = 0.0
        d = len(x)
        d_L = [i for i in range(1, d + 1)]
        for i_ in d_L:
            out += i_ * x[i_ - 1] ** 2
        return out


class SvmLandmine(ObjectiveFunction):

    def __init__(self, x_lb=np.array([1e-4, 1e-2]), x_ub=np.array([10.0, 10.0]), d=None, normalized=True, transform=None):
        assert len(x_lb.shape) == len(x_ub.shape) == 1
        super(SvmLandmine, self).__init__(x_lb, x_ub, d, normalized, transform)

    @_checker
    def __call__(self, x, x_train, y_train, x_test, y_test):
        # training svm and get the prediction with real datasets
        clf = svm.SVC(kernel="rbf", C=x[0], gamma=x[1], probability=True)
        clf.fit(x_train, y_train)
        pred = clf.predict_proba(x_test)
        score = roc_auc_score(y_test, pred[:, 1])
        return score
