from deploy.clientbase import Client
import numpy as np
from pydacefit.regr import regr_constant, regr_linear
from pydacefit.dace import DACE
from pydacefit.corr import corr_gauss
from scipy.optimize import minimize
from scipy.stats import norm


class ClientGP(Client):

    def __init__(self, client_id, group_id, params, obj_func, pb=None,
                 x_train=None, y_train=None, x_test=None, y_test=None):
        super(ClientGP, self).__init__(client_id=client_id, group_id=group_id)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_init_points = params['num_init_points']
        self.iid = params['iid']
        self.next_method = params['next_method']
        assert self.next_method in ['sampling', 'GA']
        self.x_s = None
        self.y_s = None
        self.x_s_share = None
        self.opt_obj = params['opt_obj']
        # partitioned bound
        self.pb = pb

        self.cur_x_best = None
        self.cur_y_best = None
        self.obj_func = obj_func

        if params['regr'] == 'regr_constant':
            self.regr = regr_constant
        elif params['regr'] == 'regr_linear':
            self.regr = regr_linear
        else:
            raise TypeError("{} is not expected !".format(params['regr']))
        self.global_gp_model = None
        self.local_gp_model = None
        self._init_s_data()
        self.acq_type = params['acq_type']
        self.gamma = params['gamma']

    @property
    def x_bound(self):
        # normalized or unnormalized bound
        return self.obj_func.x_bound

    @property
    def x_bound_lb(self):
        # normalized or unnormalized lower bound
        if isinstance(self.x_bound, np.ndarray):
            assert len(self.x_bound.shape) == 2
            return self.x_bound[:, 0]
        elif isinstance(self.x_bound, list):
            return self.x_bound[0]
        else:
            raise TypeError('{} is not the desired type'.format(type(self.x_bound)))

    @property
    def x_bound_ub(self):
        # normalized or unnormalized upper bound
        if isinstance(self.x_bound, np.ndarray):
            assert len(self.x_bound.shape) == 2
            return self.x_bound[:, -1]
        elif isinstance(self.x_bound, list):
            return self.x_bound[-1]
        else:
            raise TypeError('{} is not the desired type'.format(type(self.x_bound)))

    @property
    def x_lb(self):
        # unnormalized lower bound
        return self.obj_func.x_lb

    @property
    def x_ub(self):
        # unormalized upper bound
        return self.obj_func.x_ub

    @property
    def dim(self):
        return self.obj_func.d

    @property
    def global_model_params(self):
        if self.global_gp_model is not None:
            return self.global_gp_model.model['theta']

    @property
    def local_model_params(self):
        if self.local_gp_model is not None:
            return self.local_gp_model.model['theta']

    def synchronize_gp_params(self, global_params: dict):
        theta = global_params['theta']
        self.global_gp_model = DACE(regr=self.regr,
                                    corr=corr_gauss,
                                    theta=theta,
                                    thetaL=1e-5 * np.ones(self.dim),
                                    thetaU=100 * np.ones(self.dim))

    def _init_global_gp_model(self, theta=None):
        if theta is None:
            theta = 5 * np.ones(self.dim)
        self.global_gp_model = DACE(regr=self.regr,
                                    corr=corr_gauss,
                                    theta=theta,
                                    thetaL=1e-5 * np.ones(self.dim),
                                    thetaU=100 * np.ones(self.dim))

    def _init_local_gp_model(self, theta=None):
        if theta is None:
            theta = 5 * np.ones(self.dim)
        # the same as the initialized params on the server
        self.local_gp_model = DACE(regr=self.regr,
                                   corr=corr_gauss,
                                   theta=theta,
                                   thetaL=1e-5 * np.ones(self.dim),
                                   thetaU=100 * np.ones(self.dim))

    def receive_x_s(self, x_s):
        self.x_s_share = x_s

    def _init_s_data(self):
        x_init = self.obj_func.init_x_with_bound(size=self.num_init_points,
                                                 iid=self.iid,
                                                 pb=self.pb)
        y_init = []
        for x_new in x_init:
            y_init.append(self.real_obj_eval(x_new=x_new))
        self.x_s = x_init
        self.y_s = np.array(y_init)

        if self.opt_obj == 'maximize':
            self.cur_x_best = self.x_s[self.y_s.argmax()]
            self.cur_y_best = np.max(self.y_s)
        elif self.opt_obj == 'minimize':
            self.cur_x_best = self.x_s[self.y_s.argmin()]
            self.cur_y_best = np.min(self.y_s)
        else:
            raise ValueError('{} is not included in the optimization objective'.format(self.opt_obj))
        print('---- Client {} randomly samples {} initial gp datasets ----'
              .format(self.client_id, self.num_init_points))

    def real_obj_eval(self, x_new):
        # only support one datasets evaluation
        assert len(x_new.shape) == 1
        obj_value = self.obj_func(x=x_new,
                                  x_train=self.x_train, y_train=self.y_train,
                                  x_test=self.x_test, y_test=self.y_test)
        return obj_value

    def sampling_method(self, no_trials=1000):
        # find the best x to maximize the acquisition function upon the entire bound
        # TODO mimic the original implementation in DP-FTS-DE with full bound search
        x_tries = self.obj_func.random_uniform_x(size=no_trials, iid=True)
        ac_values = self.acq_eval(x_new=x_tries)
        ac_idx = ac_values.argsort()[::-1]
        ac_best = ac_values[ac_idx[0]]
        x_best = x_tries[ac_idx[0]]

        # L-BFGS-B optimization to find the best x
        x_seeds = self.obj_func.random_uniform_x(size=20, iid=True)
        for x_try in x_seeds:
            res = minimize(lambda x: -self.acq_eval(x_new=x),
                           x_try,
                           bounds=[self.x_bound for _ in range(len(x_try))],
                           method="L-BFGS-B")
            if ac_best is None or res.fun >= ac_best:
                x_best = res.x
                ac_best = res.fun
        return x_best, ac_best

    @staticmethod
    def _power(mat1, mat2):
        # To solve the problem: Numpy does not seem to allow fractional powers of negative numbers
        return np.sign(mat1) * np.power(np.abs(mat1), mat2)

    def generate_offspring(self, Parent):
        pop_size, d = Parent.shape

        # crossover(simulated binary crossover)
        # dic_c is the distribution index of crossover
        dis_c = 20
        mu = np.random.rand(int(pop_size / 2), d)
        idx1 = [i for i in range(0, pop_size, 2)]
        idx2 = [i + 1 for i in range(0, pop_size, 2)]
        parent1 = Parent[idx1, :]
        parent2 = Parent[idx2, :]
        element_min = np.minimum(parent1, parent2)
        element_max = np.maximum(parent1, parent2)
        tmp_min = np.minimum(element_min - self.x_bound_lb, self.x_bound_ub - element_max)
        beta = 1 + 2 * tmp_min / np.maximum(abs(parent2 - parent1), 1e-6)
        alpha = 2 - beta ** (-dis_c - 1)
        betaq = self._power(alpha * mu, 1 / (dis_c + 1)) * (mu <= 1 / alpha) + \
                self._power(1. / (2 - alpha * mu), 1 / (dis_c + 1)) * (mu > 1. / alpha)
        # the mutation is performed randomly on each variable
        betaq = betaq * self._power(-1, np.random.randint(0, 2, (int(pop_size / 2), d)))
        betaq[np.random.rand(int(pop_size / 2), d) > 0.5] = 1
        offspring1 = 0.5 * ((1 + betaq) * parent1 + (1 - betaq) * parent2)
        offspring2 = 0.5 * ((1 - betaq) * parent1 + (1 + betaq) * parent2)
        pop_crossover = np.vstack((offspring1, offspring2))

        # mutation (polynomial mutation)
        # dis_m is the distribution index of polynomial mutation
        dis_m = 20
        pro_m = 1
        rand_var = np.random.rand(pop_size, d)
        mu = np.random.rand(pop_size, d)
        deta = np.minimum(pop_crossover - self.x_bound_lb, self.x_bound_ub - pop_crossover) \
               / (self.x_bound_ub - self.x_bound_lb)
        detaq = np.zeros((pop_size, d))
        # use dot multiply to replace matrix & in matlab
        position1 = (rand_var <= pro_m) * (mu <= 0.5)
        position2 = (rand_var <= pro_m) * (mu > 0.5)
        tmp1 = 2 * mu[position1] + (1 - 2 * mu[position1]) * self._power(1 - deta[position1], (dis_m + 1))
        detaq[position1] = self._power(tmp1, 1 / (dis_m + 1)) - 1
        tmp2 = 2 * (1 - mu[position2]) + 2 * (mu[position2] - 0.5) * self._power(1 - deta[position2], (dis_m + 1))
        detaq[position2] = 1 - self._power(tmp2, 1 / (dis_m + 1))
        pop_mutation = pop_crossover + detaq * (self.x_bound_ub - self.x_bound_lb)
        return pop_mutation

    def ga_method(self, no_generations=20):
        using_archive = True
        if using_archive:
            # Parent = self.x_s[0:self.num_b4_d * self.dim]
            Parent = self.x_s[-self.num_init_points:]
        else:
            Parent = self.obj_func.init_x_with_bound(size=self.num_init_points,
                                                     iid=self.iid,
                                                     pb=self.pb)
        ac_best = -10000
        x_best = None
        for _ in range(no_generations):
            Offspring = self.generate_offspring(Parent)
            Popcom = np.vstack((Parent, Offspring))
            ac_values = self.acq_eval(x_new=Popcom)
            ac_idx = ac_values.argsort()[::-1]
            Parent = Popcom[ac_idx[:self.num_init_points]]
            if ac_best < ac_values[ac_idx[0]]:
                ac_best = ac_values[ac_idx[0]]
                x_best = Popcom[ac_idx[0]]
        return x_best, ac_best

    def acq_eval(self, x_new):
        if len(x_new.shape) == 1:
            x_new = x_new.reshape(1, -1)
        mean_y_new, var_y_new = self.global_gp_model.predict(x_new, return_mse=True)
        sigma_y_new = np.sqrt(var_y_new)
        if '_w' in self.acq_type:
            l_mean_y_new, l_var_y_new = self.local_gp_model.predict(x_new, return_mse=True)
            l_sigma_y_new = np.sqrt(l_var_y_new)
            if 'LCB' in self.acq_type:
                mean_y_new = mean_y_new * l_sigma_y_new / (sigma_y_new + l_sigma_y_new) \
                             + l_mean_y_new * sigma_y_new / (sigma_y_new + l_sigma_y_new)
        if self.acq_type == 'EI':
            return self.EI(mean_y_new=mean_y_new, sigma_y_new=sigma_y_new)
        if self.acq_type == 'EI_w':
            return self.EI_w(mean_y_new=mean_y_new, sigma_y_new=sigma_y_new,
                             l_mean_y_new=l_mean_y_new, l_sigma_y_new=l_sigma_y_new)
        elif self.acq_type == 'LCB' or self.acq_type == 'LCB_w':
            return self.LCB(mean_y_new=mean_y_new, sigma_y_new=sigma_y_new)
        elif self.acq_type == 'LCB_w1':
            return self.LCB_w1(mean_y_new=mean_y_new, l_sigma_y_new=l_sigma_y_new)
        elif self.acq_type == 'LCB_w2':
            return self.LCB_w2(mean_y_new=mean_y_new, sigma_y_new=sigma_y_new, l_sigma_y_new=l_sigma_y_new)
        else:
            raise TypeError('No this type of acquisition function.')

    def EI(self, mean_y_new, sigma_y_new):
        if self.opt_obj == 'maximize':
            max_y = np.max(self.y_s)
            z = (mean_y_new - max_y) / sigma_y_new
            exp_imp = (mean_y_new - max_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        else:
            # minimization problems
            min_y = np.min(self.y_s)
            z = (min_y - mean_y_new) / sigma_y_new
            exp_imp = (min_y - mean_y_new) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
        return np.squeeze(exp_imp)

    def EI_w(self, mean_y_new, sigma_y_new, l_mean_y_new, l_sigma_y_new):
        if self.opt_obj == 'maximize':
            max_y = np.max(self.y_s)
            z = (mean_y_new - max_y) / sigma_y_new
            exp_imp = (mean_y_new - max_y) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
            l_z = (l_mean_y_new - max_y) / l_sigma_y_new
            l_exp_imp = (l_mean_y_new - max_y) * norm.cdf(l_z) + l_sigma_y_new * norm.pdf(l_z)
        else:
            # minimization problems
            min_y = np.min(self.y_s)
            z = (min_y - mean_y_new) / sigma_y_new
            exp_imp = (min_y - mean_y_new) * norm.cdf(z) + sigma_y_new * norm.pdf(z)
            l_z = (min_y - l_mean_y_new) / l_sigma_y_new
            l_exp_imp = (min_y - l_mean_y_new) * norm.cdf(l_z) + l_sigma_y_new * norm.pdf(l_z)
        exp_imp = self.gamma * exp_imp + (1 - self.gamma) * l_exp_imp
        return np.squeeze(exp_imp)

    def LCB(self, mean_y_new, sigma_y_new):
        beta_t = np.log(2 * self.dim)
        if self.opt_obj == 'maximize':
            ucb_value = mean_y_new + beta_t * sigma_y_new
        else:
            # minimization problems
            ucb_value = mean_y_new - beta_t * sigma_y_new
        return -np.squeeze(ucb_value)

    def LCB_w1(self, mean_y_new, l_sigma_y_new):
        beta_t = np.log(2 * self.dim)
        if self.opt_obj == 'maximize':
            ucb_value = mean_y_new + beta_t * l_sigma_y_new
        else:
            ucb_value = mean_y_new - beta_t * l_sigma_y_new
        return -np.squeeze(ucb_value)

    def LCB_w2(self, mean_y_new, sigma_y_new, l_sigma_y_new):
        beta_t = np.log(2 * self.dim)
        if self.opt_obj == 'maximize':
            ucb_value = mean_y_new + beta_t * np.sqrt(2) * sigma_y_new**2 / (sigma_y_new + l_sigma_y_new)
        else:
            # minimization problems
            ucb_value = mean_y_new - beta_t * np.sqrt(2) * sigma_y_new**2 / (sigma_y_new + l_sigma_y_new)
        return -np.squeeze(ucb_value)

    def find_x_best(self):
        if self.next_method == 'sampling':
            x_best, ac_best = self.sampling_method(no_trials=1000)
        elif self.next_method == 'GA':
            x_best, ac_best = self.ga_method(no_generations=20)
        else:
            raise TypeError('The next method is excluded!')
        return x_best, ac_best

    def train(self, reinitialize=False, knowledge_transfer=False):
        self.train_surrogate(reinitialize)
        rank = None
        if knowledge_transfer:
            rank = self.cal_x_s_share_rank()

        x_next_best, _ = self.find_x_best()
        # check if x is a repeated entry or None
        if not self.x_s.shape[0] == 0:
            if x_next_best is None:
                x_next_best = np.squeeze(self.obj_func.random_uniform_x(size=1, iid=self.iid, pb=self.pb))
            elif np.any(np.all(self.x_s - x_next_best == 0, axis=1)):
                # if x is repeated entry, randomly select one solution
                x_next_best = np.squeeze(self.obj_func.random_uniform_x(size=1, iid=self.iid, pb=self.pb))
        # Real objective evaluation
        y_next = self.real_obj_eval(x_new=x_next_best)

        print("Client {}, Round {}, {} --> y_next: {}, y_best: {}"
              .format(self.client_id, self.round + 1, self.obj_func, y_next, self.cur_y_best))

        # update the best x and y
        if self.opt_obj == 'maximize':
            if y_next > self.cur_y_best:
                self.cur_y_best = y_next
                self.cur_x_best = x_next_best
                print('################# Find better solution ! #################')
        elif self.opt_obj == 'minimize':
            if y_next < self.cur_y_best:
                self.cur_y_best = y_next
                self.cur_x_best = x_next_best
                print('################# Find better solution ! #################')
        else:
            raise ValueError('{} is not included in the optimization objective'.format(self.opt_obj))

        # update surrogate training datasets
        self.x_s = np.vstack((self.x_s, x_next_best.reshape(1, -1)))
        self.y_s = np.append(self.y_s, y_next)

        return self.client_id, self.local_data_size, self.global_model_params, rank

    def train_surrogate(self, reinitialize=False):
        if self.global_gp_model is None or reinitialize:
            self._init_global_gp_model(theta=self.global_model_params)
        self.global_gp_model.fit(X=self.x_s, Y=self.y_s)
        if '_w' in self.acq_type:
            self._init_local_gp_model(theta=self.local_model_params)
            self.local_gp_model.fit(X=self.x_s, Y=self.y_s)

    def cal_x_s_share_rank(self):
        mean_ys_share = self.global_gp_model.predict(self.x_s_share, return_mse=False)
        # ascend ranking
        rank_idx = np.squeeze(mean_ys_share).argsort()
        # representing ascend rank of predictions
        return rank_idx.argsort()

    @property
    def local_data_size(self):
        if self.x_train is not None:
            return len(self.x_train)
        else:
            return 1

    @property
    def fitness_values(self):
        return self.y_s
