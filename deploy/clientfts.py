from deploy.clientbase import Client
import numpy as np
import GPy
from scipy.optimize import minimize


class ClientFTS(Client):

    def __init__(self, client_id, group_id, params, obj_func, partitions, partition_assignment):
        super(ClientFTS, self).__init__(client_id=client_id, group_id=group_id)

        self.num_init_points = params['num_init_points']
        self.iid = params['iid']
        self.x_s = None
        self.y_s = None
        self.cur_x_best = None
        self.cur_y_best = None
        self.obj_func = obj_func
        self.partitions = partitions
        self.partition_assignment = partition_assignment

        self.random_features = params['random_features']
        self.M_target = params['M_target']
        self.M = params['M']
        self.pt = params['pt']
        self.ARD = params['ARD']

        self.gp_mcmc = params['gp_mcmc']
        self.gp = None
        self.gp_params = None
        self.gp_opt_schedule = params['gp_opt_schedule']
        self._init_gp_data()

    @property
    def dim(self):
        return self.obj_func.d

    def _init_gp_data(self):
        l = [np.random.uniform(x[0], x[1], size=self.num_init_points)
             for x in self.partitions[self.partition_assignment]]
        x_init = np.array(list(map(list, zip(*l))))
        y_init = []
        for x_new in x_init:
            y_init.append(self.real_obj_eval(x_new=x_new))
        self.x_s = x_init
        assert self.x_s.shape[1] == self.dim
        self.y_s = np.array(y_init)
        self.cur_x_best = self.x_s[self.y_s.argmin()]
        self.cur_y_best = np.min(self.y_s)
        print('---- Client {} randomly samples {} initial gp datasets ----'
              .format(self.client_id, self.num_init_points))

    def _init_random_features(self):
        self.gp = GPy.models.GPRegression(self.x_s, self.y_s.reshape(-1, 1),
                                          GPy.kern.RBF(input_dim=self.dim, lengthscale=1.0, variance=0.1,
                                                       ARD=self.ARD))
        self.gp["Gaussian_noise.variance"][0] = 1e-6
        w_return = self.sample_w()
        return w_return

    def real_obj_eval(self, x_new):
        # only support one datasets evaluation
        assert len(x_new.shape) == 1
        obj_value = self.obj_func(x=x_new)
        return obj_value

    def sample_w(self):
        M = self.M

        s = self.random_features["s"]
        b = self.random_features["b"]
        v_kernel = self.random_features["v_kernel"]
        obs_noise = self.random_features["obs_noise"]

        Phi = np.zeros((self.x_s.shape[0], M))
        for i, x in enumerate(self.x_s):
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(v_kernel) * features

            Phi[i, :] = features

        Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.y_s.reshape(-1, 1))

        try:
            w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
        except np.linalg.LinAlgError:
            w_sample = np.random.rand(1, self.M) - 0.5

        return w_sample

    def acq_min(self, w_sample, random_features, no_trials=1000, acq_type='dp_fts_de'):
        # normalized
        x_tries = np.random.uniform(0, 1, size=(no_trials, self.dim))
        ys = []
        for x in x_tries:
            ys.append(self.acq_func(x.reshape(1, -1), w_sample, random_features, acq_type))
        ys = np.array(ys)

        x_min = x_tries[ys.argmin()]
        min_acq = ys.min()

        x_seeds = np.random.uniform(0, 1, size=(20, self.dim))
        for x_try in x_seeds:
            res = minimize(lambda x: self.acq_func(x.reshape(1, -1), w_sample, random_features, acq_type),
                           x_try.reshape(1, -1),
                           bounds=[[0, 1] for _ in range(len(x_try))],
                           method="L-BFGS-B")

            if min_acq is None or -res.fun <= min_acq:
                x_min = res.x
                min_acq = res.fun
        return x_min

    def acq_func(self, x, w_sample, random_features, acq_type):
        if acq_type == 'ts':
            return self._ts(x, random_features, w_sample)
        elif acq_type == 'dp_fts_de':
            return self._dp_fts_de(x, random_features, w_sample)
        else:
            raise TypeError("{} is not the expected acquisition function".format(acq_type))

    def _ts(self, x, random_features, w_sample):
        s = random_features["s"]
        b = random_features["b"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features  # v_kernel is set to be 1 here in the synthetic experiments
        f_value = np.squeeze(np.dot(w_sample, features))
        return f_value

    def _dp_fts_de(self, x, random_features, w_sample):
        x = np.clip(x, 0, 1)
        # below finds the index of the sub-region x belongs to
        partitions_np = np.array(self.partitions)
        N_partitions = len(self.partitions)

        tmp = np.tile(x, N_partitions).reshape(N_partitions, self.dim, 1)
        tmp_2 = np.concatenate((tmp, tmp), axis=2)

        tmp_3 = tmp_2 - partitions_np
        flag_left = tmp_3[:, :, 0]
        flag_right = tmp_3[:, :, 1]
        flag_left_count = np.nonzero(flag_left >= 0)[0]
        flag_right_count = np.nonzero(flag_right <= 0)[0]

        (indices_left, counts_left) = np.unique(flag_left_count, return_counts=True)
        indices_left_correct = indices_left[counts_left == self.dim]
        (indices_right, counts_right) = np.unique(flag_right_count, return_counts=True)
        indices_right_correct = indices_right[counts_right == self.dim]

        part_ind = np.intersect1d(indices_left_correct, indices_right_correct)[0]

        w_sample_p = w_sample[part_ind]

        s = random_features["s"]
        b = random_features["b"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features
        f_value = np.squeeze(np.dot(w_sample_p, features))
        return f_value

    def train_surrogate(self, *args, **kwargs):
        pass

    def train(self, cur_round, all_w_t=None):
        self.round = cur_round

        if cur_round == 0:
            return self._init_random_features()

        if len(self.x_s) >= self.gp_opt_schedule and len(self.x_s) % self.gp_opt_schedule == 0:
            if self.gp_mcmc:
                self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
                self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
                self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
                print("[Running MCMC for GP hyper-parameters]")
                hmc = GPy.inference.mcmc.HMC(self.gp, stepsize=5e-2)
                gp_samples = hmc.sample(num_samples=500)[-300:]  # Burnin

                gp_samples_mean = np.mean(gp_samples, axis=0)
                print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                self.gp.kern.variance.fix(gp_samples_mean[0])
                self.gp.kern.lengthscale.fix(gp_samples_mean[1])
                self.gp.likelihood.variance.fix(gp_samples_mean[2])

                self.gp_params = self.gp.parameters
            else:
                self.gp.optimize_restarts(num_restarts=10, robust=True, messages=False)
                self.gp_params = self.gp.parameters
                # print("---Optimized hyper: ", self.gp)

        if self.pt is not None:
            print("[pt: {0}]".format(self.pt[len(self.x_s) - self.num_init_points]))

        if np.random.random() < self.pt[len(self.x_s) - self.num_init_points]:
            M_target = self.M_target

            ls_target = self.gp["rbf.lengthscale"][0]
            v_kernel = self.gp["rbf.variance"][0]
            obs_noise = self.gp["Gaussian_noise.variance"][0]
            obs_noise = np.max([1e-5, obs_noise])

            try:
                s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target ** 2) * np.identity(self.dim),
                                                  M_target)
            except np.linalg.LinAlgError:
                s = np.random.rand(M_target, self.dim) - 0.5

            b = np.random.uniform(0, 2 * np.pi, M_target)
            random_features_target = {"M": M_target, "length_scale": ls_target, "s": s, "b": b, "obs_noise": obs_noise,
                                      "v_kernel": v_kernel}

            Phi = np.zeros((self.x_s.shape[0], M_target))
            for i, x in enumerate(self.x_s):
                x = np.squeeze(x).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                features = features / np.sqrt(np.inner(features, features))
                features = np.sqrt(v_kernel) * features

                Phi[i, :] = features

            Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
            Sigma_t_inv = np.linalg.inv(Sigma_t)
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.y_s.reshape(-1, 1))

            try:
                w_sample_1 = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
            except np.linalg.LinAlgError:
                w_sample_1 = np.random.rand(1, self.M) - 0.5
            x_next_best = self.acq_min(w_sample=w_sample_1, random_features=random_features_target, acq_type='ts')
            #
            # x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target,
            #                 w_sample=w_sample_1, bounds=self.bounds, partitions=None)
        else:
            w_samples = all_w_t
            x_next_best = self.acq_min(w_sample=w_samples, random_features=self.random_features, acq_type='dp_fts_de')
            # x_max = acq_max(ac=self.util_dp_fts_de.utility, M=self.M, random_features=self.random_features,
            #                 w_sample=w_samples, bounds=self.bounds, partitions=self.partitions)

        # check if x is a repeated query
        if not self.x_s.shape[0] == 0:
            if np.any(np.all(self.x_s - x_next_best == 0, axis=1)):
                x_next_best = np.random.uniform(0, 1, size=self.dim)

        y_next = self.real_obj_eval(x_new=x_next_best)
        print("Client {}, Round {}, {} --> y_next: {}, y_best: {}"
              .format(self.client_id, self.round + 1, self.obj_func, y_next, self.cur_y_best))

        if y_next < self.cur_y_best:
            self.cur_y_best = y_next
            self.cur_x_best = x_next_best
            print('################# Find better solution ! #################')

        # update surrogate training datasets
        self.x_s = np.vstack((self.x_s, x_next_best.reshape(1, -1)))
        self.y_s = np.append(self.y_s, y_next)

        self.gp.set_XY(X=self.x_s, Y=self.y_s.reshape(-1, 1))

        w_return = self.sample_w()
        return w_return

    @property
    def fitness_values(self):
        return self.y_s


class _ClientFTS(Client):

    def __init__(self, client_id, group_id, params, obj_func, partitions, partition_assignment, min_prob=False,
                 x_train=None, y_train=None, x_test=None, y_test=None):
        super(_ClientFTS, self).__init__(client_id=client_id, group_id=group_id)

        self.min_prob = min_prob

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.num_init_points = params['num_init_points']
        self.iid = params['iid']
        self.x_s = None
        self.y_s = None
        self.cur_x_best = None
        self.cur_y_best = None
        self.obj_func = obj_func
        self.partitions = partitions
        self.partition_assignment = partition_assignment

        self.random_features = params['random_features']
        self.M_target = params['M_target']
        self.M = params['M']
        self.pt = params['pt']
        self.ARD = params['ARD']

        self.gp_mcmc = params['gp_mcmc']
        self.gp = None
        self.gp_params = None
        self.gp_opt_schedule = params['gp_opt_schedule']
        self._init_gp_data()

    @property
    def dim(self):
        return self.obj_func.d

    @property
    def x_bound(self):
        # normalized or unnormalized bound
        return self.obj_func.x_bound

    def _init_gp_data(self):
        l = [np.random.uniform(x[0], x[1], size=self.num_init_points)
             for x in self.partitions[self.partition_assignment]]
        x_init = np.array(list(map(list, zip(*l))))
        y_init = []
        for x_new in x_init:
            # only valid for maximization
            y_init.append(self.real_obj_eval(x_new=x_new))
        self.x_s = x_init
        assert self.x_s.shape[1] == self.dim
        self.y_s = np.array(y_init)
        self.cur_x_best = self.x_s[self.y_s.argmax()]
        self.cur_y_best = np.max(self.y_s)
        print('---- Client {} randomly samples {} initial gp datasets ----'
              .format(self.client_id, self.num_init_points))

    def _init_random_features(self):
        self.gp = GPy.models.GPRegression(self.x_s, self.y_s.reshape(-1, 1),
                                          GPy.kern.RBF(input_dim=self.dim, lengthscale=1.0, variance=0.1,
                                                       ARD=self.ARD))
        self.gp["Gaussian_noise.variance"][0] = 1e-6
        w_return = self.sample_w()
        return w_return

    def real_obj_eval(self, x_new):
        # only support one datasets evaluation
        assert len(x_new.shape) == 1
        obj_value = self.obj_func(x=x_new,
                                  x_train=self.x_train, y_train=self.y_train,
                                  x_test=self.x_test, y_test=self.y_test)
        if self.min_prob:
            return -obj_value
        else:
            return obj_value

    def sample_w(self):
        M = self.M

        s = self.random_features["s"]
        b = self.random_features["b"]
        v_kernel = self.random_features["v_kernel"]
        obs_noise = self.random_features["obs_noise"]

        Phi = np.zeros((self.x_s.shape[0], M))
        for i, x in enumerate(self.x_s):
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(v_kernel) * features

            Phi[i, :] = features

        Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.y_s.reshape(-1, 1))

        try:
            w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
        except np.linalg.LinAlgError:
            w_sample = np.random.rand(1, self.M) - 0.5

        return w_sample

    def acq_max(self, w_sample, random_features, no_trials=1000, acq_type='dp_fts_de'):
        x_tries = np.random.uniform(self.x_bound[0], self.x_bound[1], size=(no_trials, self.dim))
        ys = []
        for x in x_tries:
            ys.append(self.acq_func(x.reshape(1, -1), w_sample, random_features, acq_type))
        ys = np.array(ys)

        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()

        x_seeds = np.random.uniform(0, 1, size=(20, self.dim))
        for x_try in x_seeds:
            res = minimize(lambda x: -self.acq_func(x.reshape(1, -1), w_sample, random_features, acq_type),
                           x_try.reshape(1, -1),
                           bounds=[[0, 1] for _ in range(len(x_try))],
                           method="L-BFGS-B")

            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun
        return x_max

    def acq_func(self, x, w_sample, random_features, acq_type):
        if acq_type == 'ts':
            return self._ts(x, random_features, w_sample)
        elif acq_type == 'dp_fts_de':
            return self._dp_fts_de(x, random_features, w_sample)
        else:
            raise TypeError("{} is not the expected acquisition function".format(acq_type))

    def _ts(self, x, random_features, w_sample):
        s = random_features["s"]
        b = random_features["b"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features  # v_kernel is set to be 1 here in the synthetic experiments
        f_value = np.squeeze(np.dot(w_sample, features))
        return f_value

    def _dp_fts_de(self, x, random_features, w_sample):
        x = np.clip(x, 0, 1)
        # below finds the index of the sub-region x belongs to
        partitions_np = np.array(self.partitions)
        N_partitions = len(self.partitions)

        tmp = np.tile(x, N_partitions).reshape(N_partitions, self.dim, 1)
        tmp_2 = np.concatenate((tmp, tmp), axis=2)

        tmp_3 = tmp_2 - partitions_np
        flag_left = tmp_3[:, :, 0]
        flag_right = tmp_3[:, :, 1]
        flag_left_count = np.nonzero(flag_left >= 0)[0]
        flag_right_count = np.nonzero(flag_right <= 0)[0]

        (indices_left, counts_left) = np.unique(flag_left_count, return_counts=True)
        indices_left_correct = indices_left[counts_left == self.dim]
        (indices_right, counts_right) = np.unique(flag_right_count, return_counts=True)
        indices_right_correct = indices_right[counts_right == self.dim]

        part_ind = np.intersect1d(indices_left_correct, indices_right_correct)[0]

        w_sample_p = w_sample[part_ind]

        s = random_features["s"]
        b = random_features["b"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features
        f_value = np.squeeze(np.dot(w_sample_p, features))
        return f_value

    def train_surrogate(self, *args, **kwargs):
        pass

    def train(self, cur_round, all_w_t=None):
        self.round = cur_round

        if cur_round == 0:
            return self._init_random_features()

        if len(self.x_s) >= self.gp_opt_schedule and len(self.x_s) % self.gp_opt_schedule == 0:
            if self.gp_mcmc:
                self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
                self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
                self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1., 10.))
                print("[Running MCMC for GP hyper-parameters]")
                hmc = GPy.inference.mcmc.HMC(self.gp, stepsize=5e-2)
                gp_samples = hmc.sample(num_samples=500)[-300:]  # Burnin

                gp_samples_mean = np.mean(gp_samples, axis=0)
                print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                self.gp.kern.variance.fix(gp_samples_mean[0])
                self.gp.kern.lengthscale.fix(gp_samples_mean[1])
                self.gp.likelihood.variance.fix(gp_samples_mean[2])

                self.gp_params = self.gp.parameters
            else:
                self.gp.optimize_restarts(num_restarts=10, robust=True, messages=False)
                self.gp_params = self.gp.parameters
                # print("---Optimized hyper: ", self.gp)

        if self.pt is not None:
            print("[pt: {0}]".format(self.pt[len(self.x_s) - self.num_init_points]))

        if np.random.random() < self.pt[len(self.x_s) - self.num_init_points]:
            M_target = self.M_target

            ls_target = self.gp["rbf.lengthscale"][0]
            v_kernel = self.gp["rbf.variance"][0]
            obs_noise = self.gp["Gaussian_noise.variance"][0]
            obs_noise = np.max([1e-5, obs_noise])

            try:
                s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target ** 2) * np.identity(self.dim),
                                                  M_target)
            except np.linalg.LinAlgError:
                s = np.random.rand(M_target, self.dim) - 0.5

            b = np.random.uniform(0, 2 * np.pi, M_target)
            random_features_target = {"M": M_target, "length_scale": ls_target, "s": s, "b": b, "obs_noise": obs_noise,
                                      "v_kernel": v_kernel}

            Phi = np.zeros((self.x_s.shape[0], M_target))
            for i, x in enumerate(self.x_s):
                x = np.squeeze(x).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                features = features / np.sqrt(np.inner(features, features))
                features = np.sqrt(v_kernel) * features

                Phi[i, :] = features

            Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
            Sigma_t_inv = np.linalg.inv(Sigma_t)
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.y_s.reshape(-1, 1))

            try:
                w_sample_1 = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
            except np.linalg.LinAlgError:
                w_sample_1 = np.random.rand(1, self.M) - 0.5
            x_next_best = self.acq_max(w_sample=w_sample_1, random_features=random_features_target, acq_type='ts')
            #
            # x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target,
            #                 w_sample=w_sample_1, bounds=self.bounds, partitions=None)
        else:
            w_samples = all_w_t
            x_next_best = self.acq_max(w_sample=w_samples, random_features=self.random_features, acq_type='dp_fts_de')
            # x_max = acq_max(ac=self.util_dp_fts_de.utility, M=self.M, random_features=self.random_features,
            #                 w_sample=w_samples, bounds=self.bounds, partitions=self.partitions)

        # check if x is a repeated query
        if not self.x_s.shape[0] == 0:
            if np.any(np.all(self.x_s - x_next_best == 0, axis=1)):
                x_next_best = np.random.uniform(0, 1, size=self.dim)

        # only valid for maximization
        y_next = self.real_obj_eval(x_new=x_next_best)
        print("Client {}, Round {}, {} --> y_next: {}, y_best: {}"
              .format(self.client_id, self.round + 1, self.obj_func, y_next, self.cur_y_best))

        if y_next > self.cur_y_best:
            self.cur_y_best = y_next
            self.cur_x_best = x_next_best
            print('################# Find better solution ! #################')

        # update surrogate training datasets
        self.x_s = np.vstack((self.x_s, x_next_best.reshape(1, -1)))
        self.y_s = np.append(self.y_s, y_next)

        self.gp.set_XY(X=self.x_s, Y=self.y_s.reshape(-1, 1))

        w_return = self.sample_w()
        return w_return

    @property
    def fitness_values(self):
        return self.y_s
