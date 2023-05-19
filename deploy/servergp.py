from deploy.serverbase import Server
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from utils.initialize import init_samples


class ServerGP(Server):

    def __init__(self, server_id: int, params, clients_per_round, enable_parallel=False):
        super(ServerGP, self).__init__(server_id)
        self.global_gp_params = {'theta': 5 * np.ones(params['dim'])}
        print('------------- Server completes global model initialization -------------')
        self.knowledge_transfer = params['knowledge_transfer']
        self.x_s = None
        if self.knowledge_transfer:
            self.kt_agg_prob = params['kt_agg_prob']
        self.enable_parallel = enable_parallel
        if self.enable_parallel:
            self.pool = ThreadPool(clients_per_round)

    def __del__(self):
        if self.enable_parallel:
            self.pool.close()

    def select_clients(self, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def create_x_s(self, clients=None):
        if clients is None:
            clients = self.selected_clients
        num_init_points = None
        dim = None
        x_bound_lb = None
        x_bound_ub = None
        for i, c in enumerate(clients):
            if i == 0:
                num_init_points = c.num_init_points
                dim = c.dim
                x_bound_lb = c.x_bound_lb
                x_bound_ub = c.x_bound_ub
            else:
                assert num_init_points == c.num_init_points
                assert dim == c.dim
                x_bound_lb, x_bound_ub = self.set_bounds(x_bound_lb, x_bound_ub, c)
        # using normalized (unnormalized) bound to avoid overflow issue
        self.x_s = init_samples(xlb=x_bound_lb,
                                xub=x_bound_ub,
                                normalized=False,
                                size=num_init_points,
                                d=dim,
                                iid=True,
                                partitions=None)

    @staticmethod
    def set_bounds(x_bound_lb, x_bound_ub, c):
        # support multi/one dimensional
        if isinstance(x_bound_lb, np.ndarray):
            assert len(x_bound_lb.shape) == 1
            # replace value for each dimension less than the lower bound
            lb_idx = np.squeeze(np.argwhere(x_bound_lb > c.x_bound_lb))
            x_bound_lb[lb_idx] = c.x_bound_lb[lb_idx]
        else:
            if x_bound_lb > c.x_bound_lb:
                x_bound_lb = c.x_bound_lb
        if isinstance(x_bound_ub, np.ndarray):
            assert len(x_bound_ub.shape) == 1
            # replace value for each dimension greater than the upper bound
            ub_idx = np.squeeze(np.argwhere(x_bound_ub < c.x_bound_ub))
            x_bound_ub[ub_idx] = c.x_bound_ub[ub_idx]
        else:
            if x_bound_ub < c.x_bound_ub:
                x_bound_ub = c.x_bound_ub
        return x_bound_lb, x_bound_ub

    def share_x_s_to(self, clients=None):
        if clients is None:
            clients = self.selected_clients
        for c in clients:
            c.receive_x_s(self.x_s)

    def train(self, clients=None):
        if clients is None:
            clients = self.selected_clients
        if self.enable_parallel:
            self.updates = self.pool.map(self.train_one_client, clients)
        else:
            for c in clients:
                self.updates.append(self.train_one_client(c))
        self.round += 1

    def train_one_client(self, c=None):
        c.round = self.round
        if len(self.global_gp_params) == 1:
            global_gp_params = self.global_gp_params
        else:
            global_gp_params = self.global_gp_params[c.client_id]
        c.synchronize_gp_params(global_params=global_gp_params)
        return c.train(knowledge_transfer=self.knowledge_transfer)

    def cal_ls_mat(self):
        n_updates = len(self.updates)
        ls_mat = np.zeros((n_updates, n_updates))
        for update in self.updates:
            for _update in self.updates:
                if update[0] == _update[0]:
                    continue
                rank, _rank = update[-1], _update[-1]
                for r1, _r1 in zip(rank, _rank):
                    ls_mat[update[0]][_update[0]] += np.sum((r1 < rank) ^ (_r1 < _rank))
        return ls_mat

    def update_model(self, agg_alg):
        if agg_alg == 'fedavg':
            if self.knowledge_transfer:
                ls_mat = self.cal_ls_mat()
                for i, ls in enumerate(ls_mat):
                    selected_idx = ls.argsort()[0:int(len(ls) * self.kt_agg_prob)]
                    # assert selected_idx[0] == i, 'ls_mat: {}'.format(ls_mat)
                    selected_updates = list(filter(lambda x: x[0] in selected_idx, self.updates))
                    self.global_gp_params[i] = {'theta': self.FedAvg(selected_updates)}
            else:
                self.global_gp_params = {'theta': self.FedAvg(self.updates)}
        else:
            self.updates = []
            raise TypeError('Does not contain aggregation algorithm: ', agg_alg)
        self.updates = []

    @staticmethod
    def FedAvg(selected_updates):
        total_samples = 0
        gp_params = 0
        for update in selected_updates:
            total_samples += update[1]
            gp_params += update[1] * update[2]
        gp_params /= total_samples
        return gp_params
