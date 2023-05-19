import copy
import numpy as np
from scipy.spatial.distance import cdist
from deploy.serverbase import Server


class ServerRBF(Server):

    def __init__(self, server_id: int, rbf):
        super(ServerRBF, self).__init__(server_id)
        self.rbf = rbf

    @property
    def rbf_params(self):
        p = self.rbf.get_params()
        return p['w'], p['b']

    def select_clients(self, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def sort_centers(self):
        self.rbf.centers, sorted_index = self.rbf.sort_centers(self.rbf.centers)
        return sorted_index

    def train(self, clients=None):
        # clear the buffer
        self.updates = []
        if clients is None:
            clients = self.selected_clients
        for c in clients:
            c.synchronize(self.rbf_params)
            update = c.train()
            self.updates.append(update)
        self.round += 1

    def update_model(self):
        tot_samples = 0
        stack_c, stack_w, stack_b, stack_s = 0, 0, 0, 0
        for update in self.updates:
            tot_samples += update[1]

            stack_c += update[1] * update[2]['centers']
            stack_w += update[1] * update[2]['w']
            stack_b += update[1] * update[2]['b']
            stack_s += update[1] * update[2]['std']

        self.rbf.centers = stack_c / tot_samples
        tmp_index = self.sort_centers()
        self.rbf.w, self.rbf.b, self.rbf.std = \
            stack_w[tmp_index] / tot_samples, stack_b / tot_samples, stack_s[tmp_index] / tot_samples
        # clear buffer
        self.updates = []

    def predict(self, x_test):
        return self.rbf.predict(x_test)
