from abc import ABC, abstractmethod


class Server(ABC):

    def __init__(self, server_id):
        self.server_id = server_id
        self.selected_clients = []
        self.updates = []
        self.round = 0

    def __del__(self):
        pass

    @property
    def master_model(self):
        return None

    @property
    def master_model_num_params(self):
        return None

    def select_clients(self, *args, **kwargs):
        pass

    def train_test(self, *args, **kwargs):
        return None

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def test(self, *args, **kwargs):
        return None

    @abstractmethod
    def update_model(self, *args, **kwargs):
        self.updates = []

    def model_aggregation(self):
        """ Master model aggregation """
        pass

    def get_server_info(self):
        server_info = {'server_id': self.server_id}
        return server_info

    def get_clients_info(self, clients=None):
        if clients is None:
            clients = self.selected_clients
        ids = [c.client_id for c in clients]
        groups = [c.client_group for c in clients]
        clients_info = {'clients_ids': ids,
                        'clients_groups': groups}
        return clients_info

    def save_model(self, path):
        pass


