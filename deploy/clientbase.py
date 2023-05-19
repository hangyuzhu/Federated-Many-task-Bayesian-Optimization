class Client:

    def __init__(self, client_id=None, group_id=None):
        self.client_id = client_id
        self.group_id = group_id
        self.round = 0

    @property
    def global_model_params(self):
        return None

    def train(self, *args, **kwargs):
        return None

    def train_surrogate(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        return None

    def encrypt(self, *args, **kwargs):
        return None

    def decrypt(self, *args, **kwargs):
        return None

    def partial_decrypt(self, *args, **kwargs):
        return None

    @property
    def local_data_size(self):
        return None

    def save(self, *args, **kwargs):
        pass
