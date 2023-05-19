import copy
import numpy as np
from deploy.clientbase import Client
from libs.EA import RCGA
from libs.AF import AF


class ClientRBF(Client):

    def __init__(self, client_id, group_id, params, rbf, obj_func, a_f=AF,
                 x_s=None, y_s=None, pb=None):
        super(ClientRBF, self).__init__(client_id=client_id, group_id=group_id)

        self.num_init_points = params['num_init_points']
        self.iid = params['iid']
        self.x_s = x_s
        self.y_s = y_s
        self.opt_obj = params['opt_obj']
        # partitioned bound
        self.pb = pb
        self.rbf = rbf

        # 20 generations & using archive
        self.s_search = RCGA(obj_fuc=obj_func,
                             a_f=a_f(self.rbf, self.opt_obj),
                             num_init_points=self.num_init_points,
                             iid=self.iid,
                             pb=self.pb)

        self.cur_x_best = None
        self.cur_y_best = None
        self.obj_func = obj_func

        self._init_rbf_data()

    def _init_rbf_data(self):
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
        obj_value = self.obj_func(x=x_new)
        return obj_value

    @staticmethod
    def sort_data(original_x, original_y, num_select):
        data_index = np.argsort(original_y.flatten())
        return original_x[data_index[0:num_select], :], original_y[data_index[0:num_select], :]

    def data_size(self):
        return len(self.x_s)

    def get_model_params(self):
        return self.rbf.get_params()

    def set_model_params(self, m_tuple):
        self.rbf.set_params(m_tuple)

    def synchronize(self, model_params: tuple):
        assert len(model_params) == 2
        self.set_model_params(model_params)

    def train(self):
        self.rbf.fit(self.x_s, self.y_s)
        x_next_best, ac_best = self.s_search.search(self.x_s, self.y_s)

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

        return self.client_id, self.local_data_size, self.get_model_params()

    @property
    def local_data_size(self):
        assert len(self.x_s) == len(self.y_s)
        return len(self.x_s)

    def predict(self):
        return self.rbf.predict(self.x_s)
