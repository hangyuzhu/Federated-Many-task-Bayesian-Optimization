import pickle
import os


def prepare_data(data_name):
    cur_dir_path = os.path.dirname(os.path.abspath(__file__))
    if data_name == 'Landmine':
        _data = pickle.load(open(os.path.join(cur_dir_path, 'landmine', 'landmine_formated_data.pkl'), 'rb'))
        # list
        all_X_train, all_Y_train, all_X_test, all_Y_test = _data["all_X_train"], _data["all_Y_train"], \
                                                           _data["all_X_test"], _data["all_Y_test"]
        return all_X_train, all_Y_train, all_X_test, all_Y_test
    else:
        raise NotImplementedError
