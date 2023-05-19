import os
import time
import pandas as pd
from deploy.servergp import ServerGP
from deploy.clientgp import ClientGP
from utils.constants import Benchmarks, Acquisitions
from utils.connection import online
from Tasks.benchmark import create_tasks_diff_func
from utils.partitions import create_normalized_partitions
import importlib


def main(iter, args):
    print('=============== Run {} ==============='.format(iter))

    # or number of evaluations (real fitness calculations)
    total_clients = args.total_clients
    clients_per_round = int(total_clients * args.C)
    num_init_points = int(args.num_b4_d * args.dim)
    if args.Total_FE == -1:
        total_rounds = 11 * args.dim - num_init_points
    else:
        total_rounds = args.Total_FE - num_init_points

    params = {'num_init_points': num_init_points,
              'iid': args.iid,
              'dim': args.dim,
              'knowledge_transfer': args.knowledge_transfer,
              'kt_agg_prob': args.kt_agg_prob,
              'gamma': args.gamma,
              'opt_obj': args.opt_obj,
              'regr': args.regr,
              'acq_type': args.acq_type,
              'next_method': args.next_method}

    # ======= Create Server ========
    pre_start = time.time()
    server = ServerGP(server_id=0,
                      params=params,
                      clients_per_round=clients_per_round,
                      enable_parallel=args.enable_parallel)

    # ======= Create Clients ========
    if args.multi_task:
        tasks = create_tasks_diff_func(args.dim)
        total_clients = len(tasks)
        clients_per_round = int(total_clients * args.C)
        if args.iid:
            partitioned_bounds = [None] * total_clients
        else:
            partitioned_bounds = create_normalized_partitions(total_clients, args.np_per_dim, args.dim)
        all_clients = [ClientGP(client_id=i, group_id=0, params=params, obj_func=t, pb=partitioned_bounds[i])
                       for i, t in enumerate(tasks)]
    else:
        # using default boundary
        func_mod = importlib.import_module('SOP.ObjectiveFunctions')
        function = getattr(func_mod, args.func)
        obj_func = function(d=args.dim, normalized=False, transform=None)
        if args.iid:
            partitioned_bounds = [None] * total_clients
        else:
            partitioned_bounds = create_normalized_partitions(total_clients, args.np_per_dim, args.dim)
        all_clients = [ClientGP(client_id=i, group_id=0, params=params, obj_func=obj_func, pb=partitioned_bounds[i])
                       for i in range(total_clients)]

    if args.knowledge_transfer:
        server.create_x_s(all_clients)
        server.share_x_s_to(all_clients)
    pre_end = time.time()
    time_per_round = [pre_end - pre_start] * num_init_points

    # ======= Federated Bayesian Optimization ========
    start = time.time()
    for i in range(total_rounds):
        # check if the communication round is correct or not
        assert i == server.round
        start_time = time.time()
        print('--------- Round %d of %d: Training %d Clients ---------'
              % (i + 1, total_rounds, clients_per_round))

        server.select_clients(online(all_clients), num_clients=clients_per_round)
        server.train()
        server.update_model(agg_alg='fedavg')

        duration_time = time.time()-start_time
        time_per_round.append(duration_time)
        print('One communication round training time: %.4fs' % duration_time)

    duration = time.time() - start
    print('Overall elapsed time : ', duration)

    # ======= save results ========
    if args.save_results:
        root_save_file = 'results_fbo'
        if args.multi_task:
            root_save_file += '_mt'
            save_dir = os.path.join(root_save_file, str(args.dim))
        else:
            save_dir = os.path.join(root_save_file, args.func, str(args.dim))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_name = 'run' + str(iter)
        if args.multi_task:
            file_name += '_DMT_' + str(clients_per_round) + '.' + str(total_clients)
        else:
            file_name += '_C' + str(clients_per_round) + '.' + str(total_clients)
        if args.knowledge_transfer:
            file_name += '_ktp' + str(args.kt_agg_prob)
        if args.iid:
            iid_file = 'iid'
        else:
            iid_file = 'niid' + str(args.np_per_dim)
        file_name += '_FE' + str(total_rounds) + '_' + args.acq_type + str(args.gamma) + '_' + iid_file + '_fedavg.csv'
        record_file_name = os.path.join(save_dir, file_name)

        record_data = {}
        for c in all_clients:
            record_data.update({'client' + str(c.client_id): c.fitness_values})
        record_data.update({'time': time_per_round})
        record_pd = pd.DataFrame(data=record_data)
        record_pd.to_csv(record_file_name)

    del server
    del all_clients


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--num_runs', default=1, type=int, help='number of runs')
    parser.add_argument('--enable_parallel', default=False, action='store_true', help='if using parallel computing')
    parser.add_argument('--multi_task', default=False, action='store_true', help='if using multi-task optimization')
    parser.add_argument('--iid', default=False, action='store_true', help='client dataset partition methods')
    parser.add_argument('--np_per_dim', default=2, type=int, help='number of partitions per feature dimension')
    parser.add_argument('--Total_FE', default=-1, type=int, help='total number of federated evaluation')
    parser.add_argument('--total_clients', default=-1, type=int, help='total number of clients')
    parser.add_argument('--C', default=1., type=float, help='connection ratio')
    parser.add_argument('--num_b4_d', default=5, type=int, help='num_b4_d')
    parser.add_argument('--dim', default=10, type=int, help='the number of dimensions of decision variables')

    parser.add_argument('--func', default='', type=str, choices=Benchmarks, help='The training dataset')
    parser.add_argument('--opt_obj', default='minimize', type=str, choices=['minimize', 'maximize'],
                        help='minimizing or maximizing the objective')
    parser.add_argument('--regr', default='regr_constant', type=str, choices=['regr_constant', 'regr_linear'],
                        help='regression type')
    parser.add_argument('--next_method', default='GA', type=str, choices=['GA', 'sampling'],
                        help='The sampling method for getting the next decision variable')
    parser.add_argument('--acq_type', default='EI', type=str, choices=Acquisitions, help='Acquisition function')
    parser.add_argument('--knowledge_transfer', default=False, action='store_true',
                        help='if sharing the shared server datasets for transfer learning')
    parser.add_argument('--kt_agg_prob', default=0.8, type=float,
                        help='the proportion of clients for global model aggregation')
    parser.add_argument('--gamma', default=0.5, type=float,
                        help='the trade-off parameters of federated ensemble acquisition function')
    parser.add_argument('--save_results', default=False, action='store_true', help='if saving the results')
    args = parser.parse_args()

    for iter in range(0, args.num_runs):
        main(iter, args)
