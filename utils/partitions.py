import numpy as np


def create_normalized_partitions(num_users, np_per_dim, dim):
    """
    :param num_users: number of clients
    :param np_per_dim: number of partitions per dimension
    :param dim: number of dimensions
    :return:
    """
    x_ub = 1
    x_lb = 0
    gap = (x_ub - x_lb) / np_per_dim
    # each row represents split bounds for each dimension
    p_mat = [[x_lb + i * gap for i in range(np_per_dim + 1)] for _ in range(dim)]
    assert dim * np_per_dim >= num_users
    p_sampled = [np.random.randint(0, np_per_dim, size=dim) for _ in range(num_users)]
    pb = []
    # sample num_user times
    for ps in p_sampled:
        lb = []
        ub = []
        for idx, p in zip(ps, p_mat):
            lb.append(p[idx])
            ub.append(p[idx + 1])
        pb.append([lb, ub])
    return pb


def create_partitions(np_per_dim, dim):
    x_ub = 1
    x_lb = 0
    gap = (x_ub - x_lb) / np_per_dim

    def convert(a):
        # let 0 -> 1
        per_dim = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        c = []
        while a:
            c.append(per_dim[a % np_per_dim])
            a = a // np_per_dim
        tot_len = len(c)
        c = [0] * (dim - tot_len) + list(reversed(c))
        return c

    # each row represents split bounds for each dimension
    p_mat = [x_lb + i * gap for i in range(np_per_dim + 1)]
    n_partitions = np_per_dim ** dim
    splits = [convert(i) for i in range(n_partitions)]

    partitions = []
    for s in splits:
        pb = []
        for _s in s:
            pb.append([p_mat[_s], p_mat[_s + 1]])
        partitions.append(pb)
    return partitions


def create_landmine_partitions(num_users=29, np_per_dim=2, dim=2):
    assert num_users == 29 and np_per_dim == 2 and dim == 2
    gap = (1 - 0) / np_per_dim
    p_mat = [[0 + i * gap for i in range(np_per_dim + 1)] for _ in range(dim)]
    partitions = [[0, 0], [0, 1], [1, 0], [1, 1]]

    partition_assignment = np.arange(num_users) % len(partitions)
    pb = []
    for pa in partition_assignment:
        ps = partitions[pa]
        lb = []
        ub = []
        for idx, p in zip(ps, p_mat):
            lb.append(p[idx])
            ub.append(p[idx + 1])
        pb.append([lb, ub])
    return pb


def create_de_weights(N_partitions, partition_assignment, N):
    T_max = 200
    a, b = 16, 1
    T_const = 10
    T_max_decay = 30

    a_s = np.linspace(a, b, T_max_decay)
    a_s = np.append(a_s, np.ones(T_max - T_max_decay))

    all_weights_all = []
    for t in range(T_const):
        all_weights = []
        for n in range(N_partitions):
            weights = np.zeros(N)
            for i in range(N):
                if partition_assignment[i] == n:
                    weights[i] = np.exp(a)
                else:
                    weights[i] = np.exp(b)
            weights = weights / np.sum(weights)
            all_weights.append(weights)
        all_weights_all.append(all_weights)

    for t in range(100):
        all_weights = []
        for n in range(N_partitions):
            weights = np.zeros(N)
            for i in range(N):
                if partition_assignment[i] == n:
                    weights[i]= np.exp(a_s[t])
                else:
                    weights[i]= np.exp(b)
            weights = weights / np.sum(weights)
            all_weights.append(weights)
        all_weights_all.append(all_weights)
    return all_weights_all


if __name__ == '__main__':
    # pb = create_landmine_partitions()
    pb = create_partitions(2, 10)
