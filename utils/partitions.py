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
