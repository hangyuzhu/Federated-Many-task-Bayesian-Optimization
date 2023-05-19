import numpy as np


def mini_batches(x_train, y_train, distance, batch_size):
    # randomly shuffle data
    rng_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)
    # loop through mini-batches
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        b_distance = distance[i:i+batch_size]
        yield (x_batch, y_batch, b_distance)


def index_bootstrap(num_data, prob):
    """
    :param num_data: the index matrix of input, int
    :param prob: the probability for one index sample to be chose, >0
    return: index of chose samples, bool

    example:
    a=np.array([[1,2,3,4],[0,0,0,0]]).T
    rand_p = np.random.rand(4)
    b=np.greater(rand_p,0.5)
    b is the output, and we can use a[b] to locate data
    """
    rand_p = np.random.rand(num_data)

    out = np.greater(rand_p, 1 - prob)

    if True not in out:
        out = index_bootstrap(num_data, prob)

    return out
