import numpy as np
import numpy.linalg as LA
from scipy.interpolate import interp1d
import time


def euclidean_dist(x1, x2):
    return LA.norm(np.array(x1) - np.array(x2))


def mapl(f, *seq):
    return list(map(f, *seq))


def timed(f):
    """Wrapper that returns a timed version of function."""

    def wrapper(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()

        return ret, (end - start)

    return wrapper


def sample_edge(v1, v2, num_samples=20, theta=None):
        sample_points = []
        x1, y1 = v1
        x2, y2 = v2
        if np.isclose((x2 - x1), 0):
            x = [x1] * num_samples
            y = np.arange(y1, y2, (y2 - y1) / num_samples)
        else:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - (m * x1)

            x = np.arange(x1, x2, (x2 - x1) / num_samples)
            y = m * x + c
        if theta is not None:
            thetas = [theta] * num_samples
            for p in zip(x, y, thetas):
                sample_points.append(p)
        else:
            for p in zip(x, y):
                sample_points.append(p)
        return sample_points


def nearest5(x):
    return round(x / 5) * 5


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


ADJ_S = {0: (1, 0), 5: (11, 1), 10: (6, 1), 15: (4, 1), 20: (3, 1), 25: (2, 1), 30: (9, 5), 35: (3, 2), 40: (6, 5), 45: (1, 1),
         50: (5, 6), 55: (2, 3), 60: (5, 9), 65: (1, 2), 70: (1, 3), 75: (1, 4), 80: (1, 6), 85: (1, 11), 90: (0, 1)}
def adjacent_square(x, y, theta):
    ax, ay = ADJ_S[theta]
    return x + ax, y + ay


if __name__ == '__main__':
    import doctest
    doctest.testmod()