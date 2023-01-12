import numpy as np


def get_cluster(x, y, l):
    queue = [(x, y)]
    r = []
    while len(queue) > 0:
        x, y = queue.pop(0)
        r.append((x, y))
        l[x, y] = 0
        if x-1 > 0 and l[x-1, y] == 1:
            l[x-1, y] = 0
            queue.append((x-1, y))

        if x+1 < l.shape[0] and l[x+1, y] == 1:
            l[x+1, y] = 0
            queue.append((x+1, y))

        if y-1 > 0 and l[x, y-1] == 1:
            l[x, y-1] = 0
            queue.append((x, y-1))

        if y+1 < l.shape[1] and l[x, y+1] == 1:
            l[x, y+1] = 0
            queue.append((x, y+1))

    return r, l


def get_clusters(l):
    clusters = []
    while True:
        x, y = np.where(l == 1)
        if len(x) == 0:
            break
        r, l = get_cluster(x[0], y[0], l)
        clusters.append(r)
    return clusters


def filter_clusters(l):
    r = get_clusters(l)
    r = sorted(r, key=lambda x: len(x), reverse=True)[0]
    result = np.zeros(l.shape)
    for x, y in r:
        result[x, y] = True
    return result

