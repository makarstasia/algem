import re
import numpy.core.numeric as NX
from numpy.core import (atleast_1d, array)


class poly1d:
    __hash__ = None


def polydiv(u, v):
    truepoly = (isinstance(u, poly1d) or isinstance(u, poly1d))
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(0, m - n + 1):
        d = scale * r[k]
        q[k] = d
        r[k:k + n + 1] -= d * v
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    if truepoly:
        return poly1d(q), poly1d(r)
    return q, r


_poly_mat = re.compile(r"[*][*]([0-9]*)")

# example
x = array([3.0, 5.0, 2.0])
y = array([2.0, 1.0])
print(polydiv(x, y))
