import torch
import numpy as np
import numpy.random as r
import autograd as ag


def allclose(a, ta):
    print(np.allclose(a, ta.detach().numpy()))
    return np.allclose(a, ta.detach().numpy())


def to_torch(*args):
    return map(lambda arg: torch.tensor(arg, requires_grad=True), args)


def to_ag(*args):
    return map(lambda arg: ag.Node(arg, requires_grad=True), args)


def test_power():
    a = r.randn(2)
    b = r.randn(1)
    print(a, b)

    ta, tb = to_torch(a, b)
    # print(ta, tb)
    # tc = (ta ** tb).sum()
    # tc.backward()

    na, nb = to_ag(a, b)
    # nc = (na ** nb).sum()
    # nc.backward()

    # print(tc, nc.value)
    print(a ** b)
    print(na ** nb)
    print(ta ** tb)

    # assert allclose(a ** b, na ** nb, ta ** tb)
    # assert allclose(nc.value, tc)
    # assert allclose(na.grad, ta.grad)


def test_log():
    pass


test_power()
test_log()
