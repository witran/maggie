import numpy as np
from autograd import Node
import torch
import time


def tg_grad_descent(data, initial_params, alpha, n_iterations):
    x = Node(data[0].copy())
    y = Node(data[1].copy())

    w1 = Node(initial_params[0].copy(), requires_grad=True)
    b1 = Node(initial_params[1].copy(), requires_grad=True)
    w2 = Node(initial_params[2].copy(), requires_grad=True)
    b2 = Node(initial_params[3].copy(), requires_grad=True)

    start = time.time()

    for _ in range(n_iterations):
        # forward
        t = (w1 @ x + b1).clamp(0, None)
        y_h = w2 @ t + b2
        l = (y_h - y).pow(2).sum()

        # backward
        params = l.backward()

        # update
        for param in params:
            param.value -= alpha * param.grad

    print("tinygrad grad descent", time.time() - start, "s")

    return list(map(lambda node: node.value, [w1, b1, w2, b2]))


def torch_grad_descent(data, initial_params, alpha, n_iterations):
    x = torch.tensor(data[0].copy())
    y = torch.tensor(data[1].copy())

    w1 = torch.tensor(initial_params[0].copy(), requires_grad=True)
    b1 = torch.tensor(initial_params[1].copy(), requires_grad=True)
    w2 = torch.tensor(initial_params[2].copy(), requires_grad=True)
    b2 = torch.tensor(initial_params[3].copy(), requires_grad=True)

    start = time.time()

    for _ in range(n_iterations):
        t = (w1.mm(x) + b1).clamp(0)
        y_h = w2.mm(t) + b2
        l = (y_h - y).pow(2).sum()
        l.backward()

        with torch.no_grad():
            w1 -= alpha * w1.grad
            b1 -= alpha * b1.grad
            w2 -= alpha * w2.grad
            b2 -= alpha * b2.grad
            w1.grad.zero_()
            b1.grad.zero_()
            w2.grad.zero_()
            b2.grad.zero_()

    print("torch grad descent", time.time() - start, "s")
    return list(map(lambda tensor: tensor.detach().numpy(), [w1, b1, w2, b2]))


def test_gradient_descent():
    batch_size = 64
    x_dim = 1000
    h_dim = 100
    y_dim = 10

    x = np.random.rand(x_dim, batch_size)
    y = np.random.rand(y_dim, batch_size)
    w1 = np.random.rand(h_dim, x_dim)
    b1 = np.random.rand(h_dim, 1)
    w2 = np.random.rand(y_dim, h_dim)
    b2 = np.random.rand(y_dim, 1)
    data = (x, y)
    initial_params = (w1, b1, w2, b2)

    alpha = 0.01
    n_iterations = 5000

    tg_trained_params = tg_grad_descent(
        data, initial_params, alpha, n_iterations)
    torch_trained_params = torch_grad_descent(
        data, initial_params, alpha, n_iterations)

    for p, t_p in list(zip(tg_trained_params, torch_trained_params)):
        print("all close", np.allclose(p, t_p))


test_gradient_descent()
