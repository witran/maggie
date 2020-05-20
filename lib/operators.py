import numpy as np
from .node import Node

""" utils """


def unbroadcast(array, original_shape):
    prefixed_axes = []
    broadcasted_axes = []
    len_diff = len(array.shape) - len(original_shape)
    for i in range(len(array.shape)):
        if i < len_diff:
            prefixed_axes.append(i)
        elif original_shape[i - len_diff] == 1:
            broadcasted_axes.append(i)

    return array.sum(axis=tuple(broadcasted_axes), keepdims=True).sum(
        axis=tuple(prefixed_axes)
    )


def as_node(*items):
    return map(lambda item: item if isinstance(item, Node) else Node(item), items)


""" primitive operators """


def add(a, b):
    a, b = as_node(a, b)
    out = Node(a.value + b.value, _inputs=[a, b], _op="add")

    def _backward(node):
        return unbroadcast(out.grad, node.value.shape)

    out._backward = _backward
    return out


def mul(a, b):
    a, b = as_node(a, b)
    out = Node(a.value * b.value, _inputs=[a, b], _op="mul")

    def _backward(node):
        if node is b:
            grad = out.grad * a.value
        else:
            grad = out.grad * b.value
        return unbroadcast(grad, node.value.shape)

    out._backward = _backward
    return out


def power(base, exp):
    base, exp = as_node(base, exp)
    out = Node(base.value ** exp.value, _inputs=[base, exp], _op="power")

    def _backward(node):
        if node is base:
            grad = out.grad * exp.value * (base.value ** (exp.value - 1.))
        else:
            grad = out.grad * np.log(base.value) * (base.value ** exp.value)
        return unbroadcast(grad, node.value.shape)

    out._backward = _backward
    return out


def log(a):
    a, = as_node(a)
    out = Node(np.log(a.value), _inputs=[a], _op="log")

    def _backward(node):
        return unbroadcast(out.grad * (node.value ** -1.), node.value.shape)

    out._backward = _backward
    return out


def neg(a):
    return mul(a, -1.)


def inv(a):
    return power(a, -1.)


def sub(a, b):
    return add(a, neg(b))


def div(a, b):
    return mul(a, inv(b))


def exp(a):
    return power(np.e, a)


def matmul(a, b):
    a, b = as_node(a, b)
    out = Node(np.matmul(a.value, b.value), _inputs=[a, b], _op="matmul")

    def _backward(node):
        if node is a:
            grad = np.matmul(out.grad, np.transpose(b.value))
        else:
            grad = np.matmul(np.transpose(a.value), out.grad)
        return unbroadcast(grad, node.value.shape)

    out._backward = _backward
    return out


def sum(a, dim=None, keepdim=False):
    a, = as_node(a)
    out = Node(np.sum(a.value, axis=dim, keepdims=keepdim),
               _inputs=[a], _op="sum")

    def _backward(node):
        if keepdim or dim is None:
            grad = out.grad
        else:
            grad = np.expand_dims(out.grad, axis=dim)
        return grad * np.ones(node.value.shape)

    out._backward = _backward
    return out


def clamp(a, a_min, a_max):
    a, = as_node(a)
    out = Node(np.clip(a.value, a_min, a_max), _inputs=[a], _op="clamp")

    def _backward(node):
        f_min = (node.value > a_min) if a_min is not None else True
        f_max = (node.value < a_max) if a_max is not None else True
        return unbroadcast(
            out.grad * np.where(np.logical_and(f_min, f_max), 1, 0,),
            node.value.shape,
        )

    out._backward = _backward
    return out


def view(a, shape):
    a, = as_node(a)
    out = Node(np.reshape(a.value, shape), _inputs=[a], _op="view")

    def _backward(node):
        return np.reshape(out.grad, a.value.shape)

    out._backward = _backward
    return out


""" support infix syntax """
Node.__add__ = lambda self, other: add(self, other)
Node.__radd__ = lambda self, other: add(other, self)
Node.__sub__ = lambda self, other: sub(self, other)
Node.__rsub__ = lambda self, other: sub(other, self)
Node.__neg__ = lambda self: neg(self)
Node.__mul__ = lambda self, other: mul(self, other)
Node.__rmul__ = lambda self, other: mul(other, self)
Node.__truediv__ = lambda self, other: div(self, other)
Node.__rtruediv__ = lambda self, other: div(other, self)
Node.__matmul__ = lambda self, other: matmul(self, other)
Node.__rmatmul__ = lambda self, other: matmul(other, self)
Node.__pow__ = lambda self, other: power(self, other)
Node.__rpow__ = lambda self, other: power(other, self)

""" support method syntax """
Node.mm = lambda self, other: matmul(self, other)
Node.pow = lambda self, other: power(self, other)
Node.log = lambda self: log(self)
Node.sum = lambda self, **kwargs: sum(self, **kwargs)
Node.clamp = lambda self, a_min, a_max: clamp(self, a_min, a_max)
Node.exp = lambda self: exp(self)
Node.view = lambda self, shape: view(self, shape)
