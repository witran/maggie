import numpy as np


class Node:
    def __init__(
        self, value, requires_grad=False,
        _inputs=[], _op="", _backward=lambda: None
    ):
        self.value = value
        self.grad = 0
        self.requires_grad = requires_grad
        self._inputs = _inputs
        self._op = _op
        self._backward = _backward

    def backward(self):
        # toposort, zero grad, find nodes that require grad
        topo = []
        visited = set()
        grad_required = set()
        params = []

        def visit(node):
            visited.add(node)
            node.grad = 0

            if node.requires_grad:
                params.append(node)
                grad_required.add(node)

            for in_node in node._inputs:
                if in_node not in visited:
                    visit(in_node)
                if in_node in grad_required:
                    grad_required.add(node)

            topo.append(node)

        visit(self)

        # only compute grad for just enough amount of nodes
        self.grad = np.ones(self.value.shape)

        for node in reversed(topo):
            # print(node._op, node.grad)
            for in_node in node._inputs:
                if in_node in grad_required:
                    # print(node._op, in_node, node._inputs,
                    #   node._backward(in_node))
                    in_node.grad += node._backward(in_node)

        return params

    def size(self):
        return self.value.shape

    def __repr__(self):
        return "Node(value={}, grad={})".format(self.value, self.grad)
