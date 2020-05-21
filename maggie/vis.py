from graphviz import Digraph
from .node import Node


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._inputs:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    print("hello world")
    nodes, edges = trace(root)
    print(nodes, edges)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        dot.node(name=str(id(n)), label="", shape="record")
        print(str(id(n)), n._op)
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.render("gout")
    # return dot
