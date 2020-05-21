from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
from matplotlib import pyplot as plt
import maggie as mg
from maggie import Node


def fetch():
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "http://deeplearning.net/data/mnist/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    x_train, y_train, x_valid, y_valid = None, None, None, None

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid),
         _) = pickle.load(f, encoding="latin-1")
        print(x_train.shape)
        plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
        print(y_train[0])
        plt.show()

    return x_train, y_train, x_valid, y_valid


def accuracy(model, x_valid, y_valid):
    y = model(Node(x_valid))
    preds = np.argmax(y.value, axis=1)
    return (preds == y_valid).astype('float').mean()


def random_test(model, x_valid):
    i = np.random.randint(len(x_valid))
    print("validation #", i)
    print("predicted: ", np.argmax(
        model(Node(x_valid[i:i + 1])).value, axis=1))
    plt.imshow(x_valid[i].reshape((28, 28)), cmap="gray")


def softmax(t):
    return t.exp() / t.exp().sum(dim=1, keepdim=True)


def nll_loss(y, y_train):
    return -(y * y_train).sum(dim=1).log().sum() / y_train.size()[0]


def train(x_train, y_train):
    n = x_train.shape[0]
    y_train = np.eye(10)[y_train]

    # model
    w1 = vg.Node(np.random.randn(784, 100) / 28, requires_grad=True)
    b1 = vg.Node(np.zeros(100), requires_grad=True)
    w2 = tg.Node(np.random.randn(100, 10) / 10, requires_grad=True)
    b2 = tg.Node(np.zeros(10), requires_grad=True)

    def model(x):
        x = x @ w1 + b1
        x = x.clamp(0, None)
        x = x @ w2 + b2
        x = softmax(x)
        return x

    # loss
    loss = nll_loss

    # hyperparams
    bs = 64
    lr = 0.05
    epochs = 20

    # learning curve
    curve = []

    # fit
    for epoch in range(epochs):
        l_total = 0
        for i in range((n - 1) // bs + 1):
            s = i * bs
            e = (i + 1) * bs
            xb = tg.Node(x_train[s:e])
            yb = tg.Node(y_train[s:e])

            y = model(xb)
            l = loss(y, yb)

            params = l.backward()

            l_total += l.value

            for p in params:
                p.value -= lr * p.grad

        print("iteration #{} - loss: {}".format(epoch, l_total))
        curve.append(l_total)

    return model, curve
