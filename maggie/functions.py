""" layers """


def linear(w, b, x):
    pass


def conv2d(w, x, stride, kernel):
    pass


def avg_pool_2d():
    pass


def max_pool_2d():
    pass


def batch_norm():
    pass


def softmax(t):
    return t.exp() / t.exp().sum(dim=-1, keepdim=True)


""" rnns """


def rnn():
    pass


def gru():
    pass


def lstm(params, sequence):
    pass


""" losses """


def nll_loss(y, y_train):
    return (-(y * y_train).sum(dim=-1).log()).sum() / y_train.size()[0]


def cce_loss(y, y_train):
    return -(
        (y_train * y).sum(dim=-1).log() +
        ((1 - y_train()) * (1 - y)).sum(dim=-1).log()
    ).sum() / y_train.size()[0]


def rmse_loss(y, y_train):
    return (y - y_train).pow(2).sum() / y_train.size()[0]
