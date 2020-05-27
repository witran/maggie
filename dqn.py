from copy import deepcopy
import gym
import numpy as np
import maggie as mg
import torch
from torch import nn
from torch import optim


# TODO:
# environment
# state tensorr
# net model - linear, dimension based on env
# double, dueling, maxq
# exp store:
#   v1 - array random sample
#   v2 - stochastic priority tree, update priority after learning

class Agent():
    def __init__(self, **kwargs):
        self.discount = kwargs["discount"]
        self.softmax_temperature = kwargs["softmax_temperature"]

        self.greedy_epsilon_max = kwargs["greedy_epsilon_max"]
        self.greedy_epsilon_decay = kwargs["greedy_epsilon_decay"]
        self.greedy_epsilon_min = kwargs["greedy_epsilon_min"]

        self.priority_epsilon = kwargs["priority_epsilon"]
        self.priority_alpha = kwargs["priority_alpha"]

        self.n_iterations = kwargs["n_iterations"]
        self.n_play_iterations = kwargs["n_play_iterations"]
        self.store_size = kwargs["store_size"]

        self.learning_rate = kwargs["learning_rate"] or 0.00025
        self.n_learn_iterations = kwargs["n_learn_iterations"]
        self.batch_size = kwargs["batch_size"] or 1 << 15
        self.epochs = kwargs["epochs"] or 1
        self.mini_batch_size = kwargs["mini_batch_size"] or 1 << 10


class Store():
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = max_size

    def add(self, item, priority):
        if len(self.buffer) == self.max_size:
            del self.buffer[0]

        self.buffer.append(item)

    def sample(self, batch_size):
        # sample_size = min(batch_size, len(self.buffer))
        indexes = np.random.choice(
            np.arange(len(self.buffer)), size=batch_size)
        return np.array(self.buffer)[indexes]

    def update_priority(self, items, priorities):
        pass


# class Net(mg.Module):
#     def __init__(self):
#         self.linear = linear
#         pass

#     def forward(self, x):
#         return x


def learn(env, agent):
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    print("input:{}, output:{}".format(input_size, output_size))

    qnet = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_size)
    )
    store = Store(agent.store_size)

    curve = []

    for i in range(agent.n_iterations):
        # print('----- iteration #{} -----'.format(i))
        # actor
        avg_reward = 0
        for j in range(agent.n_play_iterations):
            episode, priority, reward = sample_episode(env, agent, qnet)
            for k in range(len(episode)):
                store.add(episode[k], priority[k])

            avg_reward += ((1 / (j + 1)) * (reward - avg_reward))

        # learner
        loss = train(qnet, store, agent)

        # sample play

        # avg_reward = 0
        # for j in range(agent.n_play_iterations):
        #     reward = play(env, agent, qnet)
        #     avg_reward += ((1 / j) * (reward - avg_reward))

        curve.append((loss, avg_reward))

        if i % 50 == 0:
            last_episode_reward = play(env, agent, qnet, render=True)
            print("last episode reward", last_episode_reward)

        if i % 20 == 0:
            print("iteration #{}, store size: {}, loss: {}, avg_reward: {}".format(
                i, len(store.buffer), loss, avg_reward))
            last_episode_reward = play(env, agent, qnet, render=False)
            print("last episode reward", last_episode_reward)

    return qnet, curve


def play(env, agent, qnet, render=False):
    max_step = 400
    step = 0
    done = False
    s = env.reset()
    r_sum = 0
    actions = []

    while not done and step < max_step:
        if render:
            env.render()
        q = qnet(torch.tensor(s).float())
        a = q.argmax().item()
        actions.append(a)
        s_next, r, done, info = env.step(a)
        s = s_next
        step += 1
        r_sum += r

    if render:
        print(actions)

    return r_sum


def sample_episode(env, agent, qnet):
    max_step = 400
    step = 0
    total_reward = 0

    episode = []
    priority = []

    s = env.reset()

    while True:
        # qnet.eval()
        q = qnet(torch.tensor(s).float())
        pi = softmax_policy(
            q, agent.softmax_temperature).detach().numpy()
        a = sample_action(pi)
        s_next, r, done, info = env.step(a)
        # print(type(s_next), type(r), type(done), type(info))
        episode.append((s, a, r, s_next, 1. if done else 0.))
        priority.append(0)
        s = s_next
        step += 1
        total_reward += r

        if done:
            break
        elif step == max_step - 1:
            episode.append((s, a, -100, s_next, True))
            priority.append(0)
            break

    return episode, priority, total_reward


def sample_action(pi):
    return np.random.choice(np.arange(len(pi)), p=pi).item()


def softmax_policy(q_values, tau=1.):
    preferences = q_values / tau
    max_preference = preferences.max(dim=-1, keepdim=True)[0]
    numerator = (preferences - max_preference).exp()
    denominator = numerator.sum(dim=-1, keepdim=True)
    pi = (numerator / denominator).squeeze()
    return pi


# def epsilon_policy(q_values, epsilon_max, epsilon_decay, epsilon_min, step):
#     epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
#         math.exp(-LAMBDA * self.steps)
#     pass


def train(qnet, store, agent):
    discount = agent.discount
    batch_size = agent.batch_size
    mbs = agent.mini_batch_size
    n_learn_iterations = agent.n_learn_iterations
    epochs = agent.epochs
    lr = agent.learning_rate

    # loss_fn = mg.nn.huber_loss
    # loss_fn = torch.nn.L1Loss
    loss_fn = torch.nn.MSELoss()
    loss_sum = 0

    for _ in range(n_learn_iterations):
        samples = store.sample(batch_size)
        base_qnet = deepcopy(qnet)
        optimizer = optim.Adam(qnet.parameters(), lr=lr)
        for epoch in range(epochs):
            loss_sum = 0
            for i in range(batch_size // mbs):
                start = i * mbs
                end = (i + 1) * mbs

                s, a, r, s_next, done = samples[start:end].T
                s, r, s_next, done = map(
                    lambda arr: torch.tensor(np.vstack(arr)).float(),
                    (s, r, s_next, done))
                a = torch.tensor(np.vstack(a))
                # print(s.shape, a.shape, r.shape, s_next.shape, done.shape)

                # TODO: change to maggie

                # compute delta
                q_values_next = base_qnet(s_next)
                pi = softmax_policy(q_values_next)
                bootstrap_term = (
                    pi * q_values_next
                ).sum(dim=-1, keepdim=True) * (1 - done)

                q = qnet(s)

                indexes = torch.arange(q.shape[0]) * q.shape[1] + a.squeeze()
                q_a = q.take(indexes)
                q_a_target = (r + discount * bootstrap_term).squeeze()

                loss = loss_fn(q_a, q_a_target)
                loss.backward()

                loss_sum += loss.item()

                optimizer.step()
                optimizer.zero_grad()

            # print("epoch {}, loss: {}".format(epoch, loss_sum))

    return loss_sum
