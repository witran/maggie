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

        self.store_size = kwargs["store_size"]

        self.n_steps = kwargs["n_steps"]
        self.target_update_interval = kwargs["target_update_interval"]
        self.n_steps_to_start_training = kwargs["n_steps_to_start_training"]
        self.demo_interval = kwargs["demo_interval"]
        self.log_interval = kwargs["log_interval"]

        self.learning_rate = kwargs["learning_rate"] or 0.00025
        self.batch_size = kwargs["batch_size"] or 1 << 15
        self.epochs = kwargs["epochs"] or 1
        # self.mini_batch_size = kwargs["mini_batch_size"] or 1 << 10


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
        # indexes = np.random.choice(
        #     np.arange(len(self.buffer)), size=batch_size)
        batch = []
        for i in range(batch_size):
            batch.append(self.buffer[np.random.randint(len(self.buffer))])
        return np.array(batch)

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
    qnet_target = deepcopy(qnet)

    # optimizer = optim.Adam(qnet.parameters(), lr=agent.learning_rate)
    optimizer = optim.RMSprop(qnet.parameters(), lr=agent.learning_rate)

    store = Store(agent.store_size)

    s = env.reset()
    step = 0
    r_sum = 0
    n_episodes = 0
    r_sum_window = []
    r_sum_window_length = 10
    loss = 0

    curve = []

    for i in range(agent.n_steps):
        # act & store
        a, s_next, r, done = act(env, agent, qnet, s, step)
        store.add((s, a, r, s_next, done), 0)

        # loop
        if done:
            # for logging
            n_episodes += 1
            # avg_reward -= (1 / (n_episodes) * (r_sum - avg_reward))
            if len(r_sum_window) >= r_sum_window_length:
                del r_sum_window[0]
            r_sum_window.append(r_sum)

            s = env.reset()
            r_sum = 0
            step = 0
        else:
            s = s_next
            r_sum += r
            step += 1

        # learn
        if i > agent.n_steps_to_start_training:
            loss = train(qnet, qnet_target, optimizer, store, agent)

        # copy net every steps_to_target_update
        if (i + 1) % agent.target_update_interval:
            qnet_target = deepcopy(qnet)

        # debug on interval
        if step == 1 and (n_episodes + 1) % agent.demo_interval == 0:
            print("----- DEMO ----")
            last_episode_reward = play(env, agent, qnet, render=True)
            print("last episode reward", last_episode_reward)
            print("---------------")

        if (i + 1) % agent.log_interval == 0:
            print("-----")
            print("step #{}, num episodes played: {}, store size: {} \nloss: {}, avg_reward last {} episodes: {}".format(
                i + 1, n_episodes, len(store.buffer), loss, len(r_sum_window), sum(r_sum_window) / len(r_sum_window)))

            curve.append(sum(r_sum_window) / len(r_sum_window))

    return qnet, curve


def act(env, agent, qnet, s, current_step):
    max_step = 1000
    q = qnet(torch.tensor(s).float())
    pi = softmax_policy(
        q, agent.softmax_temperature).detach().numpy()
    a = sample_action(pi)
    s_next, r, done, info = env.step(a)

    if current_step > max_step:
        done = True
        r = -1000

    return a, s_next, r, done


def play(env, agent, qnet, render=False):
    max_step = 1000
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


def sample_action(pi):
    return np.random.choice(np.arange(len(pi)), p=pi).item()


def softmax_policy(q_values, tau=1.):
    preferences = q_values / tau
    max_preference = preferences.max(dim=-1, keepdim=True)[0]
    numerator = (preferences - max_preference).exp()
    denominator = numerator.sum(dim=-1, keepdim=True)
    pi = (numerator / denominator).squeeze()
    return pi


# def epsilon_policy(q_values, epsilon_max, epsilon_min, epsilon_decay, step):
#     epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * \
#         math.exp(-LAMBDA * self.steps)
#     pass


def train(qnet, qnet_target, optimizer, store, agent):
    discount = agent.discount
    batch_size = agent.batch_size
    # mbs = agent.mini_batch_size
    epochs = agent.epochs

    # loss_fn = mg.nn.huber_loss
    # loss_fn = torch.nn.L1Loss
    loss_fn = torch.nn.MSELoss()
    loss_sum = 0

    samples = store.sample(batch_size)
    for epoch in range(epochs):
        loss_sum = 0
        # for i in range(batch_size // mbs):
        # start = i * batch_size
        # end = (i + 1) * batch_size

        # TODO: cleaner code
        # s, a, r, s_next, done = samples[start:end].T
        s, a, r, s_next, done = samples.T
        s, r, s_next, done = map(
            lambda arr: torch.tensor(np.vstack(arr)).float(),
            (s, r, s_next, done))
        a = torch.tensor(np.vstack(a))

        # TODO: change to maggie

        # compute delta
        q_values_next = qnet_target(s_next)
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
