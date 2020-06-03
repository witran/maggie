from copy import deepcopy
import gym
import numpy as np
import maggie as mg
import torch
from torch import nn
from torch import optim


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
        self.n_episodes = kwargs["n_episodes"]
        self.target_update_interval = kwargs["target_update_interval"]
        self.n_steps_to_start_training = kwargs["n_steps_to_start_training"]
        self.demo_interval = kwargs["demo_interval"]
        self.log_interval = kwargs["log_interval"]

        self.timeout = kwargs["timeout"]
        self.timeout_reward = kwargs["timeout_reward"]

        self.optim_lr = kwargs["optim_lr"]
        self.optim_beta_m = kwargs["optim_beta_m"]
        self.optim_beta_v = kwargs["optim_beta_v"]
        self.optim_epsilon = kwargs["optim_epsilon"]

        self.n_batches = kwargs["n_batches"]
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]


class Store():
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = max_size
        self.rand_generator = np.random.RandomState(1)

    def add(self, item, priority):
        if len(self.buffer) == self.max_size:
            del self.buffer[0]

        self.buffer.append(item)

    def sample(self, batch_size):
        idxs = self.rand_generator.choice(
            np.arange(len(self.buffer)), size=batch_size)
        return np.array([self.buffer[idx] for idx in idxs])
        # batch = []
        # for i in range(batch_size):
        #     batch.append(self.buffer[np.random.randint(len(self.buffer))])
        # return np.array(batch)


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

    l1 = nn.Linear(input_size, 256)
    l2 = nn.Linear(256, output_size)

    torch.nn.init.orthogonal_(l1.weight)
    torch.nn.init.zeros_(l1.bias)
    torch.nn.init.orthogonal_(l2.weight)
    torch.nn.init.zeros_(l2.bias)
    qnet = nn.Sequential(l1, nn.ReLU(), l2)

    # qnet = nn.Sequential(
    #     nn.Linear(input_size, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, output_size))
    # qnet_target = deepcopy(qnet)

    # optimizer = optim.Adam(qnet.parameters(), lr=agent.learning_rate)
    # optimizer = optim.RMSprop(qnet.parameters(), lr=agent.learning_rate)
    optimizer = optim.Adam(qnet.parameters(),
                           lr=agent.optim_lr,
                           betas=(agent.optim_beta_m,
                                  agent.optim_beta_v),
                           eps=agent.optim_epsilon
                           )

    store = Store(agent.store_size)

    s = env.reset()
    step = 0
    r_sum = 0
    n_episodes = 0
    r_sum_window = []
    r_sum_window_length = 10
    loss = 0

    reward_history = []
    loss_history = []

    for i in range(agent.n_steps):
        # act & store
        a, s_next, r, done = act(env, agent, qnet, s, step)
        store.add((s, a, r, s_next, done), 0)

        # loop
        if done:
            # for logging
            n_episodes += 1
            r_sum += r
            if len(r_sum_window) >= r_sum_window_length:
                del r_sum_window[0]
            r_sum_window.append(r_sum)

            s = env.reset()
            r_sum = 0
            step = 0

            if n_episodes == agent.n_episodes:
                break

        else:
            s = s_next
            r_sum += r
            step += 1

        # learn
        if i > agent.batch_size:
            qnet_target = deepcopy(qnet)
            for _ in range(agent.n_batches):
                loss = train(qnet, qnet_target, optimizer, store, agent)
                loss_history.append(loss)

            # copy net every steps_to_target_update
            # if (i + 1) % agent.target_update_interval:
            #     qnet_target = deepcopy(qnet)

            # demo on interval
            if step == 0 and (n_episodes + 1) % agent.demo_interval == 0:
                print("----- DEMO ----")
                last_episode_reward = play(env, agent, qnet, render=True)
                print("last episode reward", last_episode_reward)
                print("---------------")

            # print debug on interval
            if (i + 1) % agent.log_interval == 0:
                print("-----")
                print("step #{}, num episodes played: {}, store size: {} \nloss: {}, last {} episodes avg={} best={} worst={}".format(
                    i + 1,
                    n_episodes,
                    len(store.buffer),
                    round(loss, 4),
                    len(r_sum_window),
                    round(sum(r_sum_window) / len(r_sum_window), 4),
                    round(max(r_sum_window), 4),
                    round(min(r_sum_window), 4)
                ))

                reward_history.append(sum(r_sum_window) / len(r_sum_window))

    return qnet, reward_history, loss_history


def act(env, agent, qnet, s, current_step):
    q = qnet(torch.tensor(s).float())
    pi = softmax_policy(
        q, agent.softmax_temperature).detach().numpy()
    # print(pi)
    a = sample_action(pi)
    s_next, r, done, info = env.step(a)

    if current_step > agent.timeout:
        done = True
        # r = agent.timeout_reward

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

    s = env.reset()

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

        # TODO: cleaner code & change to maggie
        # s, a, r, s_next, done = samples[start:end].T
        s, a, r, s_next, done = samples.T
        s, r, s_next, done = map(
            lambda arr: torch.tensor(np.vstack(arr)).float(),
            (s, r, s_next, done))
        a = torch.tensor(np.vstack(a))

        # compute delta
        q_values_next = qnet_target(s_next)
        pi = softmax_policy(q_values_next)
        bootstrap_term = (
            pi * q_values_next
        ).sum(dim=-1, keepdim=True) * (1 - done)

        q_values = qnet(s)

        indexes = torch.arange(
            q_values.shape[0]) * q_values.shape[1] + a.squeeze()

        q_a = q_values.take(indexes)
        q_a_target = (r + discount * bootstrap_term).squeeze()

        loss = loss_fn(q_a, q_a_target)
        loss.backward()

        loss_sum += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        # print("epoch {}, loss: {}".format(epoch, loss_sum))

    return loss_sum
