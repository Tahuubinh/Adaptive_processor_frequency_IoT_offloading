import pickle
from copy import deepcopy
import numpy as np
import math

from util.data_util import read_energy_data


class LinUCBAgent:
    def __init__(self, options, env):
        self.config = options
        self.is_training = True
        self.env = env
        self.K = env.action_space.n
        self.ridge_regression_para = 1
        self.matrix_H = []
        self.b = []
        self.exploration_factor = 10
        self.d = env.observation_space.shape[0]
        for arm in range(self.K):
            self.matrix_H.append(self.ridge_regression_para * np.eye(self.d))
            self.b.append(np.zeros([self.d, 1]))

    def uniform_state(self, s):
        input_s = deepcopy(s)
        input_s[0] = (input_s[0] - 0) / (24 - 0)
        input_s[1] = (input_s[1] - 0) / (self.env.battery_size - 0)
        input_s[2] = (input_s[2] - 0) / (self.env.battery_size - 0)
        input_s[3:-1] = (input_s[3:-1] - 0) / (self.env.core_number - 0)
        input_s[-1] = (input_s[-1] - (self.env.avg_data_size - 10 * 8 * 1e6)) / (
                (self.env.avg_data_size + 10 * 8 * 1e6) - (self.env.avg_data_size - 10 * 8 * 1e6))
        input_s = np.resize(input_s, (self.d, 1))
        return input_s

    def act(self, s0):
        # transfer the selection client into context
        context = self.uniform_state(s0)
        empirical_reward = np.zeros(self.K)
        exploration_reward = np.zeros(self.K)
        estimated_reward = np.zeros(self.K)
        coefficient = [[] for i in range(self.K)]
        for arm in range(self.K):
            coefficient[arm] = np.linalg.inv(self.matrix_H[arm]).dot(self.b[arm])

            empirical_reward[arm] = context.transpose().dot(coefficient[arm])
            exploration_reward[arm] = self.exploration_factor * math.sqrt(
                context.transpose().dot(np.linalg.inv(self.matrix_H[arm])).dot(context))
            # print(exploration_reward[arm])
            estimated_reward[arm] = empirical_reward[arm] + exploration_reward[arm]
            # print(select_arm)
        # print("empirical:{}".format(empirical_reward))
        # print("exploration:{}".format(exploration_reward))
        # print("estimated:{}".format(estimated_reward))
        # print(coefficient)
        # select the arm with highest index if its estimated_reward is the biggest
        invalid = self.env.filterInvalidAction(s0)
        for i in invalid:
            estimated_reward[i] = -float("inf")
        optimal_arm = np.argmax(estimated_reward)
        return optimal_arm

    def train(self):

        episode_reward = 0
        GHI_Data = read_energy_data(is_train=True)
        fr = 0

        for series in range(5):
            done = True
            ep_num = 0
            while ep_num < 10:
                if done:
                    # state = self.env.reset(is_train=True,simulation_start=0,simulation_end=100* 24, GHI_Data=GHI_Data)
                    state = self.env.reset(is_train=True, simulation_start=ep_num * 30 * 24,
                                           simulation_end=(ep_num + 1) * 30 * 24, GHI_Data=GHI_Data)
                fr += 1
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                # reward=self.gen_fake_reward(state,action)
                self.do_update(action, state, reward)
                state = next_state
                episode_reward += reward

                if done:
                    print('episode:{} rewards:{} '.format(ep_num, episode_reward))
                    self.save_model()
                    episode_reward = 0
                    ep_num += 1

    def save_model(self):
        data = [self.matrix_H, self.b]
        output = open('data/linucb_{}_{}.pkl'.format(str(self.config.lambda_r), str(self.config.tradeoff)), 'wb')
        pickle.dump(data, output)

    def load_model(self):
        output = open('data/linucb_{}_{}.pkl'.format(str(self.config.lambda_r), str(self.config.tradeoff)), 'rb')
        print('data/linucb_{}_{}.pkl'.format(str(self.config.lambda_r), str(self.config.tradeoff)))
        data = pickle.load(output)
        self.matrix_H = data[0]
        self.b = data[1]

    def do_update(self, arm, state, reward):
        # transfer the selection client into context
        context = self.uniform_state(state)
        self.matrix_H[arm] = self.matrix_H[arm] + context.dot(context.transpose())
        self.b[arm] = self.b[arm] + context * (reward)

    # function for testing purpose
    def gen_fake_reward(self, state, action):
        context = self.uniform_state(state)
        theta = np.resize(np.array([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4]) - action, (self.d, 1))
        # print(theta)
        reward = context.transpose().dot(theta)[0]
        return reward
