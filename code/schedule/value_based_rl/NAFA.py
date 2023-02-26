import random
from copy import deepcopy
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
import math
from util.data_util import read_energy_data
from schedule.value_based_rl.buffer.experience_replay import ExperienceReplay


# network architecture
class DQN(nn.Module):
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, actions_dim)
        )

    def forward(self, x):
        return self.nn(x)


class NAFA_Agent:
    def __init__(self, options, env, num_series, max_episode, ep_long):
        self.config = options
        self.is_training = True
        self.buffer = ExperienceReplay(self.config.max_buff)
        self.action_dim = env.action_space.n
        self.model = DQN(env.observation_space.shape[0], self.action_dim).to(self.config.device)
        self.target_model = DQN(env.observation_space.shape[0], self.action_dim).to(self.config.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.env = env
        self.max_episode = max_episode
        self.ep_long = ep_long
        self.num_series = num_series

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)
        
    def find_max_action(self, s0, q_values):
        actions = []
        q_values = q_values.cpu().detach().numpy()
        q_values = deepcopy(q_values)
        for row in range(len(q_values)):
            invalid = self.env.filterInvalidAction(s0[row])
            for i in invalid:
                q_values[row, i] = -float("inf")
            action = np.argmax(q_values[row])
            actions.append(action)
        return actions

    # epsilon-greedy action
    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = 0
        if random.random() > epsilon or not self.is_training:
            state_input = self.uniform_state(state.reshape(1, len(state)))
            q_value = self.model.forward(state_input)
            action = self.find_max_action(np.array([state]), q_value)[0]
        else:
            actions = self.env.getPossibleActionGivenState(state)
            action = np.random.choice(list(actions))
        return action

    # back-propagation
    def learning(self, fr):
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)
        # print(s0[0], a[0])
        # print(len(s0), len(a))
        # print(type(s0))
        # print(type(s0[0]))
        # assert 2 == 3
        r = torch.tensor(r, dtype=torch.float).to(self.config.device)
        s0_input = self.uniform_state(s0)
        s1_input = self.uniform_state(s1)
        a = torch.tensor(a, dtype=torch.long).to(self.config.device)
        # done = torch.tensor(done, dtype=torch.float).to(self.config.device)
        q_values = self.model(s0_input)
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(s1_input)
        max_q_action = self.find_max_action(s1, next_q_values)
        max_q_action = torch.tensor(max_q_action).to(self.config.device)

        next_q_state_values = self.target_model(s1_input)

        next_q_value = next_q_state_values.gather(1, max_q_action.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.discount * next_q_value
        # Notice that we need to detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    # NAFA's training stage
    def train(self):
        losses = []
        all_rewards = []
        counters = []
        fr = 0
        episode_reward = 0
        GHI_Data = read_energy_data(is_train=True)
        for series in range(self.num_series):
            done = True
            ep_num = 0
            print(f'SERIES {series}')
            while ep_num < self.max_episode:
                if done:
                    state = self.env.reset(is_train=True, simulation_start=ep_num * self.ep_long,
                                           simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)
                fr += 1
                epsilon = self.epsilon_by_frame(fr)
                action = self.act(state, epsilon)
                next_state, reward, done = self.env.step(action)
                self.buffer.add(state, action, reward, next_state, False)
                episode_reward += reward
                all_rewards.append(reward)
                counters.append(self.env.counter)
                if self.buffer.size() > self.config.batch_size:
                    loss = self.learning(fr)
                    losses.append(loss)
                else:
                    losses.append(0)
                
                # print(self.config)
                # print(self.config.print_interval)
                # print(done)
                # break
                # if fr % self.config.print_interval == 0:
                #     print("PERIOD: %d\nrewards: %f, losses: %f, episode: %d" % (
                #         self.env.counter / 24, episode_reward / ((self.env.counter - self.env.simulation_start) / 24),
                #         np.sum(losses[-100:]) / 100, ep_num))
                #     print("epsilon = {}".format(epsilon))
                state = next_state
                if done:
                    print('Episode: {}\nrewards: {}  epsilon: {} losses: {}'.format(ep_num, episode_reward, epsilon,
                                                                              np.sum(losses[-100:]) / 100))
                    self.saveModel("../result", str(self.config.lambda_r) + "_" + str(self.config.tradeoff) + "_" + str(
                        self.config.trial))
                    episode_reward = 0
                    ep_num += 1
                    # self.env.event_queue.print_queue()

            # self.save_data(all_rewards,counters,loss)

    # debug usage . check the input state.
    def check_input_state(self, input_s):
        if np.max(input_s) > 1 or np.min(input_s) < 0:
            return False
        else:
            return True

    # uniform the state to the scale of [0,1]
    def uniform_state(self, s):
        # len of s can be 1 or batch_size
        input_s = deepcopy(s)
        # 24-hour-scale local time
        input_s[:, 0] = (input_s[:, 0] - 0) / (24 - 0)
        # battery status
        input_s[:, 1] = (input_s[:, 1] - 0) / (self.env.battery_size - 0)
        # energy reservation status
        input_s[:, 2] = (input_s[:, 2] - 0) / (self.env.battery_size - 0)
        # running CPU cores in the frequency
        input_s[:, 3:-1] = (input_s[:, 3:-1] - 0) / (self.env.core_number - 0)
        # data size
        input_s[:, -1] = (input_s[:, -1] - (self.env.avg_data_size - 10 * 8 * 1e6)) / (
                (self.env.avg_data_size + 10 * 8 * 1e6) - (self.env.avg_data_size - 10 * 8 * 1e6))
        assert self.check_input_state(input_s)
        input_s = torch.tensor(input_s, dtype=torch.float).to(self.config.device)
        return input_s

    # load weight for the Q network
    def loadWeights(self, model_path):
        if model_path is None: return
        self.model.load_state_dict(torch.load(model_path))

    # def save_data(self,rewards,counters,loss):
    #     output = open('data/tr_rewards_{}_{}.pkl'.format(str(self.config.lambda_r),str(self.config.tradeoff)) , 'wb')
    #     pickle.dump(rewards,output)
    #     output = open('data/tr_counters_{}_{}.pkl'.format(str(self.config.lambda_r), str(self.config.tradeoff)), 'wb')
    #     pickle.dump(counters, output)
    #     output = open('data/tr_loss_{}_{}.pkl'.format(str(self.config.lambda_r), str(self.config.tradeoff)), 'wb')
    #     pickle.dump(loss,output)

    def saveModel(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))
