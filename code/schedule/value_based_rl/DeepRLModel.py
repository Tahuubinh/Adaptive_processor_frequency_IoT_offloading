import numpy as np
import torch
from copy import deepcopy
from util.data_util import read_energy_data
import random


class DeepRLModel:
    def __init__(self, args, env, num_series, max_episode, ep_long, **kwargs):
        self.args = args
        self.env = env
        self.max_episode = max_episode
        self.ep_long = ep_long
        self.num_series = num_series

    def findMaxAction(self, s0, q_values):
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

    # load weight for the Q network
    def loadWeights(self, model_path):
        if model_path is None:
            return
        self.model.load_state_dict(torch.load(model_path))

    def saveModel(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))

    # debug usage . check the input state.
    def checkInputState(self, input_s):
        if np.max(input_s) > 1 or np.min(input_s) < 0:
            return False
        else:
            return True

    # uniform the state to the scale of [0,1]
    def uniformState(self, s):
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
        assert self.checkInputState(input_s)
        input_s = torch.tensor(
            input_s, dtype=torch.float).to(self.args.device)
        return input_s

    def saveResults(self):
        self.saveModel(f'{self.args.link_project}/result/{self.args.save_folder}', str(self.args.lambda_r) + "_" + str(self.args.tradeoff) + "_" + str(
            self.args.trial))
        self.env.saveResults()
        print(f"Overall: {self.env.overall_results}")
        print(f"Action choices: {self.env.action_choices}")

    # epsilon-greedy action
    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = 0
        if random.random() > epsilon or not self.is_training:
            state_input = self.uniformState(state.reshape(1, len(state)))
            q_value = self.model.forward(state_input)
            action = self.findMaxAction(np.array([state]), q_value)[0]
        else:
            actions = self.env.getPossibleActionGivenState(state)
            action = np.random.choice(list(actions))
        return action

    def exploit(self, state):
        state_input = self.uniformState(state.reshape(1, len(state)))
        q_value = self.model.forward(state_input)
        action = self.findMaxAction(np.array([state]), q_value)[0]
        return action

    def remember(self, state, action, reward, next_state, done):
        pass

    def train(self):
        losses = []
        # all_rewards = []
        # counters = []
        fr = 0
        # episode_reward = 0
        GHI_Data = read_energy_data(is_train=True)
        epsilon = None
        for series in range(self.num_series):
            ep_num = 0
            self.env.replay(is_train=True, simulation_start=ep_num * self.ep_long,
                            simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)
            print(f'SERIES {series}')
            while ep_num < self.max_episode:
                state = self.env.reset(is_train=True, simulation_start=ep_num * self.ep_long,
                                       simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)
                done = False
                while not done:
                    fr += 1
                    epsilon = self.epsilon_by_frame(fr)
                    action = self.act(state, epsilon)
                    next_state, reward, done = self.env.step(action)
                    self.remember(state, action, reward, next_state, False)
                    # self.buffer.add(state, action, reward, next_state, False)
                    # episode_reward += reward
                    # all_rewards.append(reward)
                    # counters.append(self.env.counter)
                    if self.buffer.size() > self.args.batch_size:
                        loss = self.learning(fr)
                        # losses.append(loss)
                    # else:
                    #     losses.append(0)

                    state = next_state

                print(f'Episode {ep_num}')
                # print('Episode: {}\nrewards: {}  epsilon: {} losses: {}'.format(ep_num, episode_reward, epsilon,
                #                                                                 np.sum(losses[-100:]) / 100))
                self.saveResults()
                episode_reward = 0
                ep_num += 1
                # self.env.event_queue.print_queue()

    def test(self):
        GHI_Data = read_energy_data(is_train=True)
        done = False
        ep_num = 0
        self.env.replay(is_train=False, simulation_start=ep_num * self.ep_long,
                        simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)

        print('\n\n\n--------------------------------------------------')
        while ep_num < self.max_episode:
            state = self.env.reset(is_train=True, simulation_start=ep_num * self.ep_long,
                                   simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done = self.env.step(action)
                state = next_state

            print(f'Episode test {ep_num}')
            self.saveResults()
            ep_num += 1
            # self.env.event_queue.print_queue()
