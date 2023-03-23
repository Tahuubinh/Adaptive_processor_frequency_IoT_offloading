import numpy as np
import torch
from copy import deepcopy


class DeepModel:
    def __init__(self, args, env, num_series, max_episode, ep_long, **kwargs):
        self.args = args
        self.env = env
        self.max_episode = max_episode
        self.ep_long = ep_long
        self.num_series = num_series

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

    # load weight for the Q network
    def loadWeights(self, model_path):
        if model_path is None:
            return
        self.model.load_state_dict(torch.load(model_path))

    def saveModel(self, output, tag=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, tag))

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
        input_s = torch.tensor(
            input_s, dtype=torch.float).to(self.args.device)
        return input_s
