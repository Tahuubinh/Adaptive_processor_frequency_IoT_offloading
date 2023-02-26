from copy import deepcopy
import gym
from gym import spaces
import numpy as np
import pandas as pd
from environment.event import Event, EventQueue
import json


# Training/testing environment for adaptive frequency adjustment in computation offloading scenario
class CompOffloadingEnv(gym.Env):

    def __init__(self, args, task_data):
        with open(task_data, 'r') as f:
            readlines = f.readlines()
            # get generation info
            self.gentask_info = list(readlines)[0].replace(";", "," )
            self.gentask_info = json.loads('{' + self.gentask_info[:-1] + '}')
            # get task info
            # self.datatask = list(readlines)[2:]
            
        self.datatask = pd.read_csv(task_data, skiprows=[0])
        self.kappa = 1e-28
        self.complexity = self.gentask_info['complexity']
        self.avg_data_size = self.gentask_info['avg_data_size']  # 20MB
        self.battery_size = 1e6
        self.core_number = 12
        self.frequency_set = np.array([2e9, 3e9, 4e9])
        self.lambda_request = args.lambda_r
        # 24-hour-scale local time, battery status, energy reservation status, [running CPU cores in the frequency], data_size
        low = np.concatenate([[0], [0], [0], np.zeros(len(self.frequency_set)), [self.avg_data_size - 10 * 8 * 1e6]])
        high = np.concatenate(
            [[24], [self.battery_size], [self.battery_size], self.core_number * np.zeros(len(self.frequency_set)),
             [self.avg_data_size + 10 * 8 * 1e6]])
        self.action_space = spaces.Discrete(len(self.frequency_set) + 1)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.panel_size = args.panel_size
        self.args = args

    def getRequest(self):
        event_tasks = self.datatask[self.datatask['arrival_time'] >= self.simulation_start]
        event_tasks = event_tasks[event_tasks['arrival_time'] < self.simulation_end]
        #self.remain_task = event_tasks.shape[0]
        #print(self.datatask[self.datatask['arrival_time'] >= self.simulation_end].head(1))
        event_tasks = pd.concat([event_tasks, 
                                 self.datatask[self.datatask['arrival_time'] >= self.simulation_end].head(1)])
        for index, row in event_tasks.iterrows():
            self.event_queue.push(Event(row['type'], row['arrival_time'], row['data_size']))
        
    # def getRequest(self):
    #     _lambda = self.lambda_request
    #     _arrival_time = self.simulation_start
    #     while _arrival_time < self.simulation_end:
    #         # Plug it into the inverse of the CDF of Exponential(_lamnbda)
    #         _inter_arrival_time = np.random.exponential(1 / _lambda, size=1)[0]
    #         # Add the inter-arrival time to the running sum
    #         _arrival_time = _arrival_time + _inter_arrival_time
    #         datasize = np.random.uniform(self.avg_data_size - 10 * 8 * 1e6, self.avg_data_size + 10 * 8 * 1e6)
    #         self.event_queue.push(Event(0, _arrival_time, datasize))
    #         print(Event(0, _arrival_time, datasize))

    def calculateTaskFinish(self, target_core, data_size, action):
        assert action >= 1
        # floor the frequency in case of exceeding reservation energy
        frequency = self.frequency_set[action - 1]
        process_time = data_size * self.complexity / (frequency * 3600)
        # scope of target core [1, core_number], scope of task_finish event name [2, core_number+1]
        task_finish_time = self.counter + process_time
        # print(target_core, task_finish_time, action)
        self.event_queue.push(Event(target_core + 2, task_finish_time, action))
        

    # generate the event when energy consumption change
    def calculateEnergyProduceChange(self, GHI_data):
        for i in range(self.simulation_start, self.simulation_end):
            # energy produce in 1 hour
            energy_produce_rate = 3600 * self.panel_size * GHI_data[i] / 1
            self.event_queue.push(Event(1, i, energy_produce_rate))

    def updatePostActionStatus(self):

        # do not break the iteration until a request comes!
        while True:
            next_event = self.event_queue.pop()
            # update the battery status
            core_power = np.sum(self.kappa * self.core_frequency ** 3 * 3600)
            self.battery_status = max(self.battery_status - core_power * (next_event.arrival_time - self.counter), 0)
            self.battery_status = min(
                self.battery_status + self.energy_produce_rate * (next_event.arrival_time - self.counter),
                self.battery_size)
            # unlock the reservation energy
            self.reservation_status = max(
                self.reservation_status - core_power * (next_event.arrival_time - self.counter), 0)

            if next_event.name == 1:
                # update the energy produce rate if the arrived event is energy produce rate change
                new_energy_produce_rate = next_event.extra_msg
                self.energy_produce_rate = new_energy_produce_rate

            # task finish
            if 2 <= next_event.name < 2 + self.core_number:
                core = (next_event.name - 2)
                # make core into sleeping status
                action = next_event.extra_msg
                self.core_frequency[core] = 0
                self.running_instance[action - 1] -= 1
            self.counter = next_event.arrival_time
            self.event = next_event
            # print(next_event)
            if self.event.name == 0:
                # assert next_event.extra_msg > 25
                break
            
        assert self.event.name == 0
        return

    # perform action in the training environment
    def step(self, action):
        # other inner events cannot be exposed to outside
        assert self.event.name == 0
        possible_actions = self.getPossibleActionGivenState(self.convertCurrentStatusToState())
        assert action in possible_actions, "action: '{}' is invalid in state '{}'".format(action,
                                                                                          self.convertCurrentStatusToState())
        # update the statistic record data
        if (self.counter - self.simulation_start) > 24 * self.day:
            self.day += 1
        self.n_total_request[self.day - 1] += 1

        data_size = self.event.extra_msg

        if action == 0:  # the request is rejected
            reward = 0
            # rejection resulted from energy overbooked
            least_frequency = np.min(self.frequency_set)
            least_reserved_energy = self.kappa * least_frequency ** 3 * (data_size * self.complexity / least_frequency)
            if self.reservation_status + least_reserved_energy > self.battery_status:
                self.n_reject_low_power[self.day - 1] += 1
            else:
                # no core is free,then it must be rejected by overloaded
                if np.sum(self.running_instance) == self.core_number:
                    self.n_reject_high_latency[self.day - 1] += 1
                # there are other possible actions, it must be rejected due to conservation
                else:
                    assert len(possible_actions) != 0
                    self.n_reject_conservation[self.day - 1] += 1
        else:

            frequency = self.frequency_set[action - 1]
            process_time = data_size * self.complexity / (frequency * 3600)

            reward = 1 - (self.args.tradeoff * process_time)

            # print(queue_status[target-1])
            self.total_latency[self.day - 1] += process_time
            self.reservation_status += self.kappa * frequency ** 3 * data_size * self.complexity / frequency
            self.running_instance[action - 1] += 1
            # find the first sleeping core
            for i in range(len(self.core_frequency)):
                if self.core_frequency[i] == 0:
                    target_core = i
                    break
            self.core_frequency[target_core] = frequency
            self.calculateTaskFinish(target_core, data_size, action)

        self.day_rewards[self.day - 1] += reward
        self.updatePostActionStatus()
        #self.remain_task -= 1
        # print(self.remain_task)
        # if self.remain_task == 1:
        
        if self.counter > self.simulation_end:
            is_terminal = True
        else:
            is_terminal = False

        assert self.event.name == 0
        return self.convertCurrentStatusToState(), reward, is_terminal

    # reset the current environment status
    def reset(self, is_train, simulation_start, simulation_end, GHI_Data):
        if not is_train:
            np.random.seed(0)
        self.simulation_start = simulation_start
        self.simulation_end = simulation_end
        # initialize counter
        self.counter = simulation_start
        # initialize produce_rate=0
        self.energy_produce_rate = 0
        # initialize core frequency
        self.core_frequency = np.zeros(self.core_number)
        # initialize energy reservation queue
        self.reservation_status = 0
        # initialize battery_status
        self.battery_status = 0
        # initialize frequency type
        self.running_instance = np.zeros(len(self.frequency_set))

        ###########################
        # statistic data for recording purpose
        day_num = int((simulation_end - simulation_start) / 24)
        self.total_latency = np.zeros([day_num])
        self.n_reject_conservation = np.zeros([day_num])
        self.n_total_request = np.zeros([day_num])
        self.n_reject_low_power = np.zeros([day_num])
        self.n_reject_high_latency = np.zeros([day_num])
        self.day_rewards = np.zeros([day_num])
        self.day = 0
        ###########################
        # initialize request and energy event
        self.event_queue = EventQueue()
        self.getRequest()
        self.calculateEnergyProduceChange(GHI_Data)
        # find the first request arrival event
        self.updatePostActionStatus()
        return self.convertCurrentStatusToState()

    # transfer current environment status to state
    def convertCurrentStatusToState(self):
        # only usable for event 0. state is meaningful only when event=0
        event = self.event
        assert self.event.name == 0
        state = np.concatenate(
            [[self.counter % 24], [self.battery_status], [self.reservation_status], self.running_instance,
             [event.extra_msg]])
        return deepcopy(state)

    # filter invalid action given state
    def filterInvalidAction(self, state):
        battery_status = state[1]
        reservation_status = state[2]
        running_instance = state[3:-1]
        data_size = state[-1]
        actions = set()
        for frequency_index, frequency in enumerate(self.frequency_set):
            needed_reserve_energy = self.kappa * frequency ** 3 * data_size * self.complexity / frequency
            if reservation_status + needed_reserve_energy > battery_status:
                actions.add(frequency_index + 1)
            if np.sum(running_instance) == self.core_number:
                actions.add(frequency_index + 1)
        return actions

    def getPossibleActionGivenState(self, state):
        invalid = self.filterInvalidAction(state)
        possible_action = set()
        possible_action.add(0)
        for frequency_index in range(1, len(self.frequency_set) + 1):
            possible_action.add(frequency_index)
        return possible_action - invalid
