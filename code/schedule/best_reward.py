from util.data_util import read_energy_data

# best fit algorithm, which consistently choose the least frequency if applicable.


class bestReward():
    def __init__(self, env, max_episode, ep_long):
        self.env = env
        self.last_deploy_core = 0
        self.max_episode = max_episode
        self.ep_long = ep_long

    def saveResults(self):
        self.env.saveResults()
        print(f"Overall: {self.env.overall_results}")
        print(f"Action choices: {self.env.action_choices}")

    def act(self, s):
        possible_actions = self.env.getPossibleActionGivenState(s)
        action = 0
        max_value = -100
        for test_action in possible_actions:
            if test_action == 0:
                continue
            test_value = self.env.calculateRewardOnAction(test_action)
            # print(test_value)
            if max_value < test_value:
                max_value = test_value
                action = test_action

        return action

    def test(self):
        GHI_Data = read_energy_data(is_train=False)
        done = False
        ep_num = 0
        self.env.replay(is_train=False, simulation_start=ep_num * self.ep_long,
                        simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)

        print('\n\n\n--------------------------------------------------')
        while ep_num < self.max_episode:
            state = self.env.reset(is_train=False, simulation_start=ep_num * self.ep_long,
                                   simulation_end=(ep_num + 1) * self.ep_long, GHI_Data=GHI_Data)
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done = self.env.step(action)
                state = next_state

            print(f'Episode test {ep_num}')
            self.saveResults()
            ep_num += 1
            # self.env.event_queue.print_queue()
