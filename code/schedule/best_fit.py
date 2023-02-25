# best fit algorithm, which consistently choose the least frequency if applicable.
class best_fit():
    def __init__(self, env):
        self.env = env
        self.last_deploy_core = 0

    def act(self, s):
        possible_actions = self.env.possible_action_given_state(s)
        accept = False
        for frequency_index in range(len(self.env.frequency_set)):
            action = frequency_index + 1
            if action in possible_actions:
                accept = True
                break
        if not accept:
            action = 0
        return action
