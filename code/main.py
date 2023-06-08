import os
import numpy as np
import torch
from util.data_util import read_energy_data, save_data
from util.options import args_parser
from environment.comp_offload import CompOffloadingEnv
from util.model_util import trainModel


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cpu")
    args.method = 'BR'
    args.tradeoff = 6
    args.save_folder = args.method

    path = f'{args.link_project}/result/{args.save_folder}'
    if not os.path.exists(path):
        os.makedirs(path)
    print(f"The new directory is created: {path}")

    env = CompOffloadingEnv(
        args, task_data=f'{args.link_project}/data/task_data/data_30_160000000_366_80000000_20000_ver_1/train.csv')
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n
    immediate_reward = 0
    agent = trainModel(args, env, num_series=2, max_episode=2, eplong=24)
    episode = 0
    GHI_Data = read_energy_data(is_train=False)
    done = True
    accept = 0
    count_act = 0
    agent.test()

    # while episode < 1:
    #     if done:
    #         s = env.reset(is_train=False, simulation_start=0,
    #                       simulation_end=300 * 24, GHI_Data=GHI_Data)
    #     action = agent.act(s)
    #     if action != 0:
    #         accept += 1
    #     s, r, done = env.step(action)
    #     if done:
    #         print("average accept ratio of {}".format(accept))
    #         print("average reward{}".format(np.mean(env.day_rewards)))
    #         episode += 1

    # save_data(env, args)
