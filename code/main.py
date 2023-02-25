import numpy as np
import torch
from util.data_util import read_energy_data, save_data
from util.options import args_parser
from schedule.NAFA import NAFA_Agent
from schedule.best_fit import best_fit
from schedule.linUCB import LinUCBAgent
from schedule.worst_fit import worst_fit
from environment.comp_offload import CompOffloadingEnv


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device("cpu")
    print(args)
    env = CompOffloadingEnv(args)
    N_S = env.observation_space.shape[0]
    N_A = env.action_space.n
    immediate_reward = 0
    print(args.method)
    if args.method == "NAFA":
        agent = NAFA_Agent(args, env, num_series = 1, max_episode = 1, ep_long = 24)
        # agent.load_weights("data/model_{}_{}.pkl".format(args.lambda_r,args.tradeoff))
        agent.train()
        # print("data/model_{}_{}_{}.pkl".format(args.lambda_r,args.tradeoff,str(args.trial)))
        agent.load_weights("../result/model_{}_{}_{}.pkl".format(args.lambda_r, args.tradeoff, str(args.trial)))
    if args.method == "BF":  # best-fit
        agent = best_fit(env)
    if args.method == "WF":  # worst-fit
        agent = worst_fit(env)
    if args.method == "linUCB":
        agent = LinUCBAgent(args, env)
        agent.train()
        agent.load_model()
    episode = 0
    GHI_Data = read_energy_data(is_train=False)
    done = True
    accept = 0
    count_act = 0
    # while episode < 1:
    #     if done:
    #         s = env.reset(is_train=False, simulation_start=0, simulation_end=24, GHI_Data=GHI_Data)
    #     action = agent.act(s)
    #     if action != 0:
    #         accept += 1
    #     count_act += 1
    #     s, r, done = env.step(action)
    #     if done:
    #         print("average accept ratio of {}%".format(accept / count_act * 100))
    #         print("average reward of {}".format(np.mean(env.day_rewards)))
    #         episode += 1
    save_data(env, args)
