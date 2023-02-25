import numpy as np
import pickle
import json

def read_energy_data(is_train):
    if is_train:
        with open('../data/SoDa_HC3-METEO_lat0.329_lon32.499_2005-01-01_2005-12-31_1833724734.csv', 'r') as f:
            readlines = f.readlines()
            result = list(readlines)[32:]

    else:
        with open('../data/SoDa_HC3-METEO_lat0.329_lon32.499_2006-01-01_2006-12-31_1059619648.csv', 'r') as f:
            readlines = f.readlines()
            result = list(readlines)[32:]
    GHI = np.zeros(len(result))
    for i in range(len(GHI)):
        # Only get Clear-Sky 
        GHI[i] = int(result[i].split(";")[2])
    return GHI

def save_data(env, args):
    # output = open(
    #     'result/n_reject_low_power_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)), 'wb')
    # pickle.dump(env.n_reject_low_power, output)
    # output = open(
    #     'result/n_reject_conservation_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
    #     'wb')
    # pickle.dump(env.n_reject_conservation, output)
    # output = open(
    #     'result/n_reject_high_latency_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
    #     'wb')
    # pickle.dump(env.n_reject_high_latency, output)
    # output = open('result/total_latency_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
    #               'wb')
    # pickle.dump(env.total_latency, output)
    # output = open('result/n_total_request_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
    #               'wb')
    # pickle.dump(env.n_total_request, output)
    # output = open('result/day_rewards_{}_{}_{}.pkl'.format(str(args.method), str(args.lambda_r), str(args.tradeoff)),
    #               'wb')
    # pickle.dump(env.day_rewards, output)
    
    env_result = dict()
    env_result['n_reject_low_power'] = env.n_reject_low_power[0]
    env_result['n_reject_conservation'] = env.n_reject_conservation[0]
    env_result['n_reject_high_latency'] = env.n_reject_high_latency[0]
    env_result['total_latency'] = env.total_latency[0]
    env_result['n_total_request'] = env.n_total_request[0]
    env_result['day_rewards'] = env.day_rewards[0]
    with open("../result/env_result.json", "w") as outfile:
        json.dump(env_result, outfile)  
