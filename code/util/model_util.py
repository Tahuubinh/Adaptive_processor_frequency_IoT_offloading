from schedule.value_based_rl.NAFA import NAFA_Agent
from schedule.value_based_rl.DoubleDQNPer import NAFAPER_Agent
from schedule.best_fit import best_fit
from schedule.linUCB import LinUCBAgent
from schedule.worst_fit import worst_fit


def trainModel(args, env):
    if args.method == "NAFA":
        agent = NAFA_Agent(args, env, num_series=2,
                           max_episode=2, ep_long=24)
        # agent.loadWeights("data/model_{}_{}.pkl".format(args.lambda_r,args.tradeoff))
        agent.train()
        # print("data/model_{}_{}_{}.pkl".format(args.lambda_r,args.tradeoff,str(args.trial)))
        agent.loadWeights(
            f"{args.link_project}/result/model_{args.lambda_r}_{args.tradeoff}_{args.trial}.pkl")
    elif args.method == "NAFAPER":
        agent = NAFAPER_Agent(args, env, num_series=2,
                              max_episode=2, ep_long=24)
        # agent.loadWeights("data/model_{}_{}.pkl".format(args.lambda_r,args.tradeoff))
        agent.train()
        # print("data/model_{}_{}_{}.pkl".format(args.lambda_r,args.tradeoff,str(args.trial)))
        agent.loadWeights(
            f"{args.link_project}/result/model_{args.lambda_r}_{args.tradeoff}_{args.trial}.pkl")
    elif args.method == "BF":  # best-fit
        agent = best_fit(env)
    elif args.method == "WF":  # worst-fit
        agent = worst_fit(env)
    elif args.method == "linUCB":
        agent = LinUCBAgent(args, env)
        agent.train()
        agent.load_model()
    return agent
