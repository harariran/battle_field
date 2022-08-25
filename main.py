#imports
from DMs.simple_planner import Simple_DM
from DMs.simple_planner2 import Simple_DM2
from DMs.simple_DMs import Stay_DM, Do_action_DM
import time

## Main
# from environments.env_wrapper import BattleFieldSingleEnv
from environments.env_wrapper import BattleFieldSingleEnv, CreateEnvironment, CreateEnvironment_Battle

if __name__ == '__main__':
    '''
    test single agent in a scenerio
    '''

    # env = CreateEnvironment()
    env = CreateEnvironment_Battle()
    agent = "blue_80"
    action_space = env.action_spaces[agent]


    temp_env = BattleFieldSingleEnv(env, Simple_DM2(action_space,0.5), Simple_DM(action_space,0.5,red_team=True), agent)
    # temp_env = BattleFieldSingleEnv(env, Stay_DM(action_space,6), Stay_DM(action_space,6), agent)


    obs = temp_env.reset()

    simple_dm = Stay_DM(temp_env.action_space,12)

    total_reward = 0
    for i in range(5000):
        a = simple_dm.get_action(obs)
        obs,rew,done,_ = temp_env.step(a)
        if done:
            break
        temp_env.render()
        total_reward+=rew
        print(f"action: {a}, reward: {rew}, total rew: {total_reward}")
        time.sleep(0.2)
        temp_env.render()

    print(f" total rew: {total_reward}")





    # mac_BF_env = CreateEnvironment()

    # CreateCentralizedController(mac_BF_env, CreateRandomAgent(mac_BF_env))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedIdenticalAgents(mac_BF_env, RandomDecisionMaker))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedAgents(mac_BF_env, Stay_DM , Stay_DM))

    # CreateDecentralizedController(mac_BF_env, CreateDecentralizedAgents(mac_BF_env, RandomDecisionMaker, RandomDecisionMaker))


    # GDM = GreedyDecisionMaker(mac_BF_env)

    # GDM.get_action()
