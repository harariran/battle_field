# from src.agents.agent import Agent, DecisionMaker
# from src.environments.env_wrapper import*
# 'Environment Related Imports'
# import tqdm
# import gym
import PIL
import matplotlib.pyplot as plt
import time

from gym import Wrapper
from gym.spaces import MultiDiscrete, Box, Discrete

# 'Deep Model Related Imports'
from torch.nn.functional import one_hot
import torch
import numpy as np
from stable_baselines3 import DQN, PPO

from environments.env_wrapper import BattleFieldSingleEnv,CreateEnvironment, CreateEnvironment_Battle, BattleFieldHighLevelEnv
from DMs.simple_planner import Simple_DM
from utils.functions import CreateDecentralizedAgents, CreateCentralizedController, \
    CreateDecentralizedController

# del model # remove to demonstrate saving and loading
#
# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()
from DMs.simple_DMs import Do_action_DM, DecisionMaker, Stay_DM


class PPO_HL_DecisionMaker(DecisionMaker):
    def __init__(self, action_space=Discrete(3), model_name="HL_PPO3"):
        self.space = action_space
        self.model_file_name = model_name
        try:
            self.model = PPO.load(self.model_file_name)
        except:
            self.train_model()
        # print(f"{self.model.observation_space}")


        # print(f"{self.obs_size}")
        # for obs in self.model.observation_space:
        #     print(f"obs size: {obs}")
        #     print(f"obs size: { obs.n }")

# fix observation space if not fit to trained dimension
    def fit_obs(self, obs):
        temp = [obj for obj in obs[:(self.obs_size-1)]]
        temp.append(obs[-1])
        print(f"{temp}")
        return temp

    def get_action(self, observation):
        action, _states = self.model.predict(observation, deterministic=True)
        return action


    def retrain(self,env):
        self.new_model_name = "HL_PPO3"
        self.model.set_env(env)
        self.model.learn(total_timesteps=150000, n_eval_episodes=200)
        self.model.save(self.new_model_name)



    def train_model(self):
        env = CreateEnvironment_Battle()
        agent = "blue_0"
        action_space = env.action_spaces[agent]

        # temp_env = BattleFieldHighLevelEnv(env, Stay_DM(action_space), Stay_DM(action_space), agent)
        temp_env = BattleFieldHighLevelEnv(env, Simple_DM(action_space),
                                           Simple_DM(action_space, 0.5, red_team=True), agent)

        obs = temp_env.reset()
        # for i in range(20):
        #     single_env.step(15)
        #     single_env.render()
        #     time.sleep(0.2)

        # temp_env.render()

        model = PPO("MlpPolicy", temp_env, verbose=1)
        model.learn(total_timesteps=150000,n_eval_episodes=200)
        self.model = model
        self.model.save(self.model_file_name)


class PPO_Low_level_DecisionMaker(DecisionMaker):
    def __init__(self, action_space, red_team=False, model_name="HL_PPO3"):
        self.space = action_space
        self.red_team=red_team
        try:
            self.HL_model = PPO_HL_DecisionMaker(model_name)
        except:
            print("no trained model - train it first")
        self.LL_DM = Simple_DM(action_space,0.5,red_team=red_team)

        # print(f"{self.obs_size}")
        # for obs in self.model.observation_space:
        #     print(f"obs size: {obs}")
        #     print(f"obs size: { obs.n }")

    def get_action(self, observation):
        HL_action = self.HL_model.get_action(observation)
        action = self.LL_DM.get_Low_level_action(observation,HL_action)
        return action




if __name__ == '__main__':
    # check code:
    env = CreateEnvironment_Battle()
    agent = "blue_0"
    action_space = env.action_spaces[agent]

    single_env = BattleFieldHighLevelEnv(env, Simple_DM(action_space), Simple_DM(action_space,0.5,red_team=True), agent)
    # single_env = BattleFieldHighLevelEnv(env, Stay_DM(action_space), Stay_DM(action_space), agent)




    # obs = (obs[a_n] for a_n in agents)
    # im = from_RGBarray_to_image(obs)
    D_M = PPO_HL_DecisionMaker(single_env.action_space)
    #single_env.reset()
    # D_M.retrain(single_env)

    obs = single_env.reset()
    total_reward = 0
    # a = 1
    try:
        for m in range(20):
            for i in range(10):
                next_a = D_M.get_action(obs)
                # next_a = a % 3
                obs, rew, done, _ = single_env.step(next_a)
                total_reward += rew
                if done==True:
                    print(f"action: {next_a}, reward: {rew}, total rew: {total_reward}")
                    raise StopIteration
                print(f"action: {next_a}, reward: {rew}, total rew: {total_reward}, new_life {obs[6][6][2]}")
                single_env.render()
                time.sleep(0.5)
                # a -=1
    except:
        print ("agent has died or game over")



    # print(f"obs:{obs}")
    # print(f"next action: { env.index_action_dictionary[D_M.get_action(obs)]}")



