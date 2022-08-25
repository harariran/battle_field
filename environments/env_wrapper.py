from abc import ABC, abstractmethod
from copy import deepcopy
from warnings import filterwarnings

from pettingzoo.magent import battlefield_v5, battle_v4

from agents import Agent
from utils.functions import CreateDecentralizedAgents


class EnvWrapper(ABC):

    def __init__(self, env, env_agent_ids, observation_spaces, action_spaces):
        self.env = env
        self.env_agent_ids = env_agent_ids
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_env(self):
        return self.env

    def get_env_agents(self):
        return self.env_agent_ids

    @abstractmethod
    def step(self, joint_action):
        pass

    @abstractmethod
    def observation_to_dict(self, obs):
        pass

    def render(self):
        pass

class EnvWrapperPZ(EnvWrapper):

    def __init__(self, env):

        # get action space of each agent
        action_spaces = {
            agent_id: env.action_space(agent_id) for agent_id in env.agents
        }

        # get observation space for each agent
        observation_spaces = {
            agent_id: env.observation_space(agent_id) for agent_id in env.agents
        }

        super().__init__(env, env.agents, observation_spaces, action_spaces)

    def step(self, joint_action):
        return self.env.step(joint_action)

    def observation_to_dict(self, obs):
        return obs

    def render(self):
        return self.env.render()


filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


# A Wrapper for the pettingzoo environment within MAC
class BattleFieldEnv(EnvWrapperPZ):
    def __init__(self, env):
        super().__init__(env)
        self.obs = self.env.reset()
        # print(self.obs)

    def step(self, joint_action):
        return self.env.step(joint_action)

    def observation_to_dict(self, obs):
        return obs

    def reset(self):
        return self.env.reset()

class BattleFieldSingleEnv():
    def __init__(self, env,blue_other_dm, red_DM, agent="blue_0"):
        self.env = env
        self.action_space = env.action_spaces[agent]
        self.observation_space = env.observation_spaces[agent]
        self.agent_name = agent
        self.agents = CreateDecentralizedAgents(env, blue_other_dm, red_DM)
        self.others_blue_DM = blue_other_dm
        self.others_red_DM = red_DM
        self.all_obs = self.env.reset()
        self.obs = self.all_obs[agent]
        self.metadata = None
        # print(self.obs)


    def step(self, action):
        joint_action = {}
        for agent_name in self.all_obs.keys():
            act = self.agents[agent_name].get_decision_maker().get_action(self.all_obs[agent_name])
            joint_action[agent_name] = act
        joint_action[self.agent_name] = action
        self.all_obs, reward, done, self.metadata = self.env.step(joint_action)
        try:
            self.obs = self.all_obs[self.agent_name]
        except:
            return (None, None, True, self.metadata)
        return (self.obs, reward[self.agent_name], done[self.agent_name], self.metadata)

    def observation_to_dict(self, obs):
        return obs

    def render(self):
        return self.env.render()

    def reset(self):
        self.all_obs = self.env.reset()
        self.obs = self.all_obs[self.agent_name]
        return self.obs

    def print_env_info(self):
        for i, agent in enumerate(self.env.agents, 1):
            print(f'- agent {i}: {agent}')
            print(f'\t- observation space: {self.env.observation_space(agent)}')
            print(f'\t- action space: {self.env.action_space(agent)}')
            print(f'\t- action space sample: {self.env.action_space(agent).sample()}')

#create ma-env


def CreateEnvironment(minimap=False):
    # Create and reset PettingZoo environment
    BF_env = battlefield_v5.parallel_env(map_size=46, minimap_mode=minimap, step_reward=-0.005, dead_penalty=-0.1,
                                         attack_penalty=-0.01, attack_opponent_reward=0.5, max_cycles=1000,
                                         extra_features=False)
    BF_env.reset()

    # Create a MAC from the PZ environment
    return BattleFieldEnv(BF_env)

def CreateEnvironment_Battle(minimap=False):
    # Create and reset PettingZoo environment
    BF_env = battle_v4.parallel_env(map_size=45, minimap_mode=minimap, step_reward=-0.005, dead_penalty=-0.1,
                                         attack_penalty=-0.01, attack_opponent_reward=0.5, max_cycles=1000,
                                         extra_features=False)
    BF_env.reset()

    # Create a MAC from the PZ environment
    return BattleFieldEnv(BF_env)


# Create multiple agents divided into two groups of different decision makers
def CreateDecentralizedAgents(env, blue_decision_maker, red_decision_maker):
    decentralized_blue_agents = {
        agent_id: Agent(deepcopy(blue_decision_maker))
        for agent_id in env.get_env_agents() if 'blue' in agent_id
    }
    # decentralized_blue_agents["blue_0"] = Agent(blue_decision_maker(env.action_spaces["blue_0"],10))
    # decentralized_blue_agents["blue_1"] = Agent(blue_decision_maker(env.action_spaces["blue_1"], 14))

    # decentralized_red_agents = {
    #     agent_id: Agent(red_decision_maker(env.action_spaces[agent_id]))
    #     for agent_id in mac_BF_env.get_env_agents() if 'red' in agent_id
    # }

    decentralized_red_agents = {
        agent_id: Agent(deepcopy(red_decision_maker))
        for agent_id in env.get_env_agents() if 'red' in agent_id
    }

    merged_dict = {**decentralized_blue_agents, **decentralized_red_agents}
    return merged_dict

#
# class EnvWrapperMultiTaxi(EnvWrapper):
#
#     def __init__(self, env):
#
#         # get action space of each agent
#         action_spaces = {
#             agent_id: env.action_space for agent_id in env.taxis_names
#         }
#
#         # get observation space for each agent
#         observation_spaces = {
#             agent_id: env.observation_space for agent_id in env.taxis_names
#         }
#
#         super().__init__(env, env.taxis_names, observation_spaces, action_spaces)
#
#     def step(self, joint_action):
#         return self.env.step(joint_action)
#
#     def observation_to_dict(self, obs):
#         return obs
#
#     def render(self):
#         return self.env.render()


